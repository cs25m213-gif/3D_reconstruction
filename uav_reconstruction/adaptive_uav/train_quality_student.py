"""Train QualityStudent with the whole packed dataset resident in GPU memory.

Batches are assembled by tensor indexing on the GPU (no CPU data loading), so the GPU stays busy.
Losses: the existing student depth loss (LiDAR > neighbour LiDAR > map > anchors > teacher, plus
photometric/smoothness), a Laplace NLL for the per-pixel log-depth error scale, and Gaussian NLLs
for the two frame-quality exits. Truth for the quality targets is the current depth error against
LiDAR (or map depth on flights whose LiDAR is offset).

Generalisation controls:
  - areas are sampled equally (AMtown has LiDAR and many clips; HKairport has map depth only)
  - one source is dropped at random (streaming often has one or zero good past sources)
  - photometric augmentation shared by all views of a sample
  - synthetic degradations of the reference frame (blur, motion blur, exposure, noise, low resolution,
    haze); quality targets stay the real depth error, so the heads learn which frames not to select
  - checkpoints are chosen on held-out TRAINING flights (default AMtown03, HKairport03); AMvalley
    (LiDAR) and HKisland (map, never trained on) are reported every epoch but never used to choose.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .quality_student import QualityStudent
from .quality_teacher import QualityTeacher
from .frame_degradations import KINDS, degrade, random_degradation
from .student_losses import DEFAULT_WEIGHTS, student_loss
from .utility_contracts import write_json

LOW, HIGH = .25, 4.


class GpuPack:
    def __init__(self, folder: Path, device):
        self.meta = json.loads((folder / 'clips.json').read_text())
        load = lambda k: torch.from_numpy(np.load(folder / f'{k}.npy'))
        self.rgb = load('rgb').to(device)                                  # uint8 [N,3,H,W]
        self.K, self.P = load('K').to(device), load('P').to(device)        # float32, float64
        for key in ('teacher', 'anchor', 'measured', 'neighbour', 'map'):
            setattr(self, key, load(key).to(device))                      # float16 [N,H,W]
        self.teacher_ok = load('teacher_ok').to(device).bool()
        self.scale, self.src = load('scale').to(device), load('src').to(device).long()
        self.clip = load('clip').to(device).long()
        self.n = len(self.scale)
        self.area = [m['area'] for m in self.meta]
        self.flight = [m['scene'] for m in self.meta]
        self.role = [m['role'] for m in self.meta]

    def frames(self, predicate) -> torch.Tensor:
        """Frames of clips matching predicate(meta) that have at least one past source."""
        clips = torch.tensor([i for i, m in enumerate(self.meta) if predicate(m)], device=self.clip.device)
        keep = torch.isin(self.clip, clips) & (self.src[:, 0] >= 0)
        return keep.nonzero().squeeze(1)

    def batch(self, idx: torch.Tensor, drop: float = 0., augment: bool = False, n_sources: int = 2,
              degrade_prob: float = 0., kind=None, strength=None) -> dict:
        """n_sources=2: the two baseline-chosen past sources. n_sources=4: plus the two previous frames.
        Missing sources repeat the first source with weight 0."""
        b = len(idx)
        s1 = self.src[idx, 0]
        cand = [self.src[idx, 0], self.src[idx, 1]]
        if n_sources > 2:
            for lag in range(1, n_sources - 1):
                prev = (idx - lag).clamp_min(0)
                same = (idx - lag >= 0) & (self.clip[prev] == self.clip[idx])
                cand.append(torch.where(same, prev, torch.full_like(prev, -1)))
        sources = torch.stack(cand, 1)
        weights = (sources >= 0).float()
        sources = torch.where(sources >= 0, sources, s1[:, None])
        if drop > 0 and n_sources == 2:  # drop one of two available sources
            both = weights.sum(1) == 2
            hit = both & (torch.rand(b, device=idx.device) < drop)
            which = torch.randint(0, 2, (b,), device=idx.device)
            weights[hit, which[hit]] = 0.
        elif drop > 0:  # drop each source independently, always keeping at least one
            kept = weights * (torch.rand_like(weights) >= drop).float()
            empty = kept.sum(1) == 0
            first = torch.argmax(weights, 1)
            kept[empty, first[empty]] = 1.
            weights = kept
        views = torch.cat((idx[:, None], sources), 1)
        scale = self.scale[idx].double()
        P = self.P[views]                                               # [B,3,4,4] float64
        P = P @ torch.linalg.inv(P[:, :1])                              # relative to the reference (precision)
        P[:, :, :3, 3] /= scale[:, None, None]
        rgb = self.rgb[views].float() / 255.
        if augment:
            g = torch.rand(b, 3, device=idx.device)
            rgb = (rgb * (.75 + .5 * g[:, 0, None, None, None, None]) + (-.08 + .16 * g[:, 1, None, None, None, None])).clamp(0, 1)
            rgb = rgb ** (.8 + .4 * g[:, 2, None, None, None, None])
        if kind is None and degrade_prob > 0:
            kind, strength = random_degradation(b, degrade_prob, idx.device)
        if kind is not None:  # degrade the reference frame only: its own quality is what is judged
            rgb = rgb.clone()
            rgb[:, 0] = degrade(rgb[:, 0], kind, strength)
        sc = self.scale[idx].float()[:, None, None, None]
        out = {'rgb': rgb, 'K': self.K[views], 'P': P.float(), 'weights': weights,
               'degrade_kind': kind if kind is not None else torch.full((b,), -1, device=idx.device)}
        for key in ('teacher', 'anchor', 'measured', 'neighbour', 'map'):
            out[key] = getattr(self, key)[idx].float()[:, None] / sc
        for key in ('measured', 'neighbour', 'map'):
            out[key + '_valid'] = (out[key] > LOW) & (out[key] < HIGH)
        out['teacher_valid'] = self.teacher_ok[idx][:, None] & (out['teacher'] > LOW) & (out['teacher'] < HIGH)
        return out


def truth(batch: dict):
    m, n, p = batch['measured_valid'], batch['neighbour_valid'], batch['map_valid']
    known = m | n | p
    best = torch.where(m, batch['measured'], torch.where(n, batch['neighbour'], batch['map']))
    return known, best


def quality_losses(batch: dict, out: dict, min_pixels: int = 200):
    known, best = truth(batch)
    error = (out['depth'].detach().clamp_min(1e-4).log() - best.clamp_min(1e-4).log()).abs()
    s = out['log_scale']
    pixel = (error * torch.exp(-s) + s)[known].mean() if known.any() else s.sum() * 0
    count = known.flatten(1).sum(1)
    masked = torch.where(known, error, torch.full_like(error, float('nan'))).flatten(1)
    y = torch.log(masked.nanmedian(1).values + 1e-3)
    ok = count >= min_pixels
    parts = {'pixel_nll': pixel}
    for name in ('q1', 'q2'):
        mu, logvar = out[name]
        parts[name] = (.5 * (torch.exp(-logvar) * (y - mu) ** 2 + logvar))[ok].mean() if ok.any() else mu.sum() * 0
    return parts, y, ok


@torch.no_grad()
def evaluate(model, pack: GpuPack, groups: dict, batch_size: int, n_sources: int = 2) -> dict:
    model.eval()
    report = {}
    for name, idx in groups.items():
        rows = {'clip': [], 'y': [], 'q1': [], 'q2': [], 'err_px': [], 's_px': [], 'ok': []}
        for start in range(0, len(idx), batch_size):
            part = idx[start:start + batch_size]
            batch = pack.batch(part, n_sources=n_sources)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_full(batch['rgb'], batch['K'], batch['P'], batch['weights'])
            _, y, ok = quality_losses(batch, out)
            known, best = truth(batch)
            err = (out['depth'].clamp_min(1e-4).log() - best.clamp_min(1e-4).log()).abs()
            pick = known & (torch.rand_like(err) < .02)
            rows['err_px'].append(err[pick].float().cpu()); rows['s_px'].append(out['log_scale'][pick].float().cpu())
            rows['clip'].append(pack.clip[part].cpu()); rows['y'].append(y.cpu()); rows['ok'].append(ok.cpu())
            rows['q1'].append(out['q1'][0].float().cpu()); rows['q2'].append(out['q2'][0].float().cpu())
        r = {k: torch.cat(v).numpy() for k, v in rows.items()}
        ok = r['ok']
        per_clip = {'q1': [], 'q2': []}
        for c in np.unique(r['clip'][ok]):
            sel = ok & (r['clip'] == c)
            if sel.sum() >= 8 and np.ptp(r['y'][sel]) > 0:
                for q in per_clip:
                    per_clip[q].append(spearmanr(r[q][sel], r['y'][sel])[0])
        report[name] = {'frames': int(ok.sum()), 'median_abs_log_err': float(np.exp(np.median(r['y'][ok])) - 1e-3),
                        'q1_spearman_clip': float(np.nanmean(per_clip['q1'])) if per_clip['q1'] else None,
                        'q2_spearman_clip': float(np.nanmean(per_clip['q2'])) if per_clip['q2'] else None,
                        'q1_spearman_all': float(spearmanr(r['q1'][ok], r['y'][ok])[0]),
                        'q2_spearman_all': float(spearmanr(r['q2'][ok], r['y'][ok])[0]),
                        'pixel_scale_spearman': float(spearmanr(r['s_px'], r['err_px'])[0]) if len(r['s_px']) > 10 else None}
    model.train()
    return report


@torch.no_grad()
def evaluate_degradations(model, pack: GpuPack, idx: torch.Tensor, batch_size: int, n_sources: int, strength: float = .6) -> dict:
    """Same held-out frames, clean and degraded: does the true error rise, and do the exits see it?
    rank_* is the fraction of frames whose predicted error is higher for the degraded copy (0.5 = blind)."""
    model.eval()
    gen_state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1234)
    def run(kind_index):
        ys, q1, q2, oks = [], [], [], []
        for start in range(0, len(idx), batch_size):
            part = idx[start:start + batch_size]
            b = len(part)
            kind = None if kind_index is None else torch.full((b,), kind_index, device=part.device)
            batch = pack.batch(part, n_sources=n_sources, kind=kind, strength=torch.full((b,), strength, device=part.device))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_full(batch['rgb'], batch['K'], batch['P'], batch['weights'])
            _, y, ok = quality_losses(batch, out)
            ys.append(y.float()); oks.append(ok); q1.append(out['q1'][0].float()); q2.append(out['q2'][0].float())
        return torch.cat(ys), torch.cat(q1), torch.cat(q2), torch.cat(oks)
    clean = run(None)
    report = {'frames': int(clean[3].sum()), 'strength': strength, 'clean_median_err': float(torch.exp(clean[0][clean[3]].median()) - 1e-3)}
    pooled_q1, pooled_q2, pooled_y = [clean[1][clean[3]]], [clean[2][clean[3]]], [clean[0][clean[3]]]
    for i, name in enumerate(KINDS):
        y, a, c, ok = run(i)
        both = ok & clean[3]
        report[name] = {'median_err': float(torch.exp(y[both].median()) - 1e-3),
                        'true_worse': float((y[both] > clean[0][both]).float().mean()),
                        'rank_q1': float((a[both] > clean[1][both]).float().mean()),
                        'rank_q2': float((c[both] > clean[2][both]).float().mean())}
        pooled_q1.append(a[both]); pooled_q2.append(c[both]); pooled_y.append(y[both])
    y = torch.cat(pooled_y).cpu().numpy()
    report['pooled_spearman_q1'] = float(spearmanr(torch.cat(pooled_q1).cpu().numpy(), y)[0])
    report['pooled_spearman_q2'] = float(spearmanr(torch.cat(pooled_q2).cpu().numpy(), y)[0])
    torch.cuda.set_rng_state(gen_state)
    model.train()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--init', type=Path, help='DepthStudent checkpoint to start from (student model)')
    p.add_argument('--model', choices=('student', 'teacher'), default='student')
    p.add_argument('--sources', type=int, default=2, help='past sources per frame: 2, or 4 adds the two previous frames')
    p.add_argument('--eval-batch', type=int, default=64)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--holdout', nargs='+', default=['AMtown03', 'HKairport03'])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--steps-per-epoch', type=int, default=300)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--drop-source', type=float, default=.3)
    p.add_argument('--w-pixel', type=float, default=.1)
    p.add_argument('--w-quality', type=float, default=.1)
    p.add_argument('--max-epochs-without-gain', type=int, default=10)
    p.add_argument('--areas', nargs='+', help='train only on these areas (ablation), e.g. AMtown')
    p.add_argument('--degrade-prob', type=float, default=.3, help='fraction of training references degraded')
    p.add_argument('--degrade-eval', type=int, default=192, help='held-out frames used for the clean/degraded check (0 = off)')
    a = p.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')
    a.output.mkdir(parents=True, exist_ok=True)
    pack = GpuPack(a.pack, device)
    hold = set(a.holdout)
    train_idx = pack.frames(lambda m: m['role'] == 'train' and m['scene'] not in hold and (not a.areas or m['area'] in a.areas))
    scored = lambda m: not m['name'].endswith('_0')  # calibration clips are never scored
    groups = {'holdout': pack.frames(lambda m: m['role'] == 'train' and m['scene'] in hold and scored(m))}
    for area in sorted({m['area'] for m in pack.meta if m['role'] != 'train'}):
        groups[area] = pack.frames(lambda m, area=area: m['role'] != 'train' and m['area'] == area and scored(m))
    for flight in sorted(hold):
        groups[f'holdout_{flight}'] = pack.frames(lambda m, f=flight: m['scene'] == f and scored(m))
    areas = [pack.area[c] for c in pack.clip[train_idx].tolist()]
    counts = {x: areas.count(x) for x in set(areas)}
    sample_weight = torch.tensor([1. / counts[x] for x in areas], device=device)  # equal mass per area

    if a.model == 'student':
        model = QualityStudent(48, 24)
        state = torch.load(a.init, map_location='cpu', weights_only=False)
        missing = model.load_state_dict(state['model'] if 'model' in state else state, strict=False).missing_keys
        model.to(device)
    else:
        model = QualityTeacher().to(device).build(a.sources, device)
        missing = ['all (trained from scratch on a frozen DINOv2 backbone)']
    model.train()
    arch = type(model).architecture_id
    trainable = [p for p in model.parameters() if p.requires_grad]
    state_of = (lambda: model.trainable_state()) if a.model == 'teacher' else (lambda: model.state_dict())
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-4)
    total_steps = a.epochs * a.steps_per_epoch
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s + 1) / 300) * .5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    weights = {**DEFAULT_WEIGHTS, 'confidence': 0.}
    config = {'args': {k: str(v) for k, v in vars(a).items()}, 'train_frames': len(train_idx), 'area_frames': counts,
              'groups': {k: len(v) for k, v in groups.items()}, 'new_parameters': missing,
              'parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': sum(p.numel() for p in trainable), 'architecture': arch}
    write_json(a.output / 'config.json', config)
    print(json.dumps(config), flush=True)
    degrade_idx = groups['holdout'][torch.randperm(len(groups['holdout']), generator=torch.Generator().manual_seed(0))[:a.degrade_eval].to(device)] if a.degrade_eval else None
    baseline = evaluate(model, pack, groups, a.eval_batch, a.sources)
    if degrade_idx is not None:
        baseline['degradations'] = evaluate_degradations(model, pack, degrade_idx, a.eval_batch, a.sources)
    write_json(a.output / 'baseline.json', baseline)
    print(json.dumps({'epoch': 0, 'eval': baseline}), flush=True)
    best = {'quality': -1e9, 'depth': 1e9}
    stale, started, step = 0, time.monotonic(), 0
    log = (a.output / 'train.jsonl').open('a')
    for epoch in range(1, a.epochs + 1):
        sums, n = {}, 0
        t0 = time.monotonic()
        for _ in range(a.steps_per_epoch):
            idx = train_idx[torch.multinomial(sample_weight, a.batch_size, replacement=True)]
            batch = pack.batch(idx, a.drop_source, augment=True, n_sources=a.sources, degrade_prob=a.degrade_prob)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_full(batch['rgb'], batch['K'], batch['P'], batch['weights'])
            depth, conf = out['depth'].float(), out['confidence'].float()
            total, parts = student_loss(batch, depth, conf, weights)
            q, _, _ = quality_losses(batch, out)
            loss = total + a.w_pixel * q['pixel_nll'] + a.w_quality * (q['q1'] + q['q2'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 2.)
            if torch.isfinite(norm):
                optimizer.step()
            schedule.step(); step += 1
            for k, v in {'loss': loss, 'depth_loss': total, **q}.items():
                sums[k] = sums.get(k, 0.) + v.detach()
            n += 1
        train = {k: float(v / n) for k, v in sums.items()}
        seconds = time.monotonic() - t0
        ev = evaluate(model, pack, groups, a.eval_batch, a.sources)
        if degrade_idx is not None:
            ev['degradations'] = evaluate_degradations(model, pack, degrade_idx, a.eval_batch, a.sources)
        h = ev['holdout']
        quality = np.nanmean([h['q1_spearman_clip'] or np.nan, h['q2_spearman_clip'] or np.nan])
        payload = {'model': state_of(), 'architecture_id': arch, 'sources': a.sources, 'epoch': epoch, 'eval': ev,
                   'init': str(a.init), 'holdout': sorted(hold)}
        improved = False
        if quality > best['quality']:
            best['quality'], improved = quality, True
            torch.save(payload, a.output / 'best_quality.pt')
        if h['median_abs_log_err'] < best['depth']:
            best['depth'], improved = h['median_abs_log_err'], True
            torch.save(payload, a.output / 'best_depth.pt')
        torch.save(payload, a.output / 'latest.pt')
        stale = 0 if improved else stale + 1
        row = {'epoch': epoch, 'step': step, 'train': train, 'epoch_seconds': seconds, 'frames_per_s': a.steps_per_epoch * a.batch_size / seconds,
               'eval': ev, 'best': best, 'elapsed_min': (time.monotonic() - started) / 60}
        log.write(json.dumps(row) + '\n'); log.flush()
        print(json.dumps(row), flush=True)
        if stale >= a.max_epochs_without_gain:
            break
    write_json(a.output / 'report.json', {'best': best, 'baseline_holdout': baseline['holdout'], 'last_epoch': epoch,
                                          'selection': 'held-out training flights only; AMvalley and HKisland reported, never used to choose'})


if __name__ == '__main__':
    main()
