"""End-to-end selector test: run the streaming selection policy, fuse the chosen frames, score the result.

For every episode (a 35-frame UAVScenes clip with fixed reference poses, predicted depths and a
reference point population) each scorer drives the same greedy bin/occupant policy used by
evaluate_large_action_selector.py. The selected frames are TSDF-fused and scored with the
reference F metric (contribution_labels.value). Baselines at the SAME frame count k:
uniform spacing, and DINO-bin middle frames. 'all' fuses every frame.
Thresholds are tuned per scorer on the AMtown03 held flight, then applied unchanged to the
100 validation episodes (20 clips x 5 corruption variants) that no training run used.
usage: python -m uav_reconstruction.adaptive_uav.eval_selector_reconstruction OUTDIR [workers]
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import sys
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import cv2

from . import bin_memory_selector as bm
from . import reconstruction_selector as rs
from .large_selector_model import LargeActionSelector, SelectorVisualEncoder
from .contribution_labels import value
from .reconstruction_quality import ReferencePopulation
from .train_contribution_selector import dino_bins

ROOT = Path(os.environ.get('GEOSTREAM_SELECTOR_DATA', 'data/selector_large'))
C = ROOT / 'all_available_20261115/corpus'
THRESHOLDS = (-1.6, -.8, -.4, 0., .4, .8, 1.6)
TRAINED = {'start': (C / 'visual_final/latest.pt', C / 'selector/large_inner.pt'),
           'official_e27': ROOT / 'complete_20261115/best_holdout.pt',
           'expanded_e31': ROOT / 'all_available_20261115/training/best_holdout.pt',
           'expanded_e33': ROOT / 'all_available_20261115/training/latest.pt'}
M = {}


def init():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.backends.mkldnn.enabled = False
    for name, spec in TRAINED.items():
        enc, head = SelectorVisualEncoder().eval(), LargeActionSelector(36).eval()
        if isinstance(spec, tuple):
            enc.load_state_dict(torch.load(spec[0], map_location='cpu', weights_only=False)['model'])
            head.load_state_dict(torch.load(spec[1], map_location='cpu', weights_only=False)['model'])
        else:
            s = torch.load(spec, map_location='cpu', weights_only=False)
            enc.load_state_dict(s['encoder']); head.load_state_dict(s['head'])
        M[name] = (enc, head)
    for scope in ('inner', 'full'):  # previously deployed scorer on frozen DINO features
        ck = torch.load(ROOT / 'selector' / f'large_{scope}.pt', map_location='cpu', weights_only=False)
        h = LargeActionSelector(ck['state_dim']); h.load_state_dict(ck['model']); h.eval()
        M[f'old_dino_{scope}'] = (None, h)


def run_policy(c, ctx, z, feat, head, threshold):
    selected, prototypes, occupants = [], [], []
    for t in range(ctx.n):
        cos = np.asarray([p @ ctx.emb[t] for p in prototypes])
        if len(cos) and cos.max() >= .9:
            b = int(cos.argmax())
        else:
            b = len(prototypes); prototypes.append(ctx.emb[t]); occupants.append(None)
        a = occupants[b]
        x = bm.state_features(c, t, selected, len(prototypes), a, cos)
        x = np.concatenate([x, z['quality'][t], z['quality'][a] if a is not None else np.zeros(6, np.float32)]).astype(np.float32)
        with torch.no_grad():
            s = float(head(feat[t:t + 1], feat[(a or 0):(a or 0) + 1], torch.tensor([float(a is not None)]), torch.from_numpy(x)[None]))
        if s >= threshold:
            selected = selected + [t] if a is None else [s_ for s_ in selected if s_ != a] + [t]
            occupants[b] = t
    return selected


def evaluate(job):
    path, rowdir = map(Path, job)
    done = rowdir / (path.stem + '.json')
    if done.exists():
        return json.loads(done.read_text())
    out = _evaluate(path)
    done.write_text(json.dumps(out))
    return out


def _evaluate(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    z = dict(np.load(path))
    views = [SimpleNamespace(fusion_depth=d, fusion_k=k, reference_pose=p,
                             fusion_rgb=cv2.resize(rgb, d.shape[::-1], interpolation=cv2.INTER_AREA))
             for d, k, p, rgb in zip(z['fusion_depth'], z['fusion_k'], z['reference_pose'], z['rgb'])]
    ref = ReferencePopulation(z['reference_points'], meta['tolerance_m'], 'QUALIFIED', meta['quality_protocol'])
    points = [rs.back_project(v)[0] for v in views]
    seen = [np.stack([rs.consistent(p, v)[0] if i != j else np.zeros(p.shape[1], bool)
                      for j, v in enumerate(views)]) for i, p in enumerate(points)]
    cache = {}

    def F(sel):
        key = tuple(sorted(set(sel)))
        if key not in cache:
            cache[key] = value([views[i] for i in key], ref, meta['voxel_m']) if key else 0.
        return cache[key]
    ctx = SimpleNamespace(views=views, emb=z['embedding'], n=len(views), seen=seen, F=F)
    c = bm.Clip.__new__(bm.Clip)
    c.ctx, c.C, c.sharp, c.pe = ctx, z['causal'], z['sharp'], np.zeros(ctx.n, np.float32)
    c._points, c._zmed, c._mask = {}, {}, {}
    held = meta['flight'] == 'AMtown03' and meta['split'] == 'train'
    dino = torch.from_numpy(np.load(ROOT / 'utility_features' / (path.stem + '.npy')))
    rgb = torch.from_numpy(z['rgb']).permute(0, 3, 1, 2).float() / 255
    out = {'clip': meta['clip'], 'flight': meta['flight'], 'variant': meta['variant'], 'held': held,
           'n': ctx.n, 'all_f': F(range(ctx.n)), 'scorers': {}}
    for name, (enc, head) in M.items():
        if name.startswith('old_dino_') and name != ('old_dino_inner' if held else 'old_dino_full'):
            continue
        if enc is None:
            feat = dino
        else:
            with torch.no_grad():
                feat = enc(rgb)[0].float()
        rows = []
        for th in THRESHOLDS:
            sel = run_policy(c, ctx, z, feat, head, th); k = len(sel)
            middle = [int(g[len(g) // 2]) for g in dino_bins(ctx.emb, k)] if k else []
            uniform = np.linspace(0, ctx.n - 1, k).round().astype(int).tolist() if k else []
            rows.append(dict(threshold=th, k=k, f=F(sel), uniform_f=F(uniform), middle_f=F(middle), selected=sel))
        out['scorers']['old_dino' if name.startswith('old_dino_') else name] = rows
    return out


def main():
    outdir = Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    paths = sorted((ROOT / 'utility').glob('*.npz'))
    metas = {p: json.loads(p.with_suffix('.json').read_text()) for p in paths}
    hold = [p for p in paths if metas[p]['flight'] == 'AMtown03' and metas[p]['split'] == 'train']
    val = [p for p in paths if metas[p]['split'] == 'validation']
    print(len(hold), 'held episodes', len(val), 'validation episodes', flush=True)
    rowdir = outdir / 'rows'; rowdir.mkdir(exist_ok=True)
    if (outdir / 'held_rows.json').exists():
        for r, p in zip(json.loads((outdir / 'held_rows.json').read_text()), hold):
            if r['clip'] == metas[p]['clip'] and r['variant'] == metas[p]['variant']:
                (rowdir / (p.stem + '.json')).write_text(json.dumps(r))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool:
        held_rows = list(pool.map(evaluate, [(str(p), str(rowdir)) for p in hold]))
        (outdir / 'held_rows.json').write_text(json.dumps(held_rows))
        print('held done', flush=True)
        val_rows = list(pool.map(evaluate, [(str(p), str(rowdir)) for p in val]))
        (outdir / 'validation_rows.json').write_text(json.dumps(val_rows))
    scorers = list(held_rows[0]['scorers'])
    tune = {}
    for s in scorers:
        grid = [dict(threshold=th, f=float(np.mean([r['scorers'][s][i]['f'] for r in held_rows])))
                for i, th in enumerate(THRESHOLDS)]
        best = max(range(len(grid)), key=lambda i: grid[i]['f'])
        tune[s] = dict(index=best, threshold=THRESHOLDS[best], held_f=grid[best]['f'], grid=[dict(g) for g in grid])
    summary = {'scope': 'Reference-pose TSDF F on UAVScenes validation episodes (fixed poses, predicted depths); thresholds tuned on AMtown03 only',
               'tune': tune, 'validation': {}}
    for variant in ['all'] + sorted({r['variant'] for r in val_rows}):
        rows = [r for r in val_rows if variant == 'all' or r['variant'] == variant]
        case = {'episodes': len(rows), 'all_frames_f': float(np.mean([r['all_f'] for r in rows]))}
        for s in scorers:
            picks = [r['scorers'][s][tune[s]['index']] for r in rows]
            f = np.array([p['f'] for p in picks]); u = np.array([p['uniform_f'] for p in picks]); m = np.array([p['middle_f'] for p in picks])
            case[s] = dict(f=float(f.mean()), uniform_f=float(u.mean()), middle_f=float(m.mean()),
                           frames_kept=float(np.mean([p['k'] / r['n'] for p, r in zip(picks, rows)])),
                           wins_vs_uniform=float(np.mean(f > u + 1e-9)), losses_vs_uniform=float(np.mean(f < u - 1e-9)))
        summary['validation'][variant] = case
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)
    print('EVALUATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
