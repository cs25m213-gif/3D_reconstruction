"""Train a frame scorer on measured Shapley contribution; test fixed-budget selection on held-out flights.

Inputs are runtime-only: frozen DINOv2 embedding plus per-clip-normalised sharpness, brightness,
baseline to the previous frame and embedding novelty. Target: per-clip Shapley contribution
(contribution_labels.py). Loss: ListNet over the frames of a clip, so only the ranking within a
clip matters, not its absolute F.

Selection at budget K: split the clip into K contiguous DINO bins at the K-1 largest embedding
changes, then pick one frame per bin. Compared on validation flights by fused F against LiDAR
(fixed poses, fixed-context depth, as in the labels):
  uniform   K frames evenly spaced in time
  middle    middle frame of each DINO bin
  learned   highest predicted score in each DINO bin
  oracle    highest measured Shapley in each DINO bin (upper bound for this binning)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .contribution_labels import load_clip, value
from .reconstruction_quality import cluster_interval
from .student_inference import load_student
from .utility_contracts import load_contract, sha256_file, write_json


def features(z) -> np.ndarray:
    scalars = np.stack([np.log1p(z['sharpness']), z['brightness'], z['baseline_prev'], z['novelty']], 1)
    scalars = (scalars - scalars.mean(0)) / (scalars.std(0) + 1e-6)  # per clip: scene-invariant
    return np.concatenate([z['embedding'], scalars], 1).astype(np.float32)


def load(folder: Path) -> dict:
    roles = {'train': [], 'validation': []}
    for meta in sorted(folder.glob('*.json')):
        info = json.loads(meta.read_text())
        if info.get('split') not in roles:
            continue
        with np.load(folder / f"{info['clip']}.npz") as z:
            roles[info['split']].append({'clip': info['clip'], 'flight': info['flight'], 'x': features(z),
                                         'y': z['shapley'].astype(np.float32), 'se': z['shapley_se'], 'emb': z['embedding']})
    return roles


class Scorer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Dropout(.2), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def listnet(pred, target, temperature):
    return -(torch.softmax(target / temperature, 0) * torch.log_softmax(pred, 0)).sum()


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def dino_bins(emb: np.ndarray, k: int) -> list:
    change = 1. - (emb[1:] * emb[:-1]).sum(1)
    cuts = np.sort(np.argsort(-change)[:k - 1]) + 1
    return np.split(np.arange(len(emb)), cuts)


def choose(method: str, bins: list, n: int, k: int, score=None, shapley=None) -> list:
    if method == 'uniform':
        return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))
    if method == 'middle':
        return [int(b[len(b) // 2]) for b in bins]
    source = score if method == 'learned' else shapley
    return [int(b[np.argmax(source[b])]) for b in bins]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--budgets', type=int, nargs='+', default=[4, 8, 16])
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if (a.output / 'result.json').exists():
        raise SystemExit('Use a fresh output directory')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    started = time.monotonic()
    data = load(a.labels)
    train, val = data['train'], data['validation']
    if len(train) < 5 or len(val) < 2:
        raise SystemExit(f'Too few labelled clips: {len(train)} train, {len(val)} validation')
    model = Scorer(train[0]['x'].shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    tensors = [(torch.from_numpy(c['x']), torch.from_numpy(c['y']), float(c['y'].std() + 1e-6)) for c in train]
    best, log = -np.inf, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        losses = []
        for i in np.random.permutation(len(tensors)):
            x, y, t = tensors[i]
            optimiser.zero_grad()
            loss = listnet(model(x), y, t)
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            rho = float(np.mean([spearman(model(torch.from_numpy(c['x'])).numpy(), c['y']) for c in val]))
        log.append({'epoch': epoch, 'train_listnet': float(np.mean(losses)), 'val_spearman': rho})
        if rho > best:
            best = rho
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_spearman': rho, 'width': train[0]['x'].shape[1],
                        'descriptor': 'dinov2_vits14'}, a.output / 'best.pt')
    (a.output / 'train_log.jsonl').write_text('\n'.join(json.dumps(r) for r in log) + '\n')
    state = torch.load(a.output / 'best.pt', weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()

    # Held-out selection test by fused F against LiDAR.
    contract = load_contract(a.config, ('utility.quality_protocol',))
    student, voxel = load_student(a.student, 'cpu'), contract['utility']['voxel_m']
    methods = ('uniform', 'middle', 'learned', 'oracle')
    rows = []
    for c in val:
        clip, reference = load_clip(a.root.resolve(), contract, c['clip'], student, 'cpu')
        with torch.no_grad():
            score = model(torch.from_numpy(c['x'])).numpy()
        n = len(clip.views)
        for k in a.budgets:
            if k >= n:
                continue
            bins = dino_bins(c['emb'], k)
            row = {'clip': c['clip'], 'flight': c['flight'], 'budget': k, 'spearman': spearman(score, c['y'])}
            for m in methods:
                picks = choose(m, bins, n, k, score, c['y'])
                row[m] = value([clip.views[i] for i in picks], reference, voxel)
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for k in a.budgets:
        part = [r for r in rows if r['budget'] == k]
        if not part:
            continue
        entry = {m: float(np.mean([r[m] for r in part])) for m in methods}
        for base in ('uniform', 'middle'):
            gains = {}
            for r in part:
                gains.setdefault(r['flight'], []).append(r['learned'] - r[base])
            entry[f'learned_minus_{base}'] = cluster_interval(gains)
            entry[f'learned_beats_{base}_clips'] = sum(r['learned'] > r[base] for r in part)
        entry['clips'] = len(part)
        summary[str(k)] = entry
    result = {'train_clips': len(train), 'validation_clips': len(val), 'best_epoch': state['epoch'], 'val_spearman': state['val_spearman'],
              'selection_f': summary, 'labels': str(a.labels), 'student_sha256': sha256_file(a.student),
              'scope': 'Fixed poses and fixed-context depth, as in the labels; not a fresh SfM or a deployed-selector result.',
              'seconds': round(time.monotonic() - started, 1)}
    write_json(a.output / 'result.json', result)
    with (a.output / 'selection_rows.jsonl').open('w') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in rows)
    print(json.dumps({k: v for k, v in result.items() if k != 'labels'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
