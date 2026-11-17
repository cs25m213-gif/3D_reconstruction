"""Conditional contribution labels  D(f|S) = F(S + {f}) - F(S)  and a selector trained on them.

labels mode (one clip): greedy oracle against LiDAR. At each step every unselected frame is fused with the
current set S and its gain recorded; the best is added. Up to --steps steps, or until no frame helps.
Per (step, candidate) row we also store LiDAR-free state features:
  new_cover    fraction of the frame's depth points not already observed consistently (5% rel depth) by S
  agree        depth agreement with its i-2..i+2 neighbours (consistency_test.agreement)
  cos_max/mean DINO cosine of the frame to S;  sharp, bright per-clip z;  step/16
train mode: MLP on those features -> D (MSE + within-step ListNet), train AMtown, select on AMvalley rank
correlation, then run greedy with the model on validation clips and fuse against LiDAR.
Fixed reference poses and fixed-context depth, as in contribution_labels.
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

from consistency_test import agreement
from uav_reconstruction.adaptive_uav.contribution_labels import load_clip, value
from uav_reconstruction.adaptive_uav.train_contribution_selector import dino_bins, spearman, listnet
from uav_reconstruction.adaptive_uav.student_inference import load_student
from uav_reconstruction.adaptive_uav.utility_contracts import load_contract
from uav_reconstruction.adaptive_uav.reconstruction_quality import cluster_interval

M = '/home/vishnu/geostreamfusion_more_data_20261104/data'
CONFIG = Path('runs/contract_vishnu.json')
NAMES = ('new_cover', 'agree', 'cos_max', 'cos_mean', 'sharp', 'bright', 'step')
METHODS = ('uniform', 'middle', 'greedy_learned', 'greedy_learned_stop', 'greedy_oracle')


def seen_matrix(views, stride=8, tol=.05):
    """seen[i][j]: bool over frame i's sampled depth points, True where frame j sees the same surface."""
    pts = []
    for v in views:
        d, K, T = v.fusion_depth, v.fusion_k, v.reference_pose
        r, c = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[r, c]
        m = z > 0
        pix = np.stack([c[m], r[m], np.ones(m.sum())]) * z[m]
        pts.append(T[:3, :3] @ (np.linalg.inv(K) @ pix) + T[:3, 3:4])
    n, inv, seen = len(views), [np.linalg.inv(v.reference_pose) for v in views], []
    for i in range(n):
        row = np.zeros((n, pts[i].shape[1]), bool)
        for j in range(n):
            if j == i:
                continue
            dj, Kj = views[j].fusion_depth, views[j].fusion_k
            cam = inv[j][:3, :3] @ pts[i] + inv[j][:3, 3:4]
            zc = cam[2]
            ok = zc > 1e-3
            p = Kj @ cam[:, ok]
            u, w = (p[0] / p[2]).round().astype(int), (p[1] / p[2]).round().astype(int)
            ins = (u >= 0) & (u < dj.shape[1]) & (w >= 0) & (w < dj.shape[0])
            idx = np.flatnonzero(ok)[ins]
            ref = dj[w[ins], u[ins]]
            row[j, idx] = (ref > 0) & (np.abs(zc[idx] - ref) < tol * np.maximum(ref, 1e-6))
        seen.append(row)
    return seen


def state_features(S, cand, seen, agree, emb, sharp, bright, step):
    rows = []
    for f in cand:
        cover = 1. - (seen[f][S].any(0).mean() if S and seen[f].shape[1] else 0.)
        cos = emb[S] @ emb[f] if S else np.array([0.])
        rows.append([cover, agree[f], cos.max(), cos.mean(), sharp[f], bright[f], step / 16])
    return np.asarray(rows, np.float32)


def prepare(name, labels):
    contract = load_contract(CONFIG, ('utility.quality_protocol',))
    clip, ref = load_clip(Path('.').resolve(), contract, name, load_student(Path(M) / 'student.pt', 'cpu'), 'cpu')
    views = clip.views
    with np.load(labels / f'{name}.npz') as z:
        emb, sh, br = z['embedding'], np.log1p(z['sharpness']), z['brightness']
    sh, br = (sh - sh.mean()) / (sh.std() + 1e-6), (br - br.mean()) / (br.std() + 1e-6)
    agree = np.array([agreement(views, i) for i in range(len(views))])
    agree = np.where(np.isnan(agree), np.nanmedian(agree), agree)
    return contract, ref, views, emb, sh, br, agree, seen_matrix(views)


def labels_mode(a):
    target = a.output / f'{a.clip}.npz'
    if target.exists():
        return
    t0 = time.monotonic()
    contract, ref, views, emb, sh, br, agree, seen = prepare(a.clip, a.labels)
    voxel, n = contract['utility']['voxel_m'], len(views)
    S, F_S, X, Y, STEP, CAND, curve = [], 0., [], [], [], [], []
    for step in range(min(a.steps, n)):
        cand = [f for f in range(n) if f not in S]
        gains = np.array([value([views[i] for i in sorted(S + [f])], ref, voxel) - F_S for f in cand])
        X.append(state_features(S, cand, seen, agree, emb, sh, br, step))
        Y.append(gains)
        STEP += [step] * len(cand)
        CAND += cand
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        S.append(cand[best])
        F_S += gains[best]
        curve.append(float(F_S))
    a.output.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + '.partial.npz')
    np.savez_compressed(tmp, x=np.concatenate(X), y=np.concatenate(Y), step=np.array(STEP), cand=np.array(CAND),
                        order=np.array(S), curve=np.array(curve), agree=agree)
    tmp.replace(target)
    (a.output / f'{a.clip}.json').write_text(json.dumps({'clip': a.clip, 'frames': n, 'steps': len(S), 'curve': curve,
                                                          'seconds': round(time.monotonic() - t0, 1)}))


class Net(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(w, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def groups(greedy, labels, split):
    out = []
    for meta in sorted(greedy.glob('AM*.json')):
        info = json.loads((labels / meta.name).read_text())
        if info['split'] != split:
            continue
        with np.load(greedy / f'{meta.stem}.npz') as z:
            for s in np.unique(z['step']):
                m = z['step'] == s
                out.append((info['clip'], torch.from_numpy(z['x'][m]), torch.from_numpy(z['y'][m].astype(np.float32))))
    return out


def evaluate(args):
    name, labels, out = args
    torch.set_num_threads(4)
    info = json.loads((labels / f'{name}.json').read_text())
    contract, ref, views, emb, sh, br, agree, seen = prepare(name, labels)
    net = Net(len(NAMES))
    net.load_state_dict(torch.load(out / 'greedy_net.pt'))
    net.eval()
    voxel, n = contract['utility']['voxel_m'], len(views)
    F = lambda idx: value([views[i] for i in sorted(set(int(j) for j in idx))], ref, voxel)
    S, preds = [], []
    for step in range(min(16, n)):
        cand = [f for f in range(n) if f not in S]
        with torch.no_grad():
            p = net(torch.from_numpy(state_features(S, cand, seen, agree, emb, sh, br, step))).numpy()
        S.append(cand[int(np.argmax(p))])
        preds.append(float(p.max()))
    curve = json.loads((out / f'{name}.json').read_text())['curve']
    row = {'clip': name, 'flight': info['flight'], 'b': {}}
    for k in (4, 8, 16):
        if k >= n:
            continue
        bins = dino_bins(emb, k)
        stop = [s for s, q in zip(S[:k], preds[:k]) if q > 0] or S[:1]
        row['b'][str(k)] = {'uniform': F(np.linspace(0, n - 1, k).round()), 'middle': F([b[len(b) // 2] for b in bins]),
                            'greedy_learned': F(S[:k]), 'greedy_learned_stop': F(stop), 'stop_frames': len(stop),
                            'greedy_oracle': curve[min(k, len(curve)) - 1] if curve else 0.}
    print(json.dumps(row), flush=True)
    return row


def train_mode(a):
    torch.manual_seed(0)
    np.random.seed(0)
    tr, va = groups(a.output, a.labels, 'train'), groups(a.output, a.labels, 'validation')
    net, best, state, ep = Net(len(NAMES)), -np.inf, None, -1
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    for epoch in range(a.epochs):
        net.train()
        for i in np.random.permutation(len(tr)):
            _, x, y = tr[i]
            opt.zero_grad()
            p = net(x)
            loss = ((p - y * 10) ** 2).mean() + listnet(p, y, float(y.std() + 1e-6))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            rho = float(np.nanmean([spearman(net(x).numpy(), y.numpy()) for _, x, y in va if len(y) > 2]))
        print(json.dumps({'epoch': epoch, 'val_spearman': rho}), flush=True)
        if rho > best:
            best, ep, state = rho, epoch, {k: v.clone() for k, v in net.state_dict().items()}
    torch.save(state, a.output / 'greedy_net.pt')
    clips = sorted({c for c, *_ in va})
    with Pool(20) as pool:
        rows = pool.map(evaluate, [(c, a.labels, a.output) for c in clips])
    summ = {'val_spearman_within_step': best, 'epoch': ep, 'train_steps': len(tr), 'val_clips': len(clips)}
    for k in ('4', '8', '16'):
        part = [r for r in rows if k in r['b']]
        e = {m: float(np.mean([r['b'][k][m] for r in part])) for m in METHODS}
        for m in ('greedy_learned', 'greedy_learned_stop', 'greedy_oracle'):
            g = {}
            for r in part:
                g.setdefault(r['flight'], []).append(r['b'][k][m] - r['b'][k]['middle'])
            e[f'{m}_minus_middle'] = cluster_interval(g)
        e['stop_frames_mean'] = float(np.mean([r['b'][k]['stop_frames'] for r in part]))
        e['clips'] = len(part)
        summ[k] = e
    (a.output / 'eval_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (a.output / 'summary.json').write_text(json.dumps(summ, indent=1))
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('labels', 'train'))
    p.add_argument('--clip')
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--labels', type=Path, default=Path('runs/contribution_20261109/labels'))
    p.add_argument('--output', type=Path, default=Path('runs/contribution_20261109/greedy'))
    a = p.parse_args()
    labels_mode(a) if a.mode == 'labels' else train_mode(a)
