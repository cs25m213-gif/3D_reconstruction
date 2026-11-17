"""Predict each frame's student-depth error (label: median distance of its back-projected depth to LiDAR)
from runtime-only inputs, then select frames with it.

table mode (one clip): per-frame runtime features + LiDAR error label, plus timing of the runtime features.
  runtime features: DINO embedding; depth agreement with i-2..i+2 (agree) and with i-2,i-1 only (agree_past,
  causal); median depth, depth p90/p10, footprint, height above the selected-frame median height... see FEATS.
train mode: MLP -> log(error) (MSE + within-clip ListNet), train AMtown, early-stop on AMvalley Spearman, then
  fuse on validation clips against LiDAR:
    middle            middle frame of each DINO bin (current best rule)
    pred_topk         K frames with lowest predicted error
    bins_pred         lowest predicted error in each DINO bin
    bins_pred_drop    bins_pred, minus picks predicted worse than the clip median
    true_topk         K frames with lowest true LiDAR error (ceiling for this approach)
    greedy_oracle     greedy measured D(f|S) (from greedy_selector)
Height uses the camera z relative to the clip's lowest camera, so no LiDAR enters the features.
Fixed reference poses and fixed-context depth, as in the labels.
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from consistency_test import agreement
from uav_reconstruction.adaptive_uav.contribution_labels import load_clip, value
from uav_reconstruction.adaptive_uav.train_contribution_selector import dino_bins, spearman, listnet
from uav_reconstruction.adaptive_uav.student_inference import load_student
from uav_reconstruction.adaptive_uav.utility_contracts import load_contract
from uav_reconstruction.adaptive_uav.reconstruction_quality import cluster_interval

M = '/home/vishnu/geostreamfusion_more_data_20261104/data'
CONFIG = Path('runs/contract_vishnu.json')
LABELS = Path('runs/contribution_20261109/labels')
OUT = Path('runs/contribution_20261109/depth_error')
FEATS = ('agree', 'agree_past', 'log_depth_median', 'depth_range_ratio', 'log_footprint', 'rel_height', 'off_nadir',
         'sharp_z', 'bright_z', 'grad_mean')
BUDGETS = (2, 3, 4, 8)
METHODS = ('middle', 'pred_topk', 'bins_pred', 'bins_pred_drop', 'true_topk', 'greedy_oracle')


def runtime_scalars(views, i, stride=8):
    d, K, T = views[i].fusion_depth, views[i].fusion_k, views[i].reference_pose
    r, c = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
    z = d[r, c]
    m = z > 0
    pix = np.stack([c[m], r[m], np.ones(m.sum())]) * z[m]
    world = (T[:3, :3] @ (np.linalg.inv(K) @ pix) + T[:3, 3:4]).T
    zs = z[m]
    lo, hi = np.percentile(world[:, :2], [5, 95], axis=0)
    gy, gx = np.gradient(np.log(np.maximum(d, 1e-3)))
    return world, [float(np.log(np.median(zs))), float(np.percentile(zs, 90) / max(np.percentile(zs, 10), 1e-3)),
                   float(np.log1p(np.prod(hi - lo))), float(T[2, 3]),
                   float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))), float(np.hypot(gx, gy)[d > 0].mean())]


def clip_views(name):
    contract = load_contract(CONFIG, ('utility.quality_protocol',))
    clip, ref = load_clip(Path('.').resolve(), contract, name, load_student(Path(M) / 'student.pt', 'cpu'), 'cpu')
    return contract, clip, ref


def table_mode(name):
    target = OUT / f'{name}.npz'
    if target.exists():
        return
    contract, clip, ref = clip_views(name)
    views, n = clip.views, len(clip.views)
    with np.load(LABELS / f'{name}.npz') as z:
        emb, sh, br = z['embedding'], np.log1p(z['sharpness']), z['brightness']
    t0 = time.perf_counter()
    agree = np.array([agreement(views, i) for i in range(n)])
    agree_past = np.array([agreement(views, i, offsets=(-2, -1)) for i in range(n)])
    geo = [runtime_scalars(views, i) for i in range(n)]
    runtime_ms = (time.perf_counter() - t0) * 1000 / n
    tree = cKDTree(np.asarray(ref.points if hasattr(ref, 'points') else ref.tree.data))
    err = np.array([np.median(tree.query(w)[0]) if len(w) else np.nan for w, _ in geo])
    g = np.array([s for _, s in geo])
    g[:, 3] -= g[:, 3].min()  # height relative to the clip's lowest camera
    fill = lambda a: np.where(np.isnan(a), np.nanmedian(a) if np.isfinite(a).any() else 0., a)
    scal = np.column_stack([fill(agree), fill(agree_past), g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4],
                            (sh - sh.mean()) / (sh.std() + 1e-6), (br - br.mean()) / (br.std() + 1e-6), g[:, 5]])
    info = json.loads((LABELS / f'{name}.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'{name}.partial.npz', emb=emb, scal=scal.astype(np.float32), err=err)
    (OUT / f'{name}.partial.npz').replace(target)
    (OUT / f'{name}.json').write_text(json.dumps({'clip': name, 'split': info['split'], 'flight': info['flight'], 'frames': n,
                                                   'runtime_feature_ms_per_frame': runtime_ms}))


class Net(nn.Module):
    def __init__(self, n_emb, n_scal):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(n_emb, 32), nn.SiLU(), nn.Dropout(.2))
        self.f = nn.Sequential(nn.Linear(32 + n_scal, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, e, s):
        return self.f(torch.cat([self.emb(e), s], 1)).squeeze(-1)


def load_split(split, mu=None, sd=None):
    rows = []
    for meta in sorted(OUT.glob('AM*.json')):
        info = json.loads(meta.read_text())
        if info['split'] != split:
            continue
        with np.load(OUT / f'{meta.stem}.npz') as z:
            ok = np.isfinite(z['err'])
            rows.append({**info, 'emb': z['emb'], 'scal': z['scal'], 'err': z['err'], 'ok': ok})
    if mu is None:
        allscal = np.concatenate([r['scal'] for r in rows])
        mu, sd = allscal.mean(0), allscal.std(0) + 1e-6
    for r in rows:
        r['x'] = torch.from_numpy(r['emb'].astype(np.float32)), torch.from_numpy(((r['scal'] - mu) / sd).astype(np.float32))
    return rows, mu, sd


def select_eval(args):
    name, flight, pred, err, emb = args
    torch.set_num_threads(4)
    contract, clip, ref = clip_views(name)
    views, n, voxel = clip.views, len(clip.views), contract['utility']['voxel_m']
    F = lambda idx: value([views[int(i)] for i in sorted(set(int(j) for j in idx))], ref, voxel)
    true = np.where(np.isfinite(err), err, np.inf)
    curve = json.loads((Path('runs/contribution_20261109/greedy') / f'{name}.json').read_text())['curve']
    row = {'clip': name, 'flight': flight, 'b': {}}
    for k in BUDGETS:
        if k >= n:
            continue
        bins = dino_bins(emb, k)
        bp = [int(b[np.argmin(pred[b])]) for b in bins]
        drop = [i for i in bp if pred[i] <= np.median(pred)] or [int(np.argmin(pred))]
        row['b'][str(k)] = {'middle': F([b[len(b) // 2] for b in bins]), 'pred_topk': F(np.argsort(pred)[:k]),
                            'bins_pred': F(bp), 'bins_pred_drop': F(drop), 'kept_drop': len(drop),
                            'true_topk': F(np.argsort(true)[:k]),
                            'greedy_oracle': curve[min(k, len(curve)) - 1] if curve else 0.}
    print(json.dumps(row), flush=True)
    return row


def train_mode(epochs):
    torch.manual_seed(0)
    np.random.seed(0)
    tr, mu, sd = load_split('train')
    va, _, _ = load_split('validation', mu, sd)
    net = Net(tr[0]['emb'].shape[1], len(FEATS))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    best, state, ep = -np.inf, None, -1
    for epoch in range(epochs):
        net.train()
        for i in np.random.permutation(len(tr)):
            r = tr[i]
            ok = torch.from_numpy(r['ok'])
            y = torch.from_numpy(np.log(np.maximum(r['err'], 1e-3)).astype(np.float32))[ok]
            opt.zero_grad()
            p = net(*r['x'])[ok]
            loss = ((p - y) ** 2).mean() + listnet(-p, -y, float(y.std() + 1e-6))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            rho = float(np.nanmean([spearman(net(*r['x']).numpy()[r['ok']], r['err'][r['ok']]) for r in va if r['ok'].sum() > 3]))
        if rho > best:
            best, ep, state = rho, epoch, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    torch.save({'model': state, 'mu': mu, 'sd': sd, 'feats': FEATS}, OUT / 'depth_error_net.pt')
    with torch.no_grad():
        x = va[0]['x']
        t0 = time.perf_counter()
        for _ in range(200):
            net(x[0][:1], x[1][:1])
        net_ms = (time.perf_counter() - t0) * 1000 / 200
        preds = {r['clip']: net(*r['x']).numpy() for r in va}
    # baseline predictor: consistency alone (higher agree = lower error)
    rho_agree = float(np.nanmean([spearman(-r['scal'][r['ok'], 0], r['err'][r['ok']]) for r in va if r['ok'].sum() > 3]))
    with Pool(20) as pool:
        rows = pool.map(select_eval, [(r['clip'], r['flight'], preds[r['clip']], r['err'], r['emb']) for r in va])
    summ = {'val_spearman_pred_vs_true_error': best, 'val_spearman_agree_only': rho_agree, 'epoch': ep,
            'runtime_feature_ms_per_frame_median': float(np.median([r['runtime_feature_ms_per_frame'] for r in tr + va])),
            'predictor_ms_per_frame': net_ms, 'train_clips': len(tr), 'val_clips': len(va)}
    for k in map(str, BUDGETS):
        part = [r for r in rows if k in r['b']]
        e = {m: float(np.mean([r['b'][k][m] for r in part])) for m in METHODS}
        for m in ('pred_topk', 'bins_pred', 'bins_pred_drop', 'true_topk', 'greedy_oracle'):
            g = {}
            for r in part:
                g.setdefault(r['flight'], []).append(r['b'][k][m] - r['b'][k]['middle'])
            e[f'{m}_minus_middle'] = cluster_interval(g)
        e['clips'] = len(part)
        summ[k] = e
    (OUT / 'eval_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('table', 'train'))
    p.add_argument('--clip')
    p.add_argument('--epochs', type=int, default=200)
    a = p.parse_args()
    table_mode(a.clip) if a.mode == 'table' else train_mode(a.epochs)
