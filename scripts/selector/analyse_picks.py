"""What do greedy-oracle picks have in common? Per-frame geometry vs the measured conditional gain D(f|S).

For every greedy step (from greedy_selector labels) and every candidate frame: Spearman of each feature with D
across the candidates of that step, and the percentile of the picked frame within that step. Averaged over
steps, reported for all clips and for validation clips separately. depth_err_* uses LiDAR (diagnostic only,
not available at runtime); every other feature is runtime-available.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from greedy_selector import prepare, NAMES

OUT = Path('runs/contribution_20261109')


def frame_geometry(views, ref_points, stride=8):
    tree = cKDTree(ref_points)
    ground = np.median(ref_points[:, 2])
    rows = []
    for v in views:
        d, K, T = v.fusion_depth, v.fusion_k, v.reference_pose
        r, c = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[r, c]
        m = z > 0
        pix = np.stack([c[m], r[m], np.ones(m.sum())]) * z[m]
        world = (T[:3, :3] @ (np.linalg.inv(K) @ pix) + T[:3, 3:4]).T
        axis = T[:3, 2]
        zs = z[m]
        if len(world) > 20:
            err = tree.query(world)[0]
            lo, hi = np.percentile(world[:, :2], [5, 95], axis=0)
            footprint = float(np.prod(hi - lo))
            errs = (float(np.median(err)), float((err < 1.).mean()))
            dstats = (float(np.median(zs)), float(np.percentile(zs, 90) / max(np.percentile(zs, 10), 1e-3)))
        else:
            footprint, errs, dstats = 0., (np.nan, np.nan), (np.nan, np.nan)
        rows.append({'height': float(T[2, 3] - ground), 'off_nadir_deg': float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))),
                     'depth_valid_frac': float(m.mean()), 'depth_median': dstats[0], 'depth_range_ratio': dstats[1],
                     'footprint_m2': footprint, 'depth_err_median_m': errs[0], 'depth_err_inlier1m': errs[1]})
    return rows


def analyse(name):
    info = json.loads((OUT / 'labels' / f'{name}.json').read_text())
    contract, ref, views, emb, sh, br, agree, seen = prepare(name, OUT / 'labels')
    n = len(views)
    geo = frame_geometry(views, np.asarray(ref.points if hasattr(ref, 'points') else ref.tree.data))
    keys = list(geo[0]) + ['time_pos']
    per_frame = np.array([[g[k] for k in geo[0]] + [i / max(n - 1, 1)] for i, g in enumerate(geo)])
    with np.load(OUT / 'greedy' / f'{name}.npz') as z:
        x, y, step, cand, order = z['x'], z['y'], z['step'], z['cand'], z['order']
    feats = np.concatenate([x, per_frame[cand]], 1)
    names = list(NAMES) + keys
    out = []
    for s in np.unique(step):
        m = step == s
        if m.sum() < 5:
            continue
        rho = [spearmanr(feats[m, j], y[m])[0] if np.nanstd(feats[m, j]) > 0 else np.nan for j in range(len(names))]
        picked = int(np.argmax(y[m]))
        pct = [float((feats[m, j] < feats[m, j][picked]).mean()) for j in range(len(names))]
        out.append({'step': int(s), 'rho': rho, 'pct': pct, 'best_gain': float(y[m].max()), 'positive_frac': float((y[m] > 0).mean())})
    picks = per_frame[order] if len(order) else np.zeros((0, len(keys)))
    return {'clip': name, 'split': info['split'], 'names': names, 'steps': out, 'frames': n,
            'picks_time_pos': picks[:, -1].tolist() if len(picks) else [],
            'pick_vs_clip_median': {k: float(np.nanmedian(picks[:, j]) - np.nanmedian(per_frame[:, j])) for j, k in enumerate(keys)} if len(picks) else {}}


def table(rows, label):
    names = rows[0]['names']
    lines = [f'== {label}: {len(rows)} clips', f"{'feature':20s} {'rho0':>7s} {'rho1+':>7s} {'pct0':>6s} {'pct1+':>6s}"]
    for j, nme in enumerate(names):
        r0 = [s['rho'][j] for r in rows for s in r['steps'] if s['step'] == 0]
        r1 = [s['rho'][j] for r in rows for s in r['steps'] if s['step'] > 0]
        p0 = [s['pct'][j] for r in rows for s in r['steps'] if s['step'] == 0]
        p1 = [s['pct'][j] for r in rows for s in r['steps'] if s['step'] > 0]
        lines.append(f'{nme:20s} {np.nanmean(r0):7.3f} {np.nanmean(r1):7.3f} {np.nanmean(p0):6.2f} {np.nanmean(p1):6.2f}')
    gains = {}
    for r in rows:
        for s in r['steps']:
            gains.setdefault(s['step'], []).append((s['best_gain'], s['positive_frac']))
    lines.append('step: mean best gain / frac candidates with positive gain')
    lines.append(' '.join(f"{k}:{np.mean([g for g, _ in v]):.3f}/{np.mean([p for _, p in v]):.2f}" for k, v in sorted(gains.items())))
    delta = {k: np.nanmean([r['pick_vs_clip_median'][k] for r in rows if r['pick_vs_clip_median']]) for k in rows[0]['pick_vs_clip_median']}
    lines.append('picked frames minus clip median: ' + ', '.join(f'{k} {v:+.3f}' for k, v in delta.items()))
    return '\n'.join(lines)


if __name__ == '__main__':
    clips = sorted(p.stem for p in (OUT / 'greedy').glob('AM*.npz'))
    with Pool(40) as pool:
        rows = pool.map(analyse, clips)
    (OUT / 'greedy' / 'pick_analysis.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    report = table(rows, 'all') + '\n\n' + table([r for r in rows if r['split'] == 'validation'], 'validation')
    (OUT / 'greedy' / 'pick_analysis.txt').write_text(report)
    print(report)
