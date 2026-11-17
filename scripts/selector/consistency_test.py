"""LiDAR-free harmful-frame predictor: multi-view depth consistency.
Each frame's student depth is reprojected into frames i-2..i+2 (fixed poses, as in the labels) and compared
with their depth. agree = fraction of reprojected pixels within 5% relative depth. Low agreement -> drop.
Rule fixed in advance (no tuning): drop frames below the clip's 30th percentile of agree (~30% of frames had
negative Shapley). Validation clips only."""
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
from drop_test import keep, OUT, M, BUDGETS
from uav_reconstruction.adaptive_uav.contribution_labels import load_clip, value
from uav_reconstruction.adaptive_uav.train_contribution_selector import dino_bins, load, spearman
from uav_reconstruction.adaptive_uav.student_inference import load_student
from uav_reconstruction.adaptive_uav.utility_contracts import load_contract
from uav_reconstruction.adaptive_uav.reconstruction_quality import cluster_interval

def agreement(views, i, offsets=(-2, -1, 1, 2), stride=4, tol=.05):
    d, K, T = views[i].fusion_depth, views[i].fusion_k, views[i].reference_pose
    v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
    z = d[v, u]; m = z > 0
    pix = np.stack([u[m], v[m], np.ones(m.sum())]) * z[m]
    world = T[:3, :3] @ (np.linalg.inv(K) @ pix) + T[:3, 3:4]
    ok = []
    for o in offsets:
        j = i + o
        if not 0 <= j < len(views): continue
        dj, Kj, Tj = views[j].fusion_depth, views[j].fusion_k, views[j].reference_pose
        Tinv = np.linalg.inv(Tj)
        cam = Tinv[:3, :3] @ world + Tinv[:3, 3:4]
        zc = cam[2]; front = zc > 1e-3
        p = Kj @ cam[:, front]; uj, vj = (p[0] / p[2]).round().astype(int), (p[1] / p[2]).round().astype(int)
        inside = (uj >= 0) & (uj < dj.shape[1]) & (vj >= 0) & (vj < dj.shape[0])
        ref = dj[vj[inside], uj[inside]]; zz = zc[front][inside]; valid = ref > 0
        ok.append(np.abs(zz[valid] - ref[valid]) / ref[valid] < tol)
    ok = np.concatenate(ok) if ok else np.array([])
    return float(ok.mean()) if len(ok) >= 50 else np.nan

def auc(score, positive):
    a, b = score[positive], score[~positive]
    if not len(a) or not len(b): return np.nan
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())

def run(c):
    torch.set_num_threads(4)
    contract = load_contract(Path('runs/contract_vishnu.json'), ('utility.quality_protocol',))
    clip, ref = load_clip(Path('.').resolve(), contract, c['clip'], load_student(Path(M) / 'student.pt', 'cpu'), 'cpu')
    voxel, views, n = contract['utility']['voxel_m'], clip.views, len(clip.views)
    agree = np.array([agreement(views, i) for i in range(n)])
    agree = np.where(np.isnan(agree), np.nanmedian(agree), agree)
    y = c['y']; good = agree >= np.quantile(agree, .3)
    F = lambda idx: value([views[i] for i in sorted(set(idx))], ref, voxel)
    row = {'clip': c['clip'], 'flight': c['flight'], 'frames': n, 'agree': agree.round(4).tolist(),
           'spearman_agree_shapley': spearman(agree, y), 'auc_agree_vs_positive': auc(agree, y > 0),
           'all': F(range(n)), 'all_drop_consistency': F(np.flatnonzero(good)), 'budgets': {}}
    for k in BUDGETS:
        if k >= n: continue
        bins = dino_bins(c['emb'], k)
        mid = [int(b[len(b) // 2]) for b in bins]
        best = [int(b[np.argmax(agree[b])]) for b in bins]
        row['budgets'][k] = {'middle': F(mid), 'middle_drop_consistency': F(keep(mid, good)),
                             'most_consistent_in_bin': F(best), 'most_consistent_drop': F(keep(best, good)),
                             'kept': len(keep(mid, good))}
    print(json.dumps({k: v for k, v in row.items() if k != 'agree'}), flush=True)
    return row

if __name__ == '__main__':
    val = load(OUT / 'labels')['validation']
    with Pool(20) as pool:
        rows = pool.map(run, val)
    out = OUT / 'consistency_test'; out.mkdir(exist_ok=True)
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    def boot(a, b, part):
        g = {}
        for p in part: g.setdefault(p['flight'], []).append(p[a] - p[b])
        return cluster_interval(g)
    s = {'spearman_agree_shapley_mean': float(np.nanmean([r['spearman_agree_shapley'] for r in rows])),
         'auc_agree_vs_positive_mean': float(np.nanmean([r['auc_agree_vs_positive'] for r in rows])),
         'all': float(np.mean([r['all'] for r in rows])), 'all_drop_consistency': float(np.mean([r['all_drop_consistency'] for r in rows])),
         'all_drop_minus_all': boot('all_drop_consistency', 'all', rows)}
    for k in BUDGETS:
        part = [r['budgets'][k] | {'flight': r['flight']} for r in rows if k in r['budgets']]
        e = {m: float(np.mean([p[m] for p in part])) for m in ('middle', 'middle_drop_consistency', 'most_consistent_in_bin', 'most_consistent_drop', 'kept')}
        for m in ('middle_drop_consistency', 'most_consistent_in_bin', 'most_consistent_drop'):
            e[f'{m}_minus_middle'] = boot(m, 'middle', part)
        e['clips'] = len(part); s[str(k)] = e
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    print('SUMMARY', json.dumps(s))
