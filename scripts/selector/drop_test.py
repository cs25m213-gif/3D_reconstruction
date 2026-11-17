"""Does dropping negative-contribution frames help? Validation clips, fused F vs LiDAR (fixed poses/depth context)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
from uav_reconstruction.adaptive_uav.contribution_labels import load_clip, value
from uav_reconstruction.adaptive_uav.train_contribution_selector import Scorer, features, dino_bins, load
from uav_reconstruction.adaptive_uav.student_inference import load_student
from uav_reconstruction.adaptive_uav.utility_contracts import load_contract
from uav_reconstruction.adaptive_uav.reconstruction_quality import cluster_interval

OUT = Path('runs/contribution_20261109'); M = '/home/vishnu/geostreamfusion_more_data_20261104/data'
BUDGETS = (4, 8, 16)

def keep(picks, ok):
    kept = [i for i in picks if ok[i]]
    return kept or [max(picks, key=lambda i: ok[i])]

def run(c):
    torch.set_num_threads(4)
    contract = load_contract(Path('runs/contract_vishnu.json'), ('utility.quality_protocol',))
    clip, ref = load_clip(Path('.').resolve(), contract, c['clip'], load_student(Path(M) / 'student.pt', 'cpu'), 'cpu')
    voxel, views, n = contract['utility']['voxel_m'], clip.views, len(clip.views)
    state = torch.load(OUT / 'selector/best.pt', weights_only=False)
    model = Scorer(state['width']); model.load_state_dict(state['model']); model.eval()
    with torch.no_grad():
        score = model(torch.from_numpy(c['x'])).numpy()
    y, se = c['y'], c['se']
    pos, sig, low = y > 0, y > 2 * se, score >= np.quantile(score, .25)
    F = lambda idx: value([views[i] for i in sorted(set(idx))], ref, voxel)
    row = {'clip': c['clip'], 'flight': c['flight'], 'frames': n, 'all': F(range(n)),
           'all_positive': F(np.flatnonzero(pos)), 'n_positive': int(pos.sum()),
           'all_above_2se': F(keep(list(range(n)), sig)), 'n_above_2se': int(sig.sum()), 'budgets': {}}
    for k in BUDGETS:
        if k >= n: continue
        bins = dino_bins(c['emb'], k)
        mid = [int(b[len(b) // 2]) for b in bins]
        orc = [int(b[np.argmax(y[b])]) for b in bins]
        row['budgets'][k] = {'middle': F(mid), 'middle_drop_oracle': F(keep(mid, pos)), 'n_mid_kept': len(keep(mid, pos)),
                             'oracle': F(orc), 'oracle_drop': F(keep(orc, pos)),
                             'middle_drop_learned': F(keep(mid, low))}
    print(json.dumps(row), flush=True)
    return row

if __name__ == '__main__':
    val = load(OUT / 'labels')['validation']
    with Pool(20) as pool:
        rows = pool.map(run, val)
    out = OUT / 'drop_test'; out.mkdir(exist_ok=True)
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summ = {'all_frames': {m: float(np.mean([r[m] for r in rows])) for m in ('all', 'all_positive', 'all_above_2se')},
            'frac_positive': float(np.mean([r['n_positive'] / r['frames'] for r in rows]))}
    g = {}
    for r in rows: g.setdefault(r['flight'], []).append(r['all_positive'] - r['all'])
    summ['all_positive_minus_all'] = cluster_interval(g)
    for k in BUDGETS:
        part = [r['budgets'][k] | {'flight': r['flight']} for r in rows if k in r['budgets']]
        e = {m: float(np.mean([p[m] for p in part])) for m in ('middle', 'middle_drop_oracle', 'oracle', 'oracle_drop', 'middle_drop_learned')}
        for m in ('middle_drop_oracle', 'middle_drop_learned'):
            g = {}
            for p in part: g.setdefault(p['flight'], []).append(p[m] - p['middle'])
            e[f'{m}_minus_middle'] = cluster_interval(g)
        e['mean_kept_of_k'] = float(np.mean([p['n_mid_kept'] for p in part])); e['clips'] = len(part)
        summ[str(k)] = e
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
