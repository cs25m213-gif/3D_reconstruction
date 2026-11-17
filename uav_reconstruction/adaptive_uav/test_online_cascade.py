"""CPU feasibility experiment: causal online depth-aware selector and two-exit cascade.

Uses fixed-context cached student depths/reference poses. Requested depth counts are
logical demand counts, NOT measured network calls or end-to-end latency savings.
Training always measures rejected actions. No candidate LiDAR features enter a policy.
Old experiment inputs are read only; all outputs go to a fresh --work directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from . import bin_memory_selector as bm
from . import reconstruction_selector as rs
from .train_contribution_selector import dino_bins

CFG = rs.CFG
MODULE = 'uav_reconstruction.adaptive_uav.test_online_cascade'
THETAS = (.4, .5, .6, .7)
MARGINS = (0., .05, .1, .2, .35, 1.)
GEOMETRY = ('valid_fraction', 'log_depth_median', 'depth_p90_p10', 'log_footprint',
            'log_depth_gradient', 'agreement_selected', 'overlap_selected',
            'agreement_available')


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def log(message):
    row = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}'
    print(row, flush=True)
    with (CFG.work / 'progress.log').open('a') as f:
        f.write(row + '\n')


def parallel(fn, items):
    if not items:
        raise RuntimeError(f'Empty stage {fn}')
    rows = rs.parallel(fn, items, MODULE)
    if len(rows) != len(items) or any('error' in r for r in rows):
        raise RuntimeError(f'{fn}: incomplete outputs; see failures.txt')
    return rows


class Clip(bm.Clip):
    def __init__(self, name, branch=None):
        # Guard against a cache miss writing to the baseline experiment.
        original_work = CFG.work
        CFG.work = CFG.base
        try:
            cache = CFG.base / 'student_depth' / f'{rs._student_key()}_v2' / f'{name}.npz'
            if not cache.is_file() or not (CFG.base / 'tables' / f'{name}.npz').is_file():
                raise RuntimeError(f'Required baseline cache missing: {name}')
            super().__init__(name, need_pred=False)
        finally:
            CFG.work = original_work
        self.pe = (np.load(CFG.work / 'error' / branch / f'{name}.npy')
                   if branch else np.zeros(self.ctx.n, np.float32))
        self.requested = set()
        self.extra = {}

    def demand(self, t, now):
        if t > now:
            raise AssertionError('Future depth requested')
        self.requested.add(t)

    def geometry(self, t, selected):
        self.demand(t, t)
        if t not in self.extra:
            view = self.ctx.views[t]
            d = view.fusion_depth
            valid = np.isfinite(d) & (d > 0)
            z = d[valid]
            points = self.points(t)
            if len(z):
                q10, med, q90 = np.percentile(z, [10, 50, 90])
                grad = np.gradient(np.log(np.maximum(d, 1e-3)))
                slope = float(np.hypot(*grad)[valid].mean())
                box = np.percentile(points[:2], [5, 95], axis=1) if points.shape[1] else np.zeros((2, 2))
                base = [float(valid.mean()), float(np.log(max(med, 1e-3))),
                        float(q90 / max(q10, 1e-3)), float(np.log1p(np.prod(box[1] - box[0]))), slope]
            else:
                base = [0.] * 5
            self.extra[t] = base
        points = self.points(t)
        agree, checked = 0, 0
        for s in selected:
            if s >= t or s not in self.requested:
                raise AssertionError('Consistency accessed unrequested or non-past depth')
            same, seen = rs.consistent(points, self.ctx.views[s])
            agree += int(same.sum())
            checked += int(seen.sum())
        extra = self.extra[t] + [agree / max(checked, 1),
                                checked / max(len(selected) * points.shape[1], 1), float(checked > 0)]
        return np.asarray(extra, np.float32)


def policy(path, c):
    data = torch.load(path, map_location='cpu', weights_only=False)
    net = bm.BinScorer(c.ctx.emb.shape[1], data['width'])
    net.load_state_dict(data['model'])
    net.eval()
    emb = torch.from_numpy(c.ctx.emb)

    def score(t, a, x):
        with torch.no_grad():
            value = net(emb[t:t+1], emb[(a if a is not None else 0): (a if a is not None else 0)+1],
                        torch.tensor([float(a is not None)]), torch.from_numpy(x[None]))
        return float(value.sigmoid())
    return score


def rollout(c, cheap=None, deep=None, theta=.5, deep_theta=.5, margin=0.,
            measure=False, rng=None, behavior='cascade'):
    c.requested = set()
    selected, prototypes, occupants, records = [], [], [], []
    base_f, swaps, routed = 0., 0, 0
    start = time.perf_counter()
    for t in range(c.ctx.n):
        ls = np.log1p(c.sharp)
        if t >= 8 and ls[t] < np.quantile(ls[:t], .1):
            continue
        cos = np.asarray([p @ c.ctx.emb[t] for p in prototypes])
        if len(cos) and cos.max() >= CFG.tau:
            b = int(cos.argmax())
        else:
            b = len(prototypes)
            prototypes.append(c.ctx.emb[t])
            occupants.append(None)
        a = occupants[b]
        if not set(selected).issubset(c.requested):
            raise AssertionError('Selected frame depth was not counted')
        x = bm.state_features(c, t, selected, len(prototypes), a, cos)
        changed = selected + [t] if a is None else [s for s in selected if s != a] + [t]
        gain = c.ctx.F(changed) - base_f if measure else None
        p = cheap(t, a, x) if cheap else .5
        need_deep = margin >= 1. or (margin > 0 and abs(p - theta) <= margin)
        xd = np.concatenate([x, c.geometry(t, selected)]) if (measure or need_deep) else None
        if measure and deep is None:
            take = gain > 0
        elif behavior == 'cheap':
            take = p >= theta
        elif need_deep or behavior == 'deep':
            take = deep(t, a, xd) >= deep_theta
            routed += 1
        else:
            take = p >= theta
        if rng is not None and rng.random() < .15:
            take = not take
        if take:
            c.demand(t, t)  # Includes every replaced occupant, not just final selection.
            swaps += int(a is not None)
            selected, occupants[b] = changed, t
            if measure:
                base_f += gain
        if measure:
            records.append((t, -1 if a is None else a, x, xd, gain))
    elapsed = time.perf_counter() - start
    return selected, records, {'requested_depth_frames': sorted(c.requested),
                              'requested_depth_count': len(c.requested), 'routed_count': routed,
                              'swaps': swaps, 'replay_seconds': elapsed}


def prep_clip(name):
    c = Clip(name)
    np.savez_compressed(CFG.work / 'prep' / f'{name}.npz', emb=c.ctx.emb, scal=c.C,
                        err=c.ctx.err, flight=c.ctx.flight)
    return {'clip': name, 'frames': c.ctx.n, 'flight': c.ctx.flight}


def err_rows(names):
    result = []
    for name in names:
        with np.load(CFG.work / 'prep' / f'{name}.npz') as d:
            result.append({'clip': name, 'flight': str(d['flight']), 'emb': d['emb'],
                           'scal': d['scal'], 'err': d['err'], 'ok': np.isfinite(d['err'])})
    return result


def stage_prep():
    (CFG.work / 'prep').mkdir(exist_ok=True)
    rows = parallel('prep_clip', rs.clips())
    save(CFG.work / 'inventory.json', rows)
    train, val = err_rows(rs.clips('train')), err_rows(rs.clips('validation'))
    for branch in ('inner', 'full'):
        fit = [r for r in train if branch == 'full' or r['flight'] != 'AMtown03']
        folder = CFG.work / 'error' / branch
        folder.mkdir(parents=True, exist_ok=True)
        for flight in sorted({r['flight'] for r in fit}):
            other = [r for r in fit if r['flight'] != flight]
            _, predict, _, _ = bm._fit_err(other, CFG.error_epochs)
            for r in fit:
                if r['flight'] == flight:
                    np.save(folder / f'{r["clip"]}.npy', predict(r).astype(np.float32))
        net, predict, _, (mu, sd) = bm._fit_err(fit, CFG.error_epochs)
        for r in train + val:
            if r['clip'] not in {a['clip'] for a in fit}:
                np.save(folder / f'{r["clip"]}.npy', predict(r).astype(np.float32))
        torch.save({'model': net.state_dict(), 'mu': mu, 'sd': sd,
                    'training_flights': sorted({r['flight'] for r in fit})}, folder / 'model.pt')
    log('prep complete; inner error predictor excludes AMtown03 entirely')


def collect_clip(arg):
    name, branch, it = arg
    target = CFG.work / 'rows' / branch / f'it{it}' / f'{name}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    c = Clip(name, branch)
    # The holdout behavior always uses inner models, never the all-training model.
    cheap = policy(model_path(branch, 'cheap', it-1), c) if it else None
    deep = policy(model_path(branch, 'deep', it-1), c) if it else None
    rng = np.random.default_rng(zlib.crc32(f'{name}:{it}'.encode()))
    margin = float(rng.choice([0., .1, 1.]))
    _, records, _ = rollout(c, cheap, deep, margin=margin, measure=True, rng=rng)
    t, a, x, xd, y = zip(*records)
    np.savez_compressed(target, t=t, a=a, cheap=np.asarray(x, np.float32),
                        deep=np.asarray(xd, np.float32), y=np.asarray(y, np.float32))
    return {'clip': name, 'rows': len(records)}


def model_path(branch, kind, it):
    return CFG.work / 'models' / f'{branch}_{kind}_it{it}.pt'


def training_rows(names, branch, kind, it):
    result = [[] for _ in range(5)]
    for name in names:
        with np.load(CFG.work / 'prep' / f'{name}.npz') as d:
            emb = d['emb']
        for k in range(it + 1):
            with np.load(CFG.work / 'rows' / branch / f'it{k}' / f'{name}.npz') as d:
                values = (emb[d['t']], emb[np.maximum(d['a'], 0)], (d['a'] >= 0).astype(np.float32),
                          d[kind], d['y'])
                for acc, v in zip(result, values):
                    acc.append(v)
    return tuple(torch.from_numpy(np.concatenate(a)) for a in result)


def fit(data, epochs, hold=None):
    torch.manual_seed(0)
    net = bm.BinScorer(data[0].shape[1], data[3].shape[1])
    net.mu.copy_(data[3].mean(0))
    net.sd.copy_(data[3].std(0) + 1e-6)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=.01)
    best, state, best_epoch = float('inf'), None, epochs
    for epoch in range(epochs):
        net.train()
        for idx in torch.randperm(len(data[4])).split(256):
            optimizer.zero_grad()
            loss = bm._loss(net, data, idx)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
        if hold is not None:
            net.eval()
            with torch.no_grad():
                value = float(bm._loss(net, hold))
            if value < best:
                best, best_epoch = value, epoch + 1
                state = {k: v.clone() for k, v in net.state_dict().items()}
    if state is not None:
        net.load_state_dict(state)
    return net.eval(), best_epoch


def split_names():
    names = rs.clips('train')
    hold = [c for c in names if c.startswith('AMtown03_')]
    inner = [c for c in names if c not in hold]
    if not inner or not hold:
        raise RuntimeError('Training/holdout split empty')
    return names, inner, hold


def stage_train(it):
    names, inner, hold = split_names()
    (CFG.work / 'models').mkdir(exist_ok=True)
    for kind in ('cheap', 'deep'):
        data = training_rows(inner, 'inner', kind, it)
        held = training_rows(hold, 'inner', kind, it)
        net, epochs = fit(data, CFG.epochs, held)
        torch.save({'model': net.state_dict(), 'width': data[3].shape[1], 'epochs': epochs},
                   model_path('inner', kind, it))
        full = training_rows(names, 'full', kind, it)
        net, _ = fit(full, epochs)
        torch.save({'model': net.state_dict(), 'width': full[3].shape[1], 'epochs': epochs},
                   model_path('full', kind, it))
        log(f'train {kind} iteration {it}: epochs={epochs}, inner_rows={len(data[4])}')


def tune_clip(name):
    c = Clip(name, 'inner')
    rows = []
    for it in range(CFG.iterations):
        cheap, deep = (policy(model_path('inner', k, it), c) for k in ('cheap', 'deep'))
        for kind in ('cheap', 'deep'):
            for theta in THETAS:
                selected, _, cost = rollout(c, cheap, deep, theta=theta, deep_theta=theta,
                                            margin=0. if kind == 'cheap' else 1.)
                rows.append({'kind': kind, 'iteration': it, 'theta': theta,
                             'f': c.ctx.F(selected), 'depth_count': cost['requested_depth_count']})
    return {'clip': name, 'rows': rows}


def sweep_clip(arg):
    name, branch = arg
    c = Clip(name, branch)
    tune = json.loads((CFG.work / 'tune.json').read_text())
    cheap = policy(model_path(branch, 'cheap', tune['cheap']['iteration']), c)
    deep = policy(model_path(branch, 'deep', tune['deep']['iteration']), c)
    output = []
    for margin in MARGINS:
        selected, _, cost = rollout(c, cheap, deep, theta=tune['cheap']['theta'],
                                    deep_theta=tune['deep']['theta'], margin=margin)
        k = len(selected)
        middle = [b[len(b)//2] for b in dino_bins(c.ctx.emb, k)] if k else []
        uniform = np.linspace(0, c.ctx.n-1, k).round().astype(int).tolist() if k else []
        output.append({'margin': margin, 'selected': selected, 'f': c.ctx.F(selected),
                       'middle_f': c.ctx.F(middle), 'uniform_f': c.ctx.F(uniform), **cost})
    return {'clip': name, 'flight': c.ctx.flight, 'frames': c.ctx.n, 'rows': output}


def aggregate(rows):
    result = []
    total = sum(r['frames'] for r in rows)
    for j, margin in enumerate(MARGINS):
        pairs = {}
        for r in rows:
            pairs.setdefault(r['flight'], []).append(r['rows'][j]['f'] - r['rows'][j]['middle_f'])
        result.append({'margin': margin, 'f': float(np.mean([r['rows'][j]['f'] for r in rows])),
                       'middle_f': float(np.mean([r['rows'][j]['middle_f'] for r in rows])),
                       'uniform_f': float(np.mean([r['rows'][j]['uniform_f'] for r in rows])),
                       'requested_depth_fraction': sum(r['rows'][j]['requested_depth_count'] for r in rows) / total,
                       'routed_fraction': sum(r['rows'][j]['routed_count'] for r in rows) / total,
                       'selected_mean': float(np.mean([len(r['rows'][j]['selected']) for r in rows])),
                       'minus_middle_cluster_ci': rs.cluster_interval(pairs)})
    return result


def stage_tune():
    _, _, hold = split_names()
    rows = parallel('tune_clip', hold)
    save(CFG.work / 'holdout_grid.json', rows)
    chosen = {}
    for kind in ('cheap', 'deep'):
        configs = []
        for it in range(CFG.iterations):
            for theta in THETAS:
                values = [v['f'] for r in rows for v in r['rows']
                          if v['kind'] == kind and v['iteration'] == it and v['theta'] == theta]
                configs.append({'iteration': it, 'theta': theta, 'holdout_f': float(np.mean(values))})
        chosen[kind] = max(configs, key=lambda r: r['holdout_f'])
    save(CFG.work / 'tune.json', chosen)
    curves = parallel('sweep_clip', [(c, 'inner') for c in hold])
    save(CFG.work / 'holdout_sweep_rows.json', curves)
    summary = aggregate(curves)
    deep_f = summary[-1]['f']
    acceptable = [r for r in summary if r['f'] >= deep_f - .01]
    chosen['margin'] = min(acceptable, key=lambda r: (r['requested_depth_fraction'], -r['f']))['margin']
    chosen['margin_rule'] = 'minimum requested-depth fraction within 0.01 F of deep baseline on AMtown03'
    chosen['holdout_curve'] = summary
    save(CFG.work / 'tune.json', chosen)
    log(f'tuned on AMtown03 only: {chosen["cheap"]}, {chosen["deep"]}, margin={chosen["margin"]}')


def stage_eval():
    rows = parallel('sweep_clip', [(c, 'full') for c in rs.clips('validation')])
    save(CFG.work / 'validation_rows.json', rows)
    curve = aggregate(rows)
    tune = json.loads((CFG.work / 'tune.json').read_text())
    report = {'clips': len(rows), 'flights': sorted({r['flight'] for r in rows}), 'tune': tune,
              'curve': curve, 'chosen': next(r for r in curve if r['margin'] == tune['margin']),
              'limitations': ['Reference poses and fixed-context cached student depths.',
                              'Depth counts are logical unique demand accesses, not actual network calls.',
                              'Replay seconds exclude cache preparation, DINO, pose tracking and final TSDF scoring.',
                              'AMvalley is development validation, not an untouched test set.',
                              'No student update, no joint architecture or coarse depth exit tested.']}
    save(CFG.work / 'summary.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['requested_depth_fraction'] for r in curve], [r['f'] for r in curve], 'o-')
    for r in curve:
        ax.annotate(f'm={r["margin"]:g}', (r['requested_depth_fraction'], r['f']), xytext=(4, 5), textcoords='offset points')
    ax.set(xlabel='Fraction of frames requesting depth (cached replay)', ylabel='Mean LiDAR F-score',
           title='Causal two-exit cascade — development validation')
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(CFG.work / 'quality_cost_curve.png', dpi=180)
    plt.close(fig)
    log(f'evaluation complete: {report["chosen"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--base', type=Path, default=Path('runs/selector_20261111'))
    p.add_argument('--labels', type=Path, default=Path('runs/contribution_20261109/labels'))
    p.add_argument('--greedy', type=Path, default=Path('runs/contribution_20261109/greedy'))
    p.add_argument('--config', type=Path, default=Path('runs/contract_vishnu.json'))
    p.add_argument('--student', type=Path, default=Path('/home/vishnu/geostreamfusion_more_data_20261104/data/student.pt'))
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--iterations', type=int, default=2)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--error-epochs', type=int, default=48)
    p.add_argument('--tau', type=float, default=.9)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--stage', choices=('all', 'prep', 'eval'), default='all')
    a = p.parse_args()
    CFG.__dict__.update(vars(a))
    CFG.root = Path.cwd()
    if len(bm.FRAME) != 6:
        raise RuntimeError('Unset SELECTOR_CONF; this experiment excludes student confidence')
    CFG.work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    if a.stage == 'all':
        if (CFG.work / 'experiment.json').exists():
            raise RuntimeError('Use a fresh work directory; will not overwrite an existing run')
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        config['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        config['student_sha256'] = hashlib.sha256(CFG.student.read_bytes()).hexdigest()
        config['tau_note'] = 'Fixed historical tau=0.9; architecture feasibility, not a fresh test protocol'
        save(CFG.work / 'experiment.json', config)
        log('START')
        stage_prep()
        names, _, _ = split_names()
        for it in range(CFG.iterations):
            jobs = [(c, b, it) for b in ('inner', 'full') for c in names]
            done = parallel('collect_clip', jobs)
            log(f'collect iteration {it}: {len(done)} clip/branch jobs')
            stage_train(it)
        stage_tune()
        stage_eval()
        log('DONE')
    elif a.stage == 'prep':
        stage_prep()
    else:
        stage_eval()


if __name__ == '__main__':
    main()
