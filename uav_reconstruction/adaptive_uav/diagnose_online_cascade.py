"""Post-hoc diagnostics, not tuning: replay fixed policies and measure action gains."""
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from . import test_online_cascade as ex
from . import reconstruction_selector as rs

OUT = Path('runs/cascade_failure_diagnosis_20261114')
RUN = Path('runs/online_cascade_20261113')


def initialize(config):
    cfg = dict(config)
    cfg['work'] = OUT
    rs._init(cfg)
    rs.CFG.__dict__.update(config)


def job(arg):
    name, branch = arg
    c = ex.Clip(name, branch)
    tune = json.loads((RUN / 'tune.json').read_text())
    cheap = ex.policy(ex.model_path(branch, 'cheap', tune['cheap']['iteration']), c)
    deep = ex.policy(ex.model_path(branch, 'deep', tune['deep']['iteration']), c)
    theta, dt = tune['cheap']['theta'], tune['deep']['theta']
    result = {'clip': name, 'flight': c.ctx.flight, 'branch': branch, 'policies': {}}
    expected = {r['clip']: r for r in json.loads((RUN / 'validation_rows.json').read_text())}
    for mode, margin in [('cheap', 0.), ('cascade', tune['margin']), ('deep', 1.)]:
        selected, rows, _ = ex.rollout(c, cheap, deep, theta=theta, deep_theta=dt, margin=margin, measure=True)
        t, a, x, xd, gains = zip(*rows)
        pc = np.array([cheap(i, None if aa < 0 else aa, xx) for i, aa, xx in zip(t, a, x)])
        pd = np.array([deep(i, None if aa < 0 else aa, xx) for i, aa, xx in zip(t, a, xd)])
        routed = np.ones(len(rows), bool) if margin == 1 else (np.abs(pc-theta) <= margin if margin else np.zeros(len(rows), bool))
        action = np.where(routed, pd >= dt, pc >= theta)
        gains = np.asarray(gains)
        final_f = c.ctx.F(selected)
        if name in expected:
            old = next(r for r in expected[name]['rows'] if r['margin'] == margin)
            if selected != old['selected'] or abs(final_f-old['f']) > 1e-8:
                raise AssertionError(f'Replay changed for {name} {mode}')
        np.savez_compressed(OUT / f'{name}_{mode}.npz', t=t, a=a, x=x, xd=xd, gain=gains,
                            cheap=pc, deep=pd, routed=routed, action=action)
        result['policies'][mode] = {'f': final_f, 'selected': selected, 'decisions': len(rows),
                                   'accepted_harmful': int(((gains < -.005) & action).sum()),
                                   'rejected_useful': int(((gains > .005) & ~action).sum())}
    # Descriptive threshold cross-over; never changes the saved deployment choice.
    result['threshold_controls'] = {}
    for mode, ct, dtheta, margin in [('cheap_at_deep_threshold', dt, dt, 0.),
                                     ('deep_at_cheap_threshold', theta, theta, 1.)]:
        selected, _, _ = ex.rollout(c, cheap, deep, theta=ct, deep_theta=dtheta, margin=margin)
        result['threshold_controls'][mode] = c.ctx.F(selected)
    ex.save(OUT / f'{name}.json', result)
    return result


def rank(x, y):
    return float(spearmanr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else None


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    config = json.loads((RUN / 'experiment.json').read_text())
    for key in ('work', 'base', 'labels', 'greedy', 'config', 'student'):
        config[key] = Path(config[key])
    config['root'] = Path.cwd()
    rs.CFG.__dict__.update(config)
    names, _, hold = ex.split_names()
    jobs = [(c, 'inner') for c in hold] + [(c, 'full') for c in rs.clips('validation')]
    with ProcessPoolExecutor(16, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(config,)) as pool:
        rows = list(pool.map(job, jobs))
    ex.save(OUT / 'replay_summary.json', rows)
    result = {'scope': 'Post-hoc development diagnosis. Threshold controls are not a new tuned test result.', 'splits': {}}
    for branch in ('inner', 'full'):
        group = [r for r in rows if r['branch'] == branch]
        entry = {'clips': len(group), 'mean_f': {}, 'decisions': {}}
        for mode in ('cheap', 'cascade', 'deep'):
            entry['mean_f'][mode] = float(np.mean([r['policies'][mode]['f'] for r in group]))
            parts = [dict(np.load(OUT / f'{r["clip"]}_{mode}.npz')) for r in group]
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            g, pc, pd = data['gain'], data['cheap'], data['deep']
            ca, da = pc >= .4, pd >= .6
            routed = data['routed']
            swaps = data['a'] >= 0
            describe = {}
            for label, mask in [('all', np.ones(len(g), bool)), ('add', ~swaps), ('swap', swaps), ('routed', routed)]:
                if not mask.any():
                    continue
                delta = (da.astype(int)-ca.astype(int))*g
                describe[label] = {'count': int(mask.sum()), 'positive_gain_fraction': float((g[mask] > .005).mean()),
                    'cheap_gain_spearman': rank(pc[mask], g[mask]), 'deep_gain_spearman': rank(pd[mask], g[mask]),
                    'cheap_immediate_regret_mean': float(np.maximum(np.where(ca[mask], -g[mask], g[mask]), 0).mean()),
                    'deep_immediate_regret_mean': float(np.maximum(np.where(da[mask], -g[mask], g[mask]), 0).mean()),
                    'deep_override_beneficial': int(((delta > .005) & mask).sum()),
                    'deep_override_harmful': int(((delta < -.005) & mask).sum()),
                    'sum_one_step_override_gain': float(delta[mask].sum()),
                    'cheap_accepts': int((ca & mask).sum()), 'deep_accepts': int((da & mask).sum())}
            calibration = {}
            for label, pred in [('cheap', pc), ('deep', pd)]:
                bins = []
                for lo, hi in [(0., .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.000001)]:
                    mask = (pred >= lo) & (pred < hi)
                    if mask.any():
                        bins.append({'score_interval': [lo, min(hi, 1.)], 'n': int(mask.sum()),
                                     'mean_gain': float(g[mask].mean()),
                                     'positive_gain_fraction': float((g[mask] > .005).mean())})
                calibration[label] = bins
            entry['decisions'][mode] = {'comparisons': describe, 'score_gain_bins': calibration,
                'accepted_harmful': int(((g < -.005) & data['action']).sum()),
                'rejected_useful': int(((g > .005) & ~data['action']).sum())}
        for key in group[0]['threshold_controls']:
            entry['mean_f'][key] = float(np.mean([r['threshold_controls'][key] for r in group]))
        result['splits'][branch] = entry
    # Quantify raw geometric feature shift against the exact full deep training rows.
    train = []
    it = json.loads((RUN / 'tune.json').read_text())['deep']['iteration']
    for name in names:
        for k in range(it+1):
            with np.load(RUN / 'rows' / 'full' / f'it{k}' / f'{name}.npz') as z:
                train.append(z['deep'][:, -len(ex.GEOMETRY):])
    train = np.concatenate(train)
    validation = np.concatenate([np.load(OUT / f'{r["clip"]}_cascade.npz')['xd'][:, -len(ex.GEOMETRY):]
                                 for r in rows if r['branch']=='full'])
    result['geometry_shift'] = []
    for i, name in enumerate(ex.GEOMETRY):
        mean, sd = float(train[:, i].mean()), float(train[:, i].std())
        result['geometry_shift'].append({'feature': name, 'train_mean': mean, 'train_sd': sd,
            'val_mean': float(validation[:, i].mean()), 'val_median': float(np.median(validation[:, i])),
            'val_fraction_outside_train_3sd': float((np.abs(validation[:, i]-mean) > 3*max(sd, 1e-6)).mean())})
    ex.save(OUT / 'summary.json', result)
    print(json.dumps({'mean_f': {b:v['mean_f'] for b,v in result['splits'].items()},
                      'cascade_decisions': result['splits']['full']['decisions']['cascade']['comparisons']}, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
