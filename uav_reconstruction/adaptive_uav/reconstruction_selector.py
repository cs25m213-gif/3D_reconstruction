"""Reconstruction-driven frame selector: the score is the reconstruction gain, learned from the reconstruction.

Score (design notes):
    D(f | S)         = F(S + {f}) - F(S)                 gain of adding frame f to the selected set S
    D(a -> b | S)    = F(S - {a} + {b}) - F(S)            gain of swapping a selected frame a for b
                     = D(b | S - {a}) - D(a | S - {a})
F is the fused-surface F-score against the fixed LiDAR reference (TSDF fusion, fixed reference poses).
It is not differentiable, so it is the training target: a scorer Q(f | S) is regressed onto D(f | S)
measured by re-fusing (MSE + within-state ListNet ranking loss).

Training uses the reconstruction in the loop (DAgger / learning to search):
    iteration 0   states visited by the greedy oracle (measured D for every candidate; greedy_selector labels)
    iteration t   run the current selector on every training clip, re-fuse every candidate at every state it
                  visits (plus swap states S - {a}), add those measured gains, retrain on everything so far.
So the selector learns D on the states it actually reaches, including its own mistakes.

Runtime selection (no LiDAR):
    step: Q for every unselected frame; band = frames with Q >= max Q - band * |max Q| (the "within 5% of
          the max" rule); inside the band take the frame with most new coverage; stop when max Q < tau.
    then one swap pass: replace a by b when Q(b | S-a) - Q(a | S-a) > 0.
    band, swap on/off and tau are chosen on a held-out training flight, never on validation.

State features for Q(f | S), all runtime-available:
    predicted depth error of f (depth-error net, out-of-fold on training clips), depth agreement with
    neighbours (both sides and past-only), log median depth, depth p90/p10, log footprint, height above the
    clip's lowest camera, off-nadir angle, sharpness, brightness, depth-gradient; new coverage of f given S
    (fraction of f's depth points not seen consistently by any frame of S), DINO cosine max/mean to S,
    how many of S share f's DINO novelty bin (protected-prototype bins, tau .90, null slot for blur),
    |S|, and the predicted errors of S.

Backward pass (evaluation): for the final selection, every selected frame's measured leave-one-out gain,
its DINO bin and its predicted and true depth error.

Stages (python -m uav_reconstruction.adaptive_uav.reconstruction_selector <stage>):
    tables -> depthnet -> label --iteration 0 -> train --iteration 0 -> label 1 -> train 1 -> ... -> tune -> eval
Scope: fixed reference poses and fixed-context student depth, as in contribution_labels.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from .contribution_labels import load_clip, novelty_bins, value
from .reconstruction_quality import cluster_interval
from .student_inference import load_student
from .train_contribution_selector import dino_bins, listnet, spearman
from .utility_contracts import load_contract

SCALARS = ('agree', 'agree_past', 'log_depth_median', 'depth_range_ratio', 'log_footprint', 'rel_height',
           'off_nadir', 'sharp_z', 'bright_z', 'grad_mean')
FEATURES = ('pred_log_err',) + SCALARS + ('new_cover', 'cos_max', 'cos_mean', 'bin_count', 'null_bin', 'set_size',
                                          'set_min_err', 'set_mean_err', 'err_minus_set_min')
COVER = FEATURES.index('new_cover')
BIN_TAU = .90
LABEL_BUDGET = 8
SWAP_STATES = 3
BANDS = (0., .05)
EVAL_BUDGETS = (2, 3, 4, 8)
CFG = SimpleNamespace()
_STUDENT = None


# ----------------------------------------------------------------------------- geometry
def back_project(view, stride=8):
    d, K, T = view.fusion_depth, view.fusion_k, view.reference_pose
    r, c = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
    z = d[r, c]
    m = z > 0
    pix = np.stack([c[m], r[m], np.ones(m.sum())]) * z[m]
    return T[:3, :3] @ (np.linalg.inv(K) @ pix) + T[:3, 3:4], z[m]


def consistent(points, view, tol=.05):
    """For world points, True where `view` sees the same surface within tol relative depth."""
    d, K = view.fusion_depth, view.fusion_k
    inv = np.linalg.inv(view.reference_pose)
    cam = inv[:3, :3] @ points + inv[:3, 3:4]
    zc = cam[2]
    out, checked = np.zeros(points.shape[1], bool), np.zeros(points.shape[1], bool)
    front = zc > 1e-3
    p = K @ cam[:, front]
    u, w = (p[0] / p[2]).round().astype(int), (p[1] / p[2]).round().astype(int)
    ins = (u >= 0) & (u < d.shape[1]) & (w >= 0) & (w < d.shape[0])
    idx = np.flatnonzero(front)[ins]
    ref = d[w[ins], u[ins]]
    valid = ref > 0
    checked[idx[valid]] = True
    out[idx[valid]] = np.abs(zc[idx[valid]] - ref[valid]) < tol * ref[valid]
    return out, checked


def agreement(views, i, points, offsets):
    ok, seen = [], []
    for o in offsets:
        j = i + o
        if 0 <= j < len(views):
            a, c = consistent(points, views[j])
            ok.append(a[c])
    ok = np.concatenate(ok) if ok else np.array([])
    return float(ok.mean()) if len(ok) >= 50 else np.nan


def frame_table(views, emb, sharp, bright, reference_points):
    t0 = time.perf_counter()
    n, pts, rows = len(views), [], []
    for i, v in enumerate(views):
        world, z = back_project(v)
        pts.append(world)
        d = v.fusion_depth
        gy, gx = np.gradient(np.log(np.maximum(d, 1e-3)))
        if len(z) > 20:
            lo, hi = np.percentile(world[:2].T, [5, 95], axis=0)
            geo = [np.log(np.median(z)), np.percentile(z, 90) / max(np.percentile(z, 10), 1e-3), np.log1p(np.prod(hi - lo))]
        else:
            geo = [np.nan, np.nan, np.nan]
        rows.append([agreement(views, i, world, (-2, -1, 1, 2)), agreement(views, i, world, (-2, -1))] + geo +
                    [v.reference_pose[2, 3], np.degrees(np.arccos(np.clip(-v.reference_pose[2, 2], -1, 1))), 0., 0.,
                     float(np.hypot(gx, gy)[d > 0].mean()) if (d > 0).any() else 0.])
    scal = np.asarray(rows, np.float64)
    scal[:, 5] -= scal[:, 5].min()
    ls = np.log1p(sharp)
    scal[:, 7] = (ls - ls.mean()) / (ls.std() + 1e-6)
    scal[:, 8] = (bright - bright.mean()) / (bright.std() + 1e-6)
    for j in range(scal.shape[1]):
        bad = ~np.isfinite(scal[:, j])
        scal[bad, j] = np.nanmedian(scal[:, j]) if (~bad).any() else 0.
    ms = (time.perf_counter() - t0) * 1000 / n
    tree = cKDTree(reference_points)
    err = np.array([np.median(tree.query(p.T)[0]) if p.shape[1] else np.nan for p in pts])
    seen = []
    for i in range(n):
        row = np.zeros((n, pts[i].shape[1]), bool)
        for j in range(n):
            if j != i:
                row[j] = consistent(pts[i], views[j])[0]
        seen.append(row)
    return scal.astype(np.float32), err, ms, seen


# ----------------------------------------------------------------------------- clip context
def student():
    global _STUDENT
    if _STUDENT is None:
        _STUDENT = load_student(CFG.student, 'cpu')
    return _STUDENT


def _student_key():
    stat = Path(CFG.student).stat()
    return f'{zlib.crc32(str(Path(CFG.student).resolve()).encode()):08x}_{stat.st_size}_{int(stat.st_mtime)}'


def load_clip_cached(contract, name):
    """load_clip with the fixed-context student depth cached per clip (keyed by the student checkpoint).
    The student runs once per clip; later loads replay its exact outputs, so workers that only reload
    a clip never execute the CPU convolution that intermittently segfaults.
    The student's per-pixel confidence is cached too and attached as view.student_conf."""
    from . import counterfactual_replay as cr
    cache = CFG.work / 'student_depth' / f'{_student_key()}_v2' / f'{name}.npz'
    original = cr.predict

    def attach(clip, confs):
        for view, conf in zip(clip.views, confs):
            view.student_conf = np.asarray(conf, np.float32)
        return clip
    if cache.exists():
        with np.load(cache) as z:
            depths = [z[f'd{i}'] for i in range(int(z['n']))]
            confs = [z[f'c{i}'] for i in range(int(z['n']))]
        replay = iter(zip(depths, confs))
        cr.predict = lambda *args, **kwargs: next(replay)
        try:
            clip, ref = load_clip(CFG.root, contract, name, None, 'cpu')
        finally:
            cr.predict = original
        if len(depths) == len(clip.views):
            return attach(clip, confs), ref
    recorded = []

    def record(*args, **kwargs):
        depth, confidence = original(*args, **kwargs)
        recorded.append((np.asarray(depth), np.asarray(confidence, np.float32).squeeze()))
        return depth, confidence
    cr.predict = record
    try:
        clip, ref = load_clip(CFG.root, contract, name, student(), 'cpu')
    finally:
        cr.predict = original
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f'{name}.{os.getpid()}.partial.npz')
    np.savez_compressed(tmp, n=len(recorded), **{f'd{i}': d for i, (d, _) in enumerate(recorded)},
                        **{f'c{i}': c for i, (_, c) in enumerate(recorded)})
    os.replace(tmp, cache)
    return attach(clip, [c for _, c in recorded]), ref


class Context:
    def __init__(self, name, need_pred=True):
        if _FAULT_LOG is None:  # parent process; spawned workers stay single-threaded (see _init)
            torch.set_num_threads(CFG.threads)
        contract = load_contract(CFG.config, ('utility.quality_protocol',))
        clip, ref = load_clip_cached(contract, name)
        self.name, self.views, self.ref, self.voxel = name, clip.views, ref, contract['utility']['voxel_m']
        self.n = len(self.views)
        meta = json.loads((CFG.labels / f'{name}.json').read_text())
        self.split, self.flight = meta['split'], meta['flight']
        with np.load(CFG.labels / f'{name}.npz') as z:
            self.emb, sharp, bright = z['embedding'].astype(np.float32), z['sharpness'], z['brightness']
        if len(self.emb) != self.n:
            raise ValueError(f'{name}: {len(self.emb)} embeddings for {self.n} views')
        self.bins = novelty_bins(self.emb, sharp, BIN_TAU)
        cache = CFG.work / 'tables' / f'{name}.npz'
        if cache.exists():
            with np.load(cache) as z:
                self.scal, self.err, self.feature_ms = z['scal'], z['err'], float(z['ms'])
                self.seen = [np.unpackbits(z[f's{i}'], axis=1, count=int(z['p'][i])).astype(bool) for i in range(self.n)]
        else:
            points = np.asarray(ref.points if hasattr(ref, 'points') else ref.tree.data)
            self.scal, self.err, self.feature_ms, self.seen = frame_table(self.views, self.emb, sharp, bright, points)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(cache.stem + '.partial.npz')
            np.savez_compressed(tmp, scal=self.scal, err=self.err, ms=self.feature_ms, p=np.array([s.shape[1] for s in self.seen]),
                                **{f's{i}': np.packbits(s, axis=1) for i, s in enumerate(self.seen)})
            tmp.replace(cache)
        self.pred = np.load(CFG.work / 'depth' / f'{name}.npy') if need_pred else None

    def F(self, S):
        S = sorted(set(int(s) for s in S))
        return value([self.views[i] for i in S], self.ref, self.voxel) if S else 0.


def features(ctx, S, cand):
    S = [int(s) for s in S]
    pe = ctx.pred
    if S:
        se, cnt, smin, smean = ctx.emb[S], Counter(ctx.bins[S].tolist()), float(pe[S].min()), float(pe[S].mean())
    rows = []
    for f in cand:
        seen = ctx.seen[f]
        if not S:
            cover, cmax, cmean, count = 1., 0., 0., 0
        else:
            cover = 1. - float(seen[S].any(0).mean()) if seen.shape[1] else 0.
            cos = se @ ctx.emb[f]
            cmax, cmean, count = float(cos.max()), float(cos.mean()), cnt.get(int(ctx.bins[f]), 0)
        rows.append([pe[f], *ctx.scal[f], cover, cmax, cmean, count, float(ctx.bins[f] == -1), len(S) / LABEL_BUDGET,
                     smin if S else 0., smean if S else 0., (pe[f] - smin) if S else 0.])
    return np.asarray(rows, np.float32)


# ----------------------------------------------------------------------------- models
class DepthNet(nn.Module):
    def __init__(self, n_emb, n_scal):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(n_emb, 32), nn.SiLU(), nn.Dropout(.2))
        self.f = nn.Sequential(nn.Linear(32 + n_scal, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, e, s):
        return self.f(torch.cat([self.emb(e), s], 1)).squeeze(-1)


class Scorer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('mu', torch.zeros(width))
        self.register_buffer('sd', torch.ones(width))
        self.f = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.f((x - self.mu) / self.sd).squeeze(-1)


def load_scorer(path):
    net = Scorer(len(FEATURES))
    net.load_state_dict(torch.load(path, weights_only=False)['model'])
    net.eval()

    def score(x):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(x, np.float32))).numpy()
    return score


# ----------------------------------------------------------------------------- policy
def rollout(ctx, score, budget, band):
    S, qmax = [], []
    for _ in range(min(budget, ctx.n)):
        cand = [f for f in range(ctx.n) if f not in S]
        X = features(ctx, S, cand)
        q = score(X)
        top = float(q.max())
        qmax.append(top)
        inside = np.flatnonzero(q >= top - band * abs(top))
        S.append(cand[int(inside[np.argmax(X[inside, COVER])])])
    return S, qmax


def stop_length(qmax, tau):
    L = 1
    while L < len(qmax) and qmax[L] >= tau:
        L += 1
    return L


def swap_pass(ctx, score, S):
    S = list(S)
    for a in list(S):
        rest = [s for s in S if s != a]
        cand = [f for f in range(ctx.n) if f not in rest]
        q = score(features(ctx, rest, cand))
        b = cand[int(np.argmax(q))]
        if b != a and q.max() - q[cand.index(a)] > 0:
            S[S.index(a)] = b
    return S


# ----------------------------------------------------------------------------- parallel helpers
def clips(split=None):
    if split is None:
        return clips('train') + clips('validation')
    names = sorted(p.stem for p in CFG.greedy.glob('AM*.npz'))
    names = [c for c in names if json.loads((CFG.labels / f'{c}.json').read_text())['split'] == split]
    if CFG.limit and len(names) > CFG.limit:  # smoke test: spread over flights
        names = [names[i] for i in np.linspace(0, len(names) - 1, CFG.limit).round().astype(int)]
    return names


MODULE = 'uav_reconstruction.adaptive_uav.reconstruction_selector'


_FAULT_LOG = None


def _init(config):
    """Spawned worker: fill the canonical module's CFG (fork deadlocks once the parent has used torch).
    A native crash (segfault) writes its Python stack to <work>/faults/<pid>.txt."""
    global _FAULT_LOG
    import faulthandler
    import importlib
    import os
    importlib.import_module(MODULE).CFG.__dict__.update(config)
    # oneDNN CPU convolutions segfault intermittently in these workers (null jump in DepthStudent.refine);
    # the native ATen convolution gives the same depth to float precision.
    # The remaining crashes are in a native OpenMP helper thread (two OpenMP runtimes in one process),
    # so torch runs single-threaded in workers; parallelism comes from the worker pool.
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(int(os.environ.get('SELECTOR_WORKER_THREADS', '1')))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    faults = Path(config['work']) / 'faults'
    faults.mkdir(parents=True, exist_ok=True)
    _FAULT_LOG = open(faults / f'{os.getpid()}.txt', 'w')
    faulthandler.enable(file=_FAULT_LOG, all_threads=True)


def _call(job):
    import importlib
    module, fn, arg = job
    canonical = importlib.import_module(MODULE)
    try:
        return getattr(importlib.import_module(module), fn)(arg)
    except Exception:
        message = traceback.format_exc()
        with (canonical.CFG.work / 'failures.txt').open('a') as stream:
            stream.write(f'{fn} {arg}\n{message}\n')
        return {'error': message, 'arg': arg}


def _run_round(fn, items, pending, module, isolate=False):
    """Run pending items; return {index: result} for items whose worker survived.
    A worker that dies (segfault) breaks only its executor; with isolate=True each item has its own."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, wait
    import os
    ctx = multiprocessing.get_context('spawn')
    config = dict(CFG.__dict__)
    done = {}
    threads = os.environ.get('SELECTOR_WORKER_THREADS', '1')
    saved = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    os.environ.update({k: threads for k in saved})  # spawned children read this at start-up
    groups = [[i] for i in pending] if isolate else [pending]
    width = 1 if isolate else min(CFG.workers, len(pending))
    for start in range(0, len(groups), CFG.workers if isolate else 1):
        batch = groups[start:start + (CFG.workers if isolate else 1)]
        executors = [ProcessPoolExecutor(width, mp_context=ctx, initializer=_init, initargs=(config,)) for _ in batch]
        futures = {}
        for executor, group in zip(executors, batch):
            for i in group:
                futures[executor.submit(_call, (module, fn, items[i]))] = i
        wait(futures)
        for future, i in futures.items():
            try:
                done[i] = future.result()
            except Exception:  # BrokenProcessPool: a worker crashed natively
                pass
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return done


def parallel(fn, items, module=MODULE, retries=2):
    """Spawn-based map that cannot hang on a crashed worker: lost items are retried,
    then run in isolation; an item that still crashes is recorded in failures.txt."""
    items = list(items)
    if not items:
        return []
    out, pending = {}, list(range(len(items)))
    for attempt in range(retries + 2):
        isolate = attempt == retries + 1
        out.update(_run_round(fn, items, pending, module, isolate))
        lost = [i for i in pending if i not in out]
        if lost:
            print(f'{fn}: {len(lost)} item(s) lost to a worker crash (attempt {attempt + 1}'
                  f'{", isolated" if isolate else ""})', flush=True)
        pending = lost
        if not pending:
            break
    for i in pending:
        with (CFG.work / 'failures.txt').open('a') as stream:
            stream.write(f'{fn} {items[i]}\nworker crashed natively in every attempt; see faults/\n\n')
    for fault in (CFG.work / 'faults').glob('*.txt'):
        if fault.stat().st_size == 0:
            fault.unlink(missing_ok=True)
    ordered = [out[i] for i in range(len(items)) if i in out]
    good = [o for o in ordered if not (isinstance(o, dict) and 'error' in o)]
    print(f'{fn}: {len(good)}/{len(items)} ok', flush=True)
    return good


def log(message):
    with (CFG.work / 'progress.txt').open('a') as stream:
        stream.write(f'{time.strftime("%H:%M:%S")} {message}\n')
    print(message, flush=True)


# ----------------------------------------------------------------------------- stage: tables
def table_clip(name):
    ctx = Context(name, need_pred=False)
    return {'clip': name, 'frames': ctx.n, 'ms': ctx.feature_ms}


def stage_tables():
    rows = parallel('table_clip', clips())
    log(f'tables {len(rows)} clips, runtime features median {np.median([r["ms"] for r in rows]):.2f} ms/frame')


# ----------------------------------------------------------------------------- stage: depth-error net
def _depth_data(names):
    out = []
    for c in names:
        with np.load(CFG.work / 'tables' / f'{c}.npz') as z:
            scal, err = z['scal'], z['err']
        with np.load(CFG.labels / f'{c}.npz') as z:
            emb = z['embedding'].astype(np.float32)
        out.append({'clip': c, 'flight': json.loads((CFG.labels / f'{c}.json').read_text())['flight'],
                    'emb': emb, 'scal': scal, 'err': err, 'ok': np.isfinite(err)})
    return out


def _fit_depth(train, epochs, holdout=None, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allscal = np.concatenate([r['scal'] for r in train])
    mu, sd = allscal.mean(0), allscal.std(0) + 1e-6
    tensor = lambda r: (torch.from_numpy(r['emb']), torch.from_numpy(((r['scal'] - mu) / sd).astype(np.float32)))
    net = DepthNet(train[0]['emb'].shape[1], len(SCALARS))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    history = []
    for _ in range(epochs):
        net.train()
        for i in rng.permutation(len(train)):
            r = train[i]
            ok = torch.from_numpy(r['ok'])
            if ok.sum() < 3:
                continue
            y = torch.from_numpy(np.log(np.maximum(r['err'], 1e-3)).astype(np.float32))[ok]
            opt.zero_grad()
            p = net(*tensor(r))[ok]
            loss = ((p - y) ** 2).mean() + listnet(-p, -y, float(y.std() + 1e-6))
            loss.backward()
            opt.step()
        if holdout is not None:
            net.eval()
            with torch.no_grad():
                history.append({r['clip']: net(*tensor(r)).numpy() for r in holdout})
    net.eval()
    predict = lambda r: net(*tensor(r)).detach().numpy()
    return net, predict, history, (mu, sd)


def _rank(preds, rows):
    return float(np.nanmean([spearman(preds[r['clip']][r['ok']], r['err'][r['ok']]) for r in rows if r['ok'].sum() > 3]))


def stage_depthnet():
    train, val = _depth_data(clips('train')), _depth_data(clips('validation'))
    flights = sorted({r['flight'] for r in train})
    curves, oof_by_epoch = [], {}
    for fl in flights:
        fit = [r for r in train if r['flight'] != fl]
        hold = [r for r in train if r['flight'] == fl]
        _, _, history, _ = _fit_depth(fit, CFG.depth_epochs, hold)
        curves.append([_rank(h, hold) for h in history])
        for e, h in enumerate(history):
            oof_by_epoch.setdefault(e, {}).update(h)
    mean_curve = np.nanmean(np.asarray(curves), 0)
    best = int(np.nanargmax(mean_curve))
    oof = oof_by_epoch[best]
    net, predict, _, (mu, sd) = _fit_depth(train, best + 1)
    out = CFG.work / 'depth'
    out.mkdir(parents=True, exist_ok=True)
    for r in train:
        np.save(out / f'{r["clip"]}.npy', oof[r['clip']].astype(np.float32))
    val_pred = {}
    with torch.no_grad():
        for r in val:
            val_pred[r['clip']] = predict(r)
            np.save(out / f'{r["clip"]}.npy', val_pred[r['clip']].astype(np.float32))
    torch.save({'model': net.state_dict(), 'mu': mu, 'sd': sd, 'epochs': best + 1, 'scalars': SCALARS}, out / 'depth_net.pt')
    report = {'flights': flights, 'best_epoch': best + 1, 'oof_spearman_train': _rank(oof, train),
              'fold_curve_best': float(mean_curve[best]), 'val_spearman': _rank(val_pred, val),
              'val_spearman_agree_only': _rank({r['clip']: -r['scal'][:, 0] for r in val}, val)}
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    log(f'depthnet {report}')


# ----------------------------------------------------------------------------- stage: labels (reconstruction in the loop)
def label_clip(arg):
    name, it = arg
    target = CFG.work / 'rows' / f'it{it}' / f'{name}.npz'
    if target.exists():
        return {'clip': name}
    ctx = Context(name)
    X, Y, G, SIZE = [], [], [], []
    if it == 0:
        with np.load(CFG.greedy / f'{name}.npz') as z:
            step, cand, y, order = z['step'], z['cand'], z['y'], z['order']
        for g, s in enumerate(np.unique(step)):
            m = step == s
            X.append(features(ctx, order[:s], cand[m]))
            Y.append(y[m])
            G += [g] * int(m.sum())
            SIZE.append(int(s))
    else:
        score = load_scorer(CFG.work / f'scorer_full_it{it - 1}.pt')
        S, _ = rollout(ctx, score, LABEL_BUDGET, BANDS[-1])
        rng = np.random.default_rng(zlib.crc32(f'{name}{it}'.encode()))
        states = [S[:s] for s in range(1, len(S))]
        states += [[x for x in S if x != a] for a in rng.choice(S, min(SWAP_STATES, len(S)), replace=False)]
        for g, T in enumerate(states):
            cand = [f for f in range(ctx.n) if f not in T]
            base = ctx.F(T)
            X.append(features(ctx, T, cand))
            Y.append(np.array([ctx.F(T + [f]) - base for f in cand]))
            G += [g] * len(cand)
            SIZE.append(len(T))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + '.partial.npz')
    np.savez_compressed(tmp, x=np.concatenate(X), y=np.concatenate(Y).astype(np.float32), g=np.array(G), size=np.array(SIZE))
    tmp.replace(target)
    return {'clip': name, 'states': len(SIZE)}


def stage_label(it):
    rows = parallel('label_clip', [(c, it) for c in clips('train')])
    log(f'label it{it}: {len(rows)} clips, {sum(r.get("states", 0) for r in rows)} new states')


# ----------------------------------------------------------------------------- stage: train scorer
def _groups(it, names):
    out = []
    for c in names:
        flight = json.loads((CFG.labels / f'{c}.json').read_text())['flight']
        for k in range(it + 1):
            path = CFG.work / 'rows' / f'it{k}' / f'{c}.npz'
            if not path.exists():
                continue
            with np.load(path) as z:
                for g in np.unique(z['g']):
                    m = z['g'] == g
                    out.append((c, flight, torch.from_numpy(z['x'][m]), torch.from_numpy(z['y'][m])))
    return out


def _fit_scorer(groups, epochs, holdout=None, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allx = torch.cat([g[2] for g in groups])
    net = Scorer(len(FEATURES))
    net.mu.copy_(allx.mean(0))
    net.sd.copy_(allx.std(0) + 1e-6)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    best, state, curve = -np.inf, None, []
    for epoch in range(epochs):
        net.train()
        for i in rng.permutation(len(groups)):
            _, _, x, y = groups[i]
            opt.zero_grad()
            p = net(x)
            loss = ((p - 10 * y) ** 2).mean() + (listnet(p, y, float(y.std() + 1e-6)) if len(y) > 1 else 0.)
            loss.backward()
            opt.step()
        if holdout:
            net.eval()
            with torch.no_grad():
                rho = float(np.nanmean([spearman(net(x).numpy(), y.numpy()) for *_, x, y in holdout
                                        if len(y) > 2 and float(y.std()) > 0]))
            curve.append(rho)
            if rho > best:
                best, state = rho, {k: v.clone() for k, v in net.state_dict().items()}
    if holdout:
        net.load_state_dict(state)
    net.eval()
    return net, curve


def holdout_flight():
    return sorted({json.loads((CFG.labels / f'{c}.json').read_text())['flight'] for c in clips('train')})[-1]


def stage_train(it):
    names = clips('train')
    hold = holdout_flight()
    inner_names = [c for c in names if json.loads((CFG.labels / f'{c}.json').read_text())['flight'] != hold]
    hold_names = [c for c in names if c not in inner_names]
    inner, held, everything = _groups(it, inner_names), _groups(it, hold_names), _groups(it, names)
    net, curve = _fit_scorer(inner, CFG.epochs, held)
    best_epoch = int(np.nanargmax(curve)) + 1
    torch.save({'model': net.state_dict(), 'features': FEATURES, 'epoch': best_epoch, 'holdout': hold}, CFG.work / f'scorer_inner_it{it}.pt')
    full, _ = _fit_scorer(everything, best_epoch)
    torch.save({'model': full.state_dict(), 'features': FEATURES, 'epoch': best_epoch}, CFG.work / f'scorer_full_it{it}.pt')
    log(f'train it{it}: {len(everything)} states, holdout {hold} within-state spearman {max(curve):.3f} at epoch {best_epoch}')


# ----------------------------------------------------------------------------- stage: tune on the held-out training flight
def tune_clip(name):
    ctx = Context(name)
    score = load_scorer(CFG.work / f'scorer_inner_it{CFG.iterations - 1}.pt')
    out = {'clip': name}
    for band in BANDS:
        S, qmax = rollout(ctx, score, LABEL_BUDGET, band)
        out[str(band)] = {'qmax': qmax, 'plain': [ctx.F(S[:L]) for L in range(1, len(S) + 1)],
                          'swap': [ctx.F(swap_pass(ctx, score, S[:L])) for L in range(1, len(S) + 1)]}
    return out


def stage_tune():
    hold = holdout_flight()
    rows = parallel('tune_clip', [c for c in clips('train') if json.loads((CFG.labels / f'{c}.json').read_text())['flight'] == hold])
    best = None
    table = {}
    for band in map(str, BANDS):
        q = np.concatenate([r[band]['qmax'][1:] for r in rows])
        taus = [-np.inf] + sorted(set(np.quantile(q, np.linspace(0, 1, 41)).tolist())) + [np.inf]
        for mode in ('plain', 'swap'):
            for tau in taus:
                f = float(np.mean([r[band][mode][stop_length(r[band]['qmax'], tau) - 1] for r in rows]))
                frames = float(np.mean([stop_length(r[band]['qmax'], tau) for r in rows]))
                if best is None or f > best['f'] + 1e-9:
                    best = {'band': float(band), 'swap': mode == 'swap', 'tau': tau, 'f': f, 'frames': frames}
            table[f'{band}_{mode}'] = {str(k): float(np.mean([r[band][mode][min(k, len(r[band][mode])) - 1] for r in rows]))
                                       for k in (1, 2, 3, 4, 8)}
    best['tau'] = float(best['tau']) if np.isfinite(best['tau']) else (1e9 if best['tau'] > 0 else -1e9)
    result = {'holdout_flight': hold, 'clips': len(rows), 'chosen': best, 'fixed_budget_f': table}
    (CFG.work / 'tune.json').write_text(json.dumps(result, indent=1))
    log(f'tune {best}')


# ----------------------------------------------------------------------------- stage: evaluate on validation flights
def eval_clip(name):
    ctx = Context(name)
    tune = json.loads((CFG.work / 'tune.json').read_text())['chosen']
    score = load_scorer(CFG.work / f'scorer_full_it{CFG.iterations - 1}.pt')
    t0 = time.perf_counter()
    S, qmax = rollout(ctx, score, LABEL_BUDGET, tune["band"])
    decide_ms = (time.perf_counter() - t0) * 1000 / len(S)
    finish = (lambda T: swap_pass(ctx, score, T)) if tune['swap'] else (lambda T: list(T))
    L = stop_length(qmax, tune['tau'])
    auto = finish(S[:L])
    greedy = json.loads((CFG.greedy / f'{name}.json').read_text())['curve']
    row = {'clip': name, 'flight': ctx.flight, 'frames': ctx.n, 'feature_ms_per_frame': ctx.feature_ms,
           'decision_ms_per_step': decide_ms, 'all_frames': ctx.F(range(ctx.n)), 'auto_frames': len(auto),
           'auto': ctx.F(auto), 'b': {}}
    full = row['auto']
    row['backward'] = [{'frame': int(a), 'bin': int(ctx.bins[a]), 'loo_gain': full - ctx.F([x for x in auto if x != a]),
                        'pred_log_err': float(ctx.pred[a]), 'true_err_m': float(ctx.err[a])} for a in auto]
    for k in EVAL_BUDGETS:
        if k >= ctx.n:
            continue
        bins = dino_bins(ctx.emb, k)
        row['b'][str(k)] = {'uniform': ctx.F(np.linspace(0, ctx.n - 1, k).round()),
                            'middle': ctx.F([b[len(b) // 2] for b in bins]),
                            'selector': ctx.F(finish(S[:k])), 'selector_no_swap': ctx.F(S[:k]),
                            'selector_stop': ctx.F(finish(S[:min(k, L)])),
                            'greedy_oracle': greedy[min(k, len(greedy)) - 1] if greedy else 0.}
    with (CFG.work / 'eval' / 'rows.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    return row


def _interval(rows, a, b, k=None):
    g = {}
    for r in rows:
        src = r['b'][k] if k else r
        g.setdefault(r['flight'], []).append(src[a] - src[b])
    return cluster_interval(g)


def stage_eval():
    (CFG.work / 'eval').mkdir(parents=True, exist_ok=True)
    (CFG.work / 'eval' / 'rows.jsonl').write_text('')
    rows = parallel('eval_clip', clips('validation'))
    methods = ('uniform', 'middle', 'selector', 'selector_no_swap', 'selector_stop', 'greedy_oracle')
    summ = {'clips': len(rows), 'flights': sorted({r['flight'] for r in rows}), 'tune': json.loads((CFG.work / 'tune.json').read_text()),
            'depthnet': json.loads((CFG.work / 'depth' / 'report.json').read_text()),
            'feature_ms_per_frame_median': float(np.median([r['feature_ms_per_frame'] for r in rows])),
            'decision_ms_per_step_median': float(np.median([r['decision_ms_per_step'] for r in rows])),
            'auto': {'f': float(np.mean([r['auto'] for r in rows])), 'frames': float(np.mean([r['auto_frames'] for r in rows])),
                     'all_frames_f': float(np.mean([r['all_frames'] for r in rows])),
                     'auto_minus_all_frames': _interval(rows, 'auto', 'all_frames')}}
    for k in map(str, EVAL_BUDGETS):
        part = [r for r in rows if k in r['b']]
        e = {m: float(np.mean([r['b'][k][m] for r in part])) for m in methods}
        for m in ('selector', 'selector_stop', 'greedy_oracle'):
            e[f'{m}_minus_middle'] = _interval(part, m, 'middle', k)
        e['selector_minus_uniform'] = _interval(part, 'selector', 'uniform', k)
        e['selector_beats_middle_clips'] = int(sum(r['b'][k]['selector'] > r['b'][k]['middle'] + .005 for r in part))
        e['selector_worse_than_middle_clips'] = int(sum(r['b'][k]['selector'] < r['b'][k]['middle'] - .005 for r in part))
        e['clips'] = len(part)
        summ[k] = e
    back = [b for r in rows for b in r['backward']]
    summ['backward'] = {'selected_frames': len(back), 'loo_gain_median': float(np.median([b['loo_gain'] for b in back])),
                        'harmful_selected_frames': int(sum(b['loo_gain'] < -.005 for b in back)),
                        'true_err_median_m': float(np.nanmedian([b['true_err_m'] for b in back]))}
    (CFG.work / 'eval' / 'summary.json').write_text(json.dumps(summ, indent=1))
    lines = [f'# Reconstruction-driven selector: validation ({summ["clips"]} clips, flights {", ".join(summ["flights"])})', '',
             f'Depth-error net val Spearman {summ["depthnet"]["val_spearman"]:.3f} (consistency alone {summ["depthnet"]["val_spearman_agree_only"]:.3f}).',
             f'Tuned on held-out training flight {summ["tune"]["holdout_flight"]}: {summ["tune"]["chosen"]}.',
             f'Auto stop: {summ["auto"]["frames"]:.1f} frames, F {summ["auto"]["f"]:.3f} vs all frames {summ["auto"]["all_frames_f"]:.3f}.',
             f'Runtime: features {summ["feature_ms_per_frame_median"]:.1f} ms/frame, selection {summ["decision_ms_per_step_median"]:.1f} ms/step (CPU).', '',
             '| K | uniform | middle | selector | no swap | with stop | greedy oracle | selector - middle (95% CI) |', '|---|---|---|---|---|---|---|---|']
    for k in map(str, EVAL_BUDGETS):
        e = summ[k]
        ci = e['selector_minus_middle']
        lines.append(f'| {k} | {e["uniform"]:.3f} | {e["middle"]:.3f} | {e["selector"]:.3f} | {e["selector_no_swap"]:.3f} | '
                     f'{e["selector_stop"]:.3f} | {e["greedy_oracle"]:.3f} | {ci["mean"]:+.3f} [{ci["ci95"][0]:+.3f}, {ci["ci95"][1]:+.3f}] |')
    (CFG.work / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    log('eval done\n' + '\n'.join(lines))


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('tables', 'depthnet', 'label', 'train', 'tune', 'eval', 'all'))
    p.add_argument('--iteration', type=int, default=0)
    p.add_argument('--iterations', type=int, default=3, help='DAgger iterations (0 = greedy states only)')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True, help='contribution_labels output (embeddings, split)')
    p.add_argument('--greedy', type=Path, required=True, help='greedy oracle labels')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--workers', type=int, default=40)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--depth-epochs', type=int, default=150)
    p.add_argument('--limit', type=int, default=0, help='smoke test: first N clips per split')
    a = p.parse_args()
    CFG.__dict__.update(vars(a))
    CFG.root = a.root.resolve()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.stage == 'all':
        log('pipeline start')
        stage_tables()
        stage_depthnet()
        for it in range(a.iterations):
            stage_label(it)
            stage_train(it)
        stage_tune()
        stage_eval()
        log('DONE')
    elif a.stage in ('label', 'train'):
        (stage_label if a.stage == 'label' else stage_train)(a.iteration)
    else:
        globals()[f'stage_{a.stage}']()


if __name__ == '__main__':
    main()
