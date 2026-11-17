"""Online bin-memory frame selector, trained from the reconstruction. The candidate frame needs no depth call.

Forward pass (design notes):   selector -> depth student (selected frames only) -> TSDF reconstruction
Backward signal:               reconstruction F-score against LiDAR -> score targets for the selector

The selector sees frames in time order and keeps a bin memory:
    * a bin is a view cluster: a frame joins the bin whose protected prototype (frozen DINOv2 embedding of
      the first frame that opened it) is most similar, if cosine >= tau; otherwise it opens a new bin.
      A frame that could match two bins is assigned only to the best one, the one it may replace.
    * every bin holds at most one selected frame.
    * blurry frames (below the running 10% sharpness quantile) go to the null slot and are skipped.
Score, exactly as in the notes (F = fused F-score against LiDAR):
    bin empty:      Q(f | S) = F(S + {f}) - F(S)
    bin holds a:    Q(f | S) = F(S + {f} - {a}) - F(S)
The network is  frozen image embedding -> projection, concatenated with memory/context features -> FFN ->
sigmoid. It is trained so that sigmoid^-1(score) = 20 * Q, so score 0.5 means "no gain"; the frame is
taken (added or swapped in) when score >= theta.

Depth is never computed for the candidate: its inputs are the image embedding, causal sharpness and
brightness, camera motion and attitude from the tracked pose, similarity to the bin prototypes and to the
selected frames, and how much of its view the already selected frames explain (their depth points, which
exist because selected frames are reconstructed, projected into the candidate's camera).
A depth-free net also predicts the candidate's depth error from the same inputs, as one more feature.

Training keeps the reconstruction in the loop (DAgger): iteration 0 follows the measured gain with random
flips; later iterations follow the current selector with random flips; at every visited state the gain is
measured by re-fusing, and the scorer is retrained on everything so far. tau and theta are chosen on a
held-out training flight; validation flights are used once.

Stages: prep -> collect/train x iterations -> tune -> eval   (or: all)
Scope: reference poses; student depth uses a fixed dense source buffer (contribution_labels.load_clip).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import reconstruction_selector as rs
from .train_contribution_selector import dino_bins, spearman

MODULE = 'uav_reconstruction.adaptive_uav.bin_memory_selector'
CFG = rs.CFG
TAUS = (.85, .90, .95)
THETAS = (.4, .5, .6, .7)
GAIN_SCALE, EPS, GRID = 20., .15, (12, 16)
FRAME = ('sharp_c', 'bright_c', 'move_m', 'turn_deg', 'off_nadir', 'rel_height')
# Student confidence of the candidate (allowed by the user 2026-11-13): the student already runs on every
# frame for the fixed-context depth; only its confidence map is used here, no depth geometry of the candidate.
#   Off by default: tested 2026-11-13, the current student's confidence is not calibrated across areas
#   (per-clip Spearman with LiDAR depth error -0.25 on AMtown, +0.23 on AMvalley), and adding it lowered
#   validation F (0.349 vs 0.367 for middle-of-bin). Enable with SELECTOR_CONF=1 once confidence is calibrated.
CONF = ('conf_mean', 'conf_p10', 'conf_low', 'conf_z')
if os.environ.get('SELECTOR_CONF') == '1':
    FRAME = FRAME + CONF
STATE = ('pred_err', 'swap', 'cos_proto', 'cos_second', 'cos_occupant', 'cos_max_set', 'explained_set',
         'explained_occupant', 'base_ratio', 'angle_nearest', 'since_selected', 'set_size', 'bins',
         'occ_pred_err', 'err_diff', 'occ_sharp_c', 'occ_age', 'occ_agree') + FRAME


def out_dir():
    return CFG.work / getattr(CFG, 'out', 'online')


def log(message):
    out_dir().mkdir(parents=True, exist_ok=True)
    with (out_dir() / 'progress.txt').open('a') as stream:
        stream.write(f'{time.strftime("%H:%M:%S")} {message}\n')
    print(message, flush=True)


# ----------------------------------------------------------------------------- per-clip data
def causal_scalars(views, sharp, bright):
    ls, rows = np.log1p(sharp), []
    zero = views[0].reference_pose[2, 3]
    for t, v in enumerate(views):
        T = v.reference_pose
        if t >= 2:
            sc = float(np.clip((ls[t] - ls[:t + 1].mean()) / (ls[:t + 1].std() + 1e-3), -5, 5))
            bc = float(np.clip((bright[t] - bright[:t + 1].mean()) / (bright[:t + 1].std() + 1e-3), -5, 5))
        else:
            sc = bc = 0.
        if t:
            P = views[t - 1].reference_pose
            move = float(np.linalg.norm(T[:3, 3] - P[:3, 3]))
            turn = float(np.degrees(np.arccos(np.clip((np.trace(P[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1))))
        else:
            move = turn = 0.
        rows.append([sc, bc, move, turn, float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))), float(T[2, 3] - zero)])
    if 'conf_mean' in FRAME:
        conf = [v.student_conf for v in views]
        mean = np.array([float(c.mean()) for c in conf])
        for t, c in enumerate(conf):
            z = float(np.clip((mean[t] - mean[:t + 1].mean()) / (mean[:t + 1].std() + 1e-3), -5, 5)) if t >= 2 else 0.
            rows[t] += [mean[t], float(np.quantile(c, .1)), float((c < .5).mean()), z]
    return np.asarray(rows, np.float32)


class Clip:
    def __init__(self, name, need_pred=True):
        self.ctx = rs.Context(name, need_pred=False)
        with np.load(CFG.labels / f'{name}.npz') as z:
            self.sharp, bright = z['sharpness'], z['brightness']
        self.C = causal_scalars(self.ctx.views, self.sharp, bright)
        self.pe = np.load(out_dir() / 'depth' / f'{name}.npy') if need_pred else None
        self._points, self._zmed, self._mask = {}, {}, {}

    def points(self, i):
        if i not in self._points:
            world, z = rs.back_project(self.ctx.views[i])
            self._points[i], self._zmed[i] = world, float(np.median(z)) if len(z) else 1.
        return self._points[i]

    def explained(self, f, sources):
        """Fraction of f's image grid hit by the depth points of already selected frames. No depth for f."""
        total = np.zeros(GRID[0] * GRID[1], bool)
        view = self.ctx.views[f]
        h, w = view.fusion_depth.shape[:2]
        for s in sources:
            if (f, s) not in self._mask:
                p = self.points(s)
                inv = np.linalg.inv(view.reference_pose)
                cam = inv[:3, :3] @ p + inv[:3, 3:4]
                cam = cam[:, cam[2] > 1e-3]
                uv = view.fusion_k @ cam
                u, v = uv[0] / uv[2], uv[1] / uv[2]
                ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                mask = np.zeros(GRID[0] * GRID[1], bool)
                mask[(v[ok] * GRID[0] / h).astype(int) * GRID[1] + (u[ok] * GRID[1] / w).astype(int)] = True
                self._mask[(f, s)] = mask
            total |= self._mask[(f, s)]
        return float(total.mean())


def state_features(c, t, S, bins, a, cos):
    ctx, e = c.ctx, c.ctx.emb[t]
    top = np.sort(cos)[::-1] if len(cos) else np.zeros(0)
    row = [c.pe[t], float(a is not None), float(top[0]) if len(top) else 0., float(top[1]) if len(top) > 1 else 0.]
    T = ctx.views[t].reference_pose
    if S:
        centres = np.array([ctx.views[s].reference_pose[:3, 3] for s in S])
        d = np.linalg.norm(centres - T[:3, 3], axis=1)
        j = int(d.argmin())
        c.points(S[j])
        axis = ctx.views[S[j]].reference_pose[:3, 2]
        nearest = [float(d[j] / max(c._zmed[S[j]], 1e-3)), float(np.degrees(np.arccos(np.clip(axis @ T[:3, 2], -1, 1)))), float(t - max(S))]
        cos_set, explained = float((ctx.emb[S] @ e).max()), c.explained(t, S)
    else:
        nearest, cos_set, explained = [0., 0., float(t)], 0., 0.
    if a is not None:
        others = [s for s in S if s != a]
        agree = float(ctx.seen[a][others].any(0).mean()) if others and ctx.seen[a].shape[1] else 0.
        occupant = [c.pe[a], c.pe[t] - c.pe[a], c.C[a, 0], float(t - a), agree]
        cos_occ, explained_occ = float(ctx.emb[a] @ e), c.explained(t, [a])
    else:
        occupant, cos_occ, explained_occ = [0.] * 5, 0., 0.
    row += [cos_occ, cos_set, explained, explained_occ] + nearest + [len(S) / 8., bins / 8.] + occupant + c.C[t].tolist()
    return np.asarray(row, np.float32)


# ----------------------------------------------------------------------------- online policy
def rollout(c, tau, decide, measure):
    """One pass over the clip in time order. decide(t, a, x, gain) -> bool. Returns the selection and the visited rows."""
    ctx, ls = c.ctx, np.log1p(c.sharp)
    protos, occupant, S, F_S, rows = [], [], [], 0., []
    for t in range(ctx.n):
        if t >= 8 and ls[t] < np.quantile(ls[:t], .1):
            continue  # null slot
        e = ctx.emb[t]
        cos = np.array([p @ e for p in protos])
        if len(cos) and cos.max() >= tau:
            b = int(cos.argmax())
        else:
            b = len(protos)
            protos.append(e)  # protected prototype
            occupant.append(None)
        a = occupant[b]
        x = state_features(c, t, S, len(protos), a, cos)
        T = (S + [t]) if a is None else [s for s in S if s != a] + [t]
        gain = (ctx.F(T) - F_S) if measure else None
        take = bool(decide(t, a, x, gain))
        rows.append((t, -1 if a is None else a, x, gain, take))
        if take:
            S, occupant[b] = T, t
            if measure:
                F_S += gain
    return S, rows


class BinScorer(nn.Module):
    def __init__(self, n_emb, width):
        super().__init__()
        self.register_buffer('mu', torch.zeros(width))
        self.register_buffer('sd', torch.ones(width))
        self.proj = nn.Sequential(nn.Linear(n_emb, 32), nn.SiLU(), nn.Dropout(.3))
        self.f = nn.Sequential(nn.Linear(64 + width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, ef, ea, has_a, x):
        return self.f(torch.cat([self.proj(ef), self.proj(ea) * has_a[:, None], (x - self.mu) / self.sd], 1)).squeeze(-1)


def load_policy(path, c):
    state = torch.load(path, weights_only=False)
    net = BinScorer(c.ctx.emb.shape[1], len(STATE))
    net.load_state_dict(state['model'])
    net.eval()
    emb = torch.from_numpy(c.ctx.emb)

    def score(t, a, x):
        with torch.no_grad():
            z = net(emb[t][None], emb[max(a, 0) if a is not None else 0][None],
                    torch.tensor([float(a is not None)]), torch.from_numpy(x)[None])
        return float(torch.sigmoid(z))
    return score


# ----------------------------------------------------------------------------- stage: prep (causal scalars + depth-free error net)
def prep_clip(name):
    c = Clip(name, need_pred=False)
    target = out_dir() / 'frames' / f'{name}.npy'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, c.C)
    return {'clip': name}


class ErrNet(nn.Module):
    def __init__(self, n_emb, n_scal):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(n_emb, 32), nn.SiLU(), nn.Dropout(.3))
        self.f = nn.Sequential(nn.Linear(32 + n_scal, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, e, s):
        return self.f(torch.cat([self.emb(e), s], 1)).squeeze(-1)


def _err_rows(names):
    rows = []
    for c in names:
        with np.load(CFG.work / 'tables' / f'{c}.npz') as z:
            err = z['err']
        with np.load(CFG.labels / f'{c}.npz') as z:
            emb = z['embedding'].astype(np.float32)
        rows.append({'clip': c, 'flight': json.loads((CFG.labels / f'{c}.json').read_text())['flight'], 'emb': emb,
                     'scal': np.load(out_dir() / 'frames' / f'{c}.npy'), 'err': err, 'ok': np.isfinite(err)})
    return rows


def _fit_err(train, epochs, holdout=None):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    scal = np.concatenate([r['scal'] for r in train])
    mu, sd = scal.mean(0), scal.std(0) + 1e-6
    tensor = lambda r: (torch.from_numpy(r['emb']), torch.from_numpy(((r['scal'] - mu) / sd).astype(np.float32)))
    net = ErrNet(train[0]['emb'].shape[1], len(FRAME))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
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
            loss = ((net(*tensor(r))[ok] - y) ** 2).mean()
            loss.backward()
            opt.step()
        if holdout is not None:
            net.eval()
            with torch.no_grad():
                history.append({r['clip']: net(*tensor(r)).numpy() for r in holdout})
    net.eval()
    return net, (lambda r: net(*tensor(r)).detach().numpy()), history, (mu, sd)


def stage_prep():
    rs.parallel('prep_clip', rs.clips(), MODULE)
    train, val = _err_rows(rs.clips('train')), _err_rows(rs.clips('validation'))
    rank = lambda preds, rows: float(np.nanmean([spearman(preds[r['clip']][r['ok']], r['err'][r['ok']]) for r in rows if r['ok'].sum() > 3]))
    curves, oof = [], {}
    for flight in sorted({r['flight'] for r in train}):
        hold = [r for r in train if r['flight'] == flight]
        _, _, history, _ = _fit_err([r for r in train if r['flight'] != flight], CFG.depth_epochs, hold)
        curves.append([rank(h, hold) for h in history])
        for epoch, h in enumerate(history):
            oof.setdefault(epoch, {}).update(h)
    curve = np.nanmean(np.asarray(curves), 0)
    best = int(np.nanargmax(curve))
    net, predict, _, (mu, sd) = _fit_err(train, best + 1)
    folder = out_dir() / 'depth'
    folder.mkdir(parents=True, exist_ok=True)
    for r in train:
        np.save(folder / f'{r["clip"]}.npy', oof[best][r['clip']].astype(np.float32))
    val_pred = {r['clip']: predict(r) for r in val}
    for r in val:
        np.save(folder / f'{r["clip"]}.npy', val_pred[r['clip']].astype(np.float32))
    torch.save({'model': net.state_dict(), 'mu': mu, 'sd': sd, 'epochs': best + 1, 'scalars': FRAME}, folder / 'err_net.pt')
    report = {'epochs': best + 1, 'oof_spearman_train': rank(oof[best], train), 'val_spearman': rank(val_pred, val)}
    (folder / 'report.json').write_text(json.dumps(report, indent=1))
    log(f'prep: depth-free error net {report}')


# ----------------------------------------------------------------------------- stage: collect (reconstruction in the loop)
def collect_clip(arg):
    name, it = arg
    target = out_dir() / 'rows' / f'it{it}' / f'{name}.npz'
    if target.exists():
        return {'clip': name}
    c = Clip(name)
    rng = np.random.default_rng(zlib.crc32(f'{name}{it}'.encode()))
    score = load_policy(out_dir() / f'scorer_full_it{it - 1}.pt', c) if it else None
    T, A, X, Y, TAU = [], [], [], [], []
    for tau in TAUS:
        def decide(t, a, x, gain):
            choice = (gain > 0) if score is None else (score(t, a, x) >= .5)
            return (not choice) if rng.random() < EPS else choice
        _, rows = rollout(c, tau, decide, True)
        for t, a, x, gain, _ in rows:
            T.append(t), A.append(a), X.append(x), Y.append(gain), TAU.append(tau)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + '.partial.npz')
    np.savez_compressed(tmp, t=np.array(T), a=np.array(A), x=np.array(X, np.float32), y=np.array(Y, np.float32), tau=np.array(TAU))
    tmp.replace(target)
    return {'clip': name, 'rows': len(T)}


def stage_collect(it):
    done = rs.parallel('collect_clip', [(c, it) for c in rs.clips('train')], MODULE)
    log(f'collect it{it}: {len(done)} clips, {sum(r.get("rows", 0) for r in done)} measured decisions')


# ----------------------------------------------------------------------------- stage: train
def _rows(it, names):
    ef, ea, has, x, y = [], [], [], [], []
    for c in names:
        with np.load(CFG.labels / f'{c}.npz') as z:
            emb = z['embedding'].astype(np.float32)
        for k in range(it + 1):
            path = out_dir() / 'rows' / f'it{k}' / f'{c}.npz'
            if not path.exists():
                continue
            with np.load(path) as z:
                ef.append(emb[z['t']]), ea.append(emb[np.maximum(z['a'], 0)]), has.append((z['a'] >= 0).astype(np.float32))
                x.append(z['x']), y.append(z['y'])
    cat = lambda parts: torch.from_numpy(np.concatenate(parts))
    return cat(ef), cat(ea), cat(has), cat(x), cat(y)


def _loss(net, data, index=None):
    ef, ea, has, x, y = data if index is None else [d[index] for d in data]
    z = net(ef, ea, has, x)
    return nn.functional.smooth_l1_loss(z, (GAIN_SCALE * y).clamp(-4, 4)) + \
        nn.functional.binary_cross_entropy_with_logits(z, (y > .001).float())


def _fit(data, epochs, holdout=None):
    torch.manual_seed(0)
    net = BinScorer(data[0].shape[1], len(STATE))
    net.mu.copy_(data[3].mean(0))
    net.sd.copy_(data[3].std(0) + 1e-6)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    best, state, curve, n = np.inf, None, [], len(data[4])
    for _ in range(epochs):
        net.train()
        order = torch.randperm(n)
        for start in range(0, n, 256):
            opt.zero_grad()
            _loss(net, data, order[start:start + 256]).backward()
            opt.step()
        if holdout is not None:
            net.eval()
            with torch.no_grad():
                value = float(_loss(net, holdout))
            curve.append(value)
            if value < best:
                best, state = value, {k: v.clone() for k, v in net.state_dict().items()}
    if holdout is not None:
        net.load_state_dict(state)
    net.eval()
    return net, curve


def stage_train(it):
    names, hold = rs.clips('train'), rs.holdout_flight()
    flight = lambda c: json.loads((CFG.labels / f'{c}.json').read_text())['flight']
    inner, held = _rows(it, [c for c in names if flight(c) != hold]), _rows(it, [c for c in names if flight(c) == hold])
    net, curve = _fit(inner, CFG.epochs, held)
    epochs = int(np.argmin(curve)) + 1
    with torch.no_grad():
        z = net(*held[:4]).numpy()
    gain = held[4].numpy()
    report = {'rows_inner': len(inner[4]), 'rows_holdout': len(gain), 'epochs': epochs, 'holdout_spearman': spearman(z, gain),
              'holdout_decision_accuracy': float(((z > 0) == (gain > 0)).mean()), 'holdout_positive_rate': float((gain > 0).mean())}
    torch.save({'model': net.state_dict(), 'state': STATE, 'epochs': epochs, 'holdout': hold}, out_dir() / f'scorer_inner_it{it}.pt')
    full, _ = _fit(_rows(it, names), epochs)
    torch.save({'model': full.state_dict(), 'state': STATE, 'epochs': epochs}, out_dir() / f'scorer_full_it{it}.pt')
    log(f'train it{it}: {report}')


# ----------------------------------------------------------------------------- stage: tune on the held-out training flight
def tune_clip(name):
    """Grid over DAgger iteration x tau x theta, judged by fused F on the held-out training flight
    (later iterations are not always better, so the iteration is chosen like any other hyperparameter)."""
    c = Clip(name)
    out = {'clip': name, 'grid': {}}
    for it in range(CFG.iterations):
        score = load_policy(out_dir() / f'scorer_inner_it{it}.pt', c)
        for tau in TAUS:
            for theta in THETAS:
                S, _ = rollout(c, tau, lambda t, a, x, g: score(t, a, x) >= theta, False)
                out['grid'][f'{it}_{tau}_{theta}'] = [c.ctx.F(S), len(S)]
    return out


def stage_tune():
    hold = rs.holdout_flight()
    rows = rs.parallel('tune_clip', [c for c in rs.clips('train') if json.loads((CFG.labels / f'{c}.json').read_text())['flight'] == hold], MODULE)
    grid = {k: [float(np.mean([r['grid'][k][0] for r in rows])), float(np.mean([r['grid'][k][1] for r in rows]))] for k in rows[0]['grid']}
    best = max(grid, key=lambda k: grid[k][0])
    it, tau, theta = best.split('_')
    it, tau, theta = int(it), float(tau), float(theta)
    (out_dir() / 'tune.json').write_text(json.dumps({'holdout_flight': hold, 'clips': len(rows), 'iteration': it, 'tau': tau,
                                                     'theta': theta, 'f': grid[best][0], 'frames': grid[best][1],
                                                     'grid_f_frames': grid}, indent=1))
    log(f'tune: iteration {it} tau {tau} theta {theta} F {grid[best][0]:.3f} frames {grid[best][1]:.1f}')


# ----------------------------------------------------------------------------- stage: evaluate on validation flights
def eval_clip(name):
    c = Clip(name)
    ctx = c.ctx
    tune = json.loads((out_dir() / 'tune.json').read_text())
    score = load_policy(out_dir() / f'scorer_full_it{tune.get("iteration", CFG.iterations - 1)}.pt', c)
    t0 = time.perf_counter()
    S, rows = rollout(c, tune['tau'], lambda t, a, x, g: score(t, a, x) >= tune['theta'], False)
    ms = (time.perf_counter() - t0) * 1000 / ctx.n
    oracle, _ = rollout(c, tune['tau'], lambda t, a, x, g: g > 0, True)
    first, _ = rollout(c, tune['tau'], lambda t, a, x, g: a is None, False)
    k = max(1, len(S))
    greedy = json.loads((CFG.greedy / f'{name}.json').read_text())['curve']
    full = ctx.F(S)
    row = {'clip': name, 'flight': ctx.flight, 'frames': ctx.n, 'selected': len(S), 'decision_ms_per_frame': ms,
           'swaps': int(sum(take and a >= 0 for _, a, _, _, take in rows)),
           'selector': full, 'online_oracle': ctx.F(oracle), 'online_oracle_frames': len(oracle),
           'bin_first_frame': ctx.F(first), 'bin_first_frames': len(first),
           'uniform': ctx.F(np.linspace(0, ctx.n - 1, k).round()) if k < ctx.n else ctx.F(range(ctx.n)),
           'middle': ctx.F([b[len(b) // 2] for b in dino_bins(ctx.emb, k)]) if k < ctx.n else ctx.F(range(ctx.n)),
           'greedy_oracle': greedy[min(k, len(greedy)) - 1] if greedy else 0., 'all_frames': ctx.F(range(ctx.n)),
           'backward': [{'frame': int(a), 'loo_gain': full - ctx.F([s for s in S if s != a]), 'true_err_m': float(ctx.err[a])} for a in S]}
    return row


def stage_eval():
    rows = rs.parallel('eval_clip', rs.clips('validation'), MODULE)
    with (out_dir() / 'eval_rows.jsonl').open('w') as stream:
        stream.writelines(json.dumps(r) + '\n' for r in rows)
    methods = ('selector', 'online_oracle', 'bin_first_frame', 'uniform', 'middle', 'greedy_oracle', 'all_frames')

    def interval(a, b):
        g = {}
        for r in rows:
            g.setdefault(r['flight'], []).append(r[a] - r[b])
        return rs.cluster_interval(g)
    back = [b for r in rows for b in r['backward']]
    summ = {'clips': len(rows), 'tune': {k: v for k, v in json.loads((out_dir() / 'tune.json').read_text()).items() if k != 'grid_f_frames'},
            'error_net': json.loads((out_dir() / 'depth' / 'report.json').read_text()),
            'f': {m: float(np.mean([r[m] for r in rows])) for m in methods},
            'selected_mean': float(np.mean([r['selected'] for r in rows])), 'frames_mean': float(np.mean([r['frames'] for r in rows])),
            'depth_calls_saved': float(1 - np.sum([r['selected'] for r in rows]) / np.sum([r['frames'] for r in rows])),
            'swaps_mean': float(np.mean([r['swaps'] for r in rows])),
            'decision_ms_per_frame_median': float(np.median([r['decision_ms_per_frame'] for r in rows])),
            'selector_minus': {m: interval('selector', m) for m in ('uniform', 'middle', 'bin_first_frame', 'all_frames', 'online_oracle')},
            'selector_beats_middle_clips': int(sum(r['selector'] > r['middle'] + .005 for r in rows)),
            'selector_worse_than_middle_clips': int(sum(r['selector'] < r['middle'] - .005 for r in rows)),
            'backward': {'selected': len(back), 'harmful_selected': int(sum(b['loo_gain'] < -.005 for b in back)),
                         'loo_gain_median': float(np.median([b['loo_gain'] for b in back])) if back else None}}
    (out_dir() / 'summary.json').write_text(json.dumps(summ, indent=1))
    log('eval ' + json.dumps(summ))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('prep', 'collect', 'train', 'tune', 'eval', 'all'))
    p.add_argument('--iteration', type=int, default=0)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--work', type=Path, required=True, help='reconstruction_selector work dir (tables are reused)')
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--greedy', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--workers', type=int, default=40)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--depth-epochs', type=int, default=100)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', default='online', help='output folder inside --work')
    a = p.parse_args()
    CFG.__dict__.update(vars(a))
    CFG.root = a.root.resolve()
    out_dir().mkdir(parents=True, exist_ok=True)
    if a.stage == 'all':
        log('online pipeline start')
        stage_prep()
        for it in range(a.iterations):
            stage_collect(it)
            stage_train(it)
        stage_tune()
        stage_eval()
        log('DONE')
    elif a.stage in ('collect', 'train'):
        (stage_collect if a.stage == 'collect' else stage_train)(a.iteration)
    else:
        globals()[f'stage_{a.stage}']()


if __name__ == '__main__':
    main()
