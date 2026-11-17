"""Score selector checkpoints on the never-trained validation split and the AMtown03 held flight.

Rows are (candidate frame t, occupant a or none) decisions with a measured reconstruction gain y.
Metrics per split/dataset:
  loss      training utility loss (same weighting as the trainer)
  spearman  rank correlation of prediction with measured gain
  auc       ranking of beneficial (y>0.001) vs non-beneficial decisions
  acc       sign accuracy of the accept decision (pred>0 vs y>0.001)
  capture   gain of accepting pred>0 rows / gain of accepting exactly the beneficial rows (oracle)
  top20     mean gain of the top-20% predicted rows per file, as a fraction of the oracle top-20%
            (0 = random choice, 1 = perfect choice)
usage: python -m uav_reconstruction.adaptive_uav.eval_selector_checkpoints OUT.json NAME=ENC[:HEAD] ...
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .large_selector_model import SelectorVisualEncoder, LargeActionSelector
from .train_complete_selector import Decisions, utility_loss

ROOT = Path('/home/iit-t/GeoStreamFusion_GPU_20261013/GeoStreamFusion_GPU/remote/selector_large_20261114/all_available_20261115')
CORPUS = ROOT / 'corpus'


def split_files():
    out = defaultdict(list)
    for p in sorted((CORPUS / 'utility').glob('*.npz')):
        j = p.with_suffix('.json')
        if not j.exists():
            continue
        m = json.loads(j.read_text())
        if m['split'] == 'validation':
            out['validation'].append(p)
        elif m['split'] == 'train' and m['flight'] == 'AMtown03':
            out['held_AMtown03'].append(p)
    return out


def load(spec):
    enc, head = SelectorVisualEncoder().cuda().eval(), LargeActionSelector(36).cuda().eval()
    if ':' in spec:
        e, h = spec.split(':')
        enc.load_state_dict(torch.load(e, map_location='cpu', weights_only=False)['model'])
        head.load_state_dict(torch.load(h, map_location='cpu', weights_only=False)['model'])
    else:
        s = torch.load(spec, map_location='cpu', weights_only=False)
        enc.load_state_dict(s['encoder']); head.load_state_dict(s['head'])
    return enc, head


@torch.no_grad()
def predict(enc, head, ds):
    preds, ys, hs, ws = [], [], [], []
    for images, x, has, y, w in DataLoader(ds, batch_size=64, num_workers=0, pin_memory=True):
        images = images.cuda(non_blocking=True).float() / 255
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb, _ = enc(images.flatten(0, 1))
            c, o = emb.reshape(len(images), 2, -1).unbind(1)
            p = head(c, o, has.cuda().float(), x.cuda())
        preds.append(p.float().cpu()); ys.append(y.float()); hs.append(has.float()); ws.append(w.float())
    return [torch.cat(v) for v in (preds, ys, hs, ws)]


def metrics(p, y, h, w, files):
    pn, yn = p.numpy(), y.numpy()
    pos = yn > .001
    r = dict(rows=int(len(yn)), positive_rate=float(pos.mean()),
             loss=float(utility_loss(p, y, h, w)),
             spearman=float(spearmanr(pn, yn).correlation),
             auc=float(roc_auc_score(pos, pn)) if 0 < pos.sum() < len(pos) else None,
             acc=float(((pn > 0) == pos).mean()))
    oracle = yn[pos].sum()
    r['capture'] = float(yn[pn > 0].sum() / oracle) if oracle > 0 else None
    fr = []
    for f in np.unique(files):
        m = files == f
        if m.sum() < 5:
            continue
        k = max(1, int(round(.2 * m.sum())))
        yy, pp = yn[m], pn[m]
        best = np.sort(yy)[-k:].mean(); rand = yy.mean(); got = yy[np.argsort(pp)[-k:]].mean()
        if best - rand > 1e-6:
            fr.append((got - rand) / (best - rand))
    r['top20'] = float(np.mean(fr)) if fr else None
    return r


def main():
    out = Path(sys.argv[1]); models = dict(a.split('=', 1) for a in sys.argv[2:])
    groups = split_files()
    data = {}
    for name, fs in groups.items():
        ds = Decisions(fs)
        files = np.array([it[0] for it in ds.items])
        dom = np.array([json.loads(Path(it[0]).with_suffix('.json').read_text()).get('dataset', 'uavscenes') for it in ds.items]) \
            if len(fs) < 50 else None
        if dom is None:
            meta = {str(p): json.loads(p.with_suffix('.json').read_text()).get('dataset', 'uavscenes') for p in fs}
            dom = np.array([meta[f] for f in files])
        data[name] = (ds, files, dom)
        print(name, len(fs), 'files', len(ds), 'rows', flush=True)
    report = {}
    for mname, spec in models.items():
        t = time.time(); enc, head = load(spec); report[mname] = {}
        for name, (ds, files, dom) in data.items():
            p, y, h, w = predict(enc, head, ds)
            report[mname][name] = {'all': metrics(p, y, h, w, files)}
            for d in np.unique(dom):
                m = torch.from_numpy(dom == d)
                report[mname][name][d] = metrics(p[m], y[m], h[m], w[m], files[dom == d])
        print(mname, json.dumps({k: v['all'] for k, v in report[mname].items()}), f'{time.time()-t:.0f}s', flush=True)
        out.write_text(json.dumps(report, indent=1))
        del enc, head; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
