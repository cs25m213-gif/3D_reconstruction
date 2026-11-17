"""Remote CPU held-flight tuning and development validation of large selector.

Reference poses and newly predicted fixed-context depths; no deployment promotion.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import cv2

from . import bin_memory_selector as bm
from . import reconstruction_selector as rs
from .large_selector_model import LargeActionSelector
from .contribution_labels import value
from .reconstruction_quality import ReferencePopulation
from .train_contribution_selector import dino_bins

ROOT=Path('/home/vishnu/datasets/selector_large_20261114')
THRESHOLDS=(-.8,-.4,0.,.4,.8)


def init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False


def evaluate(path):
    path=Path(path)
    meta=json.loads(path.with_suffix('.json').read_text())
    z=dict(np.load(path))
    # TSDF integration requires aligned RGB even when only geometry is scored.
    # Stored RGB is the corrupted image used by this episode's visual encoder.
    views=[SimpleNamespace(fusion_depth=d,fusion_k=k,reference_pose=p,
                           fusion_rgb=cv2.resize(rgb, d.shape[::-1], interpolation=cv2.INTER_AREA))
           for d,k,p,rgb in zip(z['fusion_depth'],z['fusion_k'],z['reference_pose'],z['rgb'])]
    ref=ReferencePopulation(z['reference_points'],meta['tolerance_m'],'QUALIFIED',meta['quality_protocol'])
    points=[rs.back_project(v)[0] for v in views]
    seen=[np.stack([rs.consistent(p,v)[0] if i!=j else np.zeros(p.shape[1],bool)
                    for j,v in enumerate(views)]) for i,p in enumerate(points)]
    ctx=SimpleNamespace(views=views,emb=z['embedding'],n=len(views),seen=seen,
                        F=lambda s:value([views[i] for i in sorted(set(s))],ref,meta['voxel_m']) if len(s) else 0.)
    c=bm.Clip.__new__(bm.Clip)
    c.ctx,c.C,c.sharp,c.pe=ctx,z['causal'],z['sharp'],np.zeros(ctx.n,np.float32)
    c._points,c._zmed,c._mask={},{},{}
    feature=torch.from_numpy(np.load(ROOT/'utility_features'/(path.stem+'.npy')))
    scope='inner' if meta['flight']=='AMtown03' and meta['split']=='train' else 'full'
    tune=json.loads((ROOT/'selector/tune.json').read_text()) if (ROOT/'selector/tune.json').exists() else None
    rows=[]
    for kind in ('large','compact'):
        checkpoint=torch.load(ROOT/'selector'/f'{kind}_{scope}.pt',map_location='cpu',weights_only=False)
        model=LargeActionSelector(checkpoint['state_dim']) if kind=='large' else bm.BinScorer(384,checkpoint['state_dim'])
        model.load_state_dict(checkpoint['model']);model.eval()
        thresholds=THRESHOLDS if scope=='inner' else [tune[kind]['threshold']]
        for threshold in thresholds:
            selected,prototypes,occupants=[],[],[]
            accepted=[]
            for t in range(ctx.n):
                cos=np.asarray([p@ctx.emb[t] for p in prototypes])
                if len(cos) and cos.max()>=.9:b=int(cos.argmax())
                else:
                    b=len(prototypes);prototypes.append(ctx.emb[t]);occupants.append(None)
                a=occupants[b]
                x=bm.state_features(c,t,selected,len(prototypes),a,cos)
                x=np.concatenate([x,z['quality'][t],z['quality'][a] if a is not None else np.zeros(6,np.float32)]).astype(np.float32)
                with torch.no_grad():
                    score=float(model(feature[t:t+1],feature[(a or 0):(a or 0)+1],torch.tensor([float(a is not None)]),torch.from_numpy(x)[None]))
                if score>=threshold:
                    selected=selected+[t] if a is None else [s for s in selected if s!=a]+[t]
                    occupants[b]=t;accepted.append(t)
            k=len(selected)
            middle=[int(g[len(g)//2]) for g in dino_bins(ctx.emb,k)] if k else []
            uniform=np.linspace(0,ctx.n-1,k).round().astype(int).tolist() if k else []
            rows.append({'kind':kind,'threshold':threshold,'f':ctx.F(selected),'middle_f':ctx.F(middle),
                         'uniform_f':ctx.F(uniform),'selected':selected,'depth_requests':len(set(accepted)),
                         'candidate_frames':ctx.n})
    result={'clip':meta['clip'],'flight':meta['flight'],'split':meta['split'],'variant':meta['variant'],'rows':rows}
    (ROOT/'selector/evaluation_rows'/path.with_suffix('.json').name).write_text(json.dumps(result))
    return result


def main():
    paths=list((ROOT/'utility').glob('*.npz'))
    output=ROOT/'selector'
    (output/'evaluation_rows').mkdir(exist_ok=True)
    hold=[p for p in paths if json.loads(p.with_suffix('.json').read_text())['flight']=='AMtown03']
    val=[p for p in paths if json.loads(p.with_suffix('.json').read_text())['split']=='validation']
    with ProcessPoolExecutor(16,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        rows=list(pool.map(evaluate,hold))
    tune={}
    for kind in ('large','compact'):
        grid=[{'threshold':th,'f':float(np.mean([s['f'] for r in rows for s in r['rows'] if s['kind']==kind and s['threshold']==th]))} for th in THRESHOLDS]
        tune[kind]=dict(max(grid,key=lambda r:r['f']))
        tune[kind]['grid']=grid
    (output/'tune.json').write_text(json.dumps(tune,indent=2))
    with ProcessPoolExecutor(16,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        rows=list(pool.map(evaluate,val))
    summary={'scope':'Reference-pose development validation; not full-video mesh quality or runtime.',
             'tuning':'AMtown03 only','tune':tune,'cases':{}}
    for variant in sorted({r['variant'] for r in rows}):
        group=[r for r in rows if r['variant']==variant]
        case={}
        for kind in ('large','compact'):
            selected=[next(s for s in r['rows'] if s['kind']==kind) for r in group]
            case[kind]={'clips':len(selected),'f':float(np.mean([r['f'] for r in selected])),
                        'middle_f':float(np.mean([r['middle_f'] for r in selected])),
                        'uniform_f':float(np.mean([r['uniform_f'] for r in selected])),
                        'requested_depth_fraction':sum(r['depth_requests'] for r in selected)/sum(r['candidate_frames'] for r in selected)}
        summary['cases'][variant]=case
    (output/'evaluation_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print('EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    main()
