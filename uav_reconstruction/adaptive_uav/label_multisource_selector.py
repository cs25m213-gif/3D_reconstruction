"""Incremental CPU labels for admitted multisource reconstruction episodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch

from . import bin_memory_selector as bm, reconstruction_selector as rs
from . import prepare_large_selector_labels as old
from .contribution_labels import value
from .reconstruction_quality import ReferencePopulation
from .selector_corruptions import corrupt, frame_seed, quality_vector
from .student_inference import load_student, predict, anchor_scale

STUDENT=None


def init():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False;cv2.setNumThreads(1)


def collect(job):
    geometry,variant,output,student_path=job
    geometry,output=Path(geometry),Path(output)
    meta=json.loads(geometry.with_suffix('.json').read_text())
    name=geometry.stem;target=output/'utility'/f'{name}__{variant}.npz'
    if target.exists() and target.with_suffix('.json').exists():return str(target)
    global STUDENT
    if STUDENT is None:STUDENT=load_student(student_path,'cpu')
    with np.load(geometry) as z:source={k:z[k] for k in z.files}
    ids=np.arange(len(source['rgb']))[::3] if variant=='frame_gap' else np.arange(len(source['rgb']))
    rgb=[];faults=[]
    for t,i in enumerate(ids):
        seed=frame_seed(20261115,name+variant,int(i));rng=np.random.default_rng(seed)
        active=variant not in ('clean','frame_gap') and (len(ids)//3<=t<2*len(ids)//3 or rng.random()<.25)
        case=variant if active else 'clean'
        if case=='mixed':case=str(rng.choice(['motion_blur','defocus','dark_noise','overexposure','compression','low_texture']))
        severity=float(rng.uniform(.25,.9)) if active else 0.
        image,_=corrupt(source['rgb'][i],case,severity,seed);rgb.append(image)
        faults.append(dict(frame=int(i),case=case,severity=severity))
    views=[]
    for t,i in enumerate(ids):
        group=[t]+list(range(max(0,t-2),t))[::-1]
        anchors=source['anchors'][i]
        inputs=[(rgb[j],source['K'][ids[j]],np.linalg.inv(source['P'][ids[j]])) for j in group]
        depth,conf=predict(STUDENT,inputs,anchor_scale(anchors),'cpu',anchors=anchors)
        views.append(SimpleNamespace(fusion_depth=depth,fusion_k=source['K'][i],reference_pose=np.linalg.inv(source['P'][i]),fusion_rgb=rgb[t]))
    images=[cv2.resize(r,(224,224),interpolation=cv2.INTER_AREA) for r in rgb]
    emb=old.descriptor(images)
    grey=[cv2.cvtColor(r,cv2.COLOR_RGB2GRAY) for r in rgb]
    sharp=np.array([cv2.Laplacian(g,cv2.CV_64F).var() for g in grey]);bright=np.array([g.mean()/255 for g in grey])
    scales=[anchor_scale(source['anchors'][i]) for i in ids];scale=float(np.median(scales))
    voxel=float(np.clip(scale*.0025,.01,.5));tolerance=float(np.clip(scale*.02,.05,2.))
    ref=ReferencePopulation(source['reference_points'],tolerance,'QUALIFIED',meta['trust'])
    pts=[rs.back_project(v)[0] for v in views]
    seen=[np.stack([rs.consistent(p,v)[0] if i!=j else np.zeros(p.shape[1],bool) for j,v in enumerate(views)]) for i,p in enumerate(pts)]
    ctx=SimpleNamespace(views=views,emb=emb,n=len(views),seen=seen,
        F=lambda s:value([views[i] for i in sorted(set(s))],ref,voxel) if len(s) else 0.)
    c=bm.Clip.__new__(bm.Clip);c.ctx,c.sharp=ctx,sharp
    c.C=bm.causal_scalars(views,sharp,bright);c.pe=np.zeros(len(views),np.float32)
    c._points,c._zmed,c._mask={},{},{}
    rng=np.random.default_rng(frame_seed(20261115,name,old.VARIANTS.index(variant)))
    prototypes,occupants,selected,records=[],[],[],[];before=0.
    for t in range(len(views)):
        cos=np.array([p@emb[t] for p in prototypes])
        if len(cos) and cos.max()>=.9:b=int(cos.argmax())
        else:b=len(prototypes);prototypes.append(emb[t]);occupants.append(None)
        a=occupants[b];x=bm.state_features(c,t,selected,len(prototypes),a,cos)
        changed=selected+[t] if a is None else [s for s in selected if s!=a]+[t]
        after=ctx.F(changed);gain=after-before;records.append((t,-1 if a is None else a,x,gain))
        take=gain>0
        if rng.random()<.15:take=not take
        if take:selected,occupants[b],before=changed,t,after
    t,a,x,y=zip(*records);tmp=target.with_suffix('.partial.npz')
    np.savez_compressed(tmp,rgb=np.stack(images),embedding=emb,t=t,a=a,x=np.array(x,np.float32),y=np.array(y,np.float32),
        quality=np.stack([quality_vector(r) for r in rgb]),causal=c.C,sharp=sharp,
        reference_points=ref.points,fusion_depth=np.stack([v.fusion_depth for v in views]),
        fusion_k=np.stack([v.fusion_k for v in views]),reference_pose=np.stack([v.reference_pose for v in views]))
    tmp.replace(target)
    meta.update(variant=variant,rows=len(t),frames=len(views),faults=faults,voxel_m=voxel,tolerance_m=tolerance,
        quality_protocol=meta['trust'],source_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
        student_sha256=hashlib.sha256(Path(student_path).read_bytes()).hexdigest(),
        supervision_weight=.25 if meta['dataset']=='uavlight' else 1.,
        scope='re-fused TSDF counterfactual gain; fixed past-only student; source-specific reference proxy')
    meta_path=target.with_suffix('.json');temp=meta_path.with_suffix('.tmp');temp.write_text(json.dumps(meta,indent=2));temp.replace(meta_path)
    return str(target)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--student',required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();(a.output/'utility').mkdir(exist_ok=True,parents=True)
    pending={};scheduled=set();failed=[];completed=0
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        while True:
            for f in sorted((a.output/'geometry').glob('*.npz')):
                if not f.with_suffix('.json').exists():continue
                meta=json.loads(f.with_suffix('.json').read_text())
                if meta['split'] not in ('train','validation'):continue
                for variant in old.VARIANTS:
                    key=(str(f),variant)
                    if key in scheduled:continue
                    scheduled.add(key);future=pool.submit(collect,(*key,str(a.output),a.student));pending[future]=key
            if pending:
                ready,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in ready:
                    key=pending.pop(future)
                    try:future.result();completed+=1
                    except Exception as exc:failed.append(dict(job=key,error=repr(exc)));print(failed[-1],flush=True)
            else:time.sleep(5)
            status_path=a.output/'geometry_status.json';geometry=json.loads(status_path.read_text()) if status_path.exists() else {}
            done=geometry.get('state') in ('complete','failed') and not pending
            state=('failed' if failed or geometry.get('state')=='failed' else 'complete') if done else 'running'
            status=dict(state=state,completed=completed,scheduled=len(scheduled),pending=len(pending),failures=failed)
            temp=a.output/'utility_status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(a.output/'utility_status.json')
            if done:break
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
