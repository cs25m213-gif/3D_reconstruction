"""Remote CPU counterfactual labels: corrupted RGB -> new student depths -> fused gains.

Uses only the existing geometry-qualified cohort. Other corpus imagery is used
by visual pretraining, not assigned invented LiDAR rewards. All outputs stay remote.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch

from . import bin_memory_selector as bm
from . import reconstruction_selector as rs
from . import counterfactual_replay as cr
from .contribution_labels import load_clip, value
from .selector_corruptions import corrupt, frame_seed, quality_vector
from .student_inference import load_student

ROOT = Path('/home/vishnu/geostreamfusion_attribution_20261108')
OUT = Path('/home/vishnu/datasets/selector_large_20261114')
BASE = ROOT/'runs/selector_20261111'
VARIANTS = ('clean','motion_blur','defocus','mixed','frame_gap')
ENCODER = None
STUDENT = None


def init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    cv2.setNumThreads(1)
    import faulthandler, os
    f = (OUT/'utility_faults')
    f.mkdir(exist_ok=True)
    global FAULT
    FAULT = (f/f'{os.getpid()}.log').open('w')
    faulthandler.enable(FAULT)


def descriptor(images):
    global ENCODER
    if ENCODER is None:
        ENCODER = torch.hub.load(str(OUT/'dino_source'), 'dinov2_vits14', source='local', pretrained=False)
        ENCODER.load_state_dict(torch.load(OUT/'models/dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True))
        ENCODER.eval().requires_grad_(False)
    output = []
    for i in range(0, len(images), 8):
        x = torch.from_numpy(np.stack(images[i:i+8]).transpose(0,3,1,2)).float()/255
        mean = torch.tensor([.485,.456,.406])[None,:,None,None]
        std = torch.tensor([.229,.224,.225])[None,:,None,None]
        with torch.no_grad():
            y = ENCODER((x-mean)/std).numpy()
        output.append(y)
    result = np.concatenate(output).astype(np.float32)
    return result / np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-8)


def collect(arg):
    name, variant = arg
    target = OUT/'utility'/f'{name}__{variant}.npz'
    meta_path = target.with_suffix('.json')
    if target.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())
    global STUDENT
    if STUDENT is None:
        STUDENT = load_student('/home/vishnu/geostreamfusion_more_data_20261104/data/student.pt','cpu')
    contract = json.loads((ROOT/'runs/contract_vishnu.json').read_text())
    metadata = json.loads((ROOT/'runs/contribution_20261109/labels'/f'{name}.json').read_text())
    original = cr.Clip.use_student
    faults = []

    def altered(self, model, device):
        if variant == 'frame_gap':
            self.views = self.views[::3]
            self.choose_keyframes(contract['utility']['keyframe_baseline_ratio'])
        for i, view in enumerate(self.views):
            seed = frame_seed(20261114, name+variant, view.index)
            rng = np.random.default_rng(seed)
            active = variant not in ('clean','frame_gap') and (len(self.views)//3 <= i < 2*len(self.views)//3 or rng.random()<.25)
            case = variant if active else 'clean'
            if case == 'mixed':
                case = str(rng.choice(['motion_blur','defocus','dark_noise','overexposure','compression','low_texture']))
            severity = float(rng.uniform(.25,.9)) if active else 0.
            view.rgb, _ = corrupt(view.rgb, case, severity, seed)
            faults.append({'frame':view.index,'case':case,'severity':severity})
        return original(self, model, device)

    cr.Clip.use_student = altered
    try:
        clip, ref = load_clip(ROOT, contract, name, STUDENT, 'cpu')
    finally:
        cr.Clip.use_student = original
    images = [cv2.resize(v.rgb,(224,224),interpolation=cv2.INTER_AREA) for v in clip.views]
    emb = descriptor(images)
    grey = [cv2.cvtColor(v.rgb,cv2.COLOR_RGB2GRAY) for v in clip.views]
    sharp = np.array([cv2.Laplacian(g,cv2.CV_64F).var() for g in grey])
    bright = np.array([g.mean()/255 for g in grey])
    n = len(images)
    points = [rs.back_project(v)[0] for v in clip.views]
    seen = [np.stack([rs.consistent(p,v)[0] if j!=i else np.zeros(p.shape[1],bool)
                     for j,v in enumerate(clip.views)]) for i,p in enumerate(points)]
    ctx = SimpleNamespace(views=clip.views, emb=emb, n=n, seen=seen,
                          F=lambda s:value([clip.views[i] for i in sorted(set(s))],ref,contract['utility']['voxel_m']) if len(s) else 0.)
    c = bm.Clip.__new__(bm.Clip)
    c.ctx, c.sharp = ctx, sharp
    c.C = bm.causal_scalars(clip.views,sharp,bright)
    # No clean-only error predictor is reused on corrupt data; error scalars explicitly absent.
    c.pe = np.zeros(n,np.float32)
    c._points,c._zmed,c._mask = {},{},{}
    rng = np.random.default_rng(frame_seed(42,name,VARIANTS.index(variant)))
    prototypes, occupants, selected = [],[],[]
    before, records = 0.,[]
    # All candidates are labelled, including very blurred candidates. No sharpness veto here.
    for t in range(n):
        cos = np.asarray([p@emb[t] for p in prototypes])
        if len(cos) and cos.max()>=.9:
            b = int(cos.argmax())
        else:
            b=len(prototypes);prototypes.append(emb[t]);occupants.append(None)
        a=occupants[b]
        x=bm.state_features(c,t,selected,len(prototypes),a,cos)
        changed=selected+[t] if a is None else [s for s in selected if s!=a]+[t]
        after=ctx.F(changed)
        gain=after-before
        records.append((t,-1 if a is None else a,x,gain))
        take=gain>0
        if rng.random()<.15:take=not take
        if take:selected,occupants[b],before=changed,t,after
    t,a,x,y=zip(*records)
    temp=target.with_name(target.stem+'.partial.npz')
    np.savez_compressed(temp,rgb=np.stack(images), embedding=emb,t=np.asarray(t),a=np.asarray(a),
                        x=np.asarray(x,np.float32),y=np.asarray(y,np.float32),
                        quality=np.stack([quality_vector(v.rgb) for v in clip.views]),
                        causal=c.C, sharp=sharp, reference_points=ref.points,
                        fusion_depth=np.stack([v.fusion_depth for v in clip.views]),
                        fusion_k=np.stack([v.fusion_k for v in clip.views]),
                        reference_pose=np.stack([v.reference_pose for v in clip.views]))
    temp.replace(target)
    result={'clip':name,'flight':metadata['flight'],'split':metadata['split'],'variant':variant,
            'frames':n,'rows':len(records),'faults':faults,'scope':'fixed reference poses; newly predicted corrupted-input depths and measured TSDF F gains',
            'voxel_m':contract['utility']['voxel_m'],'tolerance_m':contract['utility']['tolerance_m'],
            'quality_protocol':contract['utility']['quality_protocol'],
            'student_sha256':hashlib.sha256(Path('/home/vishnu/geostreamfusion_more_data_20261104/data/student.pt').read_bytes()).hexdigest()}
    meta_path.write_text(json.dumps(result,indent=2))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'utility').mkdir(exist_ok=True)
    names=sorted(p.stem for p in (ROOT/'runs/contribution_20261109/greedy').glob('AM*.npz'))
    names=[c for c in names if json.loads((ROOT/'runs/contribution_20261109/labels'/f'{c}.json').read_text())['split'] in ('train','validation')]
    jobs=[(name,variant) for variant in VARIANTS for name in names]
    results=[]
    with ProcessPoolExecutor(12,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        futures={pool.submit(collect,j):j for j in jobs}
        for future in as_completed(futures):
            r=future.result();results.append(r)
            print(json.dumps({'completed':len(results),'total':len(jobs),'clip':r['clip'],'variant':r['variant']}),flush=True)
            (OUT/'utility_status.json').write_text(json.dumps({'state':'running','completed':len(results),'total':len(jobs)}))
    (OUT/'utility_status.json').write_text(json.dumps({'state':'complete','completed':len(results),'total':len(jobs),
                                                       'train_rows':sum(r['rows'] for r in results if r['split']=='train'),
                                                       'validation_rows':sum(r['rows'] for r in results if r['split']=='validation')}))


if __name__=='__main__':
    main()
