"""Experimental source-frame selector: fixed-pose depth surrogate, not global mesh selection.

Frozen depth weights, fixed cached teacher targets, hard two-of-four source selection
with a straight-through softmax gradient. Only the selector optimizer is updated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from .student_dataset import discover, UpdateDataset, view_arrays, GRID
from .student_losses import masked_log
from .student_inference import load_student
from .utility_contracts import load_contract, sha256_file, write_json


class SourceSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),
                                     nn.Conv2d(32,32,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.score = nn.Sequential(nn.Linear(76,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self, rgb, poses):
        b,v,c,h,w = rgb.shape
        features = self.encoder(F.interpolate(rgb.reshape(b*v,c,h,w),size=(96,128),mode='area')).reshape(b,v,32)
        relative = poses[:,1:] @ torch.linalg.inv(poses[:,:1])
        geometry = relative[:,:,:3,:].reshape(b,v-1,12)
        return self.score(torch.cat((features[:,1:],features[:,:1].expand(-1,v-1,-1),geometry),-1)).squeeze(-1)


def gates(logits, training, temperature=.7):
    hard = torch.zeros_like(logits).scatter(1,logits.topk(2,dim=1).indices,1.)
    soft = 2 * (logits/temperature).softmax(1)
    return hard + soft - soft.detach() if training else hard


class Candidates(Dataset):
    def __init__(self, clips, augment=False, stride=3):
        self.clips, self.augment = clips, augment
        self.base = UpdateDataset(clips,sequence=1,causal=True)
        self.items = [(c,p) for c,clip in enumerate(clips) for p in range(4,len(clip.indices),stride)]

    def __len__(self): return len(self.items)

    def __getitem__(self, item):
        c,p = self.items[item]; clip = self.clips[c]
        scale = clip.depths[clip.indices[p]]
        target = self.base.frame(clip,p,scale,None)
        positions = np.linspace(max(0,p-12),p-1,4,dtype=int).tolist()
        rgb, ks, ps = [],[],[]
        for position in [p]+positions:
            row = view_arrays(clip,clip.indices[position],('rgb','K','P'))
            ih,iw = row['rgb'].shape[:2]; h,w = GRID
            rgb.append(cv2.resize(row['rgb'],(w,h),interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/255)
            k = row['K'].copy().astype(np.float32); k[0]*=w/iw;k[1]*=h/ih;ks.append(k)
            pose = row['P'].copy().astype(np.float32);pose[:3,3]/=scale;ps.append(pose)
        target.update(rgb=torch.from_numpy(np.stack(rgb)),K=torch.from_numpy(np.stack(ks)),P=torch.from_numpy(np.stack(ps)))
        if self.augment:
            target['rgb'] = corrupt(target['rgb'].unsqueeze(0), item % 4, stochastic=True)[0]
        return {k:v for k,v in target.items() if torch.is_tensor(v)}


def corrupt(rgb, candidate=0, stochastic=False):
    """Corrupt sources only; fixed clean target/reference cannot change with selection."""
    result = rgb.clone()
    if stochastic and torch.rand(()) > .5: return result
    source = candidate+1
    result[:,source] = F.avg_pool2d(result[:,source],11,1,5)
    result[:,source] = (result[:,source]*.45).clamp(0,1)
    return result


def objective(depth, batch):
    measured = batch['measured_valid']
    teacher = batch['teacher_valid'] & ~measured
    anchor = (batch['anchor']>.25)&(batch['anchor']<4)&~measured&~teacher
    return (2*masked_log(depth,batch['measured'],measured) + masked_log(depth,batch['teacher'],teacher)
            + .2*masked_log(depth,batch['anchor'],anchor))


@torch.no_grad()
def evaluate(selector, student, loader, device):
    selector.eval(); totals={}; counts={}
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        for condition in ['clean','corrupted']:
            rgb=batch['rgb'] if condition=='clean' else corrupt(batch['rgb'])
            logits=selector(rgb,batch['P'])
            policies={'learned':gates(logits,False),'uniform':torch.tensor([1.,0.,0.,1.],device=device).expand_as(logits),
                      'recent':torch.tensor([0.,0.,1.,1.],device=device).expand_as(logits)}
            for policy,weight in policies.items():
                pred,_=student(rgb,batch['K'],batch['P'],weight)
                key=f'{condition}_{policy}'
                totals[key]=totals.get(key,0.)+float(objective(pred,batch))*len(pred)
                counts[key]=counts.get(key,0)+len(pred)
    return {k:totals[k]/counts[k] for k in totals}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--student',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args()
    if (a.output/'launch_config.json').exists(): raise SystemExit('Use a new output directory')
    torch.set_num_threads(3);cv2.setNumThreads(1)
    # Different view-batch sizes must not change convolution precision in the
    # hard-gate versus physically sliced-view equivalence check.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(.18)
    roles=discover(Path.cwd(),load_contract(a.config))
    train=Candidates(roles['train'],augment=True)
    validation=Candidates(roles['validation'],stride=6)
    if not len(train) or not len(validation): raise SystemExit('Empty qualified split')
    loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=a.workers,persistent_workers=a.workers>0)
    val=DataLoader(validation,batch_size=1,shuffle=False,num_workers=a.workers,persistent_workers=a.workers>0)
    student=load_student(a.student,device).requires_grad_(False)
    selector=SourceSelector().to(device)
    optimizer=torch.optim.AdamW(selector.parameters(),lr=1e-4,weight_decay=1e-4)
    batch={k:v.to(device) for k,v in next(iter(loader)).items()}
    weight=gates(selector(batch['rgb'],batch['P']),True)
    depth,_=student(batch['rgb'],batch['K'],batch['P'],weight)
    loss=objective(depth,batch);loss.backward()
    grads=[x.grad for x in selector.parameters() if x.grad is not None]
    norm=float(torch.sqrt(sum(g.square().sum() for g in grads)))
    if not grads or not np.isfinite(norm) or norm<=1e-10: raise RuntimeError('No usable selector gradient')
    if any(x.grad is not None for x in student.parameters()): raise RuntimeError('Depth model received parameter gradients')
    with torch.no_grad():
        rgb,k,pose=batch['rgb'],batch['K'],batch['P']
        ordinary=student(rgb,k,pose)[0]
        ones=student(rgb,k,pose,torch.ones_like(weight))[0]
        hard=student(rgb,k,pose,torch.tensor([[1.,0.,0.,1.]],device=device))[0]
        sliced=student(rgb[:,[0,1,4]],k[:,[0,1,4]],pose[:,[0,1,4]])[0]
        torch.testing.assert_close(ordinary,ones,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(hard,sliced,atol=1e-5,rtol=1e-5)
    a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/'gradient_check.json',dict(selector_gradient_norm=norm,depth_parameter_gradients=False,
               hard_selection_matches_sliced_views=True,all_ones_matches_original=True,
               hard_selection_max_absolute_difference=float((hard-sliced).abs().max()),tf32_enabled=False))
    if a.smoke_only: print('Gradient and hard-selection equivalence checks passed',flush=True);return
    optimizer.zero_grad(set_to_none=True)
    source=Path(__file__).parent
    write_json(a.output/'launch_config.json',dict(student=str(a.student),student_sha256=sha256_file(a.student),
               contract_sha256=sha256_file(a.config),train_examples=len(train),validation_examples=len(validation),
               train_clips=[c.name for c in roles['train']],validation_clips=[c.name for c in roles['validation']],
               epochs=a.epochs,seed=42,architecture='source_selector_cnn_v1',source_budget=2,candidates=4,
               sources={name:sha256_file(source/name) for name in ['train_differentiable_selector.py','depth_student.py','student_dataset.py']},
               scope='Experimental fixed-pose source selector, depth surrogate; not a global frame selector or mesh-quality result'))
    write_json(a.output/'status.json',dict(state='training',epoch=0,steps=0))
    best=float('inf'); stale=0; steps=0; started=time.time()
    with (a.output/'steps.jsonl').open('a') as log:
        for epoch in range(1,a.epochs+1):
            selector.train();losses=[]
            for batch in loader:
                batch={k:v.to(device) for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                weight=gates(selector(batch['rgb'],batch['P']),True)
                depth,_=student(batch['rgb'],batch['K'],batch['P'],weight)
                loss=objective(depth,batch)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite selector loss')
                loss.backward();norm=nn.utils.clip_grad_norm_(selector.parameters(),1.)
                if not torch.isfinite(norm): raise RuntimeError('Non-finite selector gradient')
                optimizer.step();steps+=1;losses.append(float(loss.detach()))
                if steps%25==0:
                    log.write(json.dumps(dict(epoch=epoch,steps=steps,loss=float(np.mean(losses[-25:])),gradient_norm=float(norm)))+'\n');log.flush()
                    write_json(a.output/'status.json',dict(state='training',epoch=epoch,steps=steps))
            metrics=evaluate(selector,student,val,device)
            score=(metrics['clean_learned']+metrics['corrupted_learned'])/2
            payload=dict(model=selector.state_dict(),architecture='source_selector_cnn_v1',epoch=epoch,steps=steps,
                         optimizer=optimizer.state_dict(),validation=metrics,student_sha256=sha256_file(a.student))
            tmp=a.output/'latest.tmp';torch.save(payload,tmp);tmp.replace(a.output/'latest.pt')
            if score<best:
                best=score;stale=0;tmp=a.output/'best.tmp';torch.save(payload,tmp);tmp.replace(a.output/'best.pt')
            else:stale+=1
            print(json.dumps(dict(epoch=epoch,steps=steps,validation=metrics,seconds=time.time()-started)),flush=True)
            if stale>=3:break
    write_json(a.output/'status.json',dict(state='complete',epoch=epoch,steps=steps,best_surrogate_loss=best,
               promoted=False,remaining='Independent hard-frame/mesh evaluation and global selection integration'))


if __name__=='__main__':main()
