"""Bounded joint pilot with actual subset DA3 targets and normalized 3D point loss.

Enumerates six two-of-four source sets. Both student and selector are trainable;
teacher outputs are detached. Fixed target/poses; this is not full mesh training.
"""
import argparse
import itertools
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .train_differentiable_selector import SourceSelector, Candidates, corrupt
from .student_dataset import discover
from .student_inference import load_student
from .depth_backend import DA3Backend
from .utility_contracts import load_contract, write_json, sha256_file

PAIRS=list(itertools.combinations(range(4),2))


def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def points_error(pred,target,mask,k):
    h,w=pred.shape[-2:]
    y,x=torch.meshgrid(torch.arange(h,device=pred.device),torch.arange(w,device=pred.device),indexing='ij')
    rays=torch.linalg.solve(k.float(),torch.stack((x,y,torch.ones_like(x))).float().reshape(3,-1))
    length=rays.norm(dim=0).reshape(h,w)
    if not mask.any():return pred.sum()*0
    return ((pred-target).abs()*length)[mask].mean()


def probabilities(selector,batch):
    logits=selector(batch['rgb'][None],batch['P'][None])[0]
    return torch.stack([logits[i]+logits[j] for i,j in PAIRS]).softmax(0)


def predict(student,batch,pair):
    ids=[0,pair[0]+1,pair[1]+1]
    return student(batch['rgb'][ids][None],batch['K'][ids][None],batch['P'][ids][None])[0][0,0]


def loss_terms(student,selector,b):
    p=probabilities(selector,b)
    depth_losses=[];lidar_losses=[];teacher_losses=[]
    reference=b['reference']; mask=b['reference_valid']; k=b['K'][0]
    for i,pair in enumerate(PAIRS):
        prediction=predict(student,b,pair)
        target=b['subset_teacher'][i]
        # Missing subset-teacher pixels cannot make a choice seem better.
        teacher_losses.append(points_error(target,reference,mask,k).detach())
        valid=(target>.25)&(target<4)&torch.isfinite(target)
        lidar=points_error(prediction,b['measured'][0],b['measured_valid'][0],k)
        lidar_losses.append(lidar)
        depth_losses.append(.05*points_error(prediction,target,valid,k)+2*lidar)
    lidar_values=torch.stack(lidar_losses)
    lidar_selection=(p*lidar_values.detach()).sum()
    teacher_selection=.05*(p*torch.stack(teacher_losses)).sum()
    selection=lidar_selection+teacher_selection
    depth=(p.detach()*torch.stack(depth_losses)).sum()
    joint=(p*lidar_values).sum()
    return depth+selection+.25*joint,dict(depth=depth,selection=selection,
             lidar_selection=lidar_selection,teacher_selection=teacher_selection,joint=joint)


def prepare(a,device):
    out=a.output;cache=out/'teacher_cache';cache.mkdir(parents=True,exist_ok=True)
    roles=discover(Path.cwd(),load_contract(a.config))
    datasets={r:Candidates([c for c in roles[r] if not c.map_only],stride=1) for r in ['train','validation']}
    plans=[]
    for role,limit in [('train',16),('validation',8)]:
        ds=datasets[role]
        # Spread over the ordered corpus, not only its first flight.
        chosen=np.linspace(0,len(ds.clips)-1,min(limit,len(ds.clips)),dtype=int)
        for c in chosen:
            options=[i for i,(ci,p) in enumerate(ds.items) if ci==c]
            if options:plans.append((role,ds,options[len(options)//2],ds.clips[c].name))
    teacher=DA3Backend(device=str(device),resolution=224)
    teacher.predictor.requires_grad_(False).eval()
    def infer(b,ids):
        rgb=[(b['rgb'][i].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8) for i in ids]
        with torch.inference_mode():
            result=teacher.predictor.inference(rgb,extrinsics=b['P'][ids].numpy(),intrinsics=b['K'][ids].numpy(),
                   align_to_input_ext_scale=True,process_res=224,process_res_method='upper_bound_resize',ref_view_strategy='first')
        dh,dw=result.depth[0].shape;h,w=b['rgb'].shape[-2:]
        k=np.asarray(result.intrinsics[0]).copy();k[0]*=w/dw;k[1]*=h/dh
        if not np.allclose(k,b['K'][0].numpy(),atol=1.,rtol=.005):raise ValueError('Teacher grid/intrinsic mismatch')
        d=cv2.resize(result.depth[0].astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        d[~np.isfinite(d)|(d<=0)]=0
        anchors=b['anchor'][0].numpy()
        valid=(anchors>.25)&(anchors<4)&(d>0)&np.isfinite(d)
        if valid.sum()<30:raise ValueError('Insufficient independent sparse anchors for teacher scale')
        factor=float(np.median(anchors[valid]/d[valid]))
        if not np.isfinite(factor) or not .1<factor<10:raise ValueError('Implausible teacher sparse-anchor scale')
        d*=factor
        return torch.from_numpy(d)
    manifest=[];failed=[]
    for plan_index,(role,ds,index,name) in enumerate(plans):
        write_json(out/'status.json',dict(state='preparing_teacher_targets',current=name,prepared=len(manifest),failed=failed))
        try:
            clean=ds[index]
            if clean['measured_valid'].sum()<30:raise ValueError('Insufficient independent measured validation pixels')
            reference=infer(clean,[0,1,2,3,4])
            mask=(reference>.25)&(reference<4)
            if mask.sum()<1000:raise ValueError('Insufficient richer teacher coverage')
            for condition in ['clean','corrupted']:
                b={k:v.clone() for k,v in clean.items()}
                candidate=plan_index%4
                if condition=='corrupted':b['rgb']=corrupt(b['rgb'][None],candidate)[0]
                subsets=torch.stack([infer(b,[0,i+1,j+1]) for i,j in PAIRS])
                b.update(reference=reference,reference_valid=mask,subset_teacher=subsets)
                file=f'{role}_{name}_{condition}.pt';atomic_save(b,cache/file)
                manifest.append(dict(role=role,clip=name,condition=condition,corrupted_candidate=candidate if condition=='corrupted' else None,file=file,
                                     sha256=sha256_file(cache/file)))
                write_json(out/'manifest.json',dict(rows=manifest,failed=failed))
        except Exception as exc:
            failed.append(dict(role=role,clip=name,error=repr(exc)))
            write_json(out/'manifest.json',dict(rows=manifest,failed=failed))
    del teacher
    if device.type=='cuda':torch.cuda.empty_cache()
    return manifest,failed


@torch.no_grad()
def evaluate(student,selector,rows,out,device):
    student.eval();selector.eval();totals={};counts={}
    for row in rows:
        b=torch.load(out/'teacher_cache'/row['file'],map_location=device,weights_only=True)
        learned=int(probabilities(selector,b).argmax())
        for label,pair in [('learned',PAIRS[learned]),('uniform',(0,3)),('recent',(2,3))]:
            pred=predict(student,b,pair)
            key=row['condition']+'_'+label
            metric=points_error(pred,b['measured'][0],b['measured_valid'][0],b['K'][0])
            totals[key]=totals.get(key,0)+float(metric);counts[key]=counts.get(key,0)+1
    return {k:totals[k]/counts[k] for k in totals}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--student',type=Path,required=True)
    p.add_argument('--selector',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared-cache',type=Path,help='Reuse verified teacher windows from an earlier pilot without rerunning DA3')
    p.add_argument('--epochs',type=int,default=6);a=p.parse_args()
    if (a.output/'experiment.json').exists():raise SystemExit('Use a fresh output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_json(a.output/'experiment.json',dict(student=str(a.student),student_sha256=sha256_file(a.student),selector=str(a.selector),
               selector_sha256=sha256_file(a.selector),teacher='depth-anything/DA3-BASE',teacher_resolution=224,
               contract_sha256=sha256_file(a.config),source_sha256=sha256_file(Path(__file__)),
               scope='Pilot: enumerated source subsets, known poses, normalized target-view 3D point loss, not fused mesh loss',
               loss='depth + selection + 0.25 * joint; depth = 0.05 * subset teacher + 2 * LiDAR; selection = detached selected-student LiDAR 3D error + 0.05 * teacher subset/richer discrepancy; joint = selected-student LiDAR 3D error',
               teacher_scale='median target sparse-anchor ratio; no measured reference used in scale fit',
               selector_lr=1e-4,student_lr=1e-6,epochs=a.epochs))
    if a.prepared_cache:
        source=a.prepared_cache.resolve()
        previous=json.loads((source/'manifest.json').read_text())
        rows,failed=previous['rows'],previous['failed']
        if not rows or not all(r['role'] in ('train','validation') and
               sha256_file(source/'teacher_cache'/r['file'])==r['sha256'] for r in rows):
            raise ValueError('Prepared cache missing files or checksum mismatch')
        (a.output/'teacher_cache').symlink_to(source/'teacher_cache',target_is_directory=True)
        write_json(a.output/'manifest.json',dict(rows=rows,failed=failed,source=str(source),
                   source_manifest_sha256=sha256_file(source/'manifest.json')))
    else:
        rows,failed=prepare(a,device)
    train=[r for r in rows if r['role']=='train'];val=[r for r in rows if r['role']=='validation']
    if len(train)<8 or len(val)<4:raise RuntimeError(f'Insufficient qualified pilot data: {len(train)} train, {len(val)} validation')
    student=load_student(a.student,device).requires_grad_(True)
    selector=SourceSelector().to(device)
    selector.load_state_dict(torch.load(a.selector,map_location=device,weights_only=False)['model'])
    baseline=evaluate(student,selector,val,a.output,device);write_json(a.output/'baseline.json',baseline)
    initial_student={k:v.detach().clone() for k,v in student.state_dict().items()}
    initial_selector={k:v.detach().clone() for k,v in selector.state_dict().items()}
    opt=torch.optim.AdamW([dict(params=student.parameters(),lr=1e-6),dict(params=selector.parameters(),lr=1e-4)],weight_decay=1e-4)
    steps=0;best=float('inf')
    with (a.output/'steps.jsonl').open('a') as log:
        for epoch in range(1,a.epochs+1):
            random.shuffle(train);student.train();selector.train()
            for row in train:
                b=torch.load(a.output/'teacher_cache'/row['file'],map_location=device,weights_only=True)
                opt.zero_grad(set_to_none=True);loss,parts=loss_terms(student,selector,b)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite joint loss')
                loss.backward()
                sn=nnorm(student);qn=nnorm(selector)
                if not np.isfinite(sn+qn) or min(sn,qn)<=0:raise RuntimeError('Both models must receive finite nonzero gradients')
                torch.nn.utils.clip_grad_norm_(student.parameters(),1.);torch.nn.utils.clip_grad_norm_(selector.parameters(),1.)
                opt.step();steps+=1
                if steps==1:
                    write_json(a.output/'gradient_check.json',dict(student_gradient_norm=sn,selector_gradient_norm=qn,
                        student_changed=any(not torch.equal(v,initial_student[k]) for k,v in student.state_dict().items()),
                        selector_changed=any(not torch.equal(v,initial_selector[k]) for k,v in selector.state_dict().items()),teacher_targets_detached=True))
                log.write(json.dumps(dict(epoch=epoch,steps=steps,loss=float(loss.detach()),student_grad=sn,selector_grad=qn,
                                         **{k:float(v.detach()) for k,v in parts.items()}))+'\n');log.flush()
                write_json(a.output/'status.json',dict(state='joint_training',epoch=epoch,steps=steps,train_examples=len(train),validation_examples=len(val)))
            metrics=evaluate(student,selector,val,a.output,device)
            score=(metrics['clean_learned']+metrics['corrupted_learned'])/2
            payload=dict(student=student.state_dict(),selector=selector.state_dict(),optimizer=opt.state_dict(),epoch=epoch,steps=steps,validation=metrics)
            atomic_save(payload,a.output/'latest.pt')
            if score<best:best=score;atomic_save(payload,a.output/'best.pt')
            print(json.dumps(dict(epoch=epoch,validation=metrics)),flush=True)
    write_json(a.output/'status.json',dict(state='complete',steps=steps,best_validation_point_error=best,
               baseline=baseline,promoted=False,remaining='Larger flight-level study, global selection, full video and mesh validation'))


def nnorm(model):
    return float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)))


if __name__=='__main__':main()
