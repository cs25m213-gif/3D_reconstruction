"""Train a sequential depth updater from cached DA3 and frozen student outputs.

DA3 is the training target; sparse anchors provide a second internal check.
This is not independent metric LiDAR validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .depth_student import DepthStudent
from .recurrent_depth import RecurrentDepthUpdater,confidence_average,warp_source_depth
from .train_depth_student import TeacherCacheDataset,discover,split_flights


def key(path):return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()+'.npz'


def prepare(paths,cache,checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=DepthStudent(**saved.get('architecture',{'channels':48,'planes':24})).to(device).eval()
    model.load_state_dict(saved['model']);cache.mkdir(parents=True,exist_ok=True)
    accepted=[];rejected=[]
    for index,path in enumerate(paths,1):
        destination=cache/key(path)
        if destination.is_file():accepted.append(path);continue
        try:
            sample=TeacherCacheDataset([path])[0]
            rgb=sample['rgb'][None].to(device);k=sample['K'][None].to(device);p=sample['P'][None].to(device)
            with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                depth,confidence=model(rgb,k,p)
            scale=float(sample['scale']);pose=sample['P'][0].numpy().copy();pose[:3,3]*=scale
            with np.load(path) as source:context=[str(x) for x in source['context']]
            np.savez_compressed(destination,pred=(depth[0,0].float().cpu().numpy()*scale).astype(np.float16),
                conf=confidence[0,0].float().cpu().numpy().astype(np.float16),
                teacher=(sample['teacher'][0].numpy()*scale).astype(np.float16),
                anchors=(sample['anchor'][0].numpy()*scale).astype(np.float16),
                measured=(sample['measured'][0].numpy()*scale).astype(np.float16),
                rgb=(sample['rgb'][0].numpy().transpose(1,2,0)*255).astype(np.uint8),
                K=sample['K'][0].numpy(),P=pose,context=np.array(context),scale=scale)
            accepted.append(path)
            if index%100==0:print(f'prepared student predictions {index}/{len(paths)}',flush=True)
        except (ValueError,KeyError,FileNotFoundError) as exc:rejected.append({'path':str(path),'reason':str(exc)[:200]})
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return accepted,rejected


def example(path,cache):
    with np.load(cache/key(path)) as z:target={name:z[name].copy() for name in z.files}
    target_path=Path(path);sources=[]
    for name in target['context']:
        source=target_path.parent/(str(name)+'.npz')
        if source==target_path or not (cache/key(source)).exists():continue
        with np.load(cache/key(source)) as z:
            warped=warp_source_depth(z['pred'].astype(np.float32),z['K'],z['P'],
                target['K'],target['P'],target['pred'].shape)
            warped_conf=warp_source_depth(z['pred'].astype(np.float32),z['K'],z['P'],
                target['K'],target['P'],target['pred'].shape,payload=z['conf'].astype(np.float32))
        if (warped>0).mean()>=.03:sources.append((warped,warped_conf))
        if len(sources)==2:break
    if not sources:return None
    scale=float(target['scale']);prior=target['pred'].astype(np.float32)/scale
    teacher=target['teacher'].astype(np.float32)/scale
    anchors=np.where(target['measured']>0,target['measured'],target['anchors']).astype(np.float32)/scale
    return {'prior':prior,'prior_conf':target['conf'].astype(np.float32),
        'updates':[(value/scale,conf) for value,conf in sources],
        'teacher':teacher,'anchors':anchors,'rgb':target['rgb'].transpose(2,0,1).astype(np.float32)/255.}


def tensor(array,device):return torch.from_numpy(np.asarray(array,np.float32)[None,None]).to(device)


def run_example(model,row,device,training=False):
    previous=tensor(row['prior'],device);confidence=tensor(row['prior_conf'],device)
    teacher=tensor(row['teacher'],device);anchors=tensor(row['anchors'],device)
    rgb=torch.from_numpy(row['rgb'][None]).to(device)
    valid=(teacher>.25)&(teacher<4.)&torch.isfinite(teacher)
    anchored=(anchors>.25)&(anchors<4.)
    hidden=None;losses=[];averaged=previous.detach();average_conf=confidence.detach()
    for value,conf in row['updates']:
        new=tensor(value,device);new_conf=tensor(conf,device)
        previous,confidence,hidden=model(previous,new,confidence,new_conf,rgb,hidden)
        averaged=confidence_average(averaged,new,average_conf,new_conf)
        average_conf=torch.maximum(average_conf,new_conf)
        teacher_loss=(previous[valid].log()-teacher[valid].log()).abs().mean() if valid.any() else previous.sum()*0
        anchor_loss=(previous[anchored].log()-anchors[anchored].log()).abs().mean() if anchored.any() else previous.sum()*0
        stable=((previous-new).abs()*(new<=0)).mean()
        losses.append(teacher_loss+.5*anchor_loss+.02*stable)
    def metric(depth,reference,mask):
        return float(((depth-reference).abs()/reference.clamp_min(.1))[mask].mean().detach()) if mask.any() else None
    result={'teacher_absrel':metric(previous,teacher,valid),'anchor_absrel':metric(previous,anchors,anchored),
        'average_teacher_absrel':metric(averaged,teacher,valid),
        'average_anchor_absrel':metric(averaged,anchors,anchored)}
    return torch.stack(losses).mean(),result


def evaluate(model,paths,cache,device,max_examples=0):
    model.eval();rows=[]
    with torch.no_grad():
        for path in paths[:max_examples or None]:
            sample=example(path,cache)
            if sample is None:continue
            _,item=run_example(model,sample,device);rows.append(item)
    keys=('teacher_absrel','anchor_absrel','average_teacher_absrel','average_anchor_absrel')
    return {'examples':len(rows),**{key:float(np.mean([row[key] for row in rows if row[key] is not None]))
        if any(row[key] is not None for row in rows) else None for key in keys}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=Path('runs/uavscenes_30_batch'))
    parser.add_argument('--student-checkpoint',type=Path,default=Path('runs/depth_student_20261028/best.pt'))
    parser.add_argument('--output',type=Path,default=Path('runs/recurrent_depth_20261028'))
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--max-train',type=int,default=0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits,names=split_flights(discover(args.data_root),42)
    if args.max_train:splits['train']=splits['train'][:args.max_train]
    cache=args.output/'student_predictions';start=time.monotonic();rejected=[]
    for role in ('train','validation','test'):
        splits[role],errors=prepare(splits[role],cache,args.student_checkpoint,device)
        rejected.extend(errors)
    (args.output/'dataset_manifest.json').write_text(json.dumps({'flights':names,
        'frames':{role:len(paths) for role,paths in splits.items()},'rejected':rejected,
        'teacher':'cached DA3','student':str(args.student_checkpoint),
        'measured_ground_truth':False},indent=2)+'\n')
    model=RecurrentDepthUpdater().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    best=float('inf');best_epoch=0;stale=0
    for epoch in range(1,args.epochs+1):
        model.train();order=list(splits['train']);random.shuffle(order);losses=[]
        for path in order:
            sample=example(path,cache)
            if sample is None:continue
            optimizer.zero_grad(set_to_none=True)
            loss,_=run_example(model,sample,device,training=True)
            if not torch.isfinite(loss):continue
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
            losses.append(float(loss.detach()))
        validation=evaluate(model,splits['validation'],cache,device)
        score=validation['teacher_absrel']
        if score is None:raise RuntimeError('No valid recurrent validation pairs')
        if score<best:
            best=score;best_epoch=epoch;stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'parameters':sum(p.numel() for p in model.parameters()),
                'input':'anchor-normalized student depth and warped source student depth'},args.output/'best.pt')
        else:stale+=1
        status={'epoch':epoch,'train_examples':len(losses),'train_loss':float(np.mean(losses)),
            'validation':validation,'best_epoch':best_epoch,'seconds':time.monotonic()-start}
        (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if stale>=3:break
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=device,weights_only=True)['model'])
    test=evaluate(model,splits['test'],cache,device)
    passed=(test['teacher_absrel'] is not None and test['anchor_absrel'] is not None and
        test['teacher_absrel']<test['average_teacher_absrel'] and
        test['anchor_absrel']<=test['average_anchor_absrel'])
    report={'status':'complete','best_epoch':best_epoch,'held_out':test,
        'passes_internal_baseline':passed,'independent_geometry_validated':False,
        'elapsed_seconds':time.monotonic()-start,
        'warning':'Teacher and sparse anchors share estimated poses; do not claim independent metric accuracy.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
