"""Offline pose-conditioned DA3-LARGE distillation from saved depth caches.

Split by flight, never by frame. Training creates a research checkpoint only;
the production reconstruction backend is unchanged until independent geometry
evaluation and runtime comparison pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from .depth_student import DepthStudent


def save_json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def discover(root):
    root=Path(root)
    flights={}
    for run in sorted(root.iterdir()):
        cache=run/'reconstruction/multiview/depth_cache'
        if not cache.is_dir():continue
        flight=run.name.rsplit('_',1)[0]
        for path in sorted(cache.glob('*.npz')):
            flights.setdefault(flight,[]).append(str(path.resolve()))
    if len(flights)<3:raise ValueError('At least three distinct flights are required for train/validation/test')
    return flights


def split_flights(flights,seed):
    names=sorted(flights);random.Random(seed).shuffle(names)
    n_test=max(1,len(names)//7);n_val=max(1,len(names)//7)
    groups={'train':names[n_test+n_val:],'validation':names[n_test:n_test+n_val],
            'test':names[:n_test]}
    return {key:[p for name in value for p in flights[name]] for key,value in groups.items()},groups


def add_extra_caches(splits,groups,arguments):
    rejected={}
    for role,entries in (('train',arguments.extra_train),('validation',arguments.extra_validation),
                         ('test',arguments.extra_test)):
        for entry in entries:
            if '=' not in entry:raise ValueError(f'Expected SCENE=DEPTH_CACHE: {entry}')
            scene,directory=entry.split('=',1)
            if scene in {name for names in groups.values() for name in names}:
                raise ValueError(f'Duplicate scene identity: {scene}')
            paths=[];skipped=0
            for path in sorted(Path(directory).glob('*.npz')):
                try:
                    with np.load(path) as sample:
                        anchors=sample['anchors'];valid=np.isfinite(anchors)&(anchors>0)
                        usable=(valid.sum()>=30 and np.isfinite(sample['K']).all() and
                                np.isfinite(sample['P']).all() and np.isfinite(sample['depth']).all() and
                                np.isfinite(sample['confidence']).all())
                    if usable:paths.append(str(path.resolve()))
                    else:skipped+=1
                except (KeyError,ValueError,OSError):skipped+=1
            if not paths:raise ValueError(f'No depth caches: {directory}')
            groups[role].append(scene);splits[role].extend(paths)
            rejected[scene]=skipped
    return splits,groups,rejected


class TeacherCacheDataset(Dataset):
    def __init__(self,paths,size=(192,256)):
        self.paths=[Path(p) for p in paths];self.size=size

    def __len__(self):return len(self.paths)

    def __getitem__(self,index):
        target=self.paths[index]
        with np.load(target) as data:
            context=[str(x) for x in data['context']]
        names=[target.name.removesuffix('.npz')]+[n for n in context if n+'.npz'!=target.name]
        names=names[:3]
        if len(names)<2:raise ValueError('Distillation needs at least two posed views')
        rows=[]
        for name in names:
            path=target.parent/(name+'.npz')
            if not path.is_file():raise FileNotFoundError(path)
            with np.load(path) as data:
                row={k:data[k].copy() for k in ('rgb','K','P')}
                if name==names[0]:row.update({k:data[k].copy() for k in ('depth','confidence','anchors','measured')})
            rows.append(row)
        anchors=rows[0]['anchors'];valid=anchors[np.isfinite(anchors)&(anchors>0)]
        if len(valid)<30:raise ValueError(f'Insufficient sparse scale anchors: {target}')
        scale=float(np.median(valid))
        h,w=self.size
        images=[];intrinsics=[];poses=[]
        for row in rows:
            ih,iw=row['rgb'].shape[:2]
            images.append(cv2.resize(row['rgb'],(w,h),interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/255.)
            k=row['K'].astype(np.float32).copy();k[0]*=w/iw;k[1]*=h/ih;intrinsics.append(k)
            p=row['P'].astype(np.float32).copy();p[:3,3]/=scale;poses.append(p)
        # Repeat the last context view so every sample has a fixed V=3 shape.
        while len(images)<3:
            images.append(images[-1].copy());intrinsics.append(intrinsics[-1].copy());poses.append(poses[-1].copy())
        def resized(key,interpolation):
            return cv2.resize(rows[0][key].astype(np.float32),(w,h),interpolation=interpolation)
        depth=resized('depth',cv2.INTER_AREA)/scale
        anchor=resized('anchors',cv2.INTER_NEAREST)/scale
        measured=resized('measured',cv2.INTER_NEAREST)/scale
        confidence=resized('confidence',cv2.INTER_AREA)
        teacher_valid=np.isfinite(depth)&(depth>.25)&(depth<4.)&np.isfinite(confidence)
        # Confidence is model-specific; use a conservative within-image rank.
        teacher_valid &= confidence>=np.percentile(confidence[np.isfinite(confidence)],20)
        return {'rgb':torch.from_numpy(np.stack(images)),
                'K':torch.from_numpy(np.stack(intrinsics)),
                'P':torch.from_numpy(np.stack(poses)),
                'teacher':torch.from_numpy(depth[None].astype(np.float32)),
                'teacher_valid':torch.from_numpy(teacher_valid[None]),
                'anchor':torch.from_numpy(anchor[None].astype(np.float32)),
                'measured':torch.from_numpy(measured[None].astype(np.float32)),
                'scale':scale,'path':str(target)}


def photometric_loss(rgb,K,P,depth):
    """Minimum source reprojection with an identity auto-mask; no pose gradients."""
    b,v,_,h,w=rgb.shape
    y,x=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
    pixels=torch.stack((x,y,torch.ones_like(x)),0).float().reshape(1,3,h*w).expand(b,-1,-1)
    rays=torch.linalg.solve(K[:,0].float(),pixels)
    xyz=rays*depth.flatten(2)
    errors=[]
    for source in range(1,v):
        relative=P[:,source].float()@torch.linalg.inv(P[:,0].float())
        camera=relative[:,:3,:3]@xyz+relative[:,:3,3,None]
        projected=K[:,source].float()@camera
        z=projected[:,2].clamp_min(1e-6)
        u=(projected[:,0]/z).reshape(b,h,w);vv=(projected[:,1]/z).reshape(b,h,w)
        grid=torch.stack((2*u/max(w-1,1)-1,2*vv/max(h-1,1)-1),-1)
        warped=F.grid_sample(rgb[:,source],grid,align_corners=True,padding_mode='zeros')
        diff=(rgb[:,0]-warped).abs().mean(1,keepdim=True)
        identity=(rgb[:,0]-rgb[:,source]).abs().mean(1,keepdim=True)
        valid=((camera[:,2].reshape(b,h,w)>0)&(u>=0)&(u<w)&(vv>=0)&(vv<h))[:,None]
        errors.append(torch.where(valid&(diff<identity),diff,torch.ones_like(diff)))
    best=torch.stack(errors).amin(0)
    mask=best<1.
    return best[mask].mean() if mask.any() else depth.sum()*0.


def losses(batch,depth,confidence,photo_weight=.05):
    teacher=batch['teacher'];valid=batch['teacher_valid']
    log_error=(depth.clamp_min(1e-4).log()-teacher.clamp_min(1e-4).log()).abs()
    distill=log_error[valid].mean() if valid.any() else depth.sum()*0.
    measured=batch['measured'];anchors=torch.where(measured>0,measured,batch['anchor'])
    anchor_valid=(anchors>0)&(anchors<4.)
    anchor_error=(depth.clamp_min(1e-4).log()-anchors.clamp_min(1e-4).log()).abs()
    geometry=anchor_error[anchor_valid].mean() if anchor_valid.any() else depth.sum()*0.
    dx=(depth[:,:,:,1:]-depth[:,:,:,:-1]).abs()
    dy=(depth[:,:,1:,:]-depth[:,:,:-1,:]).abs()
    image=batch['rgb'][:,0]
    sx=torch.exp(-10*(image[:,:,:,1:]-image[:,:,:,:-1]).abs().mean(1,keepdim=True))
    sy=torch.exp(-10*(image[:,:,1:,:]-image[:,:,:-1,:]).abs().mean(1,keepdim=True))
    smooth=(dx*sx).mean()+(dy*sy).mean()
    photo=photometric_loss(batch['rgb'],batch['K'],batch['P'],depth) if photo_weight else depth.sum()*0.
    # Confidence is supervised only where geometric anchors exist.
    good=(anchor_error.detach()<.05).float()
    if anchor_valid.any():
        probability=confidence[anchor_valid].clamp(1e-5,1-1e-5)
        target=good[anchor_valid]
        conf_loss=-(target*probability.log()+(1-target)*(1-probability).log()).mean()
    else:conf_loss=depth.sum()*0.
    total=distill+2*geometry+photo_weight*photo+.01*smooth+.05*conf_loss
    return total,{'distill':float(distill.detach()),'anchor':float(geometry.detach()),
                  'photo':float(photo.detach()),'smooth':float(smooth.detach()),
                  'confidence':float(conf_loss.detach())}


@torch.no_grad()
def evaluate(model,loader,device,max_samples=0):
    model.eval();rows=[];sample_count=0
    for index,batch in enumerate(loader):
        if max_samples and index>=max_samples:break
        sample_count+=len(batch['path'])
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            depth,confidence=model(batch['rgb'],batch['K'],batch['P'])
        depth=depth.float();teacher=batch['teacher'];valid=batch['teacher_valid']
        anchor=torch.where(batch['measured']>0,batch['measured'],batch['anchor'])
        geometry=(anchor>0)&(anchor<4.)
        rows.append({'teacher_absrel':float(((depth-teacher).abs()/teacher.clamp_min(1e-4))[valid].mean()) if valid.any() else None,
                     'anchor_absrel':float(((depth-anchor).abs()/anchor.clamp_min(1e-4))[geometry].mean()) if geometry.any() else None,
                     'teacher_anchor_absrel':float(((teacher-anchor).abs()/anchor.clamp_min(1e-4))[geometry].mean()) if geometry.any() else None})
    return {'samples':sample_count,**{k:float(np.mean([r[k] for r in rows if r[k] is not None]))
                                  for k in ('teacher_absrel','anchor_absrel','teacher_anchor_absrel')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('runs/uavscenes_30_batch'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--extra-train',action='append',default=[],metavar='SCENE=DEPTH_CACHE')
    parser.add_argument('--extra-validation',action='append',default=[],metavar='SCENE=DEPTH_CACHE')
    parser.add_argument('--extra-test',action='append',default=[],metavar='SCENE=DEPTH_CACHE')
    parser.add_argument('--patience',type=int,default=4)
    parser.add_argument('--max-train-samples',type=int,default=0,help='Bounded smoke test; 0 uses all training views')
    parser.add_argument('--max-eval-samples',type=int,default=0,help='Bounded smoke test; 0 uses all validation/test views')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reset-best-on-resume',action='store_true',
                        help='Reset validation score when adding new validation scenes to a copied checkpoint')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    flights=discover(args.data_root);splits,flight_split=split_flights(flights,args.seed)
    splits,flight_split,rejected=add_extra_caches(splits,flight_split,args)
    manifest={'source':str(args.data_root.resolve()),'seed':args.seed,'flights':flight_split,
              'frames':{key:len(value) for key,value in splits.items()},
              'teacher':'DA3-LARGE-1.1 cached pose-conditioned depth',
              'rejected_extra_samples':rejected,
              'metric_ground_truth_used':False,'student_deployed':False,
              'caveat':'Sparse anchors and teacher depth share estimated camera poses; held-out LiDAR is required before deployment.'}
    path=out/'dataset_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Dataset split changed; use a fresh output')
    save_json(path,manifest)
    train=splits['train'][:args.max_train_samples] if args.max_train_samples else splits['train']
    loaders={name:DataLoader(TeacherCacheDataset(paths),batch_size=args.batch_size,
                             shuffle=name=='train',num_workers=args.workers,
                             pin_memory=True,persistent_workers=args.workers>0)
             for name,paths in (('train',train),('validation',splits['validation']),('test',splits['test']))}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=DepthStudent().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start_epoch=0;best=float('inf')
    latest=out/'latest.pt'
    if args.resume and latest.is_file():
        state=torch.load(latest,map_location=device,weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']);start_epoch=state['epoch'];best=state['best_anchor_absrel']
        if args.reset_best_on_resume:best=float('inf')
    elif latest.exists():raise ValueError('Checkpoint exists; use --resume or a new output')
    save_json(out/'model_info.json',{'parameters':sum(p.numel() for p in model.parameters()),
              'architecture':'posed RGB shared CNN + geometric cost volume + depth/confidence head',
              'device':str(device),'output_units':'sparse-anchor-normalized visual depth'})
    started=time.monotonic();stale=0
    completed_epoch=start_epoch
    for epoch in range(start_epoch+1,args.epochs+1):
        model.train();totals=[];nonfinite_batches=[]
        for step,batch in enumerate(loaders['train'],1):
            batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                depth,confidence=model(batch['rgb'],batch['K'],batch['P'])
                total,parts=losses(batch,depth.float(),confidence.float())
            if not torch.isfinite(total):
                nonfinite_batches.extend(batch['path'])
                print(f'nonfinite training loss: skipped {batch["path"]}',flush=True)
                continue
            scaler.scale(total).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            scaler.step(optimizer);scaler.update()
            totals.append(float(total.detach()))
            if step%50==0:print(f'epoch {epoch} step {step}/{len(loaders["train"])} loss {np.mean(totals[-50:]):.4f}',flush=True)
        if not totals:raise RuntimeError('All training batches had nonfinite loss; inspect camera/depth inputs')
        validation=evaluate(model,loaders['validation'],device,args.max_eval_samples)
        score=validation['anchor_absrel']
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'scaler':scaler.state_dict(),'epoch':epoch,'best_anchor_absrel':min(best,score),
               'validation':validation,'architecture':{'channels':48,'planes':24}}
        temp=out/'latest.tmp.pt';torch.save(state,temp);temp.replace(latest)
        if score<best:
            best=score;stale=0;temp=out/'best.tmp.pt';torch.save(state,temp);temp.replace(out/'best.pt')
        else:stale+=1
        save_json(out/'status.json',{'status':'training','epoch':epoch,'epochs':args.epochs,
            'elapsed_seconds':time.monotonic()-started,'train_loss':float(np.mean(totals)),
            'validation':validation,'best_anchor_absrel':best,
            'nonfinite_batches':nonfinite_batches,'student_deployed':False})
        print(json.dumps({'epoch':epoch,'train_loss':float(np.mean(totals)),
                          'validation':validation,'seconds':time.monotonic()-started}),flush=True)
        completed_epoch=epoch
        if stale>=args.patience:
            print(f'early stopping at epoch {epoch}: no validation anchor improvement for {stale} epochs',flush=True)
            break
    state=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model'])
    test=evaluate(model,loaders['test'],device,args.max_eval_samples)
    save_json(out/'status.json',{'status':'complete','epochs':completed_epoch,'elapsed_seconds':time.monotonic()-started,
              'best_validation_anchor_absrel':best,'held_out_test':test,
              'student_deployed':False,'independent_geometry_validated':False})
    print(json.dumps({'held_out_test':test,'best_checkpoint':str(out/'best.pt')}),flush=True)


if __name__=='__main__':main()
