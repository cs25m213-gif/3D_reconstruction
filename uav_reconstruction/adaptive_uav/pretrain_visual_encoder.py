"""Bounded RGB-only contrastive pretraining of a lightweight CNN."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms as T


def encoder():
    model=models.mobilenet_v3_small(weights=None)
    model.classifier=nn.Identity()
    return model


class RGBPairs(Dataset):
    def __init__(self,root,files):
        self.root=Path(root);self.files=files
        self.augment=T.Compose([T.RandomResizedCrop(224,scale=(.65,1.)),T.RandomHorizontalFlip(),
            T.ColorJitter(.2,.2,.2,.05),T.RandomGrayscale(.1),T.ToTensor(),
            T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        with np.load(self.root/self.files[i]) as data:rgb=data['arr_0'].copy()
        image=Image.fromarray(rgb)
        return self.augment(image),self.augment(image)


def contrastive(a,b,projector,backbone):
    z=projector(backbone(torch.cat((a,b),0)))
    z=F.normalize(z.float(),dim=-1)
    n=len(a);logits=z@z.T/.12
    logits.fill_diagonal_(-1e4)
    labels=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(logits,labels)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('datasets/dronescapes_rgb_selector'))
    p.add_argument('--output',type=Path,default=Path('runs/dronescapes_mobilenet_pretrain'))
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=24)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    groups={}
    for row in manifest.get('files',manifest.get('pairs',[])):
        groups.setdefault(row['scene'],[]).append(row['rgb'])
    complete={k:v for k,v in groups.items() if len(v)>=64}
    if len(complete)<3:raise ValueError('At least three complete RGB scenes required')
    names=sorted(complete);validation=names[-1];training=names[:-1]
    split={'train':training,'validation':validation,'files_train':sum(len(complete[n]) for n in training),
           'files_validation':len(complete[validation]),'source':'Meehai/dronescapes RGB only'}
    (out/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    loaders={role:DataLoader(RGBPairs(args.dataset,files),batch_size=args.batch_size,
        shuffle=role=='train',num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True)
        for role,files in [('train',[f for n in training for f in complete[n]]),
                           ('validation',complete[validation])]}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone=encoder().to(device)
    projector=nn.Sequential(nn.Linear(576,256),nn.ReLU(),nn.Linear(256,128)).to(device)
    optimizer=torch.optim.AdamW(list(backbone.parameters())+list(projector.parameters()),lr=3e-4,weight_decay=1e-4)
    best=float('inf');stale=0;history=[]
    for epoch in range(1,args.epochs+1):
        results={}
        for role,loader in loaders.items():
            backbone.train(role=='train');projector.train(role=='train');losses=[]
            for a,b in loader:
                a=a.to(device,non_blocking=True);b=b.to(device,non_blocking=True)
                with torch.set_grad_enabled(role=='train'),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    loss=contrastive(a,b,projector,backbone)
                if role=='train':
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    nn.utils.clip_grad_norm_(list(backbone.parameters())+list(projector.parameters()),2.)
                    optimizer.step()
                losses.append(float(loss.detach()))
            results[role]=float(np.mean(losses))
        torch.save({'encoder':backbone.state_dict(),'projector':projector.state_dict(),
            'optimizer':optimizer.state_dict(),'epoch':epoch,'validation_loss':results['validation']},out/'latest.pt')
        if results['validation']<best:
            best=results['validation'];stale=0
            torch.save({'encoder':backbone.state_dict(),'epoch':epoch,'validation_loss':best},out/'best.pt')
        else:stale+=1
        history.append({'epoch':epoch,**results});print(json.dumps(history[-1]),flush=True)
        if stale>=3:break
    (out/'report.json').write_text(json.dumps({'status':'complete','split':split,'history':history,
        'best_validation_loss':best,'note':'RGB contrastive pretraining supplies no frame-selection labels or calibrated geometry.'},indent=2)+'\n')


if __name__=='__main__':main()
