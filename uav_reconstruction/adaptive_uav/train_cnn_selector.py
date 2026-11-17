"""Fine-tune a pretrained CNN and GRU on verified candidate-graph proxies."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from .graph_selector import CNNGraphSelector


class Windows(Dataset):
    def __init__(self,root,features,scenes,length=12):
        self.rows=[];self.length=length
        for scene in scenes:
            records=json.loads((root/scene/'candidates.json').read_text())['frames']
            with np.load(features/(scene+'.npz')) as data:
                state=data['state'].copy();label=data['label'].copy();utility=data['utility'].copy()
            if len(records)!=len(label):raise ValueError(f'Candidate/label count changed: {scene}')
            for start in range(0,len(records)-length+1,length):
                self.rows.append(([r['image'] for r in records[start:start+length]],
                    state[start:start+length],label[start:start+length],utility[start:start+length]))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        paths,state,label,utility=self.rows[i];images=[]
        for path in paths:
            bgr=cv2.imread(path)
            if bgr is None:raise FileNotFoundError(path)
            rgb=cv2.cvtColor(cv2.resize(bgr,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            images.append((rgb.astype(np.float32)/255.-[.485,.456,.406])/[.229,.224,.225])
        return (torch.from_numpy(np.stack(images).transpose(0,3,1,2).astype(np.float32)),
                torch.from_numpy(state),torch.from_numpy(label),torch.from_numpy(utility))


def metrics(model,loader,device,weights):
    model.eval();confusion=np.zeros((4,4),np.int64);bridge_hits=bridge_total=0;losses=[]
    with torch.no_grad():
        for rgb,state,label,utility in loader:
            rgb=rgb.to(device);state=state.to(device);label=label.to(device);utility=utility.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits,estimated,_,_=model(rgb,state)
            valid=label>=0
            losses.append(float(F.cross_entropy(logits.float().flatten(0,1),label.flatten(),weight=weights)+
                .3*F.smooth_l1_loss(estimated[valid],utility[valid])))
            pred=logits.argmax(-1)
            for true,guess in zip(label.flatten().cpu().tolist(),pred.flatten().cpu().tolist()):
                if true>=0:confusion[true,guess]+=1
            bridge=(utility[:,:,2]>.5)&valid
            bridge_hits+=int(((pred==0)&bridge).sum());bridge_total+=int(bridge.sum())
    f1=[]
    for i in range(3):
        tp=confusion[i,i];f1.append(float(2*tp/max(1,2*tp+confusion[:,i].sum()-tp+confusion[i].sum()-tp)))
    return {'loss':float(np.mean(losses)),'macro_f1':float(np.mean(f1)),
        'accuracy':float(np.trace(confusion)/max(1,confusion.sum())),
        'bridge_recall':bridge_hits/max(1,bridge_total),'bridge_count':bridge_total,
        'confusion':confusion.tolist(),'samples':int(confusion.sum())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate-root',type=Path,default=Path('runs/selector_candidates'))
    p.add_argument('--label-run',type=Path,default=Path('runs/cnn_gru_selector_candidates'))
    p.add_argument('--pretrained',type=Path,default=Path('runs/dronescapes_mobilenet_pretrain/best.pt'))
    p.add_argument('--output',type=Path,default=Path('runs/cnn_gru_selector_mobilenet'))
    p.add_argument('--epochs',type=int,default=16)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);np.random.seed(42)
    split=json.loads((args.label_run/'split.json').read_text())['flights']
    features=args.label_run/'features'
    loaders={key:DataLoader(Windows(args.candidate_root,features,scenes),batch_size=2,
        shuffle=key=='train',num_workers=4,pin_memory=True,persistent_workers=True)
        for key,scenes in split.items()}
    train_labels=[]
    for scene in split['train']:
        with np.load(features/(scene+'.npz')) as data:
            train_labels.extend(data['label'][data['label']>=0].tolist())
    counts=np.bincount(train_labels,minlength=4);weights=np.ones(4,np.float32)
    weights[:3]=sum(counts[:3])/(3*np.maximum(counts[:3],1))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights=torch.from_numpy(weights).to(device)
    model=CNNGraphSelector().to(device)
    state=torch.load(args.pretrained,map_location='cpu',weights_only=False)
    model.encoder.load_state_dict(state['encoder'])
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},
        {'params':list(model.visual_projector.parameters())+list(model.policy.parameters()),'lr':2e-4}],weight_decay=1e-4)
    best=float('inf');stale=0;start=time.monotonic();history=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for rgb,state,label,utility in loaders['train']:
            rgb=rgb.to(device);state=state.to(device);label=label.to(device);utility=utility.to(device)
            valid=label>=0
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits,estimated,uncertainty,_=model(rgb,state)
                ce=F.cross_entropy(logits.flatten(0,1),label.flatten(),weight=weights,reduction='none')
                loss=(ce[valid.flatten()].mean()+.3*F.smooth_l1_loss(estimated[valid],utility[valid])+
                    .05*(uncertainty.flatten()[valid.flatten()]-ce.detach()[valid.flatten()]).abs().mean())
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();losses.append(float(loss.detach()))
        validation=metrics(model,loaders['validation'],device,weights)
        current={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':validation}
        history.append(current);print(json.dumps(current),flush=True)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'validation':validation},out/'latest.pt')
        if validation['loss']<best:
            best=validation['loss'];stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation':validation},out/'best.pt')
        else:stale+=1
        if stale>=4:break
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=False)['model'])
    test=metrics(model,loaders['test'],device,weights)
    report={'status':'complete','split':split,'best_validation':torch.load(out/'best.pt',map_location='cpu',weights_only=False)['validation'],
        'test':test,'history':history,'elapsed_seconds':time.monotonic()-start,
        'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else None,
        'note':'Actions are geometric proxy labels; no synchronized IMU or verified relocalization class.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'test':test,'elapsed_seconds':report['elapsed_seconds'],'peak_gpu_gb':report['peak_gpu_gb']}),flush=True)


if __name__=='__main__':main()
