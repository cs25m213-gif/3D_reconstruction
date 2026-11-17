"""Train and evaluate an offline selector on verified camera-subset reward proxies.

The output is experimental. The live selector uses different state features and
must not load this checkpoint without a separate online validation.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .graph_selector import GraphSelector


TRAIN=('session_20261004_175217','session_20261008_130458','session_20261014_170532')
VALIDATION=('session_20261006_230626',)
TEST=('session_20261014_100642',)


def load_sequences(root,names):
    sequences=[]
    for name in names:
        with np.load(root/(name+'.npz')) as z:
            for stride in sorted(np.unique(z['stride'])):
                indices=np.flatnonzero(z['stride']==stride)
                if len(indices)<8:continue
                geometry=np.log1p(z['delta_geometry'][indices].astype(np.float32))/np.log1p(1000.)
                connectivity=np.log1p(z['utility'][indices,2])/np.log1p(20.)
                # Fixed one-frame cost cancels in equal-budget selection.
                target=.8*geometry+.2*connectivity
                state=np.nan_to_num(z['state'][indices].copy(),nan=0.,posinf=3.,neginf=0.)
                state[:,1]=np.minimum(state[:,1],3.)
                sequences.append({'name':name,'stride':int(stride),'visual':z['visual'][indices].copy(),
                    'state':state,'target':target,'bridge':z['bridge'][indices].copy(),
                    'frame_index':z['frame_index'][indices].copy(),
                    'delta_geometry':z['delta_geometry'][indices].copy()})
    return sequences


def tensors(row,device):
    return (torch.from_numpy(row['visual'][None]).to(device),
            torch.from_numpy(row['state'][None]).to(device),
            torch.from_numpy(row['target']).to(device))


def score(model,row,device):
    model.eval()
    with torch.no_grad():
        visual,state,_=tensors(row,device)
        return model(visual,state)[1][0,:,0].cpu().numpy()


def topk(values,k):
    return np.sort(np.argsort(-values,kind='stable')[:k])


def evaluate(model,rows,device):
    records=[]
    for row in rows:
        n=len(row['target']);k=max(1,int(np.ceil(n*.25)))
        predictions={'learned':score(model,row,device),
                     'uniform':np.zeros(n),
                     'heuristic':1.-row['state'][:,3]+.25*row['state'][:,2]+.15*row['state'][:,1],
                     'oracle':row['target']}
        indices={name:(np.unique(np.linspace(0,n-1,k,dtype=int)) if name=='uniform' else topk(values,k))
                 for name,values in predictions.items()}
        record={'scene':row['name'],'stride':row['stride'],'candidates':n,'budget':k,'policies':{}}
        for name,selected in indices.items():
            record['policies'][name]={'proxy_reward_sum':float(row['target'][selected].sum()),
                'bridge_recall':float(row['bridge'][selected].sum()/max(1,row['bridge'].sum())),
                'selected_frame_index':row['frame_index'][selected].tolist()}
        records.append(record)
    summary={}
    for policy in ('learned','uniform','heuristic','oracle'):
        rewards=[r['policies'][policy]['proxy_reward_sum'] for r in records]
        bridges=[r['policies'][policy]['bridge_recall'] for r in records if r['policies']['oracle']['bridge_recall']>0]
        summary[policy]={'mean_proxy_reward':float(np.mean(rewards)),
                         'mean_bridge_recall':float(np.mean(bridges)) if bridges else None}
    return {'summary':summary,'sequences':records}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('runs/reconstruction_reward'))
    parser.add_argument('--output',type=Path,default=Path('runs/reward_selector'))
    parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=load_sequences(args.data,TRAIN);validation=load_sequences(args.data,VALIDATION)
    test=load_sequences(args.data,TEST)
    model=GraphSelector().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=-float('inf');best_epoch=0;stale=0
    for epoch in range(1,args.epochs+1):
        model.train();random.shuffle(train);losses=[]
        for row in train:
            visual,state,target=tensors(row,device)
            predicted=model(visual,state)[1][0,:,0]
            regression=F.smooth_l1_loss(predicted,target)
            # Compare candidates within nearby temporal windows, where a
            # budgeted selector must distinguish useful and redundant views.
            n=len(target);a=torch.randint(0,n,(min(2048,n*16),),device=device)
            b=torch.clamp(a+torch.randint(-8,9,a.shape,device=device),0,n-1)
            difference=target[a]-target[b];valid=difference.abs()>.04
            ranking=F.softplus(-4*torch.sign(difference[valid])*(predicted[a[valid]]-
                predicted[b[valid]])).mean() if valid.any() else regression*0
            loss=regression+.15*ranking
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
            losses.append(float(loss.detach()))
        result=evaluate(model,validation,device)
        metric=result['summary']['learned']['mean_proxy_reward']
        if metric>best+1e-5:
            best=metric;best_epoch=epoch;stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,
                'training_target':'0.8 log1p(delta supported points)/log1p(1000) + 0.2 log1p(delta connected pairs)/log1p(20)',
                'state_schema':'reconstruction_reward.py; offline PnP context only'},args.output/'best.pt')
        else:stale+=1
        print(json.dumps({'epoch':epoch,'train_loss':float(np.mean(losses)),
            'validation_proxy_reward':metric,'best_epoch':best_epoch}),flush=True)
        if stale>=8:break
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=device,weights_only=True)['model'])
    report={'status':'complete','best_epoch':best_epoch,'train_scenes':TRAIN,
        'validation_scenes':VALIDATION,'test_scenes':TEST,
        'budget_fraction':.25,'validation':evaluate(model,validation,device),
        'held_out':evaluate(model,test,device),
        'limitations':'Proxy reward reuses existing SfM landmarks; no fresh COLMAP, depth or mesh measurements. Only five mapped local sessions available. Checkpoint is not deployed.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'best_epoch':best_epoch,'held_out':report['held_out']['summary']}),flush=True)


if __name__=='__main__':main()
