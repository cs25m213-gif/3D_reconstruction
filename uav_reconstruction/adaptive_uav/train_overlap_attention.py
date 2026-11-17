"""Train image-to-history overlap from verified PnP tracks, split by flight."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .overlap_attention import OverlapAttention,make_bins,overlap_pairs


TRAIN=('session_20261004_175217','session_20261008_130458','session_20261014_170532')
VALIDATION=('session_20261006_230626',)
TEST=('session_20261014_100642',)


def scene_data(candidate_root,feature_root,name,history=12):
    rows=json.loads((candidate_root/name/'candidates.json').read_text())['frames']
    with np.load(feature_root/(name+'.npz')) as file:features=file['visual'].astype(np.float32)
    if len(features)!=len(rows):raise ValueError(f'Feature/frame mismatch: {name}')
    tracks=[set(row.get('pnp_inlier_point_ids',())) if row['pnp_verified'] else set() for row in rows]
    frames=np.asarray([row['frame_index'] for row in rows],np.int32)
    examples=[]
    for i in range(1,len(rows)):
        if len(tracks[i])<12:continue
        old=[j for j in range(max(0,i-history),i) if len(tracks[j])>=12]
        if not old:continue
        target=np.asarray([len(tracks[i]&tracks[j])/min(len(tracks[i]),len(tracks[j])) for j in old],np.float32)
        h=len(old)
        examples.append((features[i],np.pad(features[old],((history-h,0),(0,0))),
            np.pad((frames[i]-frames[old]).astype(np.float32),(history-h,0)),
            np.r_[np.zeros(history-h,bool),np.ones(h,bool)],np.pad(target,(history-h,0))))
    if not examples:raise ValueError(f'No verified overlap examples: {name}')
    current,previous,lag,valid,target=(np.stack(column) for column in zip(*examples))
    return {'name':name,'features':features,'frames':frames,'tracks':tracks,
        'current':current,'previous':previous,'lag':lag,'valid':valid,'target':target}


def tensors(row,device):
    return tuple(torch.from_numpy(row[key]).to(device) for key in ('current','previous','lag','valid','target'))


@torch.no_grad()
def evaluate(model,scenes,device):
    model.eval();outputs={}
    for row in scenes:
        current,history,lag,valid,target=tensors(row,device)
        pred=model(current,history,lag,valid)
        truth=target[valid].cpu().numpy();estimate=pred[valid].cpu().numpy()
        positive=truth>=.25;selected=estimate>=.25
        tp=int(np.sum(positive&selected));fp=int(np.sum(~positive&selected));fn=int(np.sum(positive&~selected))
        outputs[row['name']]={'pairs':len(truth),'mae':float(np.mean(np.abs(truth-estimate))),
            'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),
            'f1':2*tp/max(1,2*tp+fp+fn),'baseline_positive_fraction':float(positive.mean())}
    return outputs


def bins_report(model,row,device):
    scores=overlap_pairs(row['features'],model,device,frame_indices=row['frames'])
    groups,membership=make_bins(range(len(row['features'])),scores)
    actual=[];predicted=[]
    for (i,j),value in scores.items():
        a,b=row['tracks'][i],row['tracks'][j]
        if len(a)<12 or len(b)<12:continue
        actual.append(len(a&b)/min(len(a),len(b))>=.25)
        predicted.append(bool(set(membership[i])&set(membership[j])))
    actual=np.asarray(actual,bool);predicted=np.asarray(predicted,bool)
    tp=int((actual&predicted).sum());fp=int((~actual&predicted).sum());fn=int((actual&~predicted).sum())
    return {'bins':len(groups),'frames':len(row['features']),
        'multi_bin_frames':sum(len(x)>1 for x in membership.values()),
        'groups_with_at_least_two_views':sum(len(group.members)>=2 for group in groups),
        'grouped_overlap_precision':tp/max(1,tp+fp),
        'grouped_overlap_recall':tp/max(1,tp+fn),
        'groups':[{'anchor_frame':int(row['frames'][group.members[0]]),
            'member_frames':[int(row['frames'][i]) for i in group.members]} for group in groups]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate-root',type=Path,default=Path('runs/selector_candidates'))
    parser.add_argument('--feature-root',type=Path,default=Path('runs/cnn_gru_selector_candidates/features'))
    parser.add_argument('--output',type=Path,default=Path('runs/overlap_attention'))
    parser.add_argument('--epochs',type=int,default=35)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data={name:scene_data(args.candidate_root,args.feature_root,name) for name in TRAIN+VALIDATION+TEST}
    model=OverlapAttention().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=float('inf');best_epoch=0;stale=0
    for epoch in range(1,args.epochs+1):
        model.train();names=list(TRAIN);random.shuffle(names);losses=[]
        for name in names:
            row=data[name];order=np.random.permutation(len(row['current']))
            for start in range(0,len(order),32):
                indices=order[start:start+32]
                sub={key:row[key][indices] for key in ('current','previous','lag','valid','target')}
                current,history,lag,valid,target=tensors(sub,device)
                pred=model(current,history,lag,valid)
                loss=(F.smooth_l1_loss(pred[valid],target[valid])+
                    .1*F.binary_cross_entropy(pred[valid],(target[valid]>=.25).float()))
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
                losses.append(float(loss.detach()))
        validation=evaluate(model,[data[name] for name in VALIDATION],device)
        metric=float(np.mean([item['mae'] for item in validation.values()]))
        if metric<best-1e-5:
            best=metric;best_epoch=epoch;stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,
                'target':'verified shared PnP inlier tracks / min(track counts)',
                'feature':'frozen DINOv2-S image embedding'},args.output/'best.pt')
        else:stale+=1
        print(json.dumps({'epoch':epoch,'train_loss':float(np.mean(losses)),
            'validation':validation,'best_epoch':best_epoch}),flush=True)
        if stale>=7:break
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=device,weights_only=True)['model'])
    held_out=evaluate(model,[data[name] for name in TEST],device)
    bin_reports={name:bins_report(model,data[name],device) for name in VALIDATION+TEST}
    report={'status':'complete','best_epoch':best_epoch,'train_scenes':TRAIN,
        'validation_scenes':VALIDATION,'held_out_scenes':TEST,'validation':evaluate(model,[data[name] for name in VALIDATION],device),
        'held_out':held_out,'bin_reports':bin_reports,
        'limitations':'Labels are verified SfM-track overlap, not independent depth/mesh quality. Grouping uses image similarity; calibrated pose/parallax filtering is required before depth. No production deployment.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'best_epoch':best_epoch,'held_out':held_out,
        'bins':{name:{k:v for k,v in item.items() if k!='groups'} for name,item in bin_reports.items()}}),flush=True)


if __name__=='__main__':main()
