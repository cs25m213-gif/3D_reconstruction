"""Train a bounded selector from cached camera-track graph consequences."""
import argparse
import json
from pathlib import Path
import random
import time
import cv2
import networkx as nx
import numpy as np
import pycolmap
import torch
from torch.nn import functional as F
from .graph_selector import GraphSelector,frozen_encoder,image_feature


def flight_rows(run,encoder,device,max_frames):
    ws=run/'reconstruction/dense'
    model=pycolmap.Reconstruction(ws/'sparse')
    images=sorted((i for i in model.images.values() if i.has_pose),key=lambda i:i.name)
    if len(images)<12:return None
    if max_frames and len(images)>max_frames:
        images=[images[i] for i in np.unique(np.linspace(0,len(images)-1,max_frames,dtype=int))]
    tracks=[];poses=[];rgbs=[];features=[];sharpness=[]
    for image in images:
        path=ws/'images'/image.name
        bgr=cv2.imread(str(path))
        if bgr is None:raise FileNotFoundError(path)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        rgbs.append(rgb)
        features.append(image_feature(encoder,rgb,device))
        sharpness.append(float(cv2.Laplacian(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()))
        tracks.append({p.point3D_id for p in image.points2D if p.has_point3D()})
        poses.append(image.cam_from_world().inverse().matrix())
    # Build several realistic budgeted camera graphs from actual shared tracks.
    # Each edge is verified; only the candidate set changes under subsampling.
    bridges=set();graph=nx.Graph();graph.add_nodes_from(range(len(images)))
    for stride in (1,3,4):
        for offset in range(stride):
            indices=list(range(offset,len(images),stride))
            if len(indices)<4:continue
            adjacent=[len(tracks[a]&tracks[b]) for a,b in zip(indices,indices[1:])]
            threshold=max(30,int(.6*np.median(adjacent)))
            sub=nx.Graph();sub.add_nodes_from(indices)
            for pos,i in enumerate(indices):
                for j in indices[:pos]:
                    shared=len(tracks[i]&tracks[j])
                    if shared>=threshold:sub.add_edge(i,j,weight=shared)
            if stride==1:graph=sub
            bridges.update(nx.articulation_points(sub))
    frequency={}
    for track in tracks:
        for point in track:frequency[point]=frequency.get(point,0)+1
    states=[];labels=[];utilities=[];contributions=[];overlaps=[]
    for i,image in enumerate(images):
        translation=0.;rotation=0.;overlap=0.
        if i:
            a,b=poses[i-1],poses[i]
            translation=float(np.linalg.norm(a[:3,3]-b[:3,3]))
            rotation=float(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))
            overlap=len(tracks[i]&tracks[i-1])/max(1,len(tracks[i]))
        contribution=sum(frequency[p]<=2 for p in tracks[i])/max(1,len(tracks[i]))
        connectivity=float(i in bridges)
        utility=np.array([overlap,contribution,connectivity],np.float32)
        # Labels are generated from graph removal and point survival, not from
        # the original frame selector's keep/skip record.
        contributions.append(contribution);overlaps.append(overlap)
        states.append([min(i*.25,3.),np.log1p(max(translation,0.)),
            min(rotation,1.5)/1.5,min(overlap,1.),min(sharpness[i],300.)/300.,
            0.,min(i,64)/64.,1.])
        utilities.append(utility)
    low,high=np.percentile(contributions,[30,70])
    for i,contribution in enumerate(contributions):
        if i in bridges or contribution>=high:label=0
        elif contribution<=low and overlaps[i]>=np.median(overlaps):label=2
        else:label=1
        labels.append(label)
    return {'visual':np.asarray(features,np.float32),'state':np.asarray(states,np.float32),
            'label':np.asarray(labels,np.int64),'utility':np.asarray(utilities,np.float32),
            'bridge_count':len(bridges),'connected_components':nx.number_connected_components(graph),
            'names':[i.name for i in images]}


def candidate_rows(path,encoder,device):
    """Graph counterfactuals over previously selected and skipped video frames."""
    rows=json.loads((path/'candidates.json').read_text())['frames']
    tracks=[set(row.get('pnp_inlier_point_ids',[])) for row in rows]
    verified=[i for i,row in enumerate(rows) if row['pnp_verified'] and len(tracks[i])>=12]
    if len(verified)<20:raise ValueError(f'Too few verified camera candidates: {path}')
    bridges=set()
    for stride in (1,2,3,4):
        for offset in range(stride):
            indices=verified[offset::stride]
            if len(indices)<5:continue
            adjacent=[len(tracks[a]&tracks[b]) for a,b in zip(indices,indices[1:])]
            threshold=max(12,int(.6*np.median(adjacent)))
            graph=nx.Graph();graph.add_nodes_from(indices)
            for position,i in enumerate(indices):
                for j in indices[:position]:
                    common=len(tracks[i]&tracks[j])
                    if common>=threshold:graph.add_edge(i,j,weight=common)
            bridges.update(nx.articulation_points(graph))
    frequency={}
    for i in verified:
        for point in tracks[i]:frequency[point]=frequency.get(point,0)+1
    contributions=np.array([sum(frequency.get(p,0)<=2 for p in tracks[i])/max(1,len(tracks[i]))
        for i in range(len(rows))],np.float32)
    low,high=np.percentile(contributions[verified],[30,70])
    overlap=np.array([max(row['reference_matches'].values(),default=0)/max(1,row['matched_3d_points'])
        for row in rows],np.float32)
    labels=[];utilities=[];states=[];visual=[];previous_pose=None
    for i,row in enumerate(rows):
        pose=np.array(row['world_to_camera']) if row['pnp_verified'] else None
        translation=0.;rotation=0.
        if pose is not None and previous_pose is not None:
            ca=-pose[:3,:3].T@pose[:3,3];cb=-previous_pose[:3,:3].T@previous_pose[:3,3]
            translation=float(np.linalg.norm(ca-cb))
            relative=previous_pose[:3,:3].T@pose[:3,:3]
            rotation=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
        if pose is not None:previous_pose=pose
        states.append([min(i*.5,3.),np.log1p(max(translation,0.)),min(rotation,1.5)/1.5,
            min(row['pnp_inliers'],200)/200.,min(row['sharpness'],300)/300.,0.,min(i,64)/64.,1.])
        utilities.append([overlap[i],contributions[i],float(i in bridges)])
        if i not in verified:label=-100  # Unknown, not fabricated SKIP/RELOCALIZE.
        elif i in bridges or contributions[i]>=high:label=0
        elif contributions[i]<=low and overlap[i]>=np.median(overlap[verified]):label=2
        else:label=1
        labels.append(label)
        bgr=cv2.imread(row['image'])
        if bgr is None:raise FileNotFoundError(row['image'])
        visual.append(image_feature(encoder,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),device))
    return {'visual':np.asarray(visual,np.float32),'state':np.asarray(states,np.float32),
            'label':np.asarray(labels,np.int64),'utility':np.asarray(utilities,np.float32),
            'bridge_count':len(bridges)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=Path('runs/uavscenes_30_batch'))
    p.add_argument('--candidate-root',type=Path,
                   help='Use verified original-video candidates, including frames absent from the saved map')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--max-frames-per-flight',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder=frozen_encoder(device)
    runs={}
    if args.candidate_root:
        runs={p.name:p for p in sorted(args.candidate_root.iterdir()) if (p/'candidates.json').exists()}
    else:
        for path in sorted(args.data_root.iterdir()):
            if (path/'reconstruction/dense/sparse/images.bin').exists():
                flight=path.name.rsplit('_',1)[0]
                if flight not in runs:runs[flight]=path
    names=sorted(runs);random.Random(42).shuffle(names)
    split=({'test':names[:1],'validation':names[1:2],'train':names[2:]}
        if args.candidate_root else {'test':names[:2],'validation':names[2:4],'train':names[4:]})
    cache=out/'features';cache.mkdir(exist_ok=True)
    data={}
    for flight in names:
        path=cache/(flight+'.npz')
        if path.exists():
            with np.load(path) as z:data[flight]={k:z[k] for k in ('visual','state','label','utility')}
        else:
            row=(candidate_rows(runs[flight],encoder,device) if args.candidate_root else
                 flight_rows(runs[flight],encoder,device,args.max_frames_per_flight))
            if row is None:continue
            np.savez_compressed(path,**{k:row[k] for k in ('visual','state','label','utility')})
            data[flight]={k:row[k] for k in ('visual','state','label','utility')}
            print(f'features {flight}: {len(row["label"])} frames, {row["bridge_count"]} bridge vertices',flush=True)
    del encoder
    if device.type=='cuda':torch.cuda.empty_cache()
    (out/'split.json').write_text(json.dumps({'flights':split,'frames':{s:sum(len(data[n]['label']) for n in group if n in data)
        for s,group in split.items()},
        'actions':{s:np.bincount(np.concatenate([data[n]['label'][data[n]['label']>=0] for n in group if n in data]),minlength=4).tolist()
            for s,group in split.items()},
        'verified_bridge_frames':{s:int(sum((data[n]['utility'][:,2]>.5).sum() for n in group if n in data))
            for s,group in split.items()},
        'supervision':'verified camera-track graph point survival and budgeted-subgraph articulation proxies',
        'imu':'unavailable in these caches; visual-only training'},indent=2)+'\n')
    model=GraphSelector().to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    start=0;best=float('inf');best_result=None
    if args.resume and (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        start=saved['epoch'];best=saved['best_validation']
    class_counts=np.bincount(np.concatenate([data[n]['label'][data[n]['label']>=0] for n in split['train'] if n in data]),minlength=4)
    weights=np.ones(4,np.float32)
    weights[:3]=np.sum(class_counts[:3])/(3*np.maximum(class_counts[:3],1))
    weights=torch.from_numpy(weights).to(device)
    def evaluate(group):
        model.eval();losses=[];correct=0;total=0;bridge_recall=[];confusion=np.zeros((4,4),np.int64)
        with torch.no_grad():
            for name in group:
                if name not in data:continue
                row=data[name]
                v=torch.from_numpy(row['visual'])[None].to(device)
                s=torch.from_numpy(row['state'])[None].to(device)
                label=torch.from_numpy(row['label'])[None].to(device)
                target=torch.from_numpy(row['utility'])[None].to(device)
                logits,utility,_,_=model(v,s)
                valid=label>=0
                losses.append(float(F.cross_entropy(logits.flatten(0,1),label.flatten(),weight=weights)+
                    .3*F.smooth_l1_loss(utility[valid],target[valid])))
                pred=logits.argmax(-1)
                correct+=int(((pred==label)&valid).sum());total+=int(valid.sum())
                for true,predicted in zip(label.flatten().cpu().tolist(),pred.flatten().cpu().tolist()):
                    if true>=0:confusion[true,predicted]+=1
                bridges=target[0,:,2]>.5
                if bridges.any():bridge_recall.append(float((pred[0,bridges]==0).float().mean()))
        f1=[]
        for i in range(3):
            tp=confusion[i,i];fp=confusion[:,i].sum()-tp;fn=confusion[i].sum()-tp
            f1.append(float(2*tp/max(1,2*tp+fp+fn)))
        return {'loss':float(np.mean(losses)),'accuracy':correct/max(1,total),
                'macro_f1':float(np.mean(f1)), 'confusion':confusion.tolist(),
                'bridge_recall':float(np.mean(bridge_recall)) if bridge_recall else None,'frames':total}
    stale=0
    for epoch in range(start+1,args.epochs+1):
        model.train();order=list(split['train']);random.shuffle(order)
        for name in order:
            if name not in data:continue
            row=data[name]
            v=torch.from_numpy(row['visual'])[None].to(device)
            s=torch.from_numpy(row['state'])[None].to(device)
            label=torch.from_numpy(row['label'])[None].to(device)
            target=torch.from_numpy(row['utility'])[None].to(device)
            opt.zero_grad(set_to_none=True)
            logits,utility,uncertainty,_=model(v,s)
            ce=F.cross_entropy(logits.flatten(0,1),label.flatten(),weight=weights,reduction='none')
            valid=label>=0
            loss=(ce[valid.flatten()].mean()+.3*F.smooth_l1_loss(utility[valid],target[valid])+
                .05*((uncertainty.flatten()[valid.flatten()]-ce.detach()[valid.flatten()]).abs()).mean())
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        val=evaluate(split['validation'])
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best_validation':min(best,val['loss'])}
        torch.save(state,out/'latest.pt')
        if val['loss']<best:
            best=val['loss'];best_result=val;stale=0;torch.save(state,out/'best.pt')
        else:stale+=1
        print(json.dumps({'epoch':epoch,'validation':val}),flush=True)
        if stale>=3:
            print(f'early stopping at epoch {epoch}',flush=True)
            break
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=False)['model'])
    test=evaluate(split['test'])
    (out/'report.json').write_text(json.dumps({'status':'complete','validation':best_result or val,'test':test,
        'best_validation_loss':best,'checkpoint':str(out/'best.pt'),
        'warning':'Graph proxy labels are not independent reconstruction outcomes; no real relocalization examples or synchronized IMU in this cache.'},indent=2)+'\n')
    print(json.dumps({'test':test,'checkpoint':str(out/'best.pt')}),flush=True)


if __name__=='__main__':main()
