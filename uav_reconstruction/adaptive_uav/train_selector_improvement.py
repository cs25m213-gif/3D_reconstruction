"""Training-only oracle distillation and paired selected-context clip evaluation.

Run generate -> train -> evaluate. Every geometry score is real TSDF evaluation;
no surrogate reward or final-test labels are used. The original pipeline remains
available unchanged. See --help for explicit resource and experiment settings.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .counterfactual_replay import Clip, calibration_scales, eligible, fuse, role_of
from .embedding_cache import Descriptors
from .graph_selector import load_utility_checkpoint
from .reconstruction_quality import ReferencePopulation, cluster_interval, surface_metrics
from .selector_features import MapState
from .selector_improvement import (ARRAY_KEYS, PROTOCOL, ContextScorer, WindowUtilityPolicy,
    append_selected, candidate_positions, describe_options, oracle_index, overlap_eligibility, require_training_role)
from .selector_policy import UtilityPolicy, collate, rule, uniform, windows
from .sequential_replay import StudentDepthProvider
from .student_inference import load_student
from .utility_contracts import load_contract, sha256_file, sha256_lines, write_json


def atomic_torch(value,path):
    path=Path(path)
    tmp=path.with_suffix('.partial.pt')
    torch.save(value,tmp)
    tmp.replace(path)


def atomic_npz(arrays,path):
    path=Path(path)
    tmp=path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**arrays)
    tmp.replace(path)


def source_identity():
    return sha256_lines([p.name+':'+sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))])


def seed_clip(name,seed):
    value=int.from_bytes(hashlib.sha256(f'{name}:{seed}'.encode()).digest()[:4],'little') & 0x7fffffff
    cv2.setRNGSeed(value)


def common(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    contract=load_contract(args.config,('utility.quality_protocol','utility.scalarization'))
    root=args.root.resolve()
    every=sorted(p.name for p in (root/contract['data']['cache_root']).iterdir() if (p/'selected_infos.json').exists())
    scales=calibration_scales(root,contract,every)
    gate=json.loads((root/contract['geometry_gate']).read_text())
    student=load_student(args.student_checkpoint,'cpu')
    model,data=load_utility_checkpoint(args.checkpoint,'cpu')
    descriptors=Descriptors(args.embeddings)
    if data['descriptor']!=descriptors.name:
        raise ValueError('Frozen descriptor identity does not match checkpoint')
    return root,contract,every,scales,gate,student,model,data,descriptors


def prepared(root,contract,name,scales,gate):
    if name.endswith(contract['geometry']['calibration_batch_suffix']):
        raise ValueError('Calibration clips cannot become training/evaluation examples')
    clip=Clip(root,contract,name)
    status=eligible(gate,clip.scene,False)
    if status is None or clip.scene not in scales or not len(clip.points) or not clip.views:
        raise ValueError(f'Clip {name} has no qualified reference/scale/views')
    clip.apply_scale(scales[clip.scene])
    clip.choose_keyframes(contract['utility']['keyframe_baseline_ratio'])
    reference=ReferencePopulation(clip.points,contract['utility']['tolerance_m'],status,contract['utility']['quality_protocol'])
    return clip,reference


def generate_clip(clip,scorer,model,data,descriptors,args):
    arrays,groups={},[]
    blocks=windows(clip,args.keyframes)
    for history_kind in args.histories:
        state,history=MapState(),[]
        for step,window in enumerate(blocks):
            positions=candidate_positions(window,args.candidates)
            rows=describe_options(state,clip,positions,descriptors,args.keyframes-step)
            feasible,unsupported=overlap_eligibility(rows,bool(history))
            # Only the reward teacher sees future images/reference geometry.
            suffix=[clip.views[int(w[len(w)//2])] for w in blocks[step+1:]]
            selected_prefix=[v for v,_ in history]
            results=[scorer.score(selected_prefix+[clip.views[p]]+suffix) for p in positions]
            scores=np.asarray([r['quality'] for r in results],np.float32)
            best=oracle_index(scores,feasible)
            if history_kind=='oracle':
                pick=best
            elif history_kind=='uniform':
                allowed=np.flatnonzero(feasible)
                pick=int(allowed[np.argmin(np.abs(np.asarray(positions)[allowed]-np.median(window)))])
            elif history_kind=='learner':
                with torch.no_grad():
                    pred=model(*collate([{**r,'question':data['decision_head']} for r in rows],'cpu'))['utility'][:,0,3].numpy()
                pick=oracle_index(pred,feasible)
            else:
                raise ValueError('Unknown history policy')
            prefix=f'g{len(groups):04}_'
            for key in ARRAY_KEYS:
                arrays[prefix+key]=np.stack([r[key] for r in rows]).astype(np.float32)
            arrays[prefix+'quality']=scores
            arrays[prefix+'eligible']=feasible
            groups.append({'prefix':prefix,'clip':clip.name,'flight':clip.scene,'split':'train',
                'history_policy':history_kind,'window':step,'history':[v.index for v,_ in history],
                'candidate_indices':[clip.views[p].index for p in positions],
                'candidate_metrics':results,'oracle_pick':best,'history_pick':pick,
                'unsupported_overlap':unsupported,'feasible_score_spread':float(np.ptp(scores[feasible])),
                'scope':'Training-only full-set oracle; fixed reference poses, selected-past student contexts; uniform frozen suffix'})
            append_selected(state,history,clip.views[positions[pick]],scorer.provider,descriptors,clip.name)
            print(json.dumps({'stage':'generate','clip':clip.name,'history':history_kind,'window':step,
                              'fusions':scorer.fusions,'depth_calls':scorer.provider.calls}),flush=True)
    return arrays,groups


def generate(args):
    root,contract,every,scales,gate,student,model,data,descriptors=common(args)
    names=args.clips or [n for n in every if role_of(n,contract)=='train' and not n.endswith(contract['geometry']['calibration_batch_suffix'])]
    for name in names:
        require_training_role(role_of(name,contract))
    args.output.mkdir(parents=True,exist_ok=True)
    identity=sha256_lines([sha256_file(args.config),sha256_file(args.student_checkpoint),sha256_file(args.checkpoint),
                           str(args.keyframes),str(args.candidates),','.join(args.histories),PROTOCOL,
                           source_identity()])
    for name in names:
        cache=root/contract['data']['cache_root']/name
        inputs=[cache/'selected_infos.json',*(cache/'reconstruction/multiview/depth_cache').glob('*.npz')]
        for field in ('sidecars','overlay'):
            if contract['data'].get(field):
                inputs.extend((root/contract['data'][field]/name).glob('*.npz'))
        inputs.extend([root/contract['geometry_gate'],root/contract['data']['scale_calibration'],args.embeddings/f'{name}.npz'])
        input_hash=sha256_lines([str(p)+':'+sha256_file(p) for p in sorted(set(inputs))])
        clip_identity=sha256_lines([identity,input_hash])
        folder=args.output/name;folder.mkdir(exist_ok=True)
        meta=folder/'groups.json'
        if meta.exists():
            old=json.loads(meta.read_text())
            if old['identity']!=clip_identity or old['arrays_sha256']!=sha256_file(folder/'features.npz'):
                raise ValueError(f'Incompatible existing labels: {folder}')
            continue
        clip,reference=prepared(root,contract,name,scales,gate)
        seed_clip(name,args.seed)
        provider=StudentDepthProvider(student,'cpu',128)
        scorer=ContextScorer(provider,reference,contract['utility']['voxel_m'])
        arrays,groups=generate_clip(clip,scorer,model,data,descriptors,args)
        by_candidate=collections.defaultdict(dict)
        for group in groups:
            for candidate,result in zip(group['candidate_indices'],group['candidate_metrics']):
                by_candidate[candidate][tuple(group['history'])]=result['quality']
        varied=[values for values in by_candidate.values() if len(values)>1]
        atomic_npz(arrays,folder/'features.npz')
        write_json(meta,{'identity':clip_identity,'input_sha256':input_hash,'arrays_sha256':sha256_file(folder/'features.npz'),
                         'groups':groups,'fusions':scorer.fusions,'depth_calls':provider.calls,
                         'same_candidate_multiple_histories':len(varied),
                         'median_whole_selection_score_spread_across_histories':float(np.median([np.ptp(list(v.values())) for v in varied])) if varied else None,
                         'reference_identity':reference.identity(),'causal_online_evidence':False,
                         'selection_protocol_id':PROTOCOL})
        descriptors.clips.pop(name,None)
    write_json(args.output/f'shard_{args.shard}.json',{'complete':True,'clips':names,'identity':identity})


def load_groups(labels):
    rows=[];hashes=[]
    for file in sorted(labels.glob('*/groups.json')):
        meta=json.loads(file.read_text());require_training_role('train' if all(g['split']=='train' for g in meta['groups']) else None)
        arrays=file.parent/'features.npz'
        if sha256_file(arrays)!=meta['arrays_sha256']:
            raise ValueError(f'Changed labels {arrays}')
        hashes.extend([sha256_file(file),meta['arrays_sha256']])
        with np.load(arrays) as values:
            for group in meta['groups']:
                prefix=group['prefix']
                rows.append({**group,**{key:values[prefix+key].copy() for key in (*ARRAY_KEYS,'quality','eligible')}})
    if not rows:
        raise ValueError('No complete training groups')
    return rows,sha256_lines(hashes)


def oracle_loss(model,row,head,scale):
    rows=[{key:row[key][i] for key in ARRAY_KEYS} for i in range(len(row['quality']))]
    output=model(*collate([{**r,'question':head} for r in rows],'cpu'))
    scores=output['utility'][:,0,3]
    mask=torch.as_tensor(row['eligible'],dtype=torch.bool)
    truth=torch.as_tensor(row['quality'],dtype=torch.float32)[mask]
    target=(truth-truth.mean())/scale
    pred=scores[mask]
    regression=F.smooth_l1_loss(pred,target)
    soft=F.softmax(target.detach(),dim=0)
    imitation=-(soft*F.log_softmax(pred,dim=0)).sum() if len(pred)>1 else pred.sum()*0
    # Pair ordering uses the measured gap and ignores uninformative ties.
    gap=truth[:,None]-truth[None,:]
    ordered=gap>.002
    ranks=F.softplus(-(pred[:,None]-pred[None,:]))[ordered].mean() if ordered.any() else pred.sum()*0
    return imitation+ranks+.5*regression


def train(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    groups,labels_hash=load_groups(args.labels)
    args.output.mkdir(parents=True,exist_ok=True)
    model,data=load_utility_checkpoint(args.checkpoint,'cpu')
    model.train()
    allowed=('query','token','question','memory','gru','utility')
    for name,p in model.named_parameters():
        p.requires_grad_(name.split('.')[0] in allowed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=1e-4)
    targets=np.concatenate([g['quality'][g['eligible']]-g['quality'][g['eligible']].mean() for g in groups])
    scale=max(float(targets.std()),.005)
    flights=collections.Counter(g['flight'] for g in groups)
    weights={f:len(groups)/(len(flights)*n) for f,n in flights.items()}
    settings={'schema_version':1,'selection_protocol_id':PROTOCOL,'labels_sha256':labels_hash,
              'initial_checkpoint_sha256':sha256_file(args.checkpoint),'epochs':args.epochs,'seed':args.seed,
              'learning_rate':args.learning_rate,'weight_decay':1e-4,'gradient_clip':1.,'target_scale':scale,
              'training_groups':len(groups),'clips':len({g['clip'] for g in groups}),'flight_groups':dict(flights),
              'flight_loss_weights':weights,'minibatch':'one complete candidate-ranking group',
              'stopping':'fixed epochs; best training objective, no validation oracle or reward tuning',
              'source_sha256':sha256_file(Path(__file__)),'policy_source_sha256':sha256_file(Path(__file__).with_name('selector_improvement.py')),
              'all_source_sha256':source_identity(),'torch_version':torch.__version__,'numpy_version':np.__version__,
              'initial_feature_convention':'same 384-D DINO and 17/11 state/token features as memory_utility_v1'}
    identity=sha256_lines([json.dumps(settings,sort_keys=True)])
    latest=args.output/'latest.pt'
    start,steps,skipped,best,reached=0,0,0,float('inf'),set()
    if latest.exists():
        if not args.resume:
            raise ValueError('Existing run requires --resume or a fresh output')
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['training_identity']!=identity:
            raise ValueError('Changed labels/config/source; resume rejected')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        start,steps,skipped,best=saved['next_epoch'],saved['optimizer_steps'],saved['skipped_updates'],saved['best_training_loss']
        reached=set(saved['gradient_reached'])
    elif any((args.output/n).exists() for n in ('best.pt','config.json')):
        raise ValueError('Incomplete existing output; preserve and choose a new run')
    write_json(args.output/'config.json',settings)
    started=time.monotonic()
    with (args.output/'training.jsonl').open('a') as log:
        for epoch in range(start,args.epochs):
            order=list(range(len(groups)));random.shuffle(order)
            losses=[]
            for i in order:
                row=groups[i]
                loss=oracle_loss(model,row,data['decision_head'],scale)*weights[row['flight']]
                optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    skipped+=1;log.write(json.dumps({'epoch':epoch,'group':i,'skipped':'nonfinite_loss'})+'\n');continue
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):
                    skipped+=1;log.write(json.dumps({'epoch':epoch,'group':i,'skipped':'nonfinite_gradient'})+'\n');continue
                for name,p in model.named_parameters():
                    if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0:
                        reached.add(name.split('.')[0])
                optimizer.step();steps+=1;losses.append(float(loss.detach()))
            if not losses:
                raise ValueError('No finite optimizer updates')
            mean=float(np.mean(losses));improved=mean<best;best=min(mean,best)
            saved={**data,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':None,'scheduler':None,
                   'selection_protocol_id':PROTOCOL,'training_identity':identity,'config':settings,
                   'labels_sha256':labels_hash,'next_epoch':epoch+1,'epoch':epoch,'optimizer_steps':steps,'skipped_updates':skipped,
                   'best_training_loss':best,'selection_criterion':'minimum training oracle objective; held-out quality unproven',
                   'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                   'gradient_reached':sorted(reached),'source_head_trained':False,'trained_output':'overall full-clip oracle advantage only',
                   'target_scale':scale,'target_mean':data['target_mean'],'target_std':data['target_std'],
                   'elapsed_this_invocation_seconds':time.monotonic()-started,'diagnostic_labels':False,'full_video_validated':False}
            atomic_torch(saved,latest)
            if improved:
                atomic_torch(saved,args.output/'best.pt')
            row={'epoch':epoch+1,'loss':mean,'optimizer_steps':steps,'skipped_updates':skipped,'best_training_loss':best}
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    write_json(args.output/'report.json',{'complete':True,'epochs':args.epochs,'optimizer_steps':steps,'skipped_updates':skipped,
        'gradient_reached':sorted(reached),'labels_sha256':labels_hash,'promote':False,'heldout_evaluation':'pending',
        'checkpoint_sha256':sha256_file(args.output/'best.pt'),'scope':'Training-only oracle distillation, selected-context fixed-pose clips'})


def selection_diagnostics(chosen):
    times=np.asarray([v.timestamp for v in chosen]);centres=np.asarray([v.feature_pose[:3,3] for v in chosen])
    return {'max_gap_seconds':float(np.diff(times).max()) if len(times)>1 else 0.,
            'mean_gap_seconds':float(np.diff(times).mean()) if len(times)>1 else 0.,
            'baseline_median_visual':float(np.median(np.linalg.norm(np.diff(centres,axis=0),axis=1))) if len(centres)>1 else 0.,
            'frame_indices':[v.index for v in chosen]}


def evaluate(args):
    root,contract,every,scales,gate,student,old_model,old_data,descriptors=common(args)
    new_model,new_data=load_utility_checkpoint(args.improved_checkpoint,'cpu')
    if new_data.get('selection_protocol_id')!=PROTOCOL:
        raise ValueError('Wrong improved checkpoint protocol')
    names=args.clips or [n for n in every if role_of(n,contract)=='validation' and not n.endswith(contract['geometry']['calibration_batch_suffix'])]
    if any(role_of(n,contract)!='validation' for n in names):
        raise ValueError('Development evaluation accepts validation clips only')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in names:
        clip,reference=prepared(root,contract,name,scales,gate)
        provider=StudentDepthProvider(student,'cpu',128)
        scorer=ContextScorer(provider,reference,contract['utility']['voxel_m'])
        # Historical selector receives its historical observation stream. Its final
        # geometry is also re-evaluated with selected contexts, like every rival.
        clip.use_student(student,'cpu')
        old_policy=UtilityPolicy(old_model,old_data,'cpu')
        original={v.frame_id:v for v in clip.views}
        frozen_provider=lambda v,p,h:(original[v.frame_id],0.,True)
        grid_policy=WindowUtilityPolicy(old_model,old_data,'cpu',frozen_provider,guard=False,maximum=args.candidates,allow_legacy=True)
        guarded=WindowUtilityPolicy(old_model,old_data,'cpu',frozen_provider,guard=True,maximum=args.candidates,allow_legacy=True)
        contextual=WindowUtilityPolicy(old_model,old_data,'cpu',provider,guard=True,maximum=args.candidates,allow_legacy=True)
        improved=WindowUtilityPolicy(new_model,new_data,'cpu',provider,guard=True,maximum=args.candidates)
        policies={'rule':rule,'uniform':uniform,'legacy_selector':old_policy,'full_window':grid_policy,
                  'overlap_guard':guarded,'selected_context':contextual,'oracle_distilled':improved}
        cliprows=[]
        for label,policy in policies.items():
            seed_clip(name,args.seed)
            start=time.monotonic();chosen=policy(clip,args.keyframes,descriptors)
            diagnostics=selection_diagnostics(chosen)
            selected=scorer.score(chosen)
            # A paired frozen-context score exposes whether changed contexts help/hurt.
            frozen_views=[original[v.frame_id] for v in chosen]
            frozen=surface_metrics(fuse([(v,v.reference_pose) for v in frozen_views],contract['utility']['voxel_m']),reference)
            from .reconstruction_quality import quality,DEFAULT_WEIGHTS
            cliprows.append({'clip':name,'flight':clip.scene,'policy':label,**selected,**diagnostics,
                'frozen_context_quality':quality(frozen,None,DEFAULT_WEIGHTS),'frozen_context_f_score':frozen['f_score'],
                'policy_and_score_seconds':time.monotonic()-start,'trace':getattr(policy,'trace',[]),
                'context_protocol':'selected-past student; frozen-context control separately scored',
                'fixed_reference_poses':True,'causal_online_evidence':False})
            print(json.dumps({'clip':name,'policy':label,'quality':selected['quality'],'f_score':selected['f_score']}),flush=True)
        write_json(args.output/f'{name}.json',cliprows)
        rows.extend(cliprows);descriptors.clips.pop(name,None)
    lookup={(r['clip'],r['policy']):r for r in rows}
    paired={}
    for label in ['uniform','legacy_selector','full_window','overlap_guard','selected_context','oracle_distilled']:
        gains=collections.defaultdict(list)
        for name in names:
            base=lookup[(name,'rule')];trial=lookup[(name,label)]
            if base['keyframes']!=trial['keyframes']:
                raise ValueError('Unmatched mapping-frame budget')
            gains[base['flight']].append(trial['quality']-base['quality'])
        paired[label+'_vs_rule']=cluster_interval(gains)
    write_json(args.output/'paired_metrics.json',paired)
    write_json(args.output/'report.json',{'complete':True,'clips':len(names),'paired':paired,'promote':False,
        'full_video_validated':False,'causal_online_evidence':False,'checkpoint_sha256':sha256_file(args.improved_checkpoint),
        'scope':'Offline buffered-window selected-context geometry; full-clip SfM poses/anchors; not fresh camera-to-mesh evaluation'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['generate','train','evaluate'])
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--student-checkpoint',type=Path,required=True)
    p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--clips',nargs='+')
    p.add_argument('--labels',type=Path)
    p.add_argument('--improved-checkpoint',type=Path)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=194)
    p.add_argument('--keyframes',type=int,default=12)
    p.add_argument('--candidates',type=int,default=5)
    p.add_argument('--histories',nargs='+',default=['uniform','oracle','learner'],choices=['uniform','oracle','learner'])
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.threads<1 or args.keyframes<1 or args.candidates<2 or args.epochs<1:
        p.error('Positive threads/keyframes/epochs and at least 2 candidates required')
    if args.stage=='train' and args.labels is None:
        p.error('train requires --labels')
    if args.stage=='evaluate' and args.improved_checkpoint is None:
        p.error('evaluate requires --improved-checkpoint')
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        write_json(args.output/f'{args.stage}_status_{args.shard}.json',{'state':'running','pid':os.getpid(),'stage':args.stage,'started_utc':time.time()})
        {'generate':generate,'train':train,'evaluate':evaluate}[args.stage](args)
        write_json(args.output/f'{args.stage}_status_{args.shard}.json',{'state':'complete','stage':args.stage,'finished_utc':time.time()})
    except Exception as exc:
        write_json(args.output/f'{args.stage}_status_{args.shard}.json',{'state':'failed','stage':args.stage,'error':repr(exc)})
        raise


if __name__=='__main__':
    main()
