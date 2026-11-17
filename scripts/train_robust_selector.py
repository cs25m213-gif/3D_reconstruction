"""Isolated robustness pilot: corrupt RGB -> new features/depth -> set-level reward.

Offline fixed-pose evidence only. A final bounded fresh-video check is separate.
No current dataset, checkpoint, or runtime policy is overwritten or promoted.
"""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch
from uav_reconstruction.adaptive_uav.counterfactual_replay import Clip,fuse,role_of
from uav_reconstruction.adaptive_uav.graph_selector import frozen_encoder,image_feature,load_utility_checkpoint
from uav_reconstruction.adaptive_uav.robust_selector_policy import QualityAdapter,RobustPolicy
from uav_reconstruction.adaptive_uav.selector_corruptions import CASES,corrupt,frame_fault,frame_seed,quality_vector,usable_rgb
from uav_reconstruction.adaptive_uav.selector_policy import UtilityPolicy,uniform,rule
from uav_reconstruction.adaptive_uav.student_inference import load_student,predict,anchor_scale,GRID
from uav_reconstruction.adaptive_uav.reconstruction_quality import ReferencePopulation,surface_metrics,quality,DEFAULT_WEIGHTS,depth_metrics,cluster_interval
from uav_reconstruction.adaptive_uav.utility_contracts import write_json,sha256_file
from uav_reconstruction.adaptive_uav.remote_geometry import geometry_metrics_batch,routing_provenance
EVAL_CASES = CASES + ('frozen',)

class FreshDescriptors:
    def __init__(self,clip,encoder,device):
        self.owner=clip.name;self.values={v.index:image_feature(encoder,v.rgb,device) for v in clip.views}
    def get(self,name,index,rgb):
        if name!=self.owner:raise ValueError('Descriptor belongs to another clip')
        return self.values[index]

def load_source(args,name):
    clip=Clip(ROOT,args.contract,name)
    infos=json.loads((ROOT/args.contract['data']['cache_root']/name/'selected_infos.json').read_text())
    first=float(infos[0]['OriginalImageName'].rsplit('.',1)[0])
    for view in clip.views:view.timestamp=float(infos[view.index]['OriginalImageName'].rsplit('.',1)[0])-first
    clip.raw_frame_count=len(infos);clip.reference_tolerance_m=args.contract['utility']['tolerance_m']
    return clip

def candidates(contract,cohort,role):
    gate=json.loads((ROOT/contract['geometry_gate']).read_text())['flights'];found={}
    motion_path=ROOT/'runs/motion_audit_20261104/pose_audit.json'
    motion={r['clip']:r['max_rotation_rate_deg_s'] for r in json.loads(motion_path.read_text())['clips']} if motion_path.exists() else {}
    for name in sorted(cohort[role],key=lambda n:motion.get(n,0),reverse=True):
        folder=ROOT/contract['data']['cache_root']/name;infos=json.loads((folder/'selected_infos.json').read_text());scene=infos[0]['OriginalSceneName']
        if scene in found or name.endswith('_0'):continue
        if role_of(scene,contract)!=role or gate.get(scene,{}).get('status') not in ('QUALIFIED','VERIFIED'):continue
        if len(list((folder/'reconstruction/multiview/depth_cache').glob('frame_*.jpg.npz')))<24:continue
        found[scene]=name
        if len(found)==3:break
    if len(found)<3:raise ValueError(f'Need three qualified {role} flights')
    return list(found.values())

def depth_view(view,history,student,device):
    out=copy.copy(view)
    called=False
    try:
        scale=anchor_scale(view.anchors_m);called=True
        depth,prob=predict(student,[(v.rgb,v.feature_k,v.feature_pose) for v in [view]+history[-2:][::-1]],scale,device,anchors=view.anchors_m)
        valid=usable_rgb(view.rgb)
        if not valid:depth=np.zeros_like(depth);prob=np.zeros_like(prob)
        error=None if valid else 'Unobservable RGB'
    except ValueError as failure:
        depth=np.zeros(GRID,np.float32);prob=np.zeros(GRID,np.float32);valid=False;error=str(failure)
    out.fusion_depth=depth;out.fusion_k=view.feature_k.copy();out.fusion_k[0]*=depth.shape[1]/view.rgb.shape[1];out.fusion_k[1]*=depth.shape[0]/view.rgb.shape[0]
    out.fusion_rgb=cv2.resize(view.rgb,depth.shape[::-1],interpolation=cv2.INTER_AREA)
    out.depth_m=cv2.resize(depth,view.rgb.shape[1::-1],interpolation=cv2.INTER_NEAREST);out.confidence=float(np.median(prob));out.depth_failure=error;out.depth_inference_called=called
    return out

def prepare(source,case,severity,seed,student,encoder,device,output):
    clip=copy.copy(source);clip.views=[];clip.variant=case;clip.reward_cache={};history=[];manifest=[];frozen_rgb=None
    folder=output/'derived'/f'{source.name}__{case}';folder.mkdir(parents=True,exist_ok=False)
    for position,original in enumerate(source.views):
        if case=='frame_gap' and position%3:continue
        view=copy.copy(original)
        if case=='mixed':
            rng=np.random.default_rng(frame_seed(seed,source.name,view.index));chosen_case=str(rng.choice(CASES[1:8])) if rng.random()<.6 else 'clean';level=float(rng.uniform(.2,severity))
        else:chosen_case,level=frame_fault(case,seed,source.name,view.index,getattr(source,"raw_frame_count",max(v.index for v in source.views)+1),severity)
        if case=='mixed' and source.raw_frame_count//2<=view.index<source.raw_frame_count//2+3:chosen_case='frozen'
        if chosen_case=='frozen':
            if frozen_rgb is None:frozen_rgb=source.views[max(0,position-1)].rgb.copy()
            view.rgb=frozen_rgb.copy();visible=np.zeros(view.rgb.shape[:2],bool)
        else:
            frozen_rgb=None
            view.rgb,visible=corrupt(original.rgb,chosen_case,level,frame_seed(seed,source.name,view.index))
        view.anchors_m=None if original.anchors_m is None else np.where(cv2.resize(visible.astype(np.uint8),original.anchors_m.shape[::-1],interpolation=cv2.INTER_NEAREST)>0,original.anchors_m,0)
        view.corruption_visible=visible
        view=depth_view(view,history,student,device);clip.views.append(view);history.append(view)
        np.savez_compressed(folder/f'frame_{view.index:06d}.npz',depth=view.fusion_depth,K=view.fusion_k,RGB=view.fusion_rgb,pose=view.feature_pose)
        manifest.append({'index':view.index,'case':chosen_case,'severity':level,'rgb_sha256':hashlib.sha256(view.rgb.tobytes()).hexdigest(),'depth_failure':view.depth_failure})
    clip.descriptors=FreshDescriptors(clip,encoder,device)
    write_json(folder/'manifest.json',{'source':source.name,'scenario':case,'seed':seed,'frames':manifest,'candidate_depth_inference_calls':sum(v.depth_inference_called for v in clip.views),'descriptor_inference_calls':len(clip.views),'reference_identity':ReferencePopulation(source.points,getattr(source,'reference_tolerance_m',.1),'QUALIFIED').identity(),'scope':'Fixed clean SfM pose/anchor support except occluded/frozen anchors; full-pipeline tracking not represented','descriptor_recomputed':True,'student_depth_recomputed':True})
    return clip

def reward(clip,chosen,student,device,contract):
    key=tuple(v.index for v in chosen)
    hit=key in clip.reward_cache
    if key not in clip.reward_cache:
        depth_views=[]
        for view in chosen:depth_views.append(depth_view(view,depth_views,student,device))
        reference=ReferencePopulation(clip.points,contract['utility']['tolerance_m'],'QUALIFIED','selector_robustness_v1')
        metrics=surface_metrics(fuse([(v,v.reference_pose) for v in depth_views],contract['utility']['voxel_m']),reference)
        clip.reward_cache[key]={'quality':quality(metrics,None,DEFAULT_WEIGHTS),'metrics':metrics,'selected':list(key),'prediction_failures':sum(v.depth_failure is not None for v in depth_views),'selected_context_depth_calls':sum(v.depth_inference_called for v in depth_views)}
    return {**clip.reward_cache[key],'reward_cache_hit':hit,'current_depth_inference_calls':0 if hit else clip.reward_cache[key]['selected_context_depth_calls']}

def reward_batch(clip,choices,student,device,contract):
    """Infer each unique selection on GPU, then score independent sets on remote CPUs."""
    pending={}
    for chosen in choices.values():
        key=tuple(v.index for v in chosen)
        if key in clip.reward_cache or key in pending:continue
        depth_views=[]
        for view in chosen:depth_views.append(depth_view(view,depth_views,student,device))
        pending[key]=depth_views
    if pending:
        reference=ReferencePopulation(clip.points,contract['utility']['tolerance_m'],'QUALIFIED','selector_robustness_v1')
        voxel=contract['utility']['voxel_m']
        def local():return [surface_metrics(fuse([(v,v.reference_pose) for v in group],voxel),reference) for group in pending.values()]
        measured=geometry_metrics_batch(list(pending.values()),reference,voxel,local)
        for (key,group),metrics in zip(pending.items(),measured):
            clip.reward_cache[key]={'quality':quality(metrics,None,DEFAULT_WEIGHTS),'metrics':metrics,'selected':list(key),
                'prediction_failures':sum(v.depth_failure is not None for v in group),
                'selected_context_depth_calls':sum(v.depth_inference_called for v in group)}
    result={};charged=set()
    for name,chosen in choices.items():
        key=tuple(v.index for v in chosen);hit=key not in pending or key in charged;charged.add(key)
        result[name]={**clip.reward_cache[key],'reward_cache_hit':hit,
                      'current_depth_inference_calls':0 if hit else clip.reward_cache[key]['selected_context_depth_calls']}
    return result

def paired_depth_metrics(student, teacher, measured, mask):
    """Identical measured pixels, with missing predictions counted separately and penalized."""
    population=mask & np.isfinite(measured) & (measured>0)
    valid_student=np.isfinite(student)&(student>0)
    valid_teacher=np.isfinite(teacher)&(teacher>0)
    shared=population&valid_student&valid_teacher
    def summarize(prediction,valid):
        count=int(population.sum());missing=int((population&~valid).sum())
        # A missing prediction is zero depth: 100% relative error, never silently dropped.
        complete=np.where(valid,prediction,0.)
        error=np.abs(complete[population]-measured[population])/measured[population]
        return {'common_valid_metrics':depth_metrics(prediction,measured,shared),
                'reference_pixels':count,'invalid_prediction_pixels':missing,
                'valid_fraction':float((count-missing)/count) if count else None,
                'full_population_absrel':float(error.mean()) if count else None}
    return {'student':summarize(student,valid_student),'teacher':summarize(teacher,valid_teacher),
            'shared_valid_pixels':int(shared.sum()),'missing_prediction_rule':'zero depth / unit relative error'}

def teacher_comparison(clip,chosen,student,teacher,args):
    """Same corrupted RGB, past context and sparse-anchor scaling for both depths."""
    folder=args.output/'teacher_inputs'/f'{clip.name}__{clip.variant}';folder.mkdir(parents=True,exist_ok=True);results=[]
    for position,view in enumerate(chosen):
        if position==0:results.append({'index':view.index,'status':'not_comparable','reason':'No past translated teacher context'});continue
        group=[view]+chosen[max(0,position-2):position][::-1];records={};poses={}
        for v in group:
            path=folder/f'{v.index:06d}.png'
            if not path.exists() and not cv2.imwrite(str(path),cv2.cvtColor(v.rgb,cv2.COLOR_RGB2BGR)):raise OSError('Cannot save teacher RGB')
            records[v.frame_id]={'path':str(path),'K':v.feature_k,'timestamp':v.timestamp};poses[v.frame_id]=v.feature_pose
        try:
            anchor_scale(view.anchors_m)
            predicted=teacher.infer(records,poses,view.frame_id)
            from uav_reconstruction.adaptive_uav.build_cached_lidar_targets import downsample_sparse_min
            anchors=downsample_sparse_min(view.anchors_m,predicted.depth.shape);valid=(anchors>0)&np.isfinite(predicted.depth)&(predicted.depth>0)
            if valid.sum()<30:raise ValueError('Insufficient teacher scaling anchors')
            teacher_depth=predicted.depth*float(np.median(anchors[valid]/predicted.depth[valid]));student_view=depth_view(view,group[1:][::-1],student,args.device)
            target=ROOT/args.contract['data']['sidecars']/clip.name/f'frame_{view.index:06d}.npz'
            with np.load(target) as data:
                measured=data['measured_depth_z_m'];mask=(measured>0)&np.asarray(data.get('reliable',np.ones_like(measured,bool)),dtype=bool)
                mask &= cv2.resize(view.corruption_visible.astype(np.uint8),measured.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
            teacher_depth=cv2.resize(teacher_depth,measured.shape[::-1],interpolation=cv2.INTER_NEAREST);student_depth=cv2.resize(student_view.fusion_depth,measured.shape[::-1],interpolation=cv2.INTER_NEAREST)
            results.append({'index':view.index,'status':'complete',**paired_depth_metrics(student_depth,teacher_depth,measured,mask),'student_failure':student_view.depth_failure,'teacher_rgb':'regenerated corrupted RGB','teacher_is_ground_truth':False})
        except (ValueError,OSError) as error:results.append({'index':view.index,'status':'failed','error':str(error)})
    return results

def make_corrupt_video(source,case,severity,seed,destination):
    """Fresh-check input retains native image size, intrinsics and original frame identities."""
    import subprocess
    destination.mkdir(parents=True,exist_ok=False);(destination/'session').mkdir()
    original=ROOT/source/'session/video.mp4';cap=cv2.VideoCapture(str(original));infos=json.loads((ROOT/source/'selected_infos.json').read_text());width,height=int(cap.get(3)),int(cap.get(4))
    if abs(cap.get(cv2.CAP_PROP_FPS)-2.)>.01:raise ValueError('Fresh check requires prepared 2 Hz video')
    command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r','2','-i','-','-c:v','libx264','-threads','2','-crf','17','-pix_fmt','yuv420p',str(destination/'session/video.mp4')]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for index in range(len(infos)):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Cannot decode fresh-check source')
            chosen,level=frame_fault(case,seed,Path(source).name,index,len(infos),severity);rgb,_=corrupt(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),chosen,level,frame_seed(seed,Path(source).name,index));process.stdin.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR).tobytes())
    finally:cap.release();process.stdin.close();code=process.wait()
    if code:raise RuntimeError('Corrupted-video encoding failed')
    write_json(destination/'selected_infos.json',infos)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','cohort','student-checkpoint','selector-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--rollouts',type=int,default=120);p.add_argument('--frames',type=int,default=12);p.add_argument('--fresh-checks',type=int,default=1);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--cost-weight',type=float,default=.01)
    args=p.parse_args();args.output=args.output.resolve();args.contract=json.loads(args.config.read_text());cohort=json.loads(args.cohort.read_text())['selector_clips'];names={role:candidates(args.contract,cohort,role) for role in ('train','validation')}
    if args.frames<9 or args.frames>20 or args.rollouts<1 or not 0<=args.fresh_checks<=3:p.error('Bounded pilot requires 9–20 frames, positive rollouts, 0–3 fresh checks')
    if not np.isfinite(args.cost_weight) or args.cost_weight<0:p.error('Cost weight must be finite and nonnegative')
    if set(n.rsplit('_',1)[0] for n in names['train'])&set(n.rsplit('_',1)[0] for n in names['validation']):raise ValueError('Flight leakage')
    for path in (args.student_checkpoint,args.selector_checkpoint):
        if not path.is_file():raise FileNotFoundError(path)
    if args.dry_run:print(json.dumps({'flights':names,'cases':EVAL_CASES,'rollouts':args.rollouts,'fresh_checks':args.fresh_checks,'promote':False}));return
    if args.device=='cuda':
        import subprocess
        from uav_reconstruction.adaptive_uav.reconstruct_training_clips import admission_reason
        smoke=ROOT/'runs/selector_robustness_20261104'
        while True:
            reason=admission_reason(os.getpid(),18000,10000)
            if reason is None:break
            write_json(smoke/'resource_status.json',{'state':'waiting','reason':reason});time.sleep(15)
        write_json(smoke/'resource_status.json',{'state':'admitted','minimum_free_vram_mib':18000,'minimum_available_ram_mib':10000})
        if not (smoke/'gpu_smoke.json').is_file():
            with (smoke/'gpu_smoke.log').open('a') as log:
                subprocess.run([sys.executable,str(smoke/'gpu_smoke.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    args.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(194);np.random.seed(194);cv2.setNumThreads(2)
    write_json(args.output/'protocol.json',{'names':names,'training_variants':['clean','mixed'],'evaluation_cases':EVAL_CASES,'training_seed':194,'validation_seed':9137,'training_max_severity':.65,'validation_severity':.9,'student_sha256':sha256_file(args.student_checkpoint),'selector_sha256':sha256_file(args.selector_checkpoint),'frames_budget':args.frames,'scope':'Offline fixed clean SfM geometry/anchors; augmented RGB descriptors and selected-context depth recomputed; occluded/frozen RGB loses anchor support; all inference cost counted','promotion':False})
    device=torch.device(args.device);student=load_student(args.student_checkpoint,device);encoder=frozen_encoder(device);model,data=load_utility_checkpoint(args.selector_checkpoint,device)
    if data.get('diagnostic_labels',True) or data.get('descriptor')!='dinov2_vits14':raise ValueError('Need qualified DINO selector')
    model.eval();adapter=QualityAdapter().to(device);policy=RobustPolicy(model,data,device,adapter);original=UtilityPolicy(model,data,device)
    before=copy.deepcopy(model.state_dict());empty_adapter=copy.deepcopy(adapter.state_dict());train=[];baselines={}
    for name in names['train']:
        raw=load_source(args,name)
        if not len(raw.points):raise ValueError('Empty measured training reference')
        for case in ('clean','mixed'):
            clip=prepare(raw,case,.65,194,student,encoder,device,args.output);train.append(clip)
            selected=uniform(clip,args.frames);baselines[id(clip)]=reward(clip,selected,student,device,args.contract)['quality']
    for name,parameter in model.named_parameters():parameter.requires_grad_(name.split('.')[0] in ('gru','utility'))
    model.gru.train();adapter.train();optimizer=torch.optim.AdamW([p for p in list(model.parameters())+list(adapter.parameters()) if p.requires_grad],lr=1e-5);steps=0;skipped=0
    with (args.output/'training.jsonl').open('x') as log:
        for step in range(args.rollouts):
            clip=train[step%len(train)];selected,probabilities,entropies=policy.run(clip,args.frames,clip.descriptors,1.);result=reward(clip,selected,student,device,args.contract);advantage=result['quality']-baselines[id(clip)]
            if probabilities:
                loss=-(advantage*torch.stack(probabilities).sum()+.01*torch.stack(entropies).sum());optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(adapter.parameters()),1.)
                    if torch.isfinite(norm):optimizer.step();steps+=1
                    else:skipped+=1
                else:skipped+=1
            log.write(json.dumps({'step':step,'clip':clip.name,'variant':clip.variant,'gain_vs_uniform':advantage,**result})+'\n');log.flush()
    after=copy.deepcopy(model.state_dict());trained_adapter=copy.deepcopy(adapter.state_dict());del train
    checkpoint={**data,'architecture_id':'robust_memory_utility_v1','model':after,'robustness_adapter':trained_adapter,'robustness_adapter_id':'quality_residual_v1','requires_wrapper':'RobustPolicy','promote':False};torch.save(checkpoint,args.output/'candidate.pt')
    from uav_reconstruction.adaptive_uav.depth_backend import DA3Backend
    teacher=DA3Backend('depth-anything/DA3-LARGE-1.1',device=args.device,resolution=504);evaluations=[];teacher_checks=[];fresh_jobs=[]
    for name in names['validation']:
        raw=load_source(args,name)
        if not len(raw.points):raise ValueError('Empty validation reference')
        for case in EVAL_CASES:
            clip=prepare(raw,case,.9,9137,student,encoder,device,args.output)
            choices={}
            with torch.no_grad():
                model.load_state_dict(before);model.eval();adapter.load_state_dict(empty_adapter);adapter.eval()
                choices['original']=original.run(clip,args.frames,clip.descriptors)[0];choices['before']=policy.run(clip,args.frames,clip.descriptors)[0]
                choices['uniform']=uniform(clip,args.frames);choices['rule']=rule(clip,args.frames)
                model.load_state_dict(after);adapter.load_state_dict(trained_adapter);choices['after']=policy.run(clip,args.frames,clip.descriptors)[0];trace=copy.deepcopy(policy.last_trace)
            rewards=reward_batch(clip,choices,student,device,args.contract)
            for variant,selected in choices.items():
                row={'clip':name,'flight':clip.scene,'case':case,'policy':variant,'frame_budget':args.frames,**rewards[variant]};evaluations.append(row)
                with (args.output/'evaluation.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
            if case in ('clean','motion_blur'):teacher_checks.append({'clip':name,'case':case,'frames':teacher_comparison(clip,choices['after'],student,teacher,args)})
            if case=='motion_blur' and len(fresh_jobs)<args.fresh_checks:fresh_jobs.append((clip,{k:choices[k] for k in ('original','after')},trace))
    teacher.close();summaries={}
    for case in EVAL_CASES:
        differences={}
        for row in evaluations:
            if row['case']==case and row['policy']=='after':
                base=next(x for x in evaluations if x['clip']==row['clip'] and x['case']==case and x['policy']=='original');differences.setdefault(row['flight'],[]).append(row['quality']-base['quality'])
        summaries[case]=cluster_interval(differences)
    write_json(args.output/'report.json',{'optimizer_steps':steps,'skipped_updates':skipped,'evaluations':evaluations,'paired_teacher_checks':teacher_checks,'quality_gain_by_case':summaries,'geometry_execution':routing_provenance(),'promote':False,'scope':'Bounded synthetic-corruption pilot, fixed SfM/anchor support; real rolling-shutter/dynamics/failed-tracking robustness unproven'})
    # Small fresh-video check without supplying reference trajectories to reconstruction.
    from finetune_video_mesh import execute
    mesh_args=copy.copy(args);mesh_args.contract=copy.deepcopy(args.contract);fresh_root=args.output/'fresh_sources';mesh_args.contract['data']['cache_root']=str(fresh_root);checks=[]
    for clip,choices,trace in fresh_jobs:
        source=Path(args.contract['data']['cache_root'])/clip.name;make_corrupt_video(source,'motion_blur',.9,9137,fresh_root/clip.name)
        for variant,selected in choices.items():
            result=execute(clip,selected,'student',mesh_args,f'{clip.name}_{variant}') if len(selected)>=9 else {'status':'failed','reward':-10.,'error':'Fewer than nine selected frames'}
            checks.append({'clip':clip.name,'variant':variant,**result})
        selected=choices['after'];result=execute(clip,selected,'da3',mesh_args,f'{clip.name}_teacher') if len(selected)>=9 else {'status':'failed','reward':-10.,'error':'Fewer than nine selected frames'};checks.append({'clip':clip.name,'variant':'teacher_same_selection',**result})
    write_json(args.output/'fresh_checks.json',{'checks':checks,'automatic_promotion':False})
    write_json(args.output/'status.json',{'state':'complete','optimizer_steps':steps,'promote':False})
if __name__=='__main__':main()
