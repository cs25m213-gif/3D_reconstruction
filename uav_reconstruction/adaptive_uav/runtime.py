"""Incremental relative-scale reconstruction with optional checked stereo.

The default CLI remains the legacy pipeline. This runner is explicitly selected
with --mode preview/balanced/quality and records provisional/partial states.
"""
import argparse,hashlib,json,os,queue,shutil,subprocess,sys,threading,time
from collections import deque
from pathlib import Path
import cv2,numpy as np
from .runtime_metrics import Metrics,save_json
from .runtime_types import FramePacket,PoseEstimate
from .frame_stream import video_frames
from .keyframe_policy import KeyframePolicy
from .submaps import SubmapStore,load_depth


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def read_record(record):
    rgb=cv2.cvtColor(cv2.imread(record['path']),cv2.COLOR_BGR2RGB)
    return FramePacket(record['frame_id'],record['index'],record['timestamp'],rgb,np.asarray(record['K']))


class StreamRunner:
    def __init__(self,args,config):
        self.args=args;self.config=config;self.root=Path(__file__).resolve().parents[2]
        self.out=args.output.resolve();self.out.mkdir(parents=True,exist_ok=True)
        if (self.out/'pipeline_manifest.json').exists() and not args.resume:
            raise ValueError('Output already used; select --resume or a fresh output directory')
        self.previous_report=json.loads((self.out/'run_report.json').read_text()) if args.resume and (self.out/'run_report.json').exists() else None
        attempt=self.out/'attempts'/str(time.time_ns()) if args.resume else self.out
        self.metrics=Metrics(attempt);self.gpu_lock=threading.RLock();self.state_lock=threading.RLock()
        self.stop=threading.Event();self.producer_done=threading.Event();self.errors=[]
        self.queue=queue.Queue(maxsize=config['mapping_queue']);self.records={};self.poses={};self.pose_version=0
        self.eval_records={};self.eval_poses={};self.eval_bins=set()
        self.tracking_rows=[];self.decisions=[];self.selected=[];self.map_frames=0;self.observations={};self.reused=0
        self.store=SubmapStore(self.out/'submaps',config['voxel_fraction'],config['max_active_submaps'],config['submap_frames'])
        self.backend=None;self.depth=None;self.semantic=None;self.last_depth=None;self.fallbacks=[]
        self.latest_rgb=None;self.latest_pose=None;self.phase_offload=False
        self.report={'status':'starting','execution_status':'running','source':str(args.session.resolve()),
            'mode':args.mode,'geometry_accuracy_validated':False,'scale_status':'arbitrary_visual_units',
            'units':'visual units','result_ready':False,'runtime_scope':'fresh camera tracking, depth, masks and selective stereo',
            'tracking_status':'initializing','geometry_status':'provisional','capture_status':'unverified'}
        self.started=time.monotonic();self.last_snapshot=0.
        self.validated_output=False
        self.mapping_quality=None

    def status(self,value):
        if self.mapping_quality is not None:self.report['mapping_quality']=self.mapping_quality.summary()
        self.report.update(status=value,elapsed_seconds=time.monotonic()-self.started,
            mapping_frames=self.map_frames,selected_frames=len(self.selected),fallbacks=self.fallbacks)
        save_json(self.out/'run_report.json',self.report);print(value,flush=True)

    def initialize(self):
        from .session_inputs import inspect_session
        if self.args.live_url and self.args.fresh:raise ValueError('--fresh benchmarks require saved input; live arrival cannot be replayed identically')
        with self.metrics.scope('session_validation'):
            inputs,overrides=inspect_session(self.args.session)
        self.inputs=inputs
        if getattr(self.args,'mapping_quality','off')=='adaptive':
            from .mapping_quality import MappingQualityGate
            self.mapping_quality=MappingQualityGate(max_gap_seconds=float(self.config.get('mapping_max_seconds',1.)))
        self.report['mapping_quality']=self.mapping_quality.summary() if self.mapping_quality is not None else {'enabled':False}
        self.selector_gyro=None
        gyro_reason='disabled by --imu-aware-selection off'
        if getattr(self.args,'imu_aware_selection','on')=='on' and not self.args.live_url:
            from .selector_imu import GyroHistory
            self.selector_gyro,gyro_reason=GyroHistory.from_session(self.args.session)
        elif self.args.live_url:
            gyro_reason='live video/IMU timestamp alignment unavailable'
        self.report['selector_imu']={'enabled':self.selector_gyro is not None,
                                     'reason':gyro_reason,'bridge_frames':0}
        self.learned_selector=None
        if getattr(self.args,'selector_checkpoint',None):
            from .graph_selector import SelectorRuntime
            self.learned_selector=SelectorRuntime(self.args.selector_checkpoint)
            self.report['selector_checkpoint']=str(self.args.selector_checkpoint.resolve())
        self.utility_policy=None
        if getattr(self.args,'selector_policy','rule')=='utility':
            from .graph_selector import UtilityRuntime
            self.utility_policy=UtilityRuntime(self.args.utility_checkpoint)
            self.report['utility_checkpoint']=str(self.args.utility_checkpoint.resolve())
        metadata=inputs['video'];w,h=metadata['width'],metadata['height']
        intr=overrides.get('camera_intrinsics')
        if intr:
            self.K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
            sx=w/intr.get('width',w);sy=h/intr.get('height',h);self.K[0]*=sx;self.K[1]*=sy
        else:
            f=overrides['focal_ratio']*w;self.K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
        self.distortion=np.asarray((intr or {}).get('distortion') or [],float)
        self.duration=None if self.args.live_url else metadata['duration_seconds']
        self.calibration_size=(w,h)
        if self.args.live_url:self.report['live_input']={'clock':'host_arrival','sensor_sync_used':False,'calibration':'Stored video calibration scaled to stream dimensions; stream crop/EIS remains uncertain'}
        source=self.args.live_url or str(self.args.session.resolve()/'video.mp4');self.source=source
        modules=('runtime.py','runtime_types.py','frame_stream.py','selector_imu.py','pose_dpvo.py','pose_backend.py','pose_workspace.py',
            'keyframe_policy.py','mapping_quality.py','depth_backend.py','model_versions.py','camera_orientation.py','colmap_workspace.py','temporal_depth.py','stereo_policy.py','stereo_refinement.py','submaps.py','semantic_scene.py')
        from .model_versions import model_identities
        with self.metrics.scope('model_provenance'):
            self.model_versions,self.model_paths=model_identities(self.root,getattr(self.args,'semantic_filter','on')=='on',getattr(self.args,'camera_rotation','0')=='auto')
        manifest={'schema':1,'source':source,'source_sha256':digest(source) if not self.args.live_url else None,
            'models':self.model_versions,
            'session_files':inputs['input_hashes'],'config':self.config,
            'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(self.args).items() if k not in ('output','resume','fresh','serve') and not (k=='selector_policy' and v=='rule') and not (k=='utility_checkpoint' and v is None) and not (k=='mapping_quality' and v=='off')},
            'code':{name:digest(Path(__file__).parent/name) for name in modules}}
        path=self.out/'pipeline_manifest.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if previous!=manifest:raise ValueError('Inputs, models or policy changed; use a new output')
            if not self.args.resume:raise ValueError('Output already used; select --resume or a fresh directory')
        save_json(path,manifest)
        save_json(self.out/'session_inputs_report.json',inputs);self.validated_output=True
        if self.previous_report:save_json(self.metrics.output/'previous_run_report.json',self.previous_report)
        self.manifest=manifest;self.status('initializing_models')
        import torch
        torch.set_num_threads(self.config['cpu_threads'])
        seed=int(self.config.get('seed',17));torch.manual_seed(seed);np.random.seed(seed)
        self.report['seed']=seed
        if self.config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA profile on a CPU-only machine; use stream_cpu.json')
        if self.config['device']=='cuda':
            total=torch.cuda.get_device_properties(0).total_memory/1024**2
            if total<self.config.get('gpu_memory_budget_mib',total):raise ValueError('Profile memory budget exceeds device capacity')
            self.report['hardware']={'gpu':torch.cuda.get_device_name(0),'memory_mib':total}
            torch.cuda.set_per_process_memory_fraction(min(.95,self.config.get('gpu_memory_budget_mib',total*.95)/total))
            torch.cuda.reset_peak_memory_stats()
        desired=self.args.pose_backend
        rotation=self.args.camera_rotation
        if rotation=='auto':
            if self.args.live_url:raise ValueError('Live orientation requires an explicit --camera-rotation value')
            from .camera_orientation import select_orientation
            with self.metrics.scope('orientation_selection',cuda=self.config['device']=='cuda'):
                rotation,diagnostics=select_orientation(self.source,self.config['device'])
            self.report['orientation_selection']=diagnostics
        self.config['camera_rotation']=int(rotation)
        if desired in ('dpvo','auto') and self.config['device']=='cuda':
            try:
                from .pose_dpvo import DPVOBackend
                with self.metrics.scope('pose_model_load',cuda=True):
                    if self.config['camera_rotation']:
                        from .camera_orientation import oriented_backend
                        self.backend=oriented_backend(self.root,self.config)
                    else:self.backend=DPVOBackend(self.root,self.config)
                self.report['pose_backend']=self.backend.provenance
            except Exception as error:
                if desired=='dpvo':raise
                self.fallbacks.append({'stage':'pose_initialization','from':'dpvo','to':'colmap','reason':str(error)})
        elif desired=='dpvo':raise RuntimeError('DPVO CUDA backend cannot run under the CPU profile')
        if self.backend is None:self.prepare_colmap()
        if self.args.resume:
            # The backend cannot resume its optimizer: replay tracking, retaining
            # accepted geometry only after final correction/revalidation.
            self.report['resume_scope']='replay tracking prefix; verify existing depth contexts; no native DPVO optimizer restore'
            self.report['runtime_scope']='replayed tracking and pose-context revalidation; compatible semantic/map caches retained'
        self.status('tracking_and_mapping')

    def frames(self):
        offset=None if self.args.live_url else self.inputs['timing']['video_to_session_offset_seconds']
        origin=None
        for frame in video_frames(self.source,self.K,self.config['tracking_size'],self.config['tracking_fps'],offset,live=bool(self.args.live_url),calibration_size=self.calibration_size):
            if origin is None:origin=frame.timestamp
            if self.distortion.size and np.any(self.distortion):
                # Distortion is removed in original calibration coordinates by
                # using its appropriately resized K; no unknown EIS warp is inferred.
                frame.rgb=cv2.undistort(frame.rgb,frame.K,self.distortion)
                frame.calibration_id='rectified_recorded_intrinsics'
            elapsed=frame.timestamp-origin if self.args.live_url else frame.timestamp
            if self.args.max_seconds and elapsed>self.args.max_seconds:break
            yield frame

    def record(self,frame,pose):
        directory=self.out/'tracking_rgb';directory.mkdir(exist_ok=True)
        path=directory/frame.frame_id
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:raise ValueError('Cannot encode tracking RGB')
        encoded=encoded.tobytes();rgb_hash=hashlib.sha256(encoded).hexdigest()
        if path.exists():
            if digest(path)!=rgb_hash:raise ValueError('Cached RGB differs from decoded input: '+frame.frame_id)
        else:path.write_bytes(encoded)
        return {'frame_id':frame.frame_id,'index':frame.index,'timestamp':frame.timestamp,'K':frame.K.tolist(),
            'width':frame.rgb.shape[1],'height':frame.rgb.shape[0],'path':str(path),'pose_version':pose.version,
            'submap_id':pose.submap_id,'rgb_sha256':rgb_hash,'calibration_exact':frame.calibration_exact,'pixel_transform':frame.pixel_transform.tolist()}

    def prepare_colmap(self):
        from .pose_backend import build_colmap_model
        if self.args.live_url:raise ValueError('Live COLMAP bootstrap is unavailable; choose the DPVO backend')
        records={}
        with self.metrics.scope('colmap_bootstrap_decode'):
            for frame in self.frames():
                pose=PoseEstimate(frame.frame_id,frame.timestamp,np.eye(4),'colmap','initializing')
                records[frame.frame_id]=self.record(frame,pose)
        with self.metrics.scope('colmap_fresh_bootstrap'):
            self.backend=build_colmap_model(records,self.out/'colmap_bootstrap',use_gpu=self.config['device']=='cuda',
                threads=self.config['cpu_threads'],max_seconds=self.config.get('cpu_mapping_seconds',90))
        self.report['pose_backend']={'backend':'colmap','offline_bootstrap':True,'cpu_compatible':True,'native_resume':False}

    def producer(self):
        policy=KeyframePolicy(self.config['mapping_min_seconds'],self.config['mapping_max_seconds'],self.config['mapping_parallax_pixels'])
        pending=deque();paced_start=time.monotonic();source_start=None
        trajectory=self.out/'poses';trajectory.mkdir(exist_ok=True)
        try:
            with (trajectory/'trajectory.jsonl').open('w',buffering=1) as stream,(trajectory/'corrections.jsonl').open('w',buffering=1) as corrections:
                iterator=iter(self.frames())
                while not self.stop.is_set():
                    with self.metrics.scope('decode'):
                        frame=next(iterator,None)
                    if frame is None or self.stop.is_set():break
                    if self.args.replay_paced:
                        if source_start is None:source_start=frame.timestamp
                        self.stop.wait(max(0,frame.timestamp-source_start-(time.monotonic()-paced_start)))
                    with self.gpu_lock,self.metrics.scope('pose_tracking',cuda=self.config['device']=='cuda'):
                        pose=self.backend.track(frame)
                        changes=self.backend.get_pose_updates(self.pose_version)
                    stream.write(json.dumps(pose.to_dict())+'\n')
                    self.latest_rgb=frame.rgb;self.latest_pose=pose
                    self.report['tracking_status']=pose.state
                    self.tracking_rows.append({'frame_id':frame.frame_id,'index':frame.index,'timestamp':frame.timestamp,'state':pose.state})
                    with self.state_lock:
                        for correction in changes:
                            for name,matrix in correction.poses.items():
                                if name in self.records:self.poses[name]=matrix.copy();self.records[name]['pose_version']=correction.version
                                if name in self.eval_records:self.eval_poses[name]=matrix.copy()
                            corrections.write(json.dumps({'version':correction.version,'poses':{n:p.tolist() for n,p in correction.poses.items()}})+'\n')
                        self.pose_version=pose.version
                        evaluation_bin=int(frame.timestamp/max(.1,(self.duration or 60)/48))
                        if pose.state=='tracking' and evaluation_bin not in self.eval_bins:
                            self.eval_bins.add(evaluation_bin);self.eval_records[frame.frame_id]=self.record(frame,pose)
                            self.eval_poses[frame.frame_id]=pose.T_world_from_camera.copy()
                    gyro_motion=(self.selector_gyro.motion(policy.last[0].sensor_timestamp,frame.sensor_timestamp)
                                 if self.selector_gyro is not None and policy.last is not None else None)
                    previous_last=policy.last
                    keep,stats=policy.decide(frame,pose,
                        self.queue.qsize()/self.queue.maxsize if self.args.live_url else 0.,gyro_motion)
                    if self.mapping_quality is not None:
                        geometric_keep=keep
                        keep,quality=self.mapping_quality.propose(frame,pose,stats,keep)
                        stats.update(quality)
                        if geometric_keep and not keep:policy.last=previous_last
                    quality_bridge_only=keep and not stats.get('mapping_quality',{}).get('dense_eligible',True)
                    if self.learned_selector is not None and not quality_bridge_only:
                        geometric_keep=keep
                        keep,learned=self.learned_selector.decide(frame,pose,stats,keep)
                        stats.update(learned)
                        if geometric_keep and not keep:policy.last=previous_last
                    if self.utility_policy is not None and not quality_bridge_only:
                        geometric_keep=keep
                        with self.gpu_lock:keep,learned=self.utility_policy.decide(frame,pose,stats,keep)
                        stats.update(learned)
                        if geometric_keep and not keep:policy.last=previous_last
                    if keep and self.mapping_quality is not None:self.mapping_quality.commit(frame,stats)
                    row={'frame':frame.frame_id,'timestamp':frame.timestamp,'selected':keep,**stats};self.decisions.append(row)
                    self.metrics.event('keyframe_decision',**row)
                    if keep and stats['reason']=='rotation_bridge_with_geometric_content':
                        self.report['selector_imu']['bridge_frames']+=1
                    if keep:
                        with self.state_lock:
                            record=self.record(frame,pose)
                            if self.mapping_quality is not None:
                                record['dense_eligible']=stats.get('mapping_quality',{}).get('dense_eligible',True)
                                record['mapping_quality']=stats.get('mapping_quality',{})
                            self.records[frame.frame_id]=record;self.poses[frame.frame_id]=pose.T_world_from_camera.copy()
                            self.selected.append(frame.frame_id)
                        if not quality_bridge_only:pending.append((len(self.tracking_rows),frame.frame_id))
                    while pending and len(self.tracking_rows)-pending[0][0]>=self.config['settle_tracking_frames']:
                        _,name=pending.popleft();self.enqueue(name)
                while pending and not self.stop.is_set():self.enqueue(pending.popleft()[1])
        except BaseException as error:
            self.errors.append(error);self.stop.set();self.metrics.event('tracking_failure',reason=str(error))
        finally:self.producer_done.set()

    def enqueue(self,name):
        while not self.stop.is_set():
            try:self.queue.put((name,time.monotonic()),timeout=.2);return
            except queue.Full:
                if self.args.live_url:
                    try:dropped=self.queue.get_nowait();self.metrics.event('mapping_queue_replaced',frame=dropped[0]);self.queue.task_done()
                    except queue.Empty:pass

    def context(self,name):
        with self.state_lock:
            names=[n for n,r in self.records.items() if r.get('dense_eligible',True)]
            if name not in names:return {},{}
            i=names.index(name);count=self.config['depth_views']
            # Prefer nearby settled views; only recorded, actually tracked views.
            # Past-only context is independent of decode/queue timing. The first
            # frame can use future context only at finalization, after EOF.
            if i==0 and self.producer_done.is_set():selected=names[:count]
            else:selected=names[max(0,i-count+1):i+1]
            selected=[name,*[n for n in selected if n!=name]]
            if self.utility_policy is not None:
                # Learned pairs replace the nearest views only when both are past, verified frames.
                preferred=self.utility_policy.sources(name,names[:i])
                if preferred:selected=[name,*preferred,*[n for n in selected[1:] if n not in preferred]][:max(count,3)]
            records={n:{**self.records[n],'pose_version':self.pose_version} for n in selected}
            return records,{n:self.poses[n].copy() for n in selected}

    def ensure_models(self):
        from .depth_backend import DA3Backend,SemanticWorker
        if self.depth is None:
            with self.metrics.scope('depth_model_load',cuda=self.config['device']=='cuda'):
                self.depth=DA3Backend(model=self.model_paths['depth'],device='cpu' if self.config.get('offload_inactive_models') else self.config['device'],resolution=self.config['depth_resolution'],rotation=self.config.get('camera_rotation',0))
        if self.args.semantic_filter=='on' and self.semantic is None:
            with self.metrics.scope('semantic_model_load',cuda=self.config['device']=='cuda'):
                self.semantic=SemanticWorker(self.out/'semantics','cpu' if self.config.get('offload_inactive_models') else self.config['device'],self.args.semantic_rotation)

    def activate_models(self,which):
        if not (self.config.get('offload_inactive_models') or self.phase_offload):return
        target=self.config['device']
        self.depth.predictor.to(target if which=='depth' else 'cpu')
        if self.semantic:
            self.semantic.predictor.sky.to(target if which=='semantic' else 'cpu')
            self.semantic.predictor.detector.to(target if which=='semantic' else 'cpu')
            self.semantic.predictor.device=target if which=='semantic' else 'cpu'
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()

    def process(self,name,replace=False):
        with self.state_lock:
            if not self.records[name].get('dense_eligible',True):
                self.metrics.event('depth_skipped',frame=name,reason='mapping_quality_bridge_only');return
        from .semantic_scene import remap_labels,object_position
        from .submaps import save_depth
        from .runtime_types import DepthFrame
        from .temporal_depth import warp_depth
        records,poses=self.context(name)
        if len(records)<2:
            self.metrics.event('depth_skipped',frame=name,reason='insufficient_context');return
        record=records[name];frame=read_record(record)
        pose=PoseEstimate(name,frame.timestamp,poses[name],'mapping',version=record['pose_version'],submap_id=record['submap_id'])
        with self.gpu_lock:
            self.ensure_models();classes=np.zeros(frame.rgb.shape[:2],np.uint8);ids=np.zeros_like(classes,np.uint16);observations=[]
            if self.semantic:
                mask_path=self.out/'semantics'/(name+'.npz')
                if mask_path.exists():
                    with np.load(mask_path) as data:classes=data['classes'];ids=data['instances'];observations=json.loads(str(data['observations']))
                else:
                    self.activate_models('semantic')
                    with self.metrics.scope('semantic_inference',cuda=self.config['device']=='cuda'):
                        classes,ids,observations=self.semantic.predict(name,frame.rgb,frame.K,[r['path'] for n,r in records.items() if n!=name])
            depth=None
            if self.args.temporal_reuse=='on' and self.last_depth is not None and not replace:
                from .multiview_dense import remap_camera
                target_k=self.last_depth.K;shape=self.last_depth.depth.shape
                rgb=remap_camera(frame.rgb,frame.K,target_k,shape,interpolation=cv2.INTER_LINEAR)
                target=FramePacket(frame.frame_id,frame.index,frame.timestamp,rgb,target_k)
                target_mask=remap_labels(classes,frame.K,target_k,shape)>0
                with self.metrics.scope('temporal_reprojection'):
                    depth,stats=warp_depth(self.last_depth,target,pose,target_mask,self.config['temporal_max_age'],self.config['temporal_min_coverage'])
                self.metrics.event('temporal_reuse',frame=name,**stats)
            if depth is None:
                self.activate_models('depth')
                with self.metrics.scope('depth_inference',cuda=self.config['device']=='cuda'):depth=self.depth.infer(records,poses,name)
            else:self.reused+=1
            self.activate_models(None)
        mask=remap_labels(classes,frame.K,depth.K,depth.depth.shape)>0;depth.excluded=mask
        if self.backend is not None and hasattr(self.backend,'model') and depth.source!='warped_previous_estimate':
            from .colmap_workspace import align_sparse_depth
            depth=align_sparse_depth(depth,self.out/'colmap_bootstrap',self.backend.model)
        depth.pose_version=record['pose_version'];depth.submap_id=record['submap_id']
        if depth.source=='warped_previous_estimate':depth.context_poses={**depth.context_poses,name:pose.T_world_from_camera.tolist()}
        else:depth.context_poses={n:poses[n].tolist() for n in records}
        rows=[];instance_map=remap_labels(ids,frame.K,depth.K,depth.depth.shape)
        # No stereo evidence exists at this stage. Conservative marker logic
        # keeps detections in 2D until post-stereo validation supports depth.
        for observation in observations:
            row={**observation,'frame':name,'frame_index':record['index'],'timestamp_seconds':frame.timestamp}
            row.update(object_position({'K':depth.K,'P':pose.cam_from_world,'prior':depth.depth,
                'measured':np.zeros_like(depth.depth),'validation':depth.validation},instance_map==observation['instance_id']))
            rows.append(row)
        self.observations[name]=rows
        with self.metrics.scope('fusion'):
            update=self.store.put(depth,replace=replace)
        if self.utility_policy is not None:
            seen=depth.valid
            self.utility_policy.observe(name,np.where(seen,depth.depth,0),depth.K,depth.T_world_from_camera,float(np.median(depth.confidence[seen])) if seen.any() else 0.)
        self.map_frames+=1;self.last_depth=depth
        self.metrics.event('map_update',sequence=update.sequence,frame=name,pose_version=depth.pose_version,source=depth.source)
        with self.metrics.scope('depth_cache_write'):
            save_depth(self.out/'depth_cache'/(name+'.npz'),depth)
        if time.monotonic()-self.last_snapshot>3:
            self.snapshot();self.last_snapshot=time.monotonic()
        self.status('tracking_and_mapping' if not self.producer_done.is_set() else 'draining_mapping_queue')

    def process_checked(self,name,replace=False):
        import torch
        try:return self.process(name,replace)
        except torch.cuda.OutOfMemoryError as error:
            if getattr(self,'oom_retried',False):raise
            self.oom_retried=True
            self.fallbacks.append({'stage':'inference','reason':str(error),'action':'one retry with CPU-offloaded inactive models and depth resolution <=336'})
            self.config['offload_inactive_models']=True
            self.config['depth_resolution']=min(336,self.config['depth_resolution'])
            if self.depth:
                self.depth.predictor.to('cpu');self.depth.resolution=self.config['depth_resolution']
            if self.semantic:
                self.semantic.predictor.sky.to('cpu');self.semantic.predictor.detector.to('cpu')
            import gc
            gc.collect();torch.cuda.empty_cache()
            save_json(self.out/'effective_profile_after_oom.json',self.config)
            return self.process(name,replace)

    def snapshot(self):
        for key,volume in self.store.volumes.items():self.store._export_chunk(key,volume)
        chunks=[{'id':key,'url':f'submaps/{key}.ply','version':meta.get('revision',0),
            'stale':bool(meta['stale_frames'])} for key,meta in self.store.meta.items() if (self.out/'submaps'/(key+'.ply')).exists()]
        if self.latest_rgb is not None:
            path=self.out/'current_rgb.partial.jpg';cv2.imwrite(str(path),cv2.cvtColor(self.latest_rgb,cv2.COLOR_RGB2BGR));path.replace(self.out/'current_rgb.jpg')
        if self.last_depth is not None:
            depth=self.last_depth.depth;positive=depth[depth>0]
            limit=float(np.percentile(positive,95)) if len(positive) else 1.
            display=cv2.applyColorMap(np.clip(depth/max(limit,1e-8)*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO);display[depth<=0]=0
            path=self.out/'current_depth.partial.jpg';cv2.imwrite(str(path),display);path.replace(self.out/'current_depth.jpg')
        with self.state_lock:trajectory=[p[:3,3].tolist() for p in list(self.poses.values())[-256:]]
        save_json(self.out/'map_updates.json',{'sequence':self.store.sequence,'chunks':chunks,
            'status':self.report['status'],'tracking_status':self.report['tracking_status'],'scale_status':'arbitrary_visual_units',
            'trajectory':trajectory,'camera':self.latest_pose.to_dict() if self.latest_pose else None,
            'mapping_updates_per_second':self.map_frames/max(.01,time.monotonic()-self.started),
            'selected_keyframes':len(self.selected),'queue_length':self.queue.qsize(),'resources':dict(self.metrics.resources)})

    def finish_geometry(self):
        self.status('correcting_submaps')
        with self.state_lock:poses={n:p.copy() for n,p in self.poses.items()}
        if self.args.resume:
            self.store.discard(set(self.store.index)-set(self.records))
            for name,key in self.store.index.items():
                f=load_depth(self.store._frame_path(name))
                if not set(f.context_poses)<=set(poses) and name not in self.store.meta[key]['stale_frames']:
                    self.store.meta[key]['stale_frames'].append(name)
        with self.metrics.scope('pose_correction_rebuild'):
            self.store.correct(poses,self.pose_version)
        for name in self.store.stale_frames():self.process_checked(name,replace=True)
        for name in self.selected:
            if name not in self.store.index and name not in self.observations:self.process_checked(name)
        from .pose_workspace import write_workspace
        with self.metrics.scope('workspace_export'):
            self.workspace=self.out/'dense'
            if self.backend is not None and hasattr(self.backend,'model'):
                from .colmap_workspace import copy_tracked_workspace
                copy_tracked_workspace(self.workspace,self.backend.model,self.out/'colmap_bootstrap/images')
            else:write_workspace(self.workspace,self.records,poses)
        if self.args.stereo_policy!='off' and self.config.get('stereo_references',0)>0:
            self.checked_stereo(poses)
        self.status('final_export')
        with self.metrics.scope('mesh_export'):
            result_dir=self.out/'multiview';result_dir.mkdir(exist_ok=True)
            mesh=self.store.export(result_dir/'scene_mesh.ply')
            if len(mesh.triangles):
                import open3d as o3d
                from .hybrid_dense import export_glb
                from .mesh_viewer import write_mesh_viewer
                o3d.io.write_triangle_mesh(str(result_dir/'scene_mesh_colored.ply'),mesh)
                completed=result_dir/'scene_completed.ply'
                if not completed.exists():completed.symlink_to('scene_mesh.ply')
                export_glb(mesh,result_dir/'scene_colored.glb')
                self.report.update(mesh=str(result_dir/'scene_colored.glb'),mesh_vertices=len(mesh.vertices),mesh_faces=len(mesh.triangles),result_ready=True)
                save_json(result_dir/'dense_metrics.json',{'mesh_vertices':len(mesh.vertices),'mesh_faces':len(mesh.triangles),
                    'source_model':str(self.workspace/'sparse'),'quality_warning':'Estimated relative-scale geometry; inspect tracking, coverage and depth evidence.'})
        if len(mesh.triangles):self.audit(mesh,poses)
        save_json(self.out/'keyframes.json',self.decisions)
        save_json(self.out/'frame_records.json',self.records)
        self.coverage()
        self.report['depth_network_calls']=self.depth.calls if self.depth else 0;self.report['temporal_reused_frames']=self.reused
        self.report['stale_frames']=self.store.stale_frames()
        if (self.report.get('mean_depth_agreement') is not None and self.report['mean_depth_agreement']>=.8
            and self.report.get('checked_stereo_views',0)>=4 and self.report.get('static_ray_coverage',0)>=.9):
            self.report['geometry_status']='internally_consistent_unvalidated'
        complete=self.report['capture_status']=='most_of_capture' and self.report['geometry_status']=='internally_consistent_unvalidated' and not self.report['stale_frames']
        self.report.update(execution_status='complete',tracking_status='finished',result_status='complete' if complete else 'incomplete',
            internal_quality_passed=complete,selected_result=str(self.out/'multiview'))
        self.status('complete' if complete else 'incomplete')

    def checked_stereo(self,poses):
        from .stereo_policy import plan_pose_stereo
        self.status('selective_stereo')
        # Tracking is finished. Free its GPU state and move inference weights to
        # RAM before the separate PatchMatch worker uses the same GPU.
        with self.gpu_lock:
            if self.backend:self.backend.close();self.backend=None
            self.phase_offload=True;self.activate_models(None)
        from .stereo_policy import evaluation_frames
        budget=32 if self.args.stereo_policy=='legacy' else self.config['stereo_references']
        depths={n:load_depth(self.out/'depth_cache'/(n+'.npz')) for n in evaluation_frames(sorted(self.store.index),64)}
        dense_records={n:r for n,r in self.records.items() if r.get('dense_eligible',True)}
        plan=plan_pose_stereo(self.workspace,dense_records,{n:p for n,p in poses.items() if n in dense_records},depths,budget)
        if self.args.stereo_policy=='legacy':plan['policy_note']='Legacy reference budget on pose/depth-based pair proposal, not the original sparse-track planner'
        self.report['stereo_plan']={k:v for k,v in plan.items() if k!='plan'}
        if plan['reference_views']<3:
            self.report['stereo_validation_status']='insufficient_parallax';return
        command=[sys.executable,'-u','-m','adaptive_uav.stereo_policy','--workspace',str(self.workspace),
            '--max-size',str(self.config['stereo_resolution']),'--iterations',str(self.config['stereo_iterations'])]
        with self.metrics.scope('selective_stereo'):
            with (self.out/'stereo.log').open('w') as stream:
                try:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=self.config['stereo_seconds'])
                except subprocess.TimeoutExpired:self.report['stereo_validation_status']='budget_exhausted';return
        self.report['stereo_validation_status']='computed' if result.returncode==0 else 'failed'
        self.report['stereo_exit_code']=result.returncode
        if result.returncode==0:
            from .stereo_refinement import refine_depth
            from .hybrid_dense import read_array
            from .multiview_dense import remap_camera,evidence_masks
            qualified={};diagnostics={}
            planned=sorted(row['reference'] for row in plan['plan'])
            heldout_count=min(max(4,len(planned)//3),len(planned)-3) if len(planned)>=7 else max(1,len(planned)//3)
            heldout=set(evaluation_frames(planned,heldout_count))
            self.report['held_out_stereo_frames']=sorted(heldout)
            save_json(self.out/'stereo_evaluation_split.json',{'held_out':sorted(heldout),
                'training':sorted(set(planned)-heldout),'note':'Held-out depth maps do not enter scale fitting, fusion or conflict filtering. Poses and geometric stereo remain correlated.'})
            with self.metrics.scope('stereo_anchored_refinement'):
                for name in depths:
                    if name in heldout:continue
                    path=self.workspace/'stereo/depth_maps'/(name+'.geometric.bin')
                    if not path.exists():continue
                    f=depths[name];raw=read_array(path);record=self.records[name];k=np.asarray(record['K']).copy()
                    k[0]*=raw.shape[1]/record['width'];k[1]*=raw.shape[0]/record['height']
                    measured=remap_camera(raw,k,f.K,f.depth.shape)
                    refined,check=refine_depth(f,measured);diagnostics[name]=check
                    if refined.valid.any():self.store.put(refined,replace=True)
                    else:self.store.discard([name])
                    qualified[name]={'name':name,'prior':refined.depth,'measured':np.where(refined.evidence==1,refined.depth,0),
                        'K':f.K,'P':np.linalg.inv(poses[name])}
                # Reject demonstrable free-space contradictions in other
                # predictions; unseen/occluded regions remain explicitly inferred.
                for name in list(self.store.index):
                    if name in qualified or not qualified:continue
                    f=load_depth(self.out/'depth_cache'/(name+'.npz'))
                    neighbors=sorted(qualified,key=lambda n:np.linalg.norm(poses[n][:3,3]-poses[name][:3,3]))[:3]
                    _,conflict=evidence_masks({'prior':f.depth,'K':f.K,'P':np.linalg.inv(poses[name])},[qualified[n] for n in neighbors])
                    if conflict.any():
                        f.depth[conflict]=0;f.evidence[conflict]=0
                        self.store.put(f,replace=True)
            save_json(self.out/'stereo_refinement.json',diagnostics)

    def audit(self,mesh,poses):
        from .multiview_dense import ray_audit,geometry_agreement,remap_camera
        from .hybrid_dense import read_array
        from .semantic_scene import object_position,remap_labels
        result_dir=self.out/'multiview';frames=[]
        from .stereo_policy import evaluation_frames
        stereo_names={path.name.removesuffix('.geometric.bin') for path in (self.workspace/'stereo/depth_maps').glob('*.geometric.bin')}
        audit_names=sorted((stereo_names|set(self.eval_records))&set(self.store.index))
        if not audit_names:audit_names=evaluation_frames(sorted(self.store.index),16)
        for name in audit_names:
            f=load_depth(self.out/'depth_cache'/(name+'.npz'));measured=np.zeros_like(f.depth)
            path=self.workspace/'stereo/depth_maps'/(name+'.geometric.bin')
            if path.exists():
                try:
                    raw=read_array(path);k=np.asarray(self.records[name]['K']).copy()
                    k[0]*=raw.shape[1]/self.records[name]['width'];k[1]*=raw.shape[0]/self.records[name]['height']
                    measured=remap_camera(raw,k,f.K,f.depth.shape)
                except (ValueError,OSError):pass
            object_measured=measured.copy();measured[f.excluded]=0
            frame={'name':name,'K':f.K,'P':np.linalg.inv(poses[name]),'prior':f.depth,'measured':measured,'rgb':f.rgb,
                'semantic_excluded':f.excluded,'semantic_sky':np.zeros_like(f.excluded)}
            mask_path=self.out/'semantics'/(name+'.npz')
            if mask_path.exists():
                with np.load(mask_path) as data:
                    classes=remap_labels(data['classes'],data['K'],f.K,f.depth.shape)
                    instance_ids=remap_labels(data['instances'],data['K'],f.K,f.depth.shape)
                frame['semantic_sky']=classes==1
                valid=(measured>0)&(f.depth>0);errors=np.abs(f.depth[valid]-measured[valid])/measured[valid]
                validation={'median':float(np.median(errors)) if len(errors)>=100 else None}
                for row in self.observations.get(name,[]):
                    row.update(object_position({**frame,'object_measured':object_measured,'validation':validation},instance_ids==row['instance_id']))
            frames.append(frame)
        mapping_names={f['name'] for f in frames}
        if self.semantic:
            with self.gpu_lock:self.activate_models('semantic')
        for name,record in self.eval_records.items():
            if name in mapping_names:continue
            packet=read_record(record);h,w=packet.rgb.shape[:2];factor=504/max(h,w)
            shape=(round(h*factor),round(w*factor));k=packet.K.copy();k[0]*=shape[1]/w;k[1]*=shape[0]/h
            rgb=cv2.resize(packet.rgb,(shape[1],shape[0]));empty=np.zeros(shape,np.float32)
            excluded=np.zeros(shape,bool);sky=np.zeros(shape,bool)
            if self.semantic:
                # Fixed audit views are selected independently of mapping decisions.
                with self.gpu_lock:
                    with self.metrics.scope('evaluation_semantics',cuda=self.config['device']=='cuda'):
                        classes,_,observations=self.semantic.predict(name,packet.rgb,packet.K)
                self.observations.setdefault(name,[{**row,'frame':name,'frame_index':record['index'],
                    'timestamp_seconds':record['timestamp'],'position':None,'position_status':'evaluation_only_no_validated_object_depth'} for row in observations])
                mapped=remap_labels(classes,packet.K,k,shape);excluded=mapped>0;sky=mapped==1
            frames.append({'name':name,'K':k,'P':np.linalg.inv(self.eval_poses[name]),'prior':empty,'measured':empty,
                'rgb':rgb,'semantic_excluded':excluded,'semantic_sky':sky})
        if self.semantic:
            with self.gpu_lock:self.activate_models(None)
        with self.metrics.scope('surface_audit'):
            audit=ray_audit({'completed':mesh},frames,result_dir,max_views=len(frames))
        audit['coverage_domain']='static pixels only; semantic masks applied' if self.args.semantic_filter=='on' else 'all image pixels'
        self.report['surface_audit']=audit
        self.report['all_stereo_depth_agreement']=geometry_agreement(audit)
        heldout=set(self.report.get('held_out_stereo_frames',[]))
        checked=[r for r in audit['sampled_views'] if r['frame'] in heldout]
        self.report['mean_depth_agreement']=geometry_agreement({'sampled_views':checked}) if heldout else None
        fixed=[r['meshes']['completed']['ray_hit_fraction'] for r in audit['sampled_views'] if r['frame'] in self.eval_records]
        fixed=[r for r in fixed if r is not None]
        self.report['static_ray_coverage']=float(np.mean(fixed)) if fixed else audit['mean_ray_hit']['completed']
        self.report['fixed_evaluation_views']=len(fixed)
        self.report['checked_stereo_views']=sum(r['meshes']['completed']['measured_pixels']>=100 for r in checked)
        from .semantic_scene import write_observations
        write_observations(result_dir,[row for rows in self.observations.values() for row in rows])
        # Refresh the embedded object metadata after actual stereo qualification.
        from .mesh_viewer import write_mesh_viewer
        write_mesh_viewer(result_dir,self.root/'tools/three_0.169.0')
        quality=('provisional' if self.report['mean_depth_agreement'] is None or self.report['checked_stereo_views']<4 else
            'internally_consistent_unvalidated' if self.report['mean_depth_agreement']>=.8 and self.report['static_ray_coverage']>=.9 else 'inconsistent_geometry')
        save_json(result_dir/'multiview_report.json',{'quality_status':quality,
            'surface_audit':audit,'frames':len(frames),'mean_measured_depth_agreement_under_5pct':self.report['mean_depth_agreement'],
            'geometry_accuracy_validated':False})

    def coverage(self):
        rows=self.tracking_rows;valid=[r for r in rows if r['state']=='tracking'];stamps=[r['timestamp'] for r in valid]
        fraction=len(valid)/max(1,len(rows));span=(max(stamps)-min(stamps))/self.duration if len(stamps)>1 and self.duration else 0.
        gap=float(max(np.diff(stamps))) if len(stamps)>1 else None
        passed=fraction>=.8 and span>=.9 and gap is not None and gap<=2.
        self.report['capture_coverage']={'valid_tracking_fraction':fraction,'video_span_fraction':span,
            'maximum_valid_pose_gap_seconds':gap,'tracked_frames':len(valid),'evaluated_tracking_frames':len(rows),'passed':passed,
            'note':'Includes every sampled tracking frame, including frames rejected for mapping; not geometric accuracy'}
        self.report['capture_status']='most_of_capture' if passed else 'partial'

    def run(self):
        thread=None
        try:
            self.initialize()
            from .stream_viewer import write_stream_viewer
            write_stream_viewer(self.out,self.root/'tools/three_0.169.0')
            thread=threading.Thread(target=self.producer,name='pose_tracking',daemon=True);thread.start()
            while not self.producer_done.is_set() or not self.queue.empty():
                if self.errors:raise self.errors[0]
                try:name,queued_at=self.queue.get(timeout=.2)
                except queue.Empty:continue
                self.metrics.event('queue_delay',frame=name,seconds=time.monotonic()-queued_at,queue_length=self.queue.qsize())
                self.process_checked(name,replace=name in self.store.index);self.queue.task_done()
            thread.join()
            if self.errors:raise self.errors[0]
            self.finish_geometry();self.snapshot()
        except KeyboardInterrupt:
            self.stop.set();self.report.update(execution_status='interrupted',capture_status='partial');self.status('interrupted')
        except BaseException as error:
            self.stop.set();self.report.update(execution_status='failed',error=str(error))
            if self.validated_output:self.status('failed')
            else:save_json(self.out/'initialization_error.json',{'error':str(error)})
            raise
        finally:
            self.stop.set()
            if thread:thread.join(timeout=5)
            if self.backend:self.backend.close()
            if self.depth:self.depth.close()
            if self.semantic:self.semantic.close()
            import torch
            if self.config['device']=='cuda':
                peak=torch.cuda.max_memory_allocated()/1024**2
                self.report['torch_peak_allocated_mib']=peak
                self.report['torch_peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
                self.report['gpu_memory_budget_passed']=peak<=self.config.get('gpu_memory_budget_mib',float('inf'))
            timing=self.metrics.close();self.report['timing_summary']=str(self.metrics.output/'timing_summary.json')
            self.report['elapsed_seconds']=timing['wall_seconds']
            if self.mapping_quality is not None:self.report['mapping_quality']=self.mapping_quality.summary()
            if self.validated_output:save_json(self.out/'run_report.json',self.report)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path);p.add_argument('--mode',choices=('preview','balanced','quality'),default='balanced')
    p.add_argument('--pose-backend',choices=('colmap','dpvo','auto'),default='auto')
    p.add_argument('--stereo-policy',choices=('legacy','adaptive','off'),default=None)
    p.add_argument('--temporal-reuse',choices=('on','off'),default='on');p.add_argument('--semantic-filter',choices=('on','off'),default='on')
    p.add_argument('--imu-aware-selection',choices=('on','off'),default='on',
                   help='Allow gyro-triggered bridge keyframes only with exact recorded video/IMU timing')
    p.add_argument('--selector-checkpoint',type=Path,
                   help='Optional reconstruction-derived visual selector; geometric safety gate remains active')
    p.add_argument('--selector-policy',choices=('rule','utility'),default='rule',
                   help='utility defers optional keyframes using --utility-checkpoint; the rule path is the fallback')
    p.add_argument('--utility-checkpoint',type=Path,help='memory_utility_v1 checkpoint for --selector-policy utility')
    p.add_argument('--mapping-quality',choices=('off','adaptive'),default='off',
                   help='Opt-in relative blur/feature deferral for optional mapping; tracking and geometric bridges remain protected')
    p.add_argument('--semantic-rotation',choices=('auto','0','90','180','270'),default='auto')
    p.add_argument('--camera-rotation',choices=('auto','0','90','180','270'),default='0',help='Rotate learned-model inputs with matching calibration and camera-axis transforms')
    p.add_argument('--profile',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fresh',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--replay-paced',action='store_true');p.add_argument('--refine',choices=('off','background','final'),default='off')
    p.add_argument('--serve');p.add_argument('--live-url');p.add_argument('--max-seconds',type=float,default=0.,help='Smoke-test limit, always reported as partial capture')
    args=p.parse_args(argv);root=Path(__file__).resolve().parents[2]
    if (args.selector_policy=='utility')!=bool(args.utility_checkpoint):p.error('--selector-policy utility and --utility-checkpoint go together')
    if args.selector_policy=='utility' and args.selector_checkpoint:p.error('Choose one learned selector')
    if args.fresh and args.resume:p.error('--fresh and --resume are incompatible')
    if args.max_seconds<0:p.error('--max-seconds must be nonnegative')
    if args.output.resolve().is_relative_to(args.session.resolve()):p.error('Output must be outside the input session')
    path=args.profile or root/'uav_reconstruction/configs/stream_workstation.json'
    config=json.loads(path.read_text());args.profile=path.resolve()
    if config['device']=='cpu' and (args.mode=='quality' or args.refine!='off'):
        p.error('CPU profile supports incremental preview; legacy high-quality refinement requires the CUDA stereo backend')
    args.stereo_policy=args.stereo_policy or ('off' if args.mode=='preview' else 'adaptive')
    if args.mode=='balanced' and args.stereo_policy=='off':p.error('Use preview mode for stereo-free provisional output')
    if config['device']=='cuda' and args.pose_backend in ('dpvo','auto') and Path(sys.prefix).resolve()!=root/'.venv_dpvo':
        executable=root/'.venv_dpvo/bin/python'
        if executable.exists():os.execv(str(executable),[str(executable),'-u','-m','adaptive_uav.runtime',*(sys.argv[1:] if argv is None else argv)])
    server=None
    if args.serve:
        host,port=args.serve.rsplit(':',1);port=int(port)
        if not 0<port<65536:p.error('Invalid server port')
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/'viewer_server.log').open('a') as log:
            server=subprocess.Popen([sys.executable,'-m','http.server',str(port),'--bind',host,'--directory',str(args.output.resolve())],stdout=log,stderr=subprocess.STDOUT)
        save_json(args.output/'viewer_server.json',{'pid':server.pid,'host':host,'port':port})
    try:
        runner=StreamRunner(args,config);runner.run()
    except BaseException:
        if server:server.terminate();server.wait()
        raise
    if args.refine!='off' or args.mode=='quality':
        # Optional refinement is an independent, fully timed fresh legacy pass.
        # It cannot silently replace the incremental result's quality status.
        command=[str(root/'reconstruct_final.sh'),str(args.session),'--mode','legacy','--fresh','--detail','high',
            '--semantic-filter',args.semantic_filter,'--output',str(args.output/'refinement')]
        if args.refine=='background':
            with (args.output/'refinement.log').open('w') as log:process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save_json(args.output/'refinement_job.json',{'pid':process.pid,'command':command,'status':'spawned','scope':'independent post-stream refinement'})
        else:
            start=time.monotonic()
            with (args.output/'refinement.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            save_json(args.output/'refinement_job.json',{'exit_code':result.returncode,'seconds':time.monotonic()-start,'command':command})
    if server:
        try:server.wait()
        except KeyboardInterrupt:server.terminate();server.wait()


if __name__=='__main__':main()
