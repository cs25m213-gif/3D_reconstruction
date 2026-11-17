"""Persistent pose-conditioned DA3 inference with explicit relative-scale evidence."""
import json
from pathlib import Path
import cv2,numpy as np
from .runtime_types import DepthFrame


def baseline_scale(reference_extrinsics,predicted_extrinsics):
    """Scale from camera-centre distances; valid for a line/two-view context.

    Full Sim(3) rotation alignment is degenerate for two centres. Depth scale
    itself is still observable from a nonzero baseline, independently of gauge.
    This is pose alignment, not an independent accuracy/metric-scale check.
    """
    def centers(extrinsics):
        e=np.asarray(extrinsics);return -np.einsum('nji,nj->ni',e[:,:3,:3],e[:,:3,3])
    a,b=centers(reference_extrinsics),centers(predicted_extrinsics)
    pairs=np.triu_indices(len(a),1)
    da=np.linalg.norm(a[pairs[0]]-a[pairs[1]],axis=1)
    db=np.linalg.norm(b[pairs[0]]-b[pairs[1]],axis=1)
    valid=(da>1e-5)&(db>1e-5)&np.isfinite(da)&np.isfinite(db)
    if not valid.any():raise ValueError('No nonzero baseline for depth/pose scale alignment')
    ratios=da[valid]/db[valid];scale=float(np.median(ratios))
    if np.median(np.abs(ratios-scale))/scale>.25:raise ValueError('Inconsistent pairwise depth/pose scales')
    return scale


class DA3Backend:
    def __init__(self,model='depth-anything/DA3-BASE',device='cuda',resolution=504,rotation=0):
        import torch
        try:
            from depth_anything_3.api import DepthAnything3
        except ModuleNotFoundError as exc:
            if exc.name != 'depth_anything_3':raise
            import sys
            source=Path(__file__).resolve().parents[2]/'tools/Depth-Anything-3/src'
            if not (source/'depth_anything_3').is_dir():raise
            sys.path.insert(0,str(source))
            from depth_anything_3.api import DepthAnything3
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA depth inference unavailable')
        self.device=device;self.resolution=resolution;self.model_name=model;self.rotation=rotation
        self.predictor=DepthAnything3.from_pretrained(model).to(device)
        import types
        original=self.predictor._align_to_input_extrinsics_intrinsics
        self.alignment='umeyama_pose_alignment'
        def align(instance,extrinsics,intrinsics,prediction,align_to_input_ext_scale=True,**kwargs):
            from evo.core.geometry import GeometryException
            try:
                self.alignment='umeyama_pose_alignment'
                return original(extrinsics,intrinsics,prediction,align_to_input_ext_scale,**kwargs)
            except GeometryException:
                if not align_to_input_ext_scale:raise
                scale=baseline_scale(extrinsics.numpy(),prediction.extrinsics)
                prediction.depth*=scale;prediction.extrinsics=extrinsics[...,:3,:].numpy();prediction.intrinsics=intrinsics.numpy()
                self.alignment='pairwise_baseline_scale_degenerate_rotation_gauge'
                return prediction
        self.predictor._align_to_input_extrinsics_intrinsics=types.MethodType(align,self.predictor)
        self.calls=0

    def infer(self,records,poses,target):
        import torch
        if len(records)<2:raise ValueError('Pose-conditioned depth requires two nondegenerate views')
        names=list(records);ks=np.stack([records[n]['K'] for n in names]).astype(np.float32)
        ps=np.stack([np.linalg.inv(poses[n]) for n in names]).astype(np.float32)
        images=[records[n]['path'] for n in names]
        if self.rotation:
            from .camera_orientation import rotate_calibration
            from .semantic_scene import rotate_image
            from PIL import Image
            images=[]
            for i,n in enumerate(names):
                rgb=cv2.cvtColor(cv2.imread(records[n]['path']),cv2.COLOR_BGR2RGB)
                k,q,_=rotate_calibration(ks[i],rgb.shape[1],rgb.shape[0],self.rotation)
                ks[i]=k;ps[i]=(q@ps[i]).astype(np.float32)
                images.append(Image.fromarray(rotate_image(rgb,self.rotation)))
        centers=np.stack([poses[n][:3,3] for n in names])
        if np.linalg.norm(np.ptp(centers,axis=0))<1e-5:raise ValueError('Insufficient translation to align depth and pose scale')
        with torch.inference_mode():
            result=self.predictor.inference(images,extrinsics=ps,intrinsics=ks,
                align_to_input_ext_scale=True,process_res=self.resolution,process_res_method='upper_bound_resize',ref_view_strategy='first')
        self.calls+=1;i=names.index(target)
        depth=result.depth[i].astype(np.float32)
        conf=result.conf[i] if result.conf is not None else np.ones(depth.shape,np.float32)
        rgb=result.processed_images[i];intrinsic=result.intrinsics[i]
        if self.rotation:
            from .camera_orientation import rotate_calibration
            from .semantic_scene import rotate_image
            intrinsic,_,_=rotate_calibration(intrinsic,depth.shape[1],depth.shape[0],(-self.rotation)%360)
            depth=rotate_image(depth,(-self.rotation)%360);conf=rotate_image(conf,(-self.rotation)%360)
            rgb=rotate_image(rgb,(-self.rotation)%360)
        record=records[target]
        return DepthFrame(target,record['timestamp'],depth,rgb,intrinsic,poses[target],
            observation_id=target,pose_version=record.get('pose_version',0),context_ids=tuple(names),
            context_poses={n:poses[n].tolist() for n in names},
            validation={'scale_fit_source':self.alignment,'independent_depth_agreement':None,
                        'metric_scale_validated':False,'median':None},submap_id=record.get('submap_id','0000'),
            confidence=np.asarray(conf,np.float32))

    def close(self):
        import torch
        if self.predictor is not None:
            del self.predictor._align_to_input_extrinsics_intrinsics
        self.predictor=None
        import gc
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


class SemanticWorker:
    def __init__(self,directory,device='cuda',rotation='auto'):
        from .semantic_scene import Predictor
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.predictor=Predictor(device);self.rotation=rotation;self.angle=None
        self.frames=[];self.started=None;self.previous_elapsed=0.
        report_path=self.directory/'semantic_report.json'
        if report_path.exists():
            previous=json.loads(report_path.read_text());self.frames=previous.get('frames',[])
            self.previous_elapsed=previous.get('elapsed_seconds',0.);self.angle=previous.get('rotation_clockwise')

    def predict(self,name,rgb,K,context=()):
        import time
        from .semantic_scene import rotate_image,save_json
        if self.started is None:self.started=time.monotonic()
        if self.angle is None:
            if self.rotation!='auto':self.angle=int(self.rotation)
            else:
                samples=[rgb]+[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in list(context)[:3]]
                scores={angle:float(np.mean([self.predictor.sky_prediction(rotate_image(image,angle))[1] for image in samples])) for angle in (0,90,180,270)}
                self.angle=max(scores,key=scores.get)
                save_json(self.directory/'orientation.json',{'rotation_clockwise':self.angle,'confidence_scores':scores,'selection':'semantic-confidence heuristic'})
        oriented=rotate_image(rgb,self.angle)
        sky,_=self.predictor.sky_prediction(oriented)
        sky=rotate_image(sky,(-self.angle)%360)
        classes=np.where(sky>=.65,1,0).astype(np.uint8);ids=np.zeros(classes.shape,np.uint16);observations=[]
        for label,confidence,mask in self.predictor.instances(oriented,.7):
            mask=rotate_image(mask.astype(np.uint8),(-self.angle)%360).astype(bool)&(ids==0)
            if mask.sum()<12:continue
            number=len(observations)+1;ids[mask]=number;classes[mask]=2 if label=='person' else 3
            y,x=np.where(mask)
            observations.append({'instance_id':number,'class':label,'confidence':float(confidence),
                'bbox_xyxy':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],'mask_pixels':int(mask.sum())})
        radius=max(1,round(max(classes.shape)*.0015))
        padded=cv2.dilate((classes>=2).astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))>0
        classes[padded&(classes==0)]=4
        temp=self.directory/(name+'.partial.npz');path=self.directory/(name+'.npz')
        np.savez_compressed(temp,classes=classes,instances=ids,K=K,observations=json.dumps(observations));temp.replace(path)
        self.frames=[row for row in self.frames if row['frame']!=name]
        self.frames.append({'frame':name,'sky_fraction':float((classes==1).mean()),'dynamic_fraction':float((classes>=2).mean()),'observations':observations})
        if len(self.frames)%20==1:
            overlay=rgb.copy();overlay[classes==1]=(.4*overlay[classes==1]+.6*np.array([80,130,255])).astype(np.uint8)
            overlay[classes>=2]=(.3*overlay[classes>=2]+.7*np.array([255,100,20])).astype(np.uint8)
            cv2.imwrite(str(self.directory/(name+'.overlay.jpg')),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
        save_json(self.directory/'semantic_report.json',{'status':'running','frames':self.frames,'rotation_clockwise':self.angle,'elapsed_seconds':self.previous_elapsed+time.monotonic()-self.started})
        return classes,ids,observations

    def close(self):
        import time
        from .semantic_scene import save_json,SKY_MODEL,SKY_REVISION
        save_json(self.directory/'semantic_report.json',{'status':'complete','frames':self.frames,'rotation_clockwise':self.angle,
            'sky_model':SKY_MODEL,'sky_revision':SKY_REVISION,'instance_model':'torchvision/maskrcnn_resnet50_fpn_v2/COCO_V1',
            'elapsed_seconds':self.previous_elapsed+(time.monotonic()-self.started if self.started else 0.),'object_observations':sum(len(f['observations']) for f in self.frames)})
        self.predictor=None
