"""Quarter-turn image transforms with matching camera intrinsics and axes."""
from dataclasses import replace
from pathlib import Path
import cv2,numpy as np
from .semantic_scene import rotate_image


def rotate_calibration(K,width,height,clockwise):
    K=np.asarray(K,float);fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    if clockwise==0:return K.copy(),np.eye(4),(width,height)
    if clockwise==90:
        values=(fy,fx,height-1-cy,cx);r=[[0,-1,0],[1,0,0],[0,0,1]];size=(height,width)
    elif clockwise==180:
        values=(fx,fy,width-1-cx,height-1-cy);r=[[-1,0,0],[0,-1,0],[0,0,1]];size=(width,height)
    elif clockwise==270:
        values=(fy,fx,cy,width-1-cx);r=[[0,1,0],[-1,0,0],[0,0,1]];size=(height,width)
    else:raise ValueError('Camera rotation must be a quarter turn')
    a,b,c,d=values;k=np.array([[a,0,c],[0,b,d],[0,0,1.]])
    q=np.eye(4);q[:3,:3]=r
    return k,q,size


def select_orientation(video,device='cuda'):
    """A confidence heuristic on fixed temporal samples, not gravity calibration."""
    import torch
    from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
    from .semantic_scene import SKY_MODEL,SKY_REVISION
    processor=SegformerImageProcessor.from_pretrained(SKY_MODEL,revision=SKY_REVISION,local_files_only=True)
    model=SegformerForSemanticSegmentation.from_pretrained(SKY_MODEL,revision=SKY_REVISION,local_files_only=True).to(device).eval()
    cap=cv2.VideoCapture(str(video));count=cap.get(cv2.CAP_PROP_FRAME_COUNT);samples=[]
    try:
        for index in np.linspace(0,max(0,count-1),4,dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,image=cap.read()
            if ok:samples.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if not samples:raise ValueError('No orientation samples decoded')
    scores={}
    with torch.inference_mode():
        for angle in (0,90,180,270):
            values=[]
            for rgb in samples:
                inputs=processor(images=rotate_image(rgb,angle),return_tensors='pt').to(device)
                logits=model(**inputs).logits.float();values.append(float(logits.softmax(1).amax(1).mean()))
            scores[angle]=float(np.mean(values))
    order=sorted(scores,key=scores.get,reverse=True)
    # Avoid rotating from a nearly tied heuristic without useful evidence.
    chosen=order[0] if scores[order[0]]-scores[order[1]]>=.01 else 0
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return chosen,{'scores':scores,'chosen_clockwise':chosen,'minimum_confidence_margin':.01,'method':'semantic-confidence heuristic, fixed recording orientation only'}


def oriented_backend(root,config):
    from .pose_dpvo import DPVOBackend
    class OrientedDPVO(DPVOBackend):
        def __init__(self):
            super().__init__(root,config);self.angle=int(config['camera_rotation']);self.Q=None
            self.provenance['camera_rotation_clockwise']=self.angle

        def track(self,frame,imu_samples=None):
            h,w=frame.rgb.shape[:2];k,q,_=rotate_calibration(frame.K,w,h,self.angle);self.Q=q
            pixel=k@q[:3,:3]@np.linalg.inv(frame.K)@frame.pixel_transform
            rotated=replace(frame,rgb=rotate_image(frame.rgb,self.angle),K=k,pixel_transform=pixel)
            pose=super().track(rotated,imu_samples)
            return replace(pose,T_world_from_camera=pose.T_world_from_camera@q)

        def get_pose_updates(self,since_version=0):
            return [replace(c,poses={n:p@self.Q for n,p in c.poses.items()}) for c in super().get_pose_updates(since_version)]

        def get_keyframes(self):
            if self.Q is None:return []
            return [{**row,'T_world_from_camera':(np.asarray(row['T_world_from_camera'])@self.Q).tolist()} for row in super().get_keyframes()]
    return OrientedDPVO()
