"""Visibility-aware, versioned reprojection of existing depth evidence."""
from dataclasses import replace
import cv2,numpy as np


def warp_depth(previous,current,pose,excluded=None,max_age=2,min_coverage=.9,photometric_limit=.06):
    if previous.age>=max_age:return None,{'reason':'age_limit'}
    if previous.submap_id!=pose.submap_id or pose.state!='tracking':return None,{'reason':'tracking_or_submap_change'}
    if previous.pose_version>pose.version:return None,{'reason':'pose_version_order'}
    h,w=current.rgb.shape[:2];y,x=np.where(previous.valid)
    if not len(x):return None,{'reason':'no_valid_source'}
    z=previous.depth[y,x];k=previous.K
    points=np.column_stack(((x-k[0,2])*z/k[0,0],(y-k[1,2])*z/k[1,1],z))
    rel=pose.cam_from_world@previous.T_world_from_camera
    target=points@rel[:3,:3].T+rel[:3,3]
    projected=target@current.K.T
    valid=np.isfinite(target).all(1)&(target[:,2]>0)
    uv=np.zeros((len(x),2),np.int32);uv[valid]=np.rint(projected[valid,:2]/projected[valid,2:3]).astype(np.int32)
    valid&=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    ids=np.flatnonzero(valid);flat=uv[ids,1]*w+uv[ids,0]
    order=np.argsort(target[ids,2],kind='stable');ids=ids[order];flat=flat[order]
    _,first=np.unique(flat,return_index=True);ids=ids[first];flat=flat[first]
    depth=np.zeros(h*w,np.float32);depth[flat]=target[ids,2];depth=depth.reshape(h,w)
    mask=np.zeros((h,w),bool) if excluded is None else np.asarray(excluded,bool)
    depth[mask]=0;coverage=float((depth>0).sum()/max(1,(~mask).sum()))
    # Reprojection is permitted only where current appearance agrees as well.
    a=previous.rgb[y[ids],x[ids]].astype(float)/255
    b=current.rgb[uv[ids,1],uv[ids,0]].astype(float)/255
    error=float(np.median(np.abs(a-b))) if len(ids) else 1.
    stats={'coverage':coverage,'median_rgb_error':error,'reason':'accepted'}
    if coverage<min_coverage or error>photometric_limit:
        stats['reason']='disocclusion_or_appearance_change';return None,stats
    return replace(previous,frame_id=current.frame_id,timestamp=current.timestamp,depth=depth,
        rgb=current.rgb,K=current.K,T_world_from_camera=pose.T_world_from_camera,pose_version=pose.version,
        source='warped_previous_estimate',age=previous.age+1,excluded=mask,
        evidence=np.where(depth>0,4,0).astype(np.uint8),
        confidence=np.where(depth>0,.5,0).astype(np.float32)),stats
