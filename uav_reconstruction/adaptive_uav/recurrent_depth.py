"""Pose-aligned sequential depth updates with a bounded ConvGRU."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def warp_source_depth(depth,source_K,source_P,target_K,target_P,target_shape,payload=None):
    """Forward project source camera-z depth into target camera with z-buffer."""
    depth=np.asarray(depth,np.float32);h,w=depth.shape
    y,x=np.nonzero(np.isfinite(depth)&(depth>0))
    result=np.zeros(target_shape,np.float32)
    if not len(x):return result
    z=depth[y,x];xyz=np.stack(((x-source_K[0,2])*z/source_K[0,0],
        (y-source_K[1,2])*z/source_K[1,1],z),1)
    relative=target_P@np.linalg.inv(source_P)
    xyz=xyz@relative[:3,:3].T+relative[:3,3]
    valid=np.isfinite(xyz).all(1)&(xyz[:,2]>0)
    th,tw=target_shape
    u=np.rint(xyz[:,0]/np.maximum(xyz[:,2],1e-9)*target_K[0,0]+target_K[0,2]).astype(np.int32)
    v=np.rint(xyz[:,1]/np.maximum(xyz[:,2],1e-9)*target_K[1,1]+target_K[1,2]).astype(np.int32)
    valid&=(u>=0)&(u<tw)&(v>=0)&(v<th)
    flat=v[valid]*tw+u[valid];values=xyz[valid,2]
    if len(flat):
        if payload is None:
            result.fill(np.inf);np.minimum.at(result.ravel(),flat,values)
            result[~np.isfinite(result)]=0
        else:
            payload=np.asarray(payload,np.float32)[y,x][valid]
            order=np.argsort(values,kind='stable');flat=flat[order];payload=payload[order]
            unique,first=np.unique(flat,return_index=True)
            result.ravel()[unique]=payload[first]
    return result


class ConvGRU(nn.Module):
    def __init__(self,input_channels=9,hidden=24):
        super().__init__();self.hidden=hidden
        self.gates=nn.Conv2d(input_channels+hidden,2*hidden,3,padding=1)
        self.candidate=nn.Conv2d(input_channels+hidden,hidden,3,padding=1)

    def forward(self,x,h=None):
        if h is None:h=x.new_zeros((len(x),self.hidden,*x.shape[-2:]))
        update,reset=self.gates(torch.cat((x,h),1)).sigmoid().chunk(2,1)
        candidate=torch.tanh(self.candidate(torch.cat((x,reset*h),1)))
        return (1-update)*h+update*candidate


class RecurrentDepthUpdater(nn.Module):
    def __init__(self,hidden=24):
        super().__init__();self.cell=ConvGRU(9,hidden)
        self.head=nn.Sequential(nn.Conv2d(hidden,hidden,3,padding=1),nn.SiLU(),
                                nn.Conv2d(hidden,2,3,padding=1))

    def forward(self,previous,new,previous_conf,new_conf,rgb,hidden=None):
        """Depth is anchor-normalized; new=0 denotes no visible observation."""
        valid=(new>0).float();prior=previous.clamp_min(1e-4)
        observed=torch.where(valid.bool(),new,prior).clamp_min(1e-4)
        disagreement=(observed/prior).log().clamp(-2,2)
        x=torch.cat((prior.log().clamp(-3,3),observed.log().clamp(-3,3),
            previous_conf.clamp(0,1),new_conf.clamp(0,1),valid,disagreement,rgb),1)
        hidden=self.cell(x,hidden)
        residual,confidence_logit=self.head(hidden).chunk(2,1)
        corrected=(prior*torch.exp(.35*torch.tanh(residual))).clamp(.1,10.)
        confidence=torch.sigmoid(confidence_logit)
        return corrected,confidence,hidden


def confidence_average(previous,new,previous_conf,new_conf):
    valid=(new>0).float();a=previous_conf.clamp_min(.05);b=new_conf.clamp_min(.05)*valid
    return (previous*a+new*b)/(a+b).clamp_min(1e-6)
