"""Small pose-conditioned multi-view depth student; research-only backend.

Inputs are target/source RGB, their calibrated pinhole matrices, world-to-camera
poses, and an independently supplied visual-unit scale from sparse tracks.
The network cannot establish metric scale or repair disconnected camera poses.
"""
import torch
from torch import nn
from torch.nn import functional as F


def conv2(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,1),
                         nn.GroupNorm(min(8,out_channels),out_channels),nn.SiLU())


class DepthStudent(nn.Module):
    def __init__(self, channels=48, planes=24):
        super().__init__()
        self.planes=planes
        self.encoder=nn.Sequential(conv2(3,channels//2,2),conv2(channels//2,channels,2),
                                   conv2(channels,channels),conv2(channels,channels))
        self.volume=nn.Sequential(nn.Conv3d(3,32,3,padding=1),nn.SiLU(),
                                  nn.Conv3d(32,32,3,padding=1),nn.SiLU(),
                                  nn.Conv3d(32,1,3,padding=1))
        self.refine=nn.Sequential(conv2(channels+4,channels),conv2(channels,channels),
                                  nn.Conv2d(channels,2,3,padding=1))
        self.register_buffer('depth_planes',torch.exp(torch.linspace(-1.3862944,1.3862944,planes)))

    def forward(self,rgb,K,P,source_weights=None):
        """RGB [B,V,3,H,W], K [B,V,3,3], P [B,V,4,4].

        P translations must already be divided by the sparse anchor scale.
        Output is positive depth in the same normalized units and confidence.
        """
        b,v,_,h,w=rgb.shape
        if v<2:raise ValueError('At least two posed views are required')
        if source_weights is not None and source_weights.shape != (b,v-1):
            raise ValueError('source_weights must have shape [B,V-1]')
        features=self.encoder(rgb.reshape(b*v,3,h,w)).reshape(b,v,-1,h//4,w//4)
        hf,wf=features.shape[-2:]
        k=K.clone();k[:,:,0]*=wf/w;k[:,:,1]*=hf/h
        y,x=torch.meshgrid(torch.arange(hf,device=rgb.device),torch.arange(wf,device=rgb.device),indexing='ij')
        xy=torch.stack((x,y,torch.ones_like(x)),0).float().reshape(1,1,3,hf*wf)
        rays=torch.linalg.solve(k[:,0].float(),xy.expand(b,1,-1,-1).squeeze(1))
        reference=features[:,0]
        accum=torch.zeros((b,3,self.planes,hf,wf),device=rgb.device,dtype=torch.float32)
        for source in range(1,v):
            relative=P[:,source].float()@torch.linalg.inv(P[:,0].float())
            xyz=rays[:,None]*self.depth_planes[None,:,None,None]
            xyz=relative[:,None,:3,:3]@xyz+relative[:,None,:3,3,None]
            projected=k[:,source,None].float()@xyz
            z=projected[:,:,2].clamp_min(1e-6)
            u=projected[:,:,0]/z;vv=projected[:,:,1]/z
            valid=((xyz[:,:,2]>1e-6)&(u>=0)&(u<wf)&(vv>=0)&(vv<hf)).reshape(b,self.planes,hf,wf)
            grid=torch.stack((2*u/max(wf-1,1)-1,2*vv/max(hf-1,1)-1),-1).reshape(b*self.planes,hf,wf,2)
            src=features[:,source,None].expand(-1,self.planes,-1,-1,-1).reshape(b*self.planes,-1,hf,wf)
            warped=F.grid_sample(src,grid,align_corners=True,padding_mode='zeros').reshape(b,self.planes,-1,hf,wf)
            diff=reference[:,None].float()-warped.float()
            weight=1. if source_weights is None else source_weights[:,source-1,None,None,None]
            accum[:,0]+=(-diff.square().mean(2)*valid).float()*weight
            accum[:,1]+=(reference[:,None].float()*warped.float()).mean(2)*valid*weight
            accum[:,2]+=valid.float()*weight
        denominator=max(v-1,1) if source_weights is None else source_weights.sum(1).clamp_min(1e-6)[:,None,None,None,None]
        accum=accum/denominator
        logits=self.volume(accum).squeeze(1)
        probabilities=logits.softmax(1)
        low=(probabilities*self.depth_planes[None,:,None,None]).sum(1,keepdim=True)
        entropy=-(probabilities*probabilities.clamp_min(1e-8).log()).sum(1,keepdim=True)
        guidance=torch.cat((reference,F.interpolate(rgb[:,0],size=(hf,wf),mode='bilinear',align_corners=False),low),1)
        residual=self.refine(guidance)
        depth=F.interpolate(low*torch.exp(residual[:,:1].tanh()*.5),size=(h,w),mode='bilinear',align_corners=False)
        confidence=F.interpolate(torch.sigmoid(residual[:,1:2]-entropy/self.planes),size=(h,w),mode='bilinear',align_corners=False)
        return depth,confidence
