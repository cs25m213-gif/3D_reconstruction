"""Image-only causal overlap prediction and overlapping view groups.

Features are frozen DINOv2 image embeddings. Supervision uses verified shared
SfM tracks only; PnP failures are not assigned fake negative labels.
"""
from dataclasses import dataclass,field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class OverlapAttention(nn.Module):
    def __init__(self,feature_dim=384,hidden=128,heads=4):
        super().__init__()
        self.visual=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,hidden),nn.GELU())
        self.attention=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.pair=nn.Sequential(nn.Linear(hidden*4+1,hidden),nn.GELU(),
                                nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))

    def forward(self,current,history,lag,valid):
        """current [B,384], history [B,H,384], lag [B,H] in frame steps."""
        q=self.visual(current)[:,None]
        keys=self.visual(history)
        context,_=self.attention(q,keys,keys,key_padding_mask=~valid,need_weights=False)
        q=q.expand_as(keys);context=context.expand_as(keys)
        pairs=torch.cat((q,keys,(q-keys).abs(),q*keys,
                         torch.log1p(lag.float().clamp_min(0))[...,None]/5.),-1)
        return torch.sigmoid(self.pair(pairs).squeeze(-1))


@dataclass
class ViewBin:
    anchor:int
    members:list[int]=field(default_factory=list)


def make_bins(indices,pair_scores,threshold=.28,max_memberships=2,max_anchor_gap=8,max_bin_size=8):
    """Causal bins; a frame may join two previous bins, with periodic new anchors.

    pair_scores[i,j] is predicted overlap for frame i with earlier frame j.
    This grouping is an image-space proposal; depth requires calibrated poses.
    """
    bins=[];memberships={};indices=list(indices)
    for position,frame in enumerate(indices):
        ranked=[]
        for number,group in enumerate(bins):
            if len(group.members)>=max_bin_size:continue
            recent=[j for j in group.members[-3:] if j<frame]
            if not recent:continue
            score=max(float(pair_scores.get((frame,j),0.)) for j in recent)
            if score>=threshold:ranked.append((score,number))
        ranked.sort(reverse=True)
        chosen=[number for _,number in ranked[:max_memberships]]
        # A new anchor also preserves temporal coverage through long views.
        if not chosen or position-bins[-1].anchor>=max_anchor_gap:
            bridge=None
            if chosen:
                bridge=next((member for member in reversed(bins[chosen[0]].members)
                    if len(memberships.get(member,[]))<max_memberships),None)
            elif bins:
                candidate=bins[-1].members[-1]
                if (len(memberships.get(candidate,[]))<max_memberships and
                    float(pair_scores.get((frame,candidate),0.))>=threshold):bridge=candidate
            bins.append(ViewBin(anchor=position,members=[bridge] if bridge is not None else []))
            if bridge is not None:
                memberships[bridge]=memberships.get(bridge,[])+[len(bins)-1]
            chosen=chosen[:max_memberships-1]+[len(bins)-1]
        for number in chosen:bins[number].members.append(frame)
        memberships[frame]=chosen
    return bins,memberships


def overlap_pairs(features,model,device,history=12,frame_indices=None):
    """Predict only current-to-past overlap; no future images enter attention."""
    features=np.asarray(features,np.float32)
    if frame_indices is None:frame_indices=np.arange(len(features))
    scores={};model.eval()
    with torch.inference_mode():
        for i in range(1,len(features)):
            old=list(range(max(0,i-history),i));h=len(old)
            current=torch.from_numpy(features[i:i+1]).to(device)
            previous=torch.from_numpy(features[old][None]).to(device)
            lag=torch.tensor((np.asarray(frame_indices[i])-np.asarray(frame_indices)[old])[None],device=device)
            valid=torch.ones((1,h),dtype=torch.bool,device=device)
            values=model(current,previous,lag,valid)[0].cpu().numpy()
            scores.update({(i,j):float(value) for j,value in zip(old,values)})
    return scores
