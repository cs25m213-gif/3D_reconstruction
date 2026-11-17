"""Experimental quality residual; original checkpoints and runtime stay compatible."""
import numpy as np
import torch
from torch import nn
from .selector_corruptions import quality_vector, usable_rgb
from .selector_features import MapState, as_pose
from .selector_policy import UtilityPolicy, windows, sampling_distribution

class QualityAdapter(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(12,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features):return self.net(features).squeeze(-1)

def duplicate_frame_indices(views):
    """Exact consecutive RGB repeats add no new observation; do not infer from GT poses."""
    return {current.index for previous,current in zip(views,views[1:])
            if np.array_equal(previous.rgb,current.rgb)}

class RobustPolicy(UtilityPolicy):
    def __init__(self,model,data,device,adapter):
        super().__init__(model,data,device);self.adapter=adapter;self.last_trace=[]
    def run(self,clip,count,descriptors,temperature=0.):
        state=MapState();chosen=[];logs=[];entropies=[];trace=[];duplicates=duplicate_frame_indices(clip.views)
        for position,window in enumerate(windows(clip,count)):
            options=[clip.views[int(i)] for i in window]
            observable=[v for v in options if usable_rgb(v.rgb)]
            fresh=[v for v in observable if v.index not in duplicates]
            options=[v for v in fresh if np.isfinite(v.feature_pose).all() and getattr(v,'tracking_state','tracking')=='tracking']
            if not options:
                action='ABSTAIN' if not observable else 'SKIP' if not fresh else 'RELOCALIZE'
                reason='Repeated RGB frames only' if action=='SKIP' else 'No observable RGB or trusted pose in window'
                trace.append({'window':position,'action':action,'reason':reason});continue
            # Score all candidates, including the first window. Never force a bad bootstrap frame.
            budget={'seconds':60.*(count-position),'depth_calls':count-position}
            rows=[state.features(v,as_pose(v,v.feature_pose),budget,descriptors.get(clip.name,v.index,v.rgb)) for v in options]
            extra=np.stack([np.r_[quality_vector(v.rgb),r['query'][[1,2,3,4,7,13]]] for v,r in zip(options,rows)])
            scores=self.scores(rows)+self.adapter(torch.as_tensor(extra,dtype=torch.float32,device=self.device))
            if temperature>0 and len(options)>1:
                distribution=sampling_distribution(scores,temperature);pick=distribution.sample();logs.append(distribution.log_prob(pick));entropies.append(distribution.entropy());pick=int(pick)
            else:pick=int(scores.argmax())
            view=options[pick];descriptor=descriptors.get(clip.name,view.index,view.rgb)
            state.commit(view,as_pose(view,view.feature_pose),descriptor);state.observe(view.frame_id,view.depth_m,view.K,view.feature_pose,view.confidence)
            chosen.append(view);trace.append({'window':position,'action':'SELECT','index':view.index})
        self.last_trace=trace
        return chosen,logs,entropies


def load_robust_policy(path,device='cpu'):
    """Load the adapter explicitly; the original loader rejects this architecture ID."""
    from .graph_selector import MemoryUtilitySelector
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('architecture_id')!='robust_memory_utility_v1':raise ValueError('Not a robustness adapter checkpoint')
    model=MemoryUtilitySelector(**data['model_config']).to(device).eval();model.load_state_dict(data['model'])
    adapter=QualityAdapter().to(device).eval();adapter.load_state_dict(data['robustness_adapter'])
    return RobustPolicy(model,data,device,adapter),data
