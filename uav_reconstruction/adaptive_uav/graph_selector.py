"""Optional reconstruction-derived frame selector. Default keyframe policy is unchanged."""
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn

DINO_ROOT = Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'
ACTIONS = ('NEW_KEYFRAME', 'TRACK_ONLY', 'SKIP', 'RELOCALIZE')


class GraphSelector(nn.Module):
    def __init__(self, visual_dim=384, state_dim=8, hidden=96):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(visual_dim+state_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.action=nn.Linear(hidden,4)
        self.utility=nn.Linear(hidden,3)
        self.uncertainty=nn.Linear(hidden,1)

    def forward(self,visual,state,hidden=None):
        x=self.project(torch.cat((visual,state),-1))
        x,hidden=self.gru(x,hidden)
        return self.action(x),self.utility(x),torch.sigmoid(self.uncertainty(x)),hidden


class CNNGraphSelector(nn.Module):
    """Trainable MobileNet visual encoder feeding the existing GRU policy."""
    def __init__(self):
        super().__init__()
        from .pretrain_visual_encoder import encoder
        self.encoder=encoder()
        self.visual_projector=nn.Linear(576,384)
        self.policy=GraphSelector()

    def forward(self,rgb,state,hidden=None):
        b,t=rgb.shape[:2]
        visual=self.encoder(rgb.reshape(b*t,*rgb.shape[2:]))
        visual=self.visual_projector(visual).reshape(b,t,-1)
        return self.policy(visual,state,hidden)


def frozen_encoder(device):
    if not DINO_ROOT.is_dir():
        raise FileNotFoundError(f'Cached DINOv2 source missing: {DINO_ROOT}')
    model=torch.hub.load(str(DINO_ROOT),'dinov2_vits14',source='local',pretrained=True).to(device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model


def image_feature(encoder,rgb,device):
    image=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
    tensor=torch.from_numpy(image.transpose(2,0,1).copy())[None].to(device)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
        feature=encoder((tensor-mean)/std)
    return feature.float().flatten().cpu().numpy()


class SelectorRuntime:
    """Bounded temporal and keyframe memory; geometry remains the safety gate."""
    def __init__(self,checkpoint,device=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=GraphSelector().to(self.device).eval()
        self.model.load_state_dict(data['model'])
        self.encoder=frozen_encoder(self.device)
        self.hidden=None
        self.memory=deque(maxlen=64)
        self.buffer=deque(maxlen=8)

    @torch.inference_mode()
    def decide(self,frame,pose,stats,geometric_keep):
        if pose.state!='tracking':
            return False,{'action':'RELOCALIZE','selector_uncertainty':1.}
        feature=image_feature(self.encoder,frame.rgb,self.device)
        old=self.memory[-1] if self.memory else None
        dt=frame.timestamp-old[0] if old else 0.
        translation=float(stats.get('translation_visual_units',0.))
        rotation=0.
        if old:
            relative=old[2][:3,:3].T@pose.T_world_from_camera[:3,:3]
            rotation=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
        graph_state=np.array([min(len(self.memory)*.25,3.),np.log1p(max(translation,0.)),
            min(rotation,1.5)/1.5,
            min(float(stats.get('tracked_features',0.)),200.)/200.,
            min(float(stats.get('sharpness',0.)),300.)/300.,
            float(bool(stats.get('imu_rapid_rotation',False))),
            min(len(self.memory),64)/64.,1.],np.float32)
        visual=torch.from_numpy(feature)[None,None].to(self.device)
        state=torch.from_numpy(graph_state)[None,None].to(self.device)
        logits,utility,uncertainty,self.hidden=self.model(visual,state,self.hidden)
        probability=logits.softmax(-1)[0,0]
        action=ACTIONS[int(probability.argmax())]
        uncertain=float(uncertainty[0,0,0])
        # Learned rejection is used only with high confidence and a short gap.
        # Geometric/IMU bridge decisions and gap recovery remain authoritative.
        bridge=stats.get('reason')=='rotation_bridge_with_geometric_content'
        reject=(action in ('TRACK_ONLY','SKIP') and float(probability.max())>.8
                and uncertain<.3 and len(self.memory)>=2 and dt<.75 and not bridge)
        keep=bool(geometric_keep and not reject)
        if keep:self.memory.append((frame.timestamp,feature,pose.T_world_from_camera.copy()))
        elif geometric_keep:self.buffer.append((frame.frame_id,frame.timestamp))
        return keep,{'action':action,'selector_uncertainty':uncertain,
                     'selector_probability':float(probability.max()),
                     'predicted_utility':utility[0,0].cpu().tolist(),
                     'buffered_candidates':len(self.buffer)}



class MemoryUtilitySelector(nn.Module):
    """memory_utility_v1: predicts what an action adds to this map, under this budget."""
    architecture_id = 'memory_utility_v1'

    def __init__(self, visual_dim=384, query_dim=17, token_dim=11, hidden=128, heads=4, layers=2, questions=7):
        super().__init__()
        self.query=nn.Sequential(nn.Linear(visual_dim+query_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.token=nn.Sequential(nn.Linear(visual_dim+token_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.question=nn.Embedding(questions,hidden)
        block=nn.TransformerEncoderLayer(hidden,heads,hidden*2,.1,'gelu',batch_first=True,norm_first=True)
        self.memory=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.action=nn.Linear(hidden,len(ACTIONS))
        self.utility=nn.Linear(hidden,4)  # coverage, surface, connectivity, overall
        self.cost=nn.Linear(hidden,1)
        self.log_variance=nn.Linear(hidden,1)
        self.pair=nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,visual,query,token_visual,tokens,mask,question,hidden=None,pairs=False):
        """Sequence inputs [B,T,...]; mask is True for a real token. All are causal features."""
        b,t,n=token_visual.shape[:3]
        q=self.query(torch.cat((visual,query),-1))+self.question(question)
        m=self.token(torch.cat((token_visual,tokens),-1))
        sequence=torch.cat((q.reshape(b*t,1,-1),m.reshape(b*t,n,-1)),1)
        keep=torch.cat((torch.ones(b*t,1,dtype=torch.bool,device=mask.device),mask.reshape(b*t,n)),1)
        encoded=self.memory(sequence,src_key_padding_mask=~keep)
        x,hidden=self.gru(encoded[:,0].reshape(b,t,-1),hidden)
        output={'action':self.action(x),'utility':self.utility(x),'cost':self.cost(x).squeeze(-1),
                'log_variance':self.log_variance(x).squeeze(-1),'hidden':hidden}
        if pairs:  # 256 combinations per decision, so only when source pairs are wanted.
            members=encoded[:,1:].reshape(b,t,n,-1)
            first=members[:,:,:,None].expand(-1,-1,-1,n,-1);second=members[:,:,None].expand(-1,-1,n,-1,-1)
            state=x[:,:,None,None].expand(-1,-1,n,n,-1)
            score=self.pair(torch.cat((first,second,state),-1)).squeeze(-1)
            output['pair']=(score+score.transpose(-1,-2))/2
        return output


def load_utility_checkpoint(path,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('architecture_id')!=MemoryUtilitySelector.architecture_id:
        raise ValueError(f"{path} is not a {MemoryUtilitySelector.architecture_id} checkpoint; use --selector-checkpoint for GraphSelector")
    config=dict(data['model_config'])
    config.setdefault('questions',int(data['model']['question.weight'].shape[0]))
    model=MemoryUtilitySelector(**config).to(device).eval()
    model.load_state_dict(data['model'])
    return model,data


class UtilityRuntime:
    """Learned deferral behind fixed safety checks. Anything unusual keeps the rule decision.

    Labels are independent decisions, so each call starts from a fresh recurrent state, matching
    training. Only the rule's accepted frames are questioned, and bridges are never deferred.
    """
    def __init__(self,checkpoint,device=None,seconds_budget=600.,depth_calls=256):
        import threading
        from .selector_features import MapState
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model,data=load_utility_checkpoint(checkpoint,self.device)
        if data.get('selection_protocol_id'):
            raise ValueError('Offline selection checkpoint is incompatible with online UtilityRuntime')
        self.descriptor_name=data['descriptor']
        self.encoder=frozen_encoder(self.device) if self.descriptor_name=='dinov2_vits14' else None
        self.head=data['decision_head'];self.mean=data['target_mean'][self.head];self.std=data['target_std'][self.head]
        self.limit=data['log_variance_limit']
        self.pair_trained=bool(data.get('source_head_trained'))
        self.state=MapState();self.lock=threading.RLock()
        self.seconds_budget=seconds_budget;self.calls_left=depth_calls;self.started=None;self.ranked={}

    def describe(self,frame):
        if self.encoder is not None:return image_feature(self.encoder,frame.rgb,self.device)
        from .memory_graph import thumbnail_descriptor
        return thumbnail_descriptor(cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY))

    def observe(self,frame_id,depth,K,pose,confidence=0.):
        """Called once depth exists, so later decisions see the map as it really is."""
        with self.lock:self.state.observe(frame_id,depth,K,pose,confidence)

    def sources(self,frame_id,available):
        """Best verified source pair among past frames, or None to keep the existing choice."""
        pairs=self.ranked.get(frame_id)
        if not pairs:return None
        allowed=set(available)
        return next(([a,b] for a,b,_ in pairs if a in allowed and b in allowed),None)

    @torch.inference_mode()
    def decide(self,frame,pose,stats,geometric_keep):
        from .selector_features import as_pose
        report={'policy':'utility','fallback':None}
        if not geometric_keep:return False,report
        self.started=frame.timestamp if self.started is None else self.started
        with self.lock:
            if pose.state!='tracking':return True,{**report,'fallback':f'tracking state {pose.state}'}
            descriptor=self.describe(frame)
            budget={'seconds':max(self.seconds_budget-(frame.timestamp-self.started),0.),'depth_calls':self.calls_left}
            row=self.state.features(frame,pose,budget,descriptor,stats)
            q=[torch.as_tensor(row[k],dtype=torch.float32,device=self.device)[None,None] for k in ('visual','query','token_visual','tokens')]
            question=torch.full((1,1),self.head,dtype=torch.long,device=self.device)
            output=self.model(q[0],q[1],q[2],q[3],q[3][...,-1]>0,question,pairs=self.pair_trained)
            gain=float(output['utility'][0,0,3])*self.std[3]+self.mean[3]
            log_variance=float(output['log_variance'][0,0])
            report.update(predicted_utility=gain,log_variance=log_variance)
            keep=True
            if log_variance>self.limit:report['fallback']='high uncertainty'
            elif stats.get('reason')=='rotation_bridge_with_geometric_content':report['fallback']='protected bridge'
            elif len(self.state.memory.nodes)<2:report['fallback']='map too small'
            elif gain<0:keep=False
            if keep:
                if self.pair_trained:
                    ids=row['token_ids'];scores=output['pair'][0,0].cpu().numpy()
                    ranked=sorted(((float(scores[i,j]),ids[i],ids[j]) for i in range(len(ids)) for j in range(i+1,len(ids))),reverse=True)
                    self.ranked[frame.frame_id]=[(a,b,s) for s,a,b in ranked[:8]]
                self.state.commit(frame,pose,descriptor);self.calls_left=max(self.calls_left-1,0)
        return keep,report
