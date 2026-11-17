"""Causal, opt-in image-quality deferral for optional mapping observations.

Depth networks can output depth from blurred RGB; availability is not reliability.
This adapter uses denoised focus and ORB support relative to earlier accepted
observations. It is a rule safeguard, not a newly trained blur classifier.
"""
from collections import deque

import cv2
import numpy as np


def image_evidence(rgb):
    image=np.asarray(rgb)
    if image.ndim!=3 or image.shape[2]!=3 or not image.size or not np.isfinite(image).all():
        return {'observable':False,'reason':'invalid_rgb'}
    if image.dtype!=np.uint8:
        return {'observable':False,'reason':'expected_uint8_rgb'}
    if np.all(image==0) or np.all(image==255):
        return {'observable':False,'reason':'blank_rgb'}
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray=cv2.resize(gray,(256,max(8,round(gray.shape[0]*256/gray.shape[1]))),interpolation=cv2.INTER_AREA)
    smooth=cv2.GaussianBlur(gray,(3,3),.8)
    texture=float(smooth.var())
    lap=float(cv2.Laplacian(smooth,cv2.CV_64F).var())
    features=cv2.ORB_create(nfeatures=500).detect(smooth,None)
    return {'observable':True,'reason':None,'focus':lap/max(texture,1.),
            'denoised_laplacian':lap,'texture':texture,'orb_features':len(features),
            'raw_laplacian':float(cv2.Laplacian(gray,cv2.CV_64F).var())}


class MappingQualityGate:
    def __init__(self,warmup=4,history=32,focus_ratio=.35,feature_ratio=.5,
                 max_deferrals=3,max_gap_seconds=1.):
        if warmup<1 or history<warmup or not 0<focus_ratio<1 or not 0<feature_ratio<1 or max_deferrals<1 or max_gap_seconds<=0:
            raise ValueError('Invalid mapping-quality settings')
        self.reference=deque(maxlen=history)
        self.warmup,self.focus_ratio,self.feature_ratio=warmup,focus_ratio,feature_ratio
        self.maximum,self.gap=max_deferrals,max_gap_seconds
        self.last_timestamp=None
        self.deferrals=0
        self.counts={'checked':0,'deferred':0,'protected':0,'unobservable':0,'accepted':0,'bridge_only':0}

    def propose(self,frame,pose,stats,geometric_keep):
        if not geometric_keep:
            return False,{'mapping_quality':{'checked':False,'reason':'geometric_policy_rejected'}}
        if pose.state!='tracking':
            return False,{'mapping_quality':{'checked':False,'reason':'tracking_unavailable'}}
        metrics=image_evidence(frame.rgb)
        self.counts['checked']+=1
        detail={'checked':True,**metrics,'suspected_blur':False,'protected_reason':None,
                'baseline_observations':len(self.reference),'baseline_updated':False,'dense_eligible':bool(metrics['observable'])}
        if not metrics['observable']:
            self.counts['unobservable']+=1;self.counts['deferred']+=1
            self.deferrals+=1
            return False,{'mapping_quality':detail}
        detail['baseline_eligible']=True
        poor=False
        if len(self.reference)>=self.warmup:
            focus=float(np.median([r['focus'] for r in self.reference]))
            tracks=float(np.median([r['orb_features'] for r in self.reference]))
            detail.update(focus_relative=metrics['focus']/max(focus,1e-6),
                          features_relative=metrics['orb_features']/max(tracks,1.))
            # Low texture alone cannot establish blur. Require a relative loss
            # of focus AND feature support against a textured history.
            poor=(focus>0 and tracks>=20 and metrics['texture']>=4 and
                  detail['focus_relative']<self.focus_ratio and detail['features_relative']<self.feature_ratio)
        if poor:
            detail.update(suspected_blur=True,baseline_eligible=False,dense_eligible=False,reason='relative_focus_and_feature_loss')
            protected=(stats.get('reason') in ('first_tracked_frame','rotation_bridge_with_geometric_content','relocalization_bridge')
                       or bool(stats.get('protected_bridge')))
            gap=self.last_timestamp is not None and frame.timestamp-self.last_timestamp>=self.gap
            if protected or gap or self.deferrals>=self.maximum:
                detail['protected_reason']='geometric_bridge' if protected else 'mapping_gap_limit' if gap else 'deferral_limit'
                self.counts['protected']+=1
                return True,{'mapping_quality':detail}
            self.deferrals+=1;self.counts['deferred']+=1
            return False,{'mapping_quality':detail}
        detail['reason']='warmup' if len(self.reference)<self.warmup else 'quality_supported'
        return True,{'mapping_quality':detail}

    def commit(self,frame,detail):
        """Call only after the actual final mapping decision, including learned vetoes."""
        self.last_timestamp=frame.timestamp
        self.deferrals=0
        self.counts['accepted']+=1
        evidence=detail.get('mapping_quality',{})
        if not evidence.get('dense_eligible',True):self.counts['bridge_only']+=1
        if evidence.get('baseline_eligible'):
            self.reference.append({k:evidence[k] for k in ('focus','orb_features')})
            evidence['baseline_updated']=True

    def summary(self):
        return {'enabled':True,'kind':'causal_relative_focus_and_features_v1',**self.counts,
                'thresholds_provisional':True,'warmup':self.warmup,'focus_ratio':self.focus_ratio,
                'feature_ratio':self.feature_ratio,'max_deferrals':self.maximum,'max_gap_seconds':self.gap,
                'scope':'Tracker input unchanged; protected poor-quality cameras are pose-only and excluded from dense sources/fusion'}
