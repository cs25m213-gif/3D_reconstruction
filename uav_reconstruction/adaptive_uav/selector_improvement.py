"""Opt-in clip policy for whole-selection oracle supervision.

Reuses memory_utility_v1, causal MapState, the context-versioned student provider
and ordinary TSDF. Full-clip estimated poses/anchors still qualify this as offline
development. No reference arrays enter candidate features or policy decisions.
"""
from __future__ import annotations

from collections import OrderedDict
import time

import numpy as np
import torch

from .counterfactual_replay import fuse
from .memory_graph import MIN_VERIFIED_INLIERS
from .reconstruction_quality import DEFAULT_WEIGHTS, quality, surface_metrics
from .selector_features import MapState, as_pose
from .selector_policy import collate, windows
from .utility_contracts import QUERY_FEATURES

PROTOCOL = 'selected_context_oracle_v1'
SHARED = QUERY_FEATURES.index('best_shared_tracks')
ARRAY_KEYS = ('visual', 'query', 'token_visual', 'tokens')


def candidate_positions(window, maximum=5):
    """Bounded evenly spaced candidates, including both ends (and no duplicates)."""
    if maximum < 2:
        raise ValueError('Need at least two candidates to cover a window')
    window = np.asarray(window, dtype=np.int64)
    if not len(window):
        return []
    return window[np.unique(np.rint(np.linspace(0, len(window)-1, min(maximum, len(window)))).astype(int))].tolist()


def overlap_eligibility(rows, has_history, minimum=MIN_VERIFIED_INLIERS):
    """Prefer verified edges; if none exist retain one deterministic fallback.

    An unsupported edge is logged, never called a verified bridge. No new frame
    is inserted outside the mapping budget; mandatory tracking is separate.
    """
    if not rows:
        raise ValueError('No candidates')
    if not has_history or minimum <= 0:
        return np.ones(len(rows), bool), False
    shared = np.asarray([r['query'][SHARED]*500. for r in rows])
    mask = shared >= minimum-1e-4
    if mask.any():
        return mask, False
    mask[int(np.argmax(shared))] = True
    return mask, True


def describe_options(state, clip, positions, descriptors, remaining):
    budget = {'seconds':60.*remaining, 'depth_calls':remaining}
    return [state.features(clip.views[p], as_pose(clip.views[p],clip.views[p].feature_pose), budget,
                           descriptors.get(clip.name,clip.views[p].index,clip.views[p].rgb)) for p in positions]


def append_selected(state, history, view, provider, descriptors, clip_name):
    """Depth comes from selected earlier observations, not a pre-selection stream."""
    dense, seconds, hit = provider(view,view.feature_pose,history)
    state.commit(dense,as_pose(dense,dense.feature_pose),descriptors.get(clip_name,dense.index,dense.rgb))
    state.observe(dense.frame_id,dense.depth_m,dense.K,dense.feature_pose,dense.confidence)
    history.append((dense,dense.feature_pose))
    return seconds, hit


class ContextScorer:
    """Private per-clip, bounded caches for full selected-set geometry scores."""
    def __init__(self, provider, reference, voxel, max_scores=256):
        self.provider,self.reference,self.voxel=provider,reference,voxel
        self.max_scores,self.cache=max_scores,OrderedDict()
        self.fusions=0

    def score(self, chosen):
        ids=tuple(v.frame_id for v in chosen)
        if len(ids)!=len(set(ids)) or any(a.timestamp>=b.timestamp for a,b in zip(chosen,chosen[1:])):
            raise ValueError('Selection must be unique and strictly chronological')
        if ids in self.cache:
            self.cache.move_to_end(ids)
            return dict(self.cache[ids])
        history=[]
        started=time.monotonic()
        for view in chosen:
            dense,_,_=self.provider(view,view.feature_pose,history)
            history.append((dense,view.feature_pose))
        # Fixed reference poses isolate fusion selection; no fresh registration claim.
        points=fuse([(v,v.reference_pose) for v,_ in history],self.voxel)
        metrics=surface_metrics(points,self.reference)
        result={**metrics,'quality':quality(metrics,None,DEFAULT_WEIGHTS),'keyframes':len(chosen),
                'score_seconds':time.monotonic()-started}
        self.fusions+=1
        self.cache[ids]=result
        if len(self.cache)>self.max_scores:
            self.cache.popitem(last=False)
        return dict(result)


class WindowUtilityPolicy:
    """Same network; labels, candidate distribution and depth contexts now agree.

    Full-window deliberation uses known clip boundaries and future images within
    a buffered window, so this is an offline clip policy, not a streaming tracker.
    """
    def __init__(self,model,data,device,provider,guard=True,maximum=5,allow_legacy=False):
        if not allow_legacy and data.get('selection_protocol_id')!=PROTOCOL:
            raise ValueError('Need a selected-context oracle checkpoint')
        self.model,self.data,self.device,self.provider=model,data,device,provider
        self.guard,self.maximum=guard,maximum
        self.trace=[]

    def scores(self,rows):
        with torch.no_grad():
            return self.model(*collate([{**r,'question':self.data['decision_head']} for r in rows],self.device))['utility'][:,0,3].cpu().numpy()

    def __call__(self,clip,count,descriptors):
        state,history=MapState(),[]
        self.trace=[]
        for pos,window in enumerate(windows(clip,count)):
            positions=candidate_positions(window,self.maximum)
            rows=describe_options(state,clip,positions,descriptors,count-pos)
            mask,unsupported=overlap_eligibility(rows,bool(history),MIN_VERIFIED_INLIERS if self.guard else 0)
            scores=self.scores(rows)
            finite=mask & np.isfinite(scores)
            fallback=not finite.any()
            pick=int(np.argmax(np.where(finite,scores,-np.inf))) if not fallback else int(np.flatnonzero(mask)[0])
            view=clip.views[positions[pick]]
            append_selected(state,history,view,self.provider,descriptors,clip.name)
            self.trace.append({'window':pos,'candidate_indices':[clip.views[p].index for p in positions],
                               'chosen':view.index,'eligible':mask.tolist(),'unsupported_overlap':unsupported,
                               'nonfinite_fallback':fallback,'source_ids':list(history[-1][0].source_ids)})
        return [v for v,_ in history]


def oracle_index(scores,eligible):
    scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Oracle geometry returned a nonfinite score')
    if not np.asarray(eligible).any():
        raise ValueError('Oracle has no feasible alternative')
    return int(np.argmax(np.where(eligible,scores,-np.inf)))


def require_training_role(role):
    if role!='train':
        raise ValueError('Oracle labels may only be generated on training flights')
