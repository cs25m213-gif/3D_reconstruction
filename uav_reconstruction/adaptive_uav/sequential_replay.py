"""Bounded three-action replay around the existing PnP, student and TSDF adapters.

All branches receive mandatory tracking. SKIP defers optional map work;
TRACK_ONLY retains projected existing landmarks without a new depth observation;
NEW_KEYFRAME also infers/fuses depth. Future contexts use each branch's own
committed images and poses. Initial SfM poses/anchors remain full-clip evidence:
this is explicitly an offline development experiment, never an online gate.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .counterfactual_replay import integration_seconds, register, score
from .selector_features import MapState, as_pose
from .student_inference import anchor_scale, predict
from .utility_contracts import Budget, state_identity, validate_record, write_json
from .remote_geometry import geometry_metrics_batch, routing_provenance
from .reconstruction_quality import pose_error, quality
from types import SimpleNamespace

ACTIONS = ('SKIP', 'TRACK_ONLY', 'NEW_KEYFRAME')
VERSION = 'sequential_pnp_v1'


def score_branches(branches, reference, voxel, truth, weights):
    """Only final geometry is parallel: preserve branch poses and measured action costs."""
    names = list(branches)
    choices = [[SimpleNamespace(fusion_depth=v.fusion_depth, fusion_rgb=v.fusion_rgb,
                                fusion_k=v.fusion_k, reference_pose=p)
                for v, p in branches[name].observations] for name in names]
    def local():
        return [score(branches[name].observations, reference, voxel,
                      branches[name].poses, truth, weights)['metrics'] for name in names]
    metrics = geometry_metrics_batch(choices, reference, voxel, local)
    result = {}
    for name, measured in zip(names, metrics):
        poses = branches[name].poses
        pose = pose_error(poses, truth) if poses else None
        result[name] = {'metrics': measured, 'pose': pose, 'quality': quality(measured, pose, weights)}
    return result


def copy_view(view):
    # RGB is immutable input; mutable depth/pose state is owned by a branch.
    return replace(view, depth_m=view.depth_m.copy(), fusion_depth=view.fusion_depth.copy(),
                   K=view.K.copy(), fusion_k=view.fusion_k.copy(), feature_pose=view.feature_pose.copy(),
                   feature_k=view.feature_k.copy(), anchors_m=None if view.anchors_m is None else view.anchors_m.copy())


class StudentDepthProvider:
    """Context-versioned immutable cache. Reuse equal work, charge its measured cost.

    CPU timings describe CPU experiments; they are not asserted GPU deployment costs.
    """
    def __init__(self, model, device='cpu', max_entries=128):
        self.model, self.device, self.max_entries = model, device, max_entries
        self.cache, self.calls = {}, 0

    def __call__(self, view, pose, history):
        sources = [(v, p) for v, p in history if v.timestamp < view.timestamp][-2:]
        anchors = view.anchors_m
        scale = anchor_scale(anchors)
        digest = hashlib.sha256(np.asarray(pose).tobytes()+np.asarray(anchors).tobytes()+np.asarray(view.feature_k).tobytes())
        for v, p in sources:
            digest.update(v.frame_id.encode()+np.asarray(p).tobytes()+np.asarray(v.feature_k).tobytes())
        key = (view.frame_id, digest.hexdigest())
        if key in self.cache:
            result, seconds = self.cache[key]
            return copy_view(result), seconds, True
        started = time.monotonic()
        depth, confidence = predict(self.model, [(view.rgb, view.feature_k, pose)] +
                                    [(v.rgb, v.feature_k, p) for v, p in reversed(sources)],
                                    scale, self.device, anchors=anchors)
        seconds = time.monotonic() - started
        h, w = view.rgb.shape[:2]
        k = view.feature_k.copy()
        k[0] *= depth.shape[1] / w
        k[1] *= depth.shape[0] / h
        result = replace(view, depth_m=cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST),
                         K=view.feature_k.copy(), feature_pose=np.asarray(pose).copy(),
                         fusion_depth=depth.copy(), fusion_k=k,
                         fusion_rgb=cv2.resize(view.rgb, depth.shape[::-1], interpolation=cv2.INTER_AREA),
                         confidence=float(np.median(confidence)), source_ids=tuple(v.frame_id for v, _ in sources))
        self.calls += 1
        if len(self.cache) >= self.max_entries:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (copy_view(result), seconds)
        return result, seconds, False


def projected_tracking_view(view, pose, history):
    """Project already observed landmarks; never invent candidate depth or fuse it."""
    h, w = view.rgb.shape[:2]
    depth = np.full(h*w, np.inf, np.float32)
    inverse = np.linalg.inv(pose)
    for older, old_pose in history[-3:]:
        ys, xs = np.mgrid[0:older.depth_m.shape[0]:4, 0:older.depth_m.shape[1]:4]
        z = older.depth_m[ys, xs]
        valid = np.isfinite(z) & (z > 0)
        points = np.c_[(xs[valid]-older.K[0, 2])/older.K[0, 0]*z[valid],
                       (ys[valid]-older.K[1, 2])/older.K[1, 1]*z[valid], z[valid]]
        world = points @ old_pose[:3, :3].T + old_pose[:3, 3]
        current = world @ inverse[:3, :3].T + inverse[:3, 3]
        current = current[np.isfinite(current).all(1) & (current[:, 2] > 0)]
        if not len(current):
            continue
        x = np.floor(view.feature_k[0, 0]*current[:, 0]/current[:, 2]+view.feature_k[0, 2]).astype(int)
        y = np.floor(view.feature_k[1, 1]*current[:, 1]/current[:, 2]+view.feature_k[1, 2]).astype(int)
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        np.minimum.at(depth, y[inside]*w+x[inside], current[inside, 2])
    depth[~np.isfinite(depth)] = 0
    return replace(view, depth_m=depth.reshape(h, w), K=view.feature_k.copy(),
                   feature_pose=np.asarray(pose).copy(), fusion_depth=np.zeros_like(view.fusion_depth),
                   source_ids=tuple(v.frame_id for v, _ in history[-3:]))


class Branch:
    def __init__(self, prefix, limits, descriptors, clip_name, observations=None, voxel=.5):
        self.history = [(copy_view(v), np.asarray(p).copy()) for v, p in prefix]
        self.observations = [(copy_view(v), np.asarray(p).copy()) for v, p in (prefix if observations is None else observations)]
        self.voxel = voxel
        self.budget, self.memory = Budget(**limits), MapState()
        self.descriptors, self.clip_name = descriptors, clip_name
        self.poses, self.failures, self.depth_contexts = {}, 0, []
        self.hidden = None  # Frozen rule rollout; future learned rollouts must fork this too.
        self.last_pose = prefix[-1][1].copy()
        self.status, self.shared_seconds = 'complete', 0.
        self.optional_seconds, self.cache_hits = 0., 0
        dense_ids = {v.frame_id for v, _ in self.observations}
        for v, p in self.history:
            self.memory.commit(v, as_pose(v, p), self.descriptor(v))
            if v.frame_id in dense_ids:
                self.memory.observe(v.frame_id, v.depth_m, v.K, p, v.confidence)

    def descriptor(self, view):
        return self.descriptors.get(self.clip_name, view.index, view.rgb)

    def fork(self):
        other=object.__new__(Branch)
        other.__dict__=self.__dict__.copy()
        other.history=[(copy_view(v),p.copy()) for v,p in self.history]
        other.observations=[(copy_view(v),p.copy()) for v,p in self.observations]
        other.memory=self.memory.fork()
        other.budget=Budget(**self.budget.initial)
        other.budget.spent = self.budget.spent.copy()
        other.hidden = copy.deepcopy(self.hidden)
        other.last_pose = self.last_pose.copy()
        other.poses, other.depth_contexts = copy.deepcopy(self.poses), copy.deepcopy(self.depth_contexts)
        other.failures, other.status = self.failures, self.status
        other.shared_seconds, other.optional_seconds = self.shared_seconds, self.optional_seconds
        return other

    def commit(self, view, pose, dense):
        self.history.append((view, np.asarray(pose).copy()))
        self.history = self.history[-64:]
        self.memory.commit(view, as_pose(view, pose), self.descriptor(view))
        if dense:
            self.observations.append((view, np.asarray(pose).copy()))
            self.memory.observe(view.frame_id, view.depth_m, view.K, pose, view.confidence)

    def probe(self, view, min_inliers):
        pairs = min(3, int(self.budget.remaining('matches')))
        if not pairs or self.budget.remaining('seconds') <= 0:
            return {'ok':False, 'inliers':0, 'pose':None, 'matches':0}, 0.
        started = time.monotonic()
        found = register(view, self.history[-pairs:], min_inliers)
        return found, time.monotonic()-started

    def act(self, view, action, found, probe_seconds, provider):
        if action not in ACTIONS:
            raise ValueError(f'Unknown replay action {action}')
        if not self.budget.feasible(seconds=probe_seconds, matches=found.get('matches', 0)):
            self.status = 'over_budget'
            return
        self.budget.charge(seconds=probe_seconds, matches=found.get('matches', 0))
        self.shared_seconds += probe_seconds
        if not found['ok']:
            self.failures += 1
            return
        pose = found['pose']
        self.last_pose = pose.copy()
        self.poses[view.index] = pose.copy()
        if action == 'SKIP':
            return  # Mandatory tracking already ran, but optional map state is untouched.
        started = time.monotonic()
        if action == 'NEW_KEYFRAME':
            if self.budget.remaining('depth_calls') < 1:
                self.status = 'over_budget'
                return
            prepared, inference_seconds, cached = provider(view, pose, self.history)
            self.cache_hits += int(cached)
            self.budget.charge(seconds=inference_seconds, depth_calls=1)
            self.optional_seconds += inference_seconds
            self.depth_contexts.append({'frame':view.frame_id, 'sources':list(prepared.source_ids)})
            started = time.monotonic()  # Cached inference still costs its uncached measured work.
        else:
            prepared = projected_tracking_view(view, pose, self.history)
        if self.budget.spent['seconds'] > self.budget.initial['seconds']:
            self.status = 'over_budget'
            return
        if action == 'NEW_KEYFRAME':
            integration_seconds(prepared, pose, self.voxel)
        self.commit(prepared, pose, action == 'NEW_KEYFRAME')
        work = time.monotonic()-started
        self.budget.charge(seconds=work)
        self.optional_seconds += work
        if self.budget.spent['seconds'] > self.budget.initial['seconds']:
            self.status = 'over_budget'

    def continue_with(self, future, provider, min_inliers, baseline_ratio):
        for view in future:
            if self.status == 'over_budget' or self.budget.remaining('seconds') <= 0:
                break
            found, seconds = self.probe(view, min_inliers)
            action = 'TRACK_ONLY'
            if found['ok'] and self.budget.remaining('depth_calls'):
                last, pose = self.observations[-1]
                valid = last.depth_m[np.isfinite(last.depth_m) & (last.depth_m > 0)]
                ratio = np.linalg.norm(found['pose'][:3, 3]-pose[:3, 3])/np.median(valid) if valid.size else 0.
                if ratio >= baseline_ratio:
                    action = 'NEW_KEYFRAME'
            self.act(view, action, found, seconds, provider)


def run_sequential_clip(clip, settings, reference, recorder, descriptors, output, limit, provider):
    from .counterfactual_replay import plan
    report = {'states':0, 'decisions':0, 'branches':0, 'replay':VERSION,
              'causal_online_evidence':False, 'reason':'Bootstrap poses and RGB anchors use full-clip SfM'}
    for prefix, candidate, future in plan(clip, settings):
        for budget_id, limits in settings['budgets'].items():
            if report['decisions'] >= limit:
                return report
            # Same initial map for all actions. No cached DA3 depth enters this bootstrap.
            prepared = []
            try:
                for view in prefix:
                    pose = view.aligned_pose if view.aligned_pose is not None else view.feature_pose
                    dense, _, _ = provider(view, pose, prepared)
                    prepared.append((dense, pose))
            except (ValueError, OSError, MemoryError) as exc:
                report.setdefault('skipped', []).append({'candidate':candidate.index,'budget':budget_id,'error':str(exc)})
                report['decisions'] += 1
                continue
            initial = Branch(prepared, limits, descriptors, clip.name, voxel=settings['voxel_m'])
            found, seconds = initial.probe(candidate, settings['registration_min_inliers'])
            estimated = found['pose'] if found['ok'] else initial.last_pose
            row = initial.memory.features(candidate, as_pose(candidate, estimated), initial.budget.state(), initial.descriptor(candidate))
            state = state_identity(clip=clip.name, prefix=[v.index for v in prefix], budget=limits,
                                   backend=recorder.backend, reference=reference.identity(), replay=VERSION)
            write_json(output/'snapshots'/f'{state}.json', {
                'state_id':state, 'clip':clip.name, 'prefix':[v.frame_id for v,_ in prepared],
                'poses':[p.tolist() for _,p in prepared], 'contexts':[list(v.source_ids) for v,_ in prepared],
                'budget':initial.budget.to_dict(), 'backend':recorder.backend,
                'candidate':candidate.frame_id, 'future':[v.frame_id for v in future],
                'replay':VERSION, 'causal_online_evidence':False,
                'restore':'Rebuild prefix in recorded order with pinned backend; no native optimizer snapshot is claimed'})
            feature = output/'features'/f'{state}_{candidate.index}_{budget_id}.npz'
            np.savez_compressed(feature, **{k:v for k,v in row.items() if k!='token_ids'}, token_ids=np.array(row['token_ids']))
            branches = {}
            for action in ACTIONS:
                branch = initial.fork()
                try:
                    branch.act(candidate, action, found, seconds, provider)
                    branch.continue_with(future, provider, settings['registration_min_inliers'], settings['keyframe_baseline_ratio'])
                except (ValueError, OSError, MemoryError) as exc:
                    branch.status = 'censored'
                    report.setdefault('censored', []).append({'candidate':candidate.index, 'action':action, 'error':str(exc)})
                branches[action] = branch
            truth = {v.index:v.reference_pose for v in [candidate]+future}
            results = score_branches(branches, reference, settings['voxel_m'], truth, recorder.weights)
            report['geometry_execution'] = routing_provenance()
            control = branches['SKIP']
            for action, intervention in [('NEW_KEYFRAME','add'), ('TRACK_ONLY','track_only'), ('SKIP','skip')]:
                branch = branches[action]
                identity = state_identity(state=state, candidate=candidate.index, action=action)
                if identity in recorder.done:
                    continue
                delta_cost = branch.budget.spent['seconds']-control.budget.spent['seconds']
                connectivity = (control.failures-branch.failures)/max(1,len(future)+1)
                before, after = results['SKIP'], results[action]
                target = after['quality']-before['quality']+recorder.weights['connectivity']*connectivity
                target -= recorder.weights['cost_per_budget_fraction']*float(np.clip(delta_cost/max(limits['seconds'],1e-3),-10,10))
                record = {'schema_version':3, 'experiment_id':identity, 'state_id':state,
                          'candidate':candidate.index, 'action':action, 'control_action':'SKIP',
                          'intervention':intervention, 'fidelity':'R_ROLLOUT', 'budget_id':budget_id,
                          'clip':clip.name, 'flight':clip.scene, 'split':settings['_role'],
                          'evaluation_protocol_id':reference.protocol, 'reference_identity':reference.identity(),
                          'reference_status':recorder.status, 'execution_status':branch.status,
                          'control':before, 'intervened':after, 'features':str(feature),
                          'prefix':[v.index for v in prefix], 'scalarization':recorder.weights['version'],
                          'cost':{'incremental_seconds':delta_cost, 'control':control.budget.to_dict(),
                                  'intervention':branch.budget.to_dict(), 'scope':'measured replay backend; shared prefix excluded'},
                          'targets':{'delta_coverage':after['metrics']['recall']-before['metrics']['recall'],
                                     'delta_surface':after['quality']-before['quality'], 'delta_connectivity':connectivity,
                                     'delta_cost':delta_cost, 'utility':target},
                          'detail':{**recorder.backend, 'replay':VERSION, 'causal_online_evidence':False,
                                    'local_bundle_adjustment':False, 'mandatory_tracking_in_all_branches':True,
                                    'candidate_registration_failed':not found['ok'], 'control_failures':control.failures,
                                    'intervention_failures':branch.failures, 'horizon_frames':len(future),
                                    'depth_contexts':branch.depth_contexts, 'reference_used_for_pose_alignment':True}}
                validate_record(record)
                recorder.stream.write(json.dumps(record,allow_nan=False)+'\n')
                recorder.stream.flush()
                recorder.written += 1
            report['states'] += 1
            report['decisions'] += 1
            report['branches'] += 3
    return report
