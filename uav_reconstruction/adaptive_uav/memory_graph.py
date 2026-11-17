"""Stage 3: bounded causal keyframe memory and verified covisibility graph.

Appearance only proposes pairs: an edge needs geometric evidence. Images are evicted
independently of graph identity, and bridges cannot be lost to eviction. `replay`
runs in shadow mode and must reproduce the unchanged rule decisions.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .utility_contracts import load_contract, sha256_lines, state_identity, write_json, write_status

LIMITS = {'active_keyframes': 64, 'recent_candidates': 4, 'historical_candidates': 12, 'verified_pair_budget': 16,
          'policy_tokens': 16, 'descriptor_shard': 4096, 'candidate_buffer': 8}
MIN_VERIFIED_INLIERS = 20


@dataclass
class Node:
    frame_id: str
    timestamp: float
    descriptor: np.ndarray
    pose: np.ndarray
    component: int
    observation_count: int = 0
    depth_confidence: float = 0.


@dataclass
class Edge:
    shared_tracks: int
    parallax_px: float
    timestamp: float
    causal: bool = True
    pose_version: int = 0


@dataclass
class Counters:
    lookups: int = 0
    matcher_calls: int = 0
    evictions: int = 0
    descriptors_spilled: int = 0
    rejected_future: int = 0
    rejected_merges: int = 0


class CausalMemory:
    def __init__(self, limits: dict | None = None):
        self.limits = {**LIMITS, **(limits or {})}
        self.nodes: dict[str, Node] = OrderedDict()
        self.edges: dict[tuple, Edge] = {}
        self.residency: OrderedDict[str, object] = OrderedDict()
        self.cutoff = -np.inf
        self.next_component = 0
        self.version = 0
        self.counters = Counters()

    def add(self, frame_id: str, timestamp: float, descriptor, pose, payload=None, component: int | None = None) -> Node:
        if timestamp < self.cutoff:
            raise ValueError(f'{frame_id} is earlier than the causal cutoff')
        if frame_id in self.nodes:
            raise ValueError(f'{frame_id} is already in memory')
        if component is None:
            component = self.next_component
            self.next_component += 1
        self.next_component = max(self.next_component, component + 1)
        node = Node(frame_id, float(timestamp), np.asarray(descriptor, np.float32), np.asarray(pose, np.float64), component)
        self.nodes[frame_id] = node
        self.cutoff = float(timestamp)
        self.version += 1
        if payload is not None:
            self.residency[frame_id] = payload
        self._enforce_bounds()
        return node

    def retrieve(self, descriptor, timestamp: float) -> list[str]:
        """Bounded proposals, recent first. Later evidence can never appear here."""
        self.counters.lookups += 1
        visible = [n for n in self.nodes.values() if n.timestamp <= timestamp]
        self.counters.rejected_future += len(self.nodes) - len(visible)
        recent = [n.frame_id for n in visible[-self.limits['recent_candidates']:]]
        older = [n for n in visible[:-self.limits['recent_candidates']] if n.descriptor is not None and n.descriptor.size]
        scored = []
        if older:
            query = np.asarray(descriptor, np.float32)
            query = query / (np.linalg.norm(query) + 1e-9)
            matrix = np.stack([n.descriptor for n in older])
            matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
            order = np.argsort(-(matrix @ query), kind='stable')[:self.limits['historical_candidates']]
            scored = [older[i].frame_id for i in order]
        return (recent[::-1] + scored)[:self.limits['verified_pair_budget']]

    def connect(self, first: str, second: str, shared: int, parallax: float, timestamp: float,
                causal: bool = True, aligned: bool = False) -> bool:
        if not causal:
            raise ValueError('Non-causal evidence cannot enter the online graph')
        a, b = self.nodes[first], self.nodes[second]
        if shared < MIN_VERIFIED_INLIERS:
            return False
        if a.component != b.component:
            if not aligned:
                # Retrieval across components can suggest relocalization, never a merge.
                self.counters.rejected_merges += 1
                return False
            keep, drop = sorted((a.component, b.component))
            for node in self.nodes.values():
                if node.component == drop:
                    node.component = keep
        self.edges[tuple(sorted((first, second)))] = Edge(shared, parallax, timestamp)
        self.version += 1
        return True

    def neighbours(self) -> dict:
        table = defaultdict(set)
        for a, b in self.edges:
            table[a].add(b)
            table[b].add(a)
        return table

    def bridges(self) -> set[str]:
        """Articulation nodes: removing one would split a verified component."""
        table, order, low, found = self.neighbours(), {}, {}, set()
        for root in self.nodes:
            if root in order:
                continue
            stack = [(root, None, iter(sorted(table[root])))]
            order[root] = low[root] = len(order)
            children = 0
            while stack:
                node, parent, remaining = stack[-1]
                advanced = False
                for other in remaining:
                    if other == parent:
                        continue
                    if other in order:
                        low[node] = min(low[node], order[other])
                        continue
                    order[other] = low[other] = len(order)
                    stack.append((other, node, iter(sorted(table[other]))))
                    children += parent is None
                    advanced = True
                    break
                if not advanced:
                    stack.pop()
                    if parent is not None:
                        low[parent] = min(low[parent], low[node])
                        if low[node] >= order[parent] and stack[-1][1] is not None:
                            found.add(parent)
            if children > 1:
                found.add(root)
        return found

    def _enforce_bounds(self) -> None:
        protected = self.bridges()
        while len(self.residency) > self.limits['active_keyframes']:
            victim = next((name for name in self.residency if name not in protected), next(iter(self.residency)))
            del self.residency[victim]  # Image/depth only; identity and graph stay.
            self.counters.evictions += 1
        with_descriptor = [n for n in self.nodes.values() if n.descriptor.size]
        for node in with_descriptor[:max(len(with_descriptor) - self.limits['descriptor_shard'], 0)]:
            node.descriptor = np.empty(0, np.float32)
            self.counters.descriptors_spilled += 1

    def tokens(self, reference_time: float) -> list[Node]:
        """The bounded policy view: bridges first, then the most recent."""
        bridges = self.bridges()
        protected = [n for n in self.nodes.values() if n.frame_id in bridges]
        recent = [n for n in reversed(self.nodes.values()) if n not in protected]
        return (protected + recent)[:self.limits['policy_tokens']]

    def components(self) -> int:
        return len({n.component for n in self.nodes.values()})

    def snapshot(self, budget: dict, extra: dict | None = None) -> dict:
        return {'state_id': state_identity(nodes=list(self.nodes), edges=sorted(map(list, self.edges)),
                                           version=self.version, budget=budget, extra=extra or {}),
                'cutoff': self.cutoff, 'keyframes': list(self.nodes), 'memory_version': self.version,
                'edges': len(self.edges), 'components': self.components(), 'budget': budget,
                'resident': len(self.residency), 'counters': self.counters.__dict__.copy()}


def detect(gray: np.ndarray, features: int = 1500) -> tuple:
    return cv2.ORB_create(features).detectAndCompute(gray, None)


def verify_features(first: tuple, second: tuple, k: np.ndarray) -> dict:
    """Geometric check with the same ORB/essential evidence for every candidate pair."""
    points_a, descriptors_a = first
    points_b, descriptors_b = second
    if descriptors_a is None or descriptors_b is None or len(points_a) < 8 or len(points_b) < 8:
        return {'shared_tracks': 0, 'parallax_px': 0.}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(descriptors_a, descriptors_b)
    if len(pairs) < 8:
        return {'shared_tracks': 0, 'parallax_px': 0.}
    a = np.float32([points_a[m.queryIdx].pt for m in pairs])
    b = np.float32([points_b[m.trainIdx].pt for m in pairs])
    _, inliers = cv2.findEssentialMat(a, b, k, cv2.RANSAC, .999, 1.5)
    if inliers is None:
        return {'shared_tracks': 0, 'parallax_px': 0.}
    good = inliers.ravel().astype(bool)
    moved = float(np.median(np.linalg.norm(a[good] - b[good], axis=1))) if good.any() else 0.
    return {'shared_tracks': int(good.sum()), 'parallax_px': moved}


def verify_pair(gray_a: np.ndarray, gray_b: np.ndarray, k: np.ndarray) -> dict:
    return verify_features(detect(gray_a), detect(gray_b), k)


def thumbnail_descriptor(gray: np.ndarray) -> np.ndarray:
    """384-D appearance stand-in for CPU runs. Recorded by name, never called DINO."""
    small = cv2.resize(gray, (24, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (small - small.mean()) / (small.std() + 1e-6)


@dataclass
class ShadowFrame:
    frame_id: str
    timestamp: float
    rgb: np.ndarray
    K: np.ndarray
    extra: dict = field(default_factory=dict)


def shadow_clip(frames, policy_factory, poses, descriptor, limits=None) -> dict:
    """Same frames through the rule path and through propose/commit; decisions must match."""
    from .keyframe_policy import KeyframePolicy  # noqa: F401
    legacy, shadow = policy_factory(), policy_factory()
    memory, states, mismatches, started = CausalMemory(limits), [], 0, time.monotonic()
    for frame, pose in zip(frames, poses):
        before = memory.snapshot({}, {'frame': frame.frame_id})
        kept, _ = legacy.decide(frame, pose)
        proposed, stats = shadow.propose(frame, pose)
        mismatches += kept != proposed
        if proposed:
            shadow.commit(frame, pose)
            gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
            value = descriptor(gray)
            candidates = memory.retrieve(value, frame.timestamp)
            # Tracking continuity decides the component. Appearance only proposes pairs.
            component = next(reversed(memory.nodes.values())).component if memory.nodes else None
            found = detect(gray)
            memory.add(frame.frame_id, frame.timestamp, value, pose.T_world_from_camera, found, component)
            for name in candidates[:memory.limits['verified_pair_budget']]:
                other = memory.residency.get(name)
                if other is None:
                    continue
                memory.counters.matcher_calls += 1
                result = verify_features(other, found, frame.K)
                memory.connect(name, frame.frame_id, result['shared_tracks'], result['parallax_px'], frame.timestamp)
        states.append({**before, 'frame': frame.frame_id, 'action': 'NEW_KEYFRAME' if proposed else 'TRACK_ONLY',
                       'reason': stats.get('reason')})
    return {'states': states, 'mismatches': mismatches, 'keyframes': len(memory.nodes), 'edges': len(memory.edges),
            'components': memory.components(), 'bridges': sorted(memory.bridges()),
            'seconds': time.monotonic() - started, 'counters': memory.counters.__dict__.copy(),
            'memory_digest': sha256_lines(f'{a}|{b}' for a, b in sorted(memory.edges))}


def replay(contract: dict, root: Path, output: Path, clip_limit: int, frame_limit: int) -> dict:
    from .keyframe_policy import KeyframePolicy
    from .runtime_types import PoseEstimate
    output.mkdir(parents=True, exist_ok=True)
    write_status(output, 'stage3_memory', 'running', None, None)
    rows, cache_root = [], root / contract['data']['cache_root']
    batches = sorted(p for p in cache_root.iterdir() if (p / 'selected_infos.json').exists() and 'GNSS' not in p.name)
    with (output / 'states.jsonl').open('w') as stream:
        for batch in batches[:clip_limit]:
            files = sorted((batch / 'reconstruction/multiview/depth_cache').glob('frame_*.jpg.npz'))[:frame_limit]
            frames, poses = [], []
            for i, file in enumerate(files):
                with np.load(file) as cached:
                    rgb, k, p = cached['rgb'], cached['K'].astype(np.float64), cached['P'].astype(np.float64)
                if not np.isfinite(p).all():
                    continue
                index = int(file.name.split('_')[1].split('.')[0])
                frames.append(ShadowFrame(f'{batch.name}/{index}', index / 2., rgb, k))
                poses.append(PoseEstimate(frames[-1].frame_id, index / 2., np.linalg.inv(p), 'colmap_full_clip_diagnostic'))
            make = lambda: KeyframePolicy(.25, 1., 8.)
            first, second = (shadow_clip(frames, make, poses, thumbnail_descriptor, contract['memory']) for _ in (0, 1))
            for state in first.pop('states'):
                stream.write(json.dumps({'clip': batch.name, **state}) + '\n')
            repeat = first['memory_digest'] == second['memory_digest']
            rows.append({'clip': batch.name, 'frames': len(frames), 'repeatable': repeat,
                         **{k: v for k, v in first.items() if k != 'memory_digest'}})
    failures = [r['clip'] for r in rows if r['mismatches'] or not r['repeatable']]
    bounded = all(r['counters']['matcher_calls'] <= contract['memory']['verified_pair_budget'] * r['frames'] for r in rows)
    gate = {'status': 'PASS' if rows and not failures and bounded else 'FAIL', 'clips': rows, 'failures': failures,
            'bounded_matching': bounded, 'descriptor': 'thumbnail384 (CPU stand-in, not DINO)',
            'poses': 'Full-clip SfM poses: a named shadow diagnostic, not prefix-only camera evidence.',
            'next_eligible_stage': 'stage4_labels'}
    write_json(output / 'replay_cost_report.json', gate)
    write_status(output, 'stage3_memory', 'complete', gate['status'], gate['next_eligible_stage'] if gate['status'] == 'PASS' else None)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('replay',))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--clips', type=int, default=6)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    if args.dry_run:
        print(json.dumps({'limits': contract['memory'], 'cache_root': contract['data']['cache_root']}))
        return
    report = replay(contract, args.root.resolve(), args.output, args.clips, args.frames)
    print(json.dumps({k: v for k, v in report.items() if k != 'clips'}, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
