"""Causal policy features, shared by label generation and runtime so the two cannot diverge.

Everything is scale-free or normalized: the runtime works in arbitrary visual units and at other
image sizes than the training caches. CoverageMap answers the question the selector exists for,
"how much of this view does the current map already explain?", and needs no depth call for the
candidate, so asking costs almost nothing.
"""
import cv2
import copy
import numpy as np

from .keyframe_policy import KeyframePolicy
from .memory_graph import CausalMemory, detect, verify_features
from .runtime_types import PoseEstimate
from .utility_contracts import QUERY_FEATURES, TOKEN_FEATURES, feature_vector

OFFSET = 1 << 20


def pack(keys: np.ndarray) -> np.ndarray:
    keys = keys.astype(np.int64) + OFFSET
    return (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]


def unpack(packed: np.ndarray) -> np.ndarray:
    mask = (1 << 21) - 1
    return np.stack(((packed >> 42) & mask, (packed >> 21) & mask, packed & mask), 1).astype(np.int64) - OFFSET


class CoverageMap:
    """Voxels the map has observed and how often. The voxel size follows scene depth."""

    def __init__(self, voxels_per_depth: float = 40., max_voxels: int = 400000, stride: int = 8):
        self.voxels_per_depth, self.max_voxels, self.stride = voxels_per_depth, max_voxels, stride
        self.counts: dict = {}
        self.depths: list = []
        self.voxel = None
        self._dirty, self._centres, self._values = True, None, None

    @property
    def depth_scale(self) -> float | None:
        return float(np.median(self.depths)) if self.depths else None

    def observe(self, depth: np.ndarray, k: np.ndarray, pose: np.ndarray) -> None:
        ys, xs = np.mgrid[self.stride // 2:depth.shape[0]:self.stride, self.stride // 2:depth.shape[1]:self.stride]
        z = depth[ys, xs]
        good = np.isfinite(z) & (z > 0)
        if not good.any():
            return
        self.depths = (self.depths + [float(np.median(z[good]))])[-64:]
        if self.voxel is None:
            self.voxel = self.depths[-1] / self.voxels_per_depth
        rays = np.c_[(xs[good] - k[0, 2]) / k[0, 0], (ys[good] - k[1, 2]) / k[1, 1], np.ones(int(good.sum()))] * z[good, None]
        world = rays @ pose[:3, :3].T + pose[:3, 3]
        for key in np.unique(pack(np.floor(world / self.voxel))).tolist():
            self.counts[key] = self.counts.get(key, 0) + 1
        while len(self.counts) > self.max_voxels:
            self.counts.pop(next(iter(self.counts)))
        self._dirty = True

    def view_stats(self, k: np.ndarray, pose: np.ndarray, width: int, height: int, grid: tuple = (12, 16)) -> tuple:
        """(fraction of the view already mapped, how often that part was seen, 0 to 1)."""
        if not self.counts:
            return 0., 0.
        if self._dirty:
            keys = np.fromiter(self.counts.keys(), np.int64, len(self.counts))
            self._centres = (unpack(keys) + .5) * self.voxel
            self._values = np.fromiter(self.counts.values(), np.float64, len(self.counts))
            self._dirty = False
        camera = (self._centres - pose[:3, 3]) @ pose[:3, :3]
        front = camera[:, 2] > 1e-6
        u = k[0, 0] * camera[front, 0] / camera[front, 2] + k[0, 2]
        v = k[1, 1] * camera[front, 1] / camera[front, 2] + k[1, 2]
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not inside.any():
            return 0., 0.
        rows, columns = grid
        cells = (v[inside] * rows / height).astype(np.int64) * columns + (u[inside] * columns / width).astype(np.int64)
        best = np.zeros(rows * columns)
        np.maximum.at(best, cells, self._values[front][inside])
        seen = best > 0
        return float(seen.mean()), float(np.minimum(best[seen], 8).mean() / 8)


def image_sharpness(rgb: np.ndarray) -> float:
    """Laplacian variance at a fixed width, so it means the same at any source resolution."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (256, max(int(256 * gray.shape[0] / gray.shape[1]), 8)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def rotation_angle(a, b) -> float:
    return float(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))


def query_values(memory, frame, pose, stats: dict, budget: dict, coverage: CoverageMap, overlap: dict) -> dict:
    last = next(reversed(memory.nodes.values()), None) if memory.nodes else None
    matrix, depth = pose.T_world_from_camera, coverage.depth_scale
    height, width = frame.rgb.shape[:2]
    covered, redundant = coverage.view_stats(frame.K, matrix, width, height)
    return {'age_s': min(frame.timestamp - last.timestamp, 10.) / 10. if last else 1.,
            'baseline_over_depth': min(np.linalg.norm(matrix[:3, 3] - last.pose[:3, 3]) / depth, 1.) if last and depth else 0.,
            'rotation_from_last': min(rotation_angle(matrix, last.pose), 1.5) / 1.5 if last else 0.,
            'parallax': min(float(stats.get('rotation_compensated_flow_pixels', 0.)) / width * 10, 1.),
            'tracked_features': min(float(stats.get('tracked_features', 0.)), 200.) / 200.,
            'sharpness': min(image_sharpness(frame.rgb), 300.) / 300.,
            'map_keyframes': min(len(memory.nodes), 64) / 64., 'map_coverage': covered, 'map_redundancy': redundant,
            'graph_edges': min(len(memory.edges), 256) / 256., 'graph_components': min(memory.components(), 8) / 8.,
            'remaining_seconds': min(np.log1p(max(budget['seconds'], 0.)) / np.log1p(3600.), 1.),
            'remaining_depth_calls': min(max(budget['depth_calls'], 0), 64) / 64.,
            'tracking_ok': float(pose.state == 'tracking'), 'queue_load': float(stats.get('queue_load', 0.)),
            'best_overlap': min(float(overlap.get('parallax_px', 0.)) / width * 10, 1.),
            'best_shared_tracks': min(float(overlap.get('shared_tracks', 0.)), 500.) / 500.}


def token_rows(memory, frame, pose, visual_dim: int, depth_scale: float | None) -> tuple:
    """Bounded memory view for the policy, plus which frame sits in each slot."""
    limit = memory.limits['policy_tokens']
    rows = np.zeros((limit, len(TOKEN_FEATURES)), np.float32)
    visual = np.zeros((limit, visual_dim), np.float32)
    table, bridges, ids = memory.neighbours(), memory.bridges(), []
    current = next(reversed(memory.nodes.values()), None) if memory.nodes else None
    matrix, width = pose.T_world_from_camera, frame.rgb.shape[1]
    for i, node in enumerate(memory.tokens(frame.timestamp)):
        edges = [memory.edges[tuple(sorted((node.frame_id, other)))] for other in table[node.frame_id]]
        baseline = np.linalg.norm(matrix[:3, 3] - node.pose[:3, 3])
        rows[i] = (min(frame.timestamp - node.timestamp, 60.) / 60., min(baseline / depth_scale, 2.) / 2. if depth_scale else 0.,
                   min(rotation_angle(matrix, node.pose), 1.5) / 1.5, min(sum(e.shared_tracks for e in edges), 500) / 500.,
                   min(max((e.parallax_px for e in edges), default=0.) / width * 10, 1.), node.depth_confidence,
                   min(len(edges), 16) / 16., float(node.frame_id in bridges), float(bool(edges)),
                   float(current is not None and node.component == current.component), 1.)
        if node.descriptor.size == visual_dim:
            visual[i] = node.descriptor
        ids.append(node.frame_id)
    return rows, visual, ids


def candidate_features(memory, coverage: CoverageMap, frame, pose, stats: dict, budget: dict, descriptor: np.ndarray, overlap: dict) -> dict:
    values = query_values(memory, frame, pose, stats, budget, coverage, overlap)
    rows, visual, ids = token_rows(memory, frame, pose, descriptor.size, coverage.depth_scale)
    return {'query': np.float32(feature_vector(values, QUERY_FEATURES)), 'visual': np.float32(descriptor),
            'tokens': rows, 'token_visual': visual, 'token_ids': ids}


class MapState:
    """Everything a policy may know, updated the same way offline and in the running pipeline.

    commit() happens at decision time. observe() happens once depth exists, which in the runtime is
    later and on another thread, so the map a decision sees is the map that really existed then.
    """

    def __init__(self, limits: dict | None = None):
        self.memory, self.coverage = CausalMemory(limits), CoverageMap()
        self.last = None

    def fork(self):
        """Copy map/graph state exactly, without rerunning stochastic verification.

        ORB keypoint payloads and input images are immutable lookup evidence and
        may be shared. Nodes, edges, poses, counters and coverage are private.
        """
        other=MapState(self.memory.limits)
        other.memory=copy.copy(self.memory)
        other.memory.nodes=copy.deepcopy(self.memory.nodes)
        other.memory.edges=copy.deepcopy(self.memory.edges)
        other.memory.counters=copy.deepcopy(self.memory.counters)
        other.memory.limits=self.memory.limits.copy()
        other.memory.residency=self.memory.residency.copy()
        other.coverage=copy.deepcopy(self.coverage)
        if self.last is not None:
            other.last={**self.last,'pose':copy.deepcopy(self.last['pose'])}
        return other

    def commit(self, frame, pose, descriptor: np.ndarray) -> None:
        found = detect(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY))
        previous = self.last
        component = self.memory.nodes[previous['frame'].frame_id].component if previous else None
        self.memory.add(frame.frame_id, frame.timestamp, descriptor, pose.T_world_from_camera, found, component)
        if previous:
            result = verify_features(previous['found'], found, frame.K)
            self.memory.connect(previous['frame'].frame_id, frame.frame_id, result['shared_tracks'], result['parallax_px'], frame.timestamp)
        self.last = {'frame': frame, 'pose': pose, 'found': found}

    def observe(self, frame_id: str, depth: np.ndarray, k: np.ndarray, pose: np.ndarray, confidence: float = 0.) -> None:
        self.coverage.observe(depth, k, pose)
        if frame_id in self.memory.nodes:
            self.memory.nodes[frame_id].depth_confidence = float(confidence)

    def features(self, frame, pose, budget: dict, descriptor: np.ndarray, stats: dict | None = None) -> dict:
        previous = self.last
        overlap = {}
        if previous:
            found = detect(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY))
            overlap = verify_features(previous['found'], found, frame.K)
            if stats is None:
                policy = KeyframePolicy(.25, 1., 8.)
                policy.commit(previous['frame'], previous['pose'])
                stats = policy.propose(frame, pose)[1]
        return candidate_features(self.memory, self.coverage, frame, pose, stats or {}, budget, descriptor, overlap)


def as_pose(frame, matrix: np.ndarray, state: str = 'tracking'):
    return PoseEstimate(frame.frame_id, frame.timestamp, matrix, 'offline', state)
