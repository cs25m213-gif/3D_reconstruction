"""Views with corrected geometry for updating the existing student.

Supervision stays separated by trust: reliable LiDAR, neighbouring LiDAR, dense map, then sparse
anchors and teacher. Normalization matches the student's contract exactly: depth and translation
are divided by the median sparse anchor. Sources can be restricted to past frames so training sees
what streaming will see, and a clip without LiDAR still trains through self-supervised losses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .build_cached_lidar_targets import downsample_sparse_min

GRID = (192, 256)
VALID_RANGE = (.25, 4.)
MIN_ANCHORS = 30
BASELINE_TARGETS = (.05, .12)


@dataclass
class ClipRecord:
    name: str
    scene: str
    role: str
    alpha: float | None
    cache: Path
    targets: Path | None
    overlay: Path | None
    map_only: bool = False
    indices: list = field(default_factory=list)
    poses: dict = field(default_factory=dict)
    depths: dict = field(default_factory=dict)


def view_arrays(clip: ClipRecord, index: int, keys: tuple) -> dict:
    """Read only what is asked for: the images are by far the slowest part of a cache file."""
    with np.load(clip.cache / f'frame_{index:06d}.jpg.npz') as cached:
        row = {key: cached[key].copy() for key in keys}
    file = clip.overlay / f'frame_{index:06d}.npz' if clip.overlay else None
    if file is not None and file.exists():
        with np.load(file) as overlay:
            row['P'] = overlay['P']
            if 'K' in keys:
                row['K'] = overlay['K']
            if 'anchors' in keys:
                row['anchors'] = overlay['anchors']
            if 'depth' in keys:
                teacher = float(overlay['teacher_scale'])
                row['depth'] = row['depth'] * teacher if np.isfinite(teacher) else np.zeros_like(row['depth'])
    return row


def index_clip(clip: ClipRecord) -> ClipRecord:
    """Read poses and anchor scale once, so source selection never touches image data."""
    for file in sorted(clip.cache.glob('frame_*.jpg.npz')):
        index = int(file.name.split('_')[1].split('.')[0])
        row = view_arrays(clip, index, ('P', 'anchors'))
        pose, anchors = row['P'].astype(np.float64), row['anchors']
        valid = anchors[np.isfinite(anchors) & (anchors > 0)]
        if np.isfinite(pose).all() and len(valid) >= MIN_ANCHORS:
            clip.indices.append(index)
            clip.poses[index] = np.linalg.inv(pose)
            clip.depths[index] = float(np.median(valid))
    return clip


def discover(root: Path, contract: dict, trusted_only: bool = True) -> dict:
    data, splits = contract['data'], contract['splits']
    scales = json.loads((root / data['scale_calibration']).read_text())
    gate_file = root / contract['geometry_gate']
    gate = json.loads(gate_file.read_text()).get('flights', {}) if gate_file.exists() else {}
    roles: dict = {role: [] for role in splits}
    for folder in sorted((root / data['cache_root']).iterdir()):
        name = folder.name
        if not (folder / 'selected_infos.json').exists() or 'GNSS' in name or not scales.get(name, {}).get('usable'):
            continue
        scene = scales[name]['scene']
        role = next((r for r, areas in splits.items() if any(scene.startswith(a) for a in areas)), None)
        if role is None:
            continue
        if (role in ('train', 'validation') and contract.get('student', {}).get('exclude_calibration_clips', False)
                and name.endswith(contract['geometry']['calibration_batch_suffix'])):
            continue
        flight = gate.get(scene, {})
        passed = flight.get('status') in ('VERIFIED', 'QUALIFIED')
        # Map depth reaches pixels through camera poses, so it survives a LiDAR registration miss.
        map_only = role == 'train' and not passed and flight.get('registration_outcome') == 'offset' and not flight.get('blocking_reasons')
        if trusted_only and role in ('train', 'validation') and not passed and not map_only:
            continue
        overlay = root / data['overlay'] / name if data.get('overlay') else None
        clip = ClipRecord(name, scene, role, scales[name]['scale_m_per_visual_unit'], folder / 'reconstruction/multiview/depth_cache',
                          root / data['sidecars'] / name, overlay if overlay is not None and overlay.exists() else None, trusted_only and map_only)
        roles[role].append(index_clip(clip))
    return roles


def unlabelled_clip(cache: Path) -> ClipRecord:
    return index_clip(ClipRecord(cache.parent.parent.parent.name, 'unlabelled', 'unlabelled', None, cache, None, None))


def auxiliary_manifest(path: Path) -> list:
    """Admit only training-role caches from an explicitly versioned auxiliary manifest."""
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'geostreamfusion.auxiliary.v1':
        raise ValueError(f'Unsupported auxiliary manifest: {path}')
    status = json.loads(path.with_name('status.json').read_text())
    if status.get('state') != 'complete' or manifest.get('failures'):
        raise ValueError('Auxiliary preparation must complete before training')
    clips, seen = [], set()
    for row in manifest['clips']:
        if row['role'] != 'train':
            continue
        cache = (path.parent / row['cache']).resolve()
        if not cache.is_relative_to(path.parent.resolve()) or cache in seen:
            raise ValueError('Escaping or duplicate auxiliary cache path')
        seen.add(cache)
        if row['supervision'] != 'sfm_anchors_and_photometric':
            raise ValueError('Unsupported auxiliary supervision')
        clip = index_clip(ClipRecord(row['name'], row['scene'], 'train', None, cache, None, None))
        if len(clip.indices) != row['frames'] or len(clip.indices) < 3:
            raise ValueError(f'Incomplete auxiliary cache: {cache}')
        clips.append(clip)
    if not clips:
        raise ValueError('No admitted auxiliary training clips')
    return clips


def choose_sources(clip: ClipRecord, position: int, causal: bool) -> list:
    """Two sources at useful baselines. Past-only in causal mode, like the streaming runtime."""
    index = clip.indices[position]
    centre, depth = clip.poses[index][:3, 3], clip.depths[index]
    pool = clip.indices[:position] if causal else [i for i in clip.indices if i != index]
    if not pool:
        return []
    ratios = np.array([np.linalg.norm(clip.poses[i][:3, 3] - centre) / depth for i in pool])
    picks = []
    for target in BASELINE_TARGETS:
        order = np.argsort(np.abs(ratios - target), kind='stable')
        picks.append(next((pool[j] for j in order if pool[j] not in picks), pool[int(order[0])]))
    return picks


class UpdateDataset(Dataset):
    """Each item is a short chronological sequence; length one is a single view."""

    def __init__(self, clips: list, sequence: int = 1, causal: bool = True, augment: bool = False):
        self.clips, self.sequence, self.causal, self.augment = clips, sequence, causal, augment
        first = 1 if causal else 0
        self.items = [(c, p) for c, clip in enumerate(clips) for p in range(first + sequence - 1, len(clip.indices))]

    def __len__(self):
        return len(self.items)

    def targets(self, clip: ClipRecord, index: int, scale: float) -> dict:
        h, w = GRID
        empty = np.zeros((h, w), np.float32)
        row = {'measured': empty, 'neighbour': empty, 'map': empty}
        file = clip.targets / f'frame_{index:06d}.npz' if clip.targets else None
        if file is not None and file.exists() and clip.alpha:
            factor = clip.alpha * scale
            with np.load(file) as data:
                if not clip.map_only:
                    row['measured'] = downsample_sparse_min(data['measured_depth_z_m'], GRID) / factor
                    if 'neighbour_depth_z_m' in data:
                        row['neighbour'] = downsample_sparse_min(data['neighbour_depth_z_m'], GRID) / factor
                if 'map_depth_z_m' in data:
                    row['map'] = cv2.resize(data['map_depth_z_m'], (w, h), interpolation=cv2.INTER_NEAREST) / factor
        return row

    def frame(self, clip: ClipRecord, position: int, scale: float, gain) -> dict:
        index = clip.indices[position]
        names = [index] + choose_sources(clip, position, self.causal)
        if len(names) < 2:
            raise ValueError('A view needs at least one source')
        h, w = GRID
        images, intrinsics, poses, reference = [], [], [], None
        for n in names[:3]:
            row = view_arrays(clip, n, ('rgb', 'K', 'P', 'depth', 'confidence', 'anchors') if n == index else ('rgb', 'K', 'P'))
            reference = row if n == index else reference
            ih, iw = row['rgb'].shape[:2]
            images.append(cv2.resize(row['rgb'], (w, h), interpolation=cv2.INTER_AREA).transpose(2, 0, 1).astype(np.float32) / 255.)
            k = row['K'].astype(np.float32).copy()
            k[0] *= w / iw
            k[1] *= h / ih
            intrinsics.append(k)
            p = row['P'].astype(np.float32).copy()
            p[:3, 3] /= scale
            poses.append(p)
        while len(images) < 3:
            images.append(images[-1].copy()); intrinsics.append(intrinsics[-1].copy()); poses.append(poses[-1].copy())
        stack = np.stack(images)
        if gain is not None:
            stack = np.clip(stack * gain[0] + gain[1], 0, 1) ** gain[2]
        teacher = cv2.resize(reference['depth'].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) / scale
        anchor = downsample_sparse_min(np.where(np.isfinite(reference['anchors']), reference['anchors'], 0).astype(np.float32), GRID) / scale
        confidence = cv2.resize(reference['confidence'].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        low, high = VALID_RANGE
        keep = (teacher > low) & (teacher < high) & np.isfinite(confidence)
        keep &= confidence >= np.percentile(confidence[np.isfinite(confidence)], 20)
        targets = self.targets(clip, index, scale)
        item = {'rgb': torch.from_numpy(stack.astype(np.float32)), 'K': torch.from_numpy(np.stack(intrinsics)), 'P': torch.from_numpy(np.stack(poses)),
                'teacher': torch.from_numpy(teacher[None]), 'teacher_valid': torch.from_numpy(keep[None]), 'anchor': torch.from_numpy(anchor[None].astype(np.float32)),
                'metric_per_normalized': float((clip.alpha or 1.) * scale)}
        for key, value in targets.items():
            value = np.asarray(value, np.float32)
            item[key] = torch.from_numpy(value[None])
            item[key + '_valid'] = torch.from_numpy(((value > low) & (value < high))[None])
        return item

    def __getitem__(self, i):
        c, position = self.items[i]
        clip = self.clips[c]
        scale = clip.depths[clip.indices[position]]
        gain = None
        if self.augment:  # One transform for all views keeps their correspondence.
            gain = (.75 + .5 * float(torch.rand(())), -.08 + .16 * float(torch.rand(())), .8 + .4 * float(torch.rand(())))
        frames = [self.frame(clip, p, scale, gain) for p in range(position - self.sequence + 1, position + 1)]
        joined = {key: torch.stack([f[key] for f in frames]) for key in frames[0] if key != 'metric_per_normalized'}
        joined['metric_per_normalized'] = frames[-1]['metric_per_normalized']
        joined['scene'] = clip.scene
        return joined
