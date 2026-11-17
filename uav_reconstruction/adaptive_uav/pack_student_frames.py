"""Pack every usable frame at the student grid (192x256) into flat arrays that fit in GPU memory.

One pass over the depth caches replaces per-step JPEG/npz decoding, so training can assemble
batches by tensor indexing on the GPU. Values are stored in the same units the student uses
before per-reference normalisation (visual units); the trainer divides by each reference's
anchor scale on the GPU, exactly as student_dataset.UpdateDataset does on the CPU.

Supervision trust follows the geometry gate: flights whose LiDAR registration did not pass
keep only map depth (their LiDAR is offset), whatever their role.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .build_cached_lidar_targets import downsample_sparse_min
from .student_dataset import GRID, UpdateDataset, choose_sources, discover, view_arrays
from .utility_contracts import load_contract

_CLIPS = None


def _pack_clip(c: int) -> dict:
    clip = _CLIPS[c]
    h, w = GRID
    data = UpdateDataset([clip], 1, True)
    n = len(clip.indices)
    out = {'rgb': np.zeros((n, 3, h, w), np.uint8), 'K': np.zeros((n, 3, 3), np.float32), 'P': np.zeros((n, 4, 4), np.float64),
           'teacher': np.zeros((n, h, w), np.float16), 'teacher_ok': np.zeros((n, h, w), np.uint8), 'anchor': np.zeros((n, h, w), np.float16),
           'measured': np.zeros((n, h, w), np.float16), 'neighbour': np.zeros((n, h, w), np.float16), 'map': np.zeros((n, h, w), np.float16),
           'scale': np.zeros(n, np.float32), 'src': np.full((n, 2), -1, np.int32), 'position': np.arange(n, dtype=np.int32)}
    for p, index in enumerate(clip.indices):
        row = view_arrays(clip, index, ('rgb', 'K', 'P', 'depth', 'confidence', 'anchors'))
        ih, iw = row['rgb'].shape[:2]
        out['rgb'][p] = cv2.resize(row['rgb'], (w, h), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        k = row['K'].astype(np.float32).copy()
        k[0] *= w / iw
        k[1] *= h / ih
        out['K'][p], out['P'][p] = k, row['P'].astype(np.float64)
        out['teacher'][p] = np.clip(cv2.resize(row['depth'].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA), 0, 6e4)
        conf = cv2.resize(row['confidence'].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        finite = np.isfinite(conf)
        out['teacher_ok'][p] = finite & (conf >= np.percentile(conf[finite], 20)) if finite.any() else 0
        out['anchor'][p] = np.clip(downsample_sparse_min(np.where(np.isfinite(row['anchors']), row['anchors'], 0).astype(np.float32), GRID), 0, 6e4)
        for key, value in data.targets(clip, index, 1.).items():  # visual units (metres / alpha)
            out[key][p] = np.clip(np.nan_to_num(np.asarray(value, np.float32)), 0, 6e4)
        out['scale'][p] = clip.depths[index]
        sources = choose_sources(clip, p, True)
        out['src'][p, :len(sources)] = [clip.indices.index(s) for s in sources]
    return out


def main():
    global _CLIPS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--workers', type=int, default=24)
    a = p.parse_args()
    contract = load_contract(a.config)
    root = a.root.resolve()
    gate = json.loads((root / contract['geometry_gate']).read_text()).get('flights', {})
    roles = discover(root, contract, True)
    clips, meta = [], []
    for role, members in roles.items():
        for clip in members:
            passed = gate.get(clip.scene, {}).get('status') in ('VERIFIED', 'QUALIFIED')
            clip.map_only = clip.map_only or not passed  # offset LiDAR is never used as truth
            clips.append(clip)
            meta.append({'name': clip.name, 'scene': clip.scene, 'area': clip.scene.rstrip('0123456789'), 'role': role,
                         'lidar': passed, 'alpha': clip.alpha, 'frames': len(clip.indices)})
    _CLIPS = clips
    with Pool(a.workers) as pool:
        parts = pool.map(_pack_clip, range(len(clips)), chunksize=1)
    a.output.mkdir(parents=True, exist_ok=True)
    offset, clip_id = 0, []
    for c, part in enumerate(parts):
        part['src'] = np.where(part['src'] >= 0, part['src'] + offset, -1)
        meta[c]['start'] = offset
        offset += len(part['scale'])
        clip_id.append(np.full(len(part['scale']), c, np.int32))
    for key in parts[0]:
        np.save(a.output / f'{key}.npy', np.concatenate([part[key] for part in parts]))
    np.save(a.output / 'clip.npy', np.concatenate(clip_id))
    (a.output / 'clips.json').write_text(json.dumps(meta, indent=1))
    summary = {}
    for m in meta:
        s = summary.setdefault(f"{m['role']}/{m['area']}", {'clips': 0, 'frames': 0, 'lidar': m['lidar']})
        s['clips'] += 1
        s['frames'] += m['frames']
    (a.output / 'summary.json').write_text(json.dumps({'frames': offset, 'groups': summary}, indent=1))
    print(json.dumps({'frames': offset, 'groups': summary}))


if __name__ == '__main__':
    main()
