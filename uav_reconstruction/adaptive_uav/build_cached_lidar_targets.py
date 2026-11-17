"""Attach measured LiDAR targets to existing DA3 cache frame identities.

Writes sidecars only; original teacher caches and student checkpoint are untouched.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .build_uavscenes_targets import pose_at_time
from .lidar_depth_supervision import project_returns


def downsample_sparse_min(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    source_h, source_w = depth.shape
    ys, xs = np.nonzero(depth > 0)
    y = np.minimum((ys * h / source_h).astype(np.int64), h - 1)
    x = np.minimum((xs * w / source_w).astype(np.int64), w - 1)
    output = np.full((h, w), np.inf, np.float32)
    np.minimum.at(output, (y, x), depth[ys, xs])
    output[~np.isfinite(output)] = 0
    return output


def build(cache_root: Path, lidar_root: Path, output: Path) -> dict:
    reports = []
    info_cache = {}
    output.mkdir(parents=True, exist_ok=True)
    for batch in sorted(p for p in cache_root.iterdir() if (p / 'selected_infos.json').exists()):
        infos = json.loads((batch / 'selected_infos.json').read_text())
        scene = infos[0]['OriginalSceneName']
        if 'GNSS' in scene:
            reports.append({'batch': batch.name, 'scene': scene, 'status': 'skipped',
                            'reason': 'separate GNSS extrinsic not yet verified'})
            continue
        lidar_dir = lidar_root / batch.name
        if not lidar_dir.exists():
            continue
        if scene not in info_cache:
            info_cache[scene] = json.loads((lidar_root / 'all_infos' / f'{scene}.json').read_text())
        lidar_rows = {r['frame']: r for r in json.loads((lidar_dir / 'manifest.json').read_text())}
        cache = batch / 'reconstruction/multiview/depth_cache'
        batch_output = output / batch.name
        batch_output.mkdir(parents=True, exist_ok=True)
        for file in sorted(cache.glob('frame_*.jpg.npz')):
            index = int(file.name.split('_')[1].split('.')[0])
            if index not in lidar_rows:
                continue
            row = lidar_rows[index]
            info = infos[index]
            try:
                camera_at_lidar = pose_at_time(info_cache[scene], row['lidar_time'])
                xyz = np.loadtxt(lidar_dir / f'frame_{index:06d}.txt', dtype=np.float32)
                depth, stats = project_returns(xyz, info, camera_at_lidar=camera_at_lidar)
                with np.load(file) as data:
                    shape = data['depth'].shape
                    rgb_shape = data['rgb'].shape[:2]
                    if shape != rgb_shape:
                        raise ValueError('RGB/depth cache grid mismatch')
                small = downsample_sparse_min(depth, shape)
            except (ValueError, OSError, KeyError) as error:
                reports.append({'batch': batch.name, 'scene': scene, 'frame': index,
                                'status': 'rejected', 'reason': str(error)})
                continue
            target = batch_output / f'frame_{index:06d}.npz'
            np.savez_compressed(target, measured_depth_z_m=small,
                                measured_valid=small > 0,
                                metric_reference_pose=np.asarray(info['T4x4'], dtype=np.float64),
                                K_native=np.asarray(info['P3x3'], dtype=np.float64),
                                native_shape=np.asarray(depth.shape))
            reports.append({'batch': batch.name, 'scene': scene, 'frame': index,
                            'status': 'provisional', 'target': str(target.resolve()),
                            'pixels': int((small > 0).sum()), 'sync_offset_ms': row['sync_offset_ms'],
                            'native_coverage_percent': stats['coverage_percent']})
    result = {'frames': sum(r['status'] == 'provisional' for r in reports),
              'measured_pixels_cache_grid': sum(r.get('pixels', 0) for r in reports),
              'scenes': sorted({r['scene'] for r in reports if r['status'] == 'provisional'}),
              'rows': reports,
              'warning': 'Provisional LiDAR targets; no dynamic mask or independent image alignment gate yet.'}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--lidar-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.cache_root, args.lidar_root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))
