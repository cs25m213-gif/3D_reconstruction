"""Project synchronized UAVScenes LiDAR returns to measured camera-Z targets.

This module deliberately keeps measured returns separate from teacher predictions.
The interval5 RGB metadata declares an undistorted map camera (P3x3, zero
distortion). Raw sensor calibration is available for diagnostics, not silently
substituted for the released image calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


R_LIDAR_TO_CAMERA = np.array([
    [.00352762, -.999765, -.0213775],
    [-.0111803, .0213369, -.99971],
    [.999931, .00376561, -.0111025],
], dtype=np.float64)
T_LIDAR_TO_CAMERA = np.array([-.0025563, .0470454, -.0513375])
K_RAW = np.array([[1444.43, 0, 1177.8], [0, 1444.34, 1043.6], [0, 0, 1.]])
DIST_RAW = np.array([-.053, .121, .00127, .00043, -.06495])

# Published MARS-LVIG camera/LiDAR extrinsics, grouped by UAVScenes flight.
# These are distinct from each frame's camera-to-map P3x3 intrinsics.
SCENE_EXTRINSICS = {
    'AMtown': (
        [[.00298088, -.999728, -.0231416], [-.00504636, .0231263, -.99972],
         [.999983, .00309683, -.00497605]], [.0025563, .0567484, -.0512149]),
    'AMvalley': (
        [[.00298068, -.999735, -.0231428], [-.00504595, .023132, -.99974],
         [.999985, .00309701, -.00497598]], [-.0025563, .0567484, -.0512149]),
    'HKairport': (
        [[.00363212, -.999819, -.0213618], [-.0111679, .0214512, -.999591],
         [.999879, .00375613, -.0111134]], [-.0021928, .0470312, -.0513126]),
    'HKisland': (R_LIDAR_TO_CAMERA, T_LIDAR_TO_CAMERA),
}


def extrinsics_for_scene(scene: str) -> tuple[np.ndarray, np.ndarray]:
    if 'GNSS' in scene:
        raise ValueError(f'GNSS flight {scene} needs its separate published calibration')
    for prefix, (r, t) in SCENE_EXTRINSICS.items():
        if scene.startswith(prefix):
            return np.asarray(r, dtype=np.float64), np.asarray(t, dtype=np.float64)
    raise ValueError(f'No published camera/LiDAR calibration for {scene}')


def project_returns(xyz: np.ndarray, info: dict, *, inverse_extrinsic: bool = False,
                    raw_calibration: bool = False,
                    camera_at_lidar: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Return a native-resolution sparse Z map and projection diagnostics."""
    xyz = np.asarray(xyz, dtype=np.float64)[:, :3]
    r, t = extrinsics_for_scene(info['OriginalSceneName'])
    if inverse_extrinsic:
        camera = (xyz - t) @ r
    else:
        camera = xyz @ r.T + t
    if camera_at_lidar is not None:
        camera_at_image = np.asarray(info['T4x4'], dtype=np.float64)
        relative = np.linalg.inv(camera_at_image) @ camera_at_lidar
        camera = camera @ relative[:3, :3].T + relative[:3, 3]
    front = (camera[:, 2] > 0.5) & (camera[:, 2] < 250) & np.isfinite(camera).all(1)
    camera = camera[front]
    k = K_RAW if raw_calibration else np.asarray(info['P3x3'], dtype=np.float64)
    distortion = DIST_RAW if raw_calibration else np.zeros(5)
    uv = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k, distortion)[0].reshape(-1, 2)
    w, h = int(info['Width']), int(info['Height'])
    good = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, z = uv[good], camera[good, 2]
    pixel = np.floor(uv[:, 1]).astype(np.int64) * w + np.floor(uv[:, 0]).astype(np.int64)
    depth = np.zeros(h * w, dtype=np.float32)
    # Nearest Z wins within each pixel. Zero remains the invalid sentinel.
    if len(pixel):
        order = np.argsort(z)
        unique, first = np.unique(pixel[order], return_index=True)
        depth[unique] = z[order[first]].astype(np.float32)
    depth = depth.reshape(h, w)
    valid = depth > 0
    stats = {'input_returns': int(len(xyz)), 'positive_z_returns': int(front.sum()),
             'in_image_returns': int(good.sum()), 'valid_pixels': int(valid.sum()),
             'coverage_percent': float(100 * valid.mean()),
             'depth_min_m': float(depth[valid].min()) if valid.any() else None,
             'depth_max_m': float(depth[valid].max()) if valid.any() else None,
             'depth_median_m': float(np.median(depth[valid])) if valid.any() else None,
             'mean_range_minus_z_m': float(np.mean(np.linalg.norm(camera[good], axis=1) - z)) if len(z) else None}
    return depth, stats


def generate(source: Path, infos_file: Path, output: Path) -> dict:
    infos = json.loads(infos_file.read_text())
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, info in enumerate(infos):
        lidar = source / 'lidar' / f'frame_{index:06d}.txt'
        rgb = source / 'frames' / f'frame_{index:06d}.jpg'
        if not lidar.exists() or not rgb.exists():
            continue
        xyz = np.loadtxt(lidar, dtype=np.float32)
        depth, stats = project_returns(xyz, info)
        target = output / f'frame_{index:06d}.npz'
        np.savez_compressed(target, measured_depth_z_m=depth, measured_valid=depth > 0,
                            K=np.asarray(info['P3x3'], dtype=np.float64),
                            camera_to_world=np.asarray(info['T4x4'], dtype=np.float64))
        rows.append({'frame': index, 'scene': info['OriginalSceneName'],
                     'rgb': str(rgb.resolve()), 'target': str(target.resolve()),
                     'lidar': str(lidar.resolve()), 'timestamp_name': info['OriginalImageName'],
                     'calibration': 'UAVScenes interval5 map P3x3; zero distortion',
                     'extrinsic': 'published HKisland LiDAR-to-camera; direction under diagnostic review',
                     **stats})
    report = {'frames': len(rows), 'measured_pixels': sum(r['valid_pixels'] for r in rows),
              'mean_coverage_percent': float(np.mean([r['coverage_percent'] for r in rows])) if rows else 0,
              'scenes': sorted({r['scene'] for r in rows}), 'rows': rows,
              'warning': 'No dynamic-object mask or motion compensation; use as provisional measured targets until reprojection checks pass.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--infos', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.source, args.infos, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
