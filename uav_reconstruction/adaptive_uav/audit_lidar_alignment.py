"""Check whether projected LiDAR depth jumps coincide with RGB edges."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .build_uavscenes_targets import pose_at_time
from .lidar_depth_supervision import extrinsics_for_scene


RAW_CAMERA = {
    'AMtown': ([[1453.72, 0, 1172.18], [0, 1453.28, 1041.78], [0, 0, 1]],
               [-.121, .1113, .0016, .00013, -.062353]),
    'AMvalley': ([[1453.88, 0, 1182.53], [0, 1452.85, 1045.82], [0, 0, 1]],
                [-.052, .1168, .0015, .00013, -.068564]),
    'HKairport': ([[1444.43, 0, 1179.5], [0, 1444.34, 1044.9], [0, 0, 1]],
                  [-.056, .118, .00122, .00064, -.0627]),
    'HKisland': ([[1444.43, 0, 1177.8], [0, 1444.34, 1043.6], [0, 0, 1]],
                 [-.053, .121, .00127, .00043, -.06495]),
}


def score_pairs(camera, k, distortion, gradient):
    h, w = gradient.shape
    uv = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k, distortion)[0].reshape(-1, 2)
    midpoint = .5 * (uv[:-1] + uv[1:])
    pixel_jump = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    depth_jump = np.abs(np.diff(camera[:, 2]))
    valid = ((camera[:-1, 2] > 1) & (camera[1:, 2] > 1) &
             (pixel_jump > 2) & (pixel_jump < 30) & (depth_jump > 2) &
             np.isfinite(midpoint).all(1))
    xy = midpoint[valid]
    inside = (xy[:, 0] > 55) & (xy[:, 0] < w - 55) & (xy[:, 1] > 55) & (xy[:, 1] < h - 55)
    xy = xy[inside].astype(int)
    if len(xy) < 100:
        return None
    boundary = gradient[xy[:, 1], xy[:, 0]]
    controls = [gradient[xy[:, 1] + dy, xy[:, 0] + dx].mean()
                for dx, dy in ((-50, 0), (-25, 0), (-10, 0), (10, 0), (25, 0), (50, 0),
                               (0, -50), (0, -25), (0, 25), (0, 50))]
    displaced = float(np.median(controls))
    return {'pairs': int(len(xy)), 'edge_mean': float(boundary.mean()),
            'shifted_edge_mean': displaced, 'negative_control_means': [float(x) for x in controls],
            'edge_gain': float(boundary.mean() / max(displaced, 1e-6))}


def audit(root: Path, output: Path, frames_per_scene: int) -> dict:
    rows = []
    for folder in sorted(root.iterdir()):
        if not (folder / 'manifest.json').exists():
            continue
        scene = folder.name
        manifest = json.loads((folder / 'manifest.json').read_text())
        all_infos = json.loads((folder / 'all_sampleinfos.json').read_text())
        prefix = next(p for p in RAW_CAMERA if scene.startswith(p))
        raw_k, raw_dist = (np.asarray(x, dtype=np.float64) for x in RAW_CAMERA[prefix])
        indices = np.linspace(2, len(manifest)-3, frames_per_scene).astype(int)
        r, t = extrinsics_for_scene(scene)
        for index in indices:
            row = manifest[index]
            image = cv2.imread(str(folder / 'rgb' / f'frame_{index:06d}.jpg'), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            gradient = cv2.magnitude(cv2.Sobel(image, cv2.CV_32F, 1, 0),
                                     cv2.Sobel(image, cv2.CV_32F, 0, 1))
            gradient = cv2.dilate(gradient, np.ones((5, 5), np.uint8))
            xyz = np.loadtxt(folder / 'lidar' / f'frame_{index:06d}.txt', dtype=np.float32)
            lidar_camera = xyz @ r.T + t
            reference = np.asarray(row['info']['T4x4'])
            at_lidar = pose_at_time(all_infos, row['lidar_timestamp'])
            relative = np.linalg.inv(reference) @ at_lidar
            compensated = lidar_camera @ relative[:3, :3].T + relative[:3, 3]
            map_k = np.asarray(row['info']['P3x3'])
            results = {
                'map_compensated': score_pairs(compensated, map_k, np.zeros(5), gradient),
                'map_uncompensated': score_pairs(lidar_camera, map_k, np.zeros(5), gradient),
                'raw_compensated': score_pairs(compensated, raw_k, raw_dist, gradient),
            }
            rows.append({'scene': scene, 'frame': int(index),
                         'sync_offset_ms': row['sync_offset_ms'], **results})
    summary = {}
    for scene in sorted({row['scene'] for row in rows}):
        subset = [row for row in rows if row['scene'] == scene]
        summary[scene] = {'frames': len(subset), 'median_abs_sync_offset_ms': float(np.median([abs(r['sync_offset_ms']) for r in subset]))}
        for name in ('map_compensated', 'map_uncompensated', 'raw_compensated'):
            values = [r[name]['edge_gain'] for r in subset if r[name]]
            summary[scene][name + '_median_edge_gain'] = float(np.median(values)) if values else None
    report = {'summary': summary, 'rows': rows,
              'interpretation': 'Edge gain >1 means projected LiDAR depth jumps coincide with RGB edges better than a 25-pixel horizontal shift; it is a weak alignment check, not proof of exact calibration.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/uavscenes_multiflight'))
    parser.add_argument('--output', type=Path, default=Path('runs/uavscenes_multiflight_measured/alignment_audit.json'))
    parser.add_argument('--frames-per-scene', type=int, default=8)
    args = parser.parse_args()
    report = audit(args.root, args.output, args.frames_per_scene)
    print(json.dumps(report['summary'], indent=2))
