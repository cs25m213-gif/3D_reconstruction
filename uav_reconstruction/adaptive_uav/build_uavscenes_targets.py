"""Make measured UAVScenes depth targets with timestamp-aware pose interpolation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .lidar_depth_supervision import project_returns


def pose_at_time(infos: list[dict], timestamp: float) -> np.ndarray:
    times = np.array([float(Path(i['OriginalImageName']).stem) for i in infos])
    position = np.searchsorted(times, timestamp)
    if position == 0 or position == len(times):
        raise ValueError('LiDAR timestamp outside reference camera trajectory')
    lo, hi = position - 1, position
    if times[hi] - times[lo] > 1.0:
        raise ValueError('Reference pose interpolation spans >1 second')
    left = np.asarray(infos[lo]['T4x4'], dtype=np.float64)
    right = np.asarray(infos[hi]['T4x4'], dtype=np.float64)
    alpha = (timestamp - times[lo]) / (times[hi] - times[lo])
    rotation = Slerp([times[lo], times[hi]], Rotation.from_matrix(np.stack((left[:3, :3], right[:3, :3]))))([timestamp])[0]
    result = np.eye(4)
    result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = (1 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    return result


def build(root: Path, output: Path) -> dict:
    splits = json.loads((root / 'split_manifest.json').read_text())['splits']
    rows = []
    for role, scenes in splits.items():
        for scene_row in scenes:
            scene = scene_row['scene']
            folder = root / scene
            manifest = json.loads((folder / 'manifest.json').read_text())
            all_infos = json.loads((folder / 'all_sampleinfos.json').read_text())
            scene_output = output / scene
            scene_output.mkdir(parents=True, exist_ok=True)
            for row in manifest:
                info = row['info']
                lidar = folder / 'lidar' / f"frame_{row['index']:06d}.txt"
                rgb = folder / 'rgb' / f"frame_{row['index']:06d}.jpg"
                try:
                    camera_at_lidar = pose_at_time(all_infos, row['lidar_timestamp'])
                    xyz = np.loadtxt(lidar, dtype=np.float32)
                    depth, stats = project_returns(xyz, info, camera_at_lidar=camera_at_lidar)
                except (ValueError, OSError) as error:
                    rows.append({'scene': scene, 'role': role, 'frame': row['index'],
                                 'status': 'rejected', 'reason': str(error)})
                    continue
                target = scene_output / f"frame_{row['index']:06d}.npz"
                np.savez_compressed(target, measured_depth_z_m=depth,
                                    measured_valid=depth > 0,
                                    K=np.asarray(info['P3x3'], dtype=np.float64),
                                    camera_to_world=np.asarray(info['T4x4'], dtype=np.float64),
                                    camera_at_lidar=camera_at_lidar)
                rows.append({'scene': scene, 'role': role, 'frame': row['index'],
                             'status': 'provisional', 'rgb': str(rgb.resolve()),
                             'target': str(target.resolve()), 'sync_offset_ms': row['sync_offset_ms'],
                             'motion_compensation': 'reference camera pose interpolated at LiDAR timestamp',
                             **stats})
    output.mkdir(parents=True, exist_ok=True)
    summary = {'frames': sum(r['status'] == 'provisional' for r in rows),
               'measured_pixels': sum(r.get('valid_pixels', 0) for r in rows),
               'splits': {role: {'frames': sum(r['role'] == role and r['status'] == 'provisional' for r in rows),
                                'pixels': sum(r.get('valid_pixels', 0) for r in rows if r['role'] == role)}
                          for role in splits},
               'rows': rows,
               'warning': 'Targets remain provisional until image-edge alignment, dynamic-object and occlusion checks pass.'}
    (output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))
