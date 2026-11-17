"""Recover frame identities and supervision availability for v1/v4/v5."""
import json
from pathlib import Path

import numpy as np

from .direct_lidar_dataset import DirectLidarDataset
from .train_depth_student_lidar import discover


def run(output=Path('runs/depth_student_diagnosis/dataset_manifest_comparison.json')):
    cache_root = Path('runs/uavscenes_30_batch')
    sidecars = Path('runs/uavscenes_cached_lidar_targets')
    direct_root = Path('runs/uavscenes_multiflight')
    direct_targets = Path('runs/uavscenes_multiflight_measured')
    scales = json.loads((sidecars / 'scale_calibration.json').read_text())
    paths, _ = discover(cache_root, sidecars, scales)
    cached = []
    info_cache = {}
    for path in paths['train']:
        file = Path(path)
        batch = file.parents[3].name
        frame = int(file.name.split('_')[1].split('.')[0])
        if batch not in info_cache:
            info_cache[batch] = json.loads((cache_root / batch / 'selected_infos.json').read_text())
        info = info_cache[batch][frame]
        sidecar = sidecars / batch / f'frame_{frame:06d}.npz'
        with np.load(sidecar) as data:
            depth = data['measured_depth_z_m']
            valid = depth[depth > 0]
        with np.load(file) as data:
            teacher = np.asarray(data['depth'])
            anchors = np.asarray(data['anchors'])
            rgb_shape = data['rgb'].shape[:2]
        cached.append({'kind': 'cached', 'batch': batch, 'scene': info['OriginalSceneName'],
                       'frame_id': info['OriginalImageName'], 'rgb': str((cache_root / batch / 'source/frames' / f'frame_{frame:06d}.jpg').resolve()),
                       'target': str(sidecar.resolve()), 'lidar': str((Path('runs/uavscenes_cached_lidar') / batch / f'frame_{frame:06d}.txt').resolve()),
                       'calibration': 'per-frame map P3x3, zero distortion; published LiDAR-to-camera extrinsic',
                       'camera_to_world': info['T4x4'], 'K': info['P3x3'],
                       'measured_pixels': int(len(valid)), 'median_depth_m': float(np.median(valid)),
                       'depth_range_m': [float(valid.min()), float(valid.max())],
                       'rgb_resolution': list(rgb_shape),
                       'teacher_available': bool(np.isfinite(teacher).all() and np.any(teacher > 0)),
                       'sparse_anchors': int(np.sum(anchors > 0)),
                       'normalization': 'median sparse visual-unit anchor; metric conversion from camera-centre Sim3'})
    direct_sets = {}
    for label, maximum, neighbors in (('v4', .4, False), ('v5', .16, True)):
        dataset = DirectLidarDataset(direct_root, direct_targets,
                                     max_normalized_baseline=maximum, use_neighbors=neighbors,
                                     augment=False)
        records = []
        for scene, frame, sources, _ in dataset.groups:
            info = dataset.manifests[scene][frame]['info']
            target = direct_targets / scene / f'frame_{frame:06d}.npz'
            with np.load(target) as data:
                depth = data['measured_depth_z_m']
                valid = depth[depth > 0]
            records.append({'kind': 'direct', 'scene': scene, 'frame_id': info['OriginalImageName'],
                            'rgb': str((direct_root / scene / 'rgb' / f'frame_{frame:06d}.jpg').resolve()),
                            'target': str(target.resolve()),
                            'lidar': str((direct_root / scene / 'lidar' / f'frame_{frame:06d}.txt').resolve()),
                            'calibration': 'per-frame map P3x3, zero distortion; published LiDAR-to-camera extrinsic',
                            'camera_to_world': info['T4x4'], 'K': info['P3x3'],
                            'measured_pixels': int(len(valid)), 'median_depth_m': float(np.median(valid)),
                            'depth_range_m': [float(valid.min()), float(valid.max())],
                            'rgb_resolution': [info['Height'], info['Width']],
                            'teacher_available': False, 'sparse_anchors': 0,
                            'source_ids': [source['info']['OriginalImageName'] for source in sources],
                            'normalization': 'median disjoint LiDAR pixels; reference metric camera poses',
                            'quality_flags': ['LiDAR/RGB alignment unresolved', 'no DA3 teacher', 'no SfM anchors']})
        direct_sets[label] = records
    key = lambda r: (r['scene'], r['frame_id'])
    cache_ids = {key(r) for r in cached}
    v4_ids = {key(r) for r in direct_sets['v4']}
    v5_ids = {key(r) for r in direct_sets['v5']}
    result = {'method': 'v4/v5 direct frame IDs reconstructed from the historical selection rules; exact source list was not saved in run checkpoints',
              'summary': {'cached_v1_frames': len(cached), 'v4_direct_groups': len(direct_sets['v4']),
                          'v5_direct_groups': len(direct_sets['v5']),
                          'cached_and_v4_same_rgb': len(cache_ids & v4_ids),
                          'cached_and_v5_same_rgb': len(cache_ids & v5_ids),
                          'v4_v5_shared_direct_rgb': len(v4_ids & v5_ids),
                          'v4_only_direct_rgb': len(v4_ids - v5_ids),
                          'v5_only_direct_rgb': len(v5_ids - v4_ids),
                          'empty_measured': sum(r['measured_pixels'] == 0 for r in cached + direct_sets['v4'] + direct_sets['v5']),
                          'direct_no_teacher': len(direct_sets['v4']) + len(direct_sets['v5'])},
              'v1_cached': cached, 'v4_direct': direct_sets['v4'], 'v5_direct': direct_sets['v5']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    run()
