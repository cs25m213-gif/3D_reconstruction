"""Check adjacent LiDAR scans in the released metric camera world frame."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .build_uavscenes_targets import pose_at_time
from .lidar_depth_supervision import extrinsics_for_scene


def audit(root: Path, lidar_root: Path, output: Path):
    rows = []
    all_info_cache = {}
    for batch in sorted(root.iterdir()):
        if not (batch / 'selected_infos.json').exists():
            continue
        infos = json.loads((batch / 'selected_infos.json').read_text())
        scene = infos[0]['OriginalSceneName']
        if 'GNSS' in scene or scene not in ('AMvalley01', 'AMvalley02', 'AMvalley03',
                                          'HKisland01', 'HKisland02', 'HKisland03'):
            continue
        if scene not in all_info_cache:
            all_info_cache[scene] = json.loads((lidar_root / 'all_infos' / f'{scene}.json').read_text())
        records = json.loads((lidar_root / batch.name / 'manifest.json').read_text())
        by_frame = {record['frame']: record for record in records}
        r, t = extrinsics_for_scene(scene)
        for i in (5, 15, 25, 35, 45):
            if i not in by_frame or i+1 not in by_frame:
                continue
            worlds = []
            for j in (i, i+1):
                xyz = np.loadtxt(lidar_root / batch.name / f'frame_{j:06d}.txt', dtype=np.float32)
                camera = xyz @ r.T + t
                pose = pose_at_time(all_info_cache[scene], by_frame[j]['lidar_time'])
                world = camera @ pose[:3, :3].T + pose[:3, 3]
                # Deterministic bounded sample keeps the audit inexpensive.
                worlds.append(world[::max(1, len(world)//5000)])
            target = cKDTree(worlds[1])
            distances = target.query(worlds[0], k=1, workers=1)[0]
            shifted = cKDTree(worlds[1] + np.array([0., 0., 5.]))
            negative = shifted.query(worlds[0], k=1, workers=1)[0]
            rows.append({'scene': scene, 'batch': batch.name, 'frame_pair': [i, i+1],
                         'scan_time_delta_s': by_frame[i+1]['lidar_time']-by_frame[i]['lidar_time'],
                         'median_nearest_m': float(np.median(distances)),
                         'p25_nearest_m': float(np.percentile(distances, 25)),
                         'negative_5m_vertical_median_m': float(np.median(negative))})
    summary = {}
    for scene in sorted({row['scene'] for row in rows}):
        subset = [row for row in rows if row['scene'] == scene]
        summary[scene] = {'pairs': len(subset),
                          'median_pair_distance_m': float(np.median([r['median_nearest_m'] for r in subset])),
                          'median_negative_control_m': float(np.median([r['negative_5m_vertical_median_m'] for r in subset]))}
    report = {'summary': summary, 'rows': rows,
              'note': 'Nearest-neighbour world-point consistency is affected by sparse scan pattern, occlusions and moving objects. It does not prove RGB pixel alignment.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, default=Path('runs/uavscenes_30_batch'))
    parser.add_argument('--lidar-root', type=Path, default=Path('runs/uavscenes_cached_lidar'))
    parser.add_argument('--output', type=Path, default=Path('runs/depth_student_diagnosis/lidar_temporal_consistency.json'))
    args = parser.parse_args()
    audit(args.cache_root, args.lidar_root, args.output)
