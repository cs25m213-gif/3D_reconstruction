"""Build RGB-triangulated sparse anchors for added LiDAR training groups."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit_direct_sparse_scale import triangulated_depths
from .direct_lidar_dataset import DirectLidarDataset


def run(output: Path):
    dataset = DirectLidarDataset(Path('runs/uavscenes_multiflight'),
                                 Path('runs/uavscenes_multiflight_measured'),
                                 max_normalized_baseline=.16, augment=False)
    rows = []
    for index, (scene, frame, _, _) in enumerate(dataset.groups):
        item = dataset[index]
        rgb, k, p = (item[name].numpy() for name in ('rgb', 'K', 'P'))
        matches = [triangulated_depths(rgb[0], rgb[j], k[0], k[j], p[0], p[j]) for j in (1, 2)]
        xy = np.concatenate([x for x, _ in matches])
        depth = np.concatenate([z for _, z in matches])
        if len(depth) < 24:
            rows.append({'scene': scene, 'frame': frame, 'status': 'rejected', 'matches': len(depth)})
            continue
        scale_lidar = float(item['metric_per_normalized'])
        anchor_m = np.zeros((192, 256), np.float32)
        x = np.clip(np.floor(xy[:, 0]).astype(int), 0, 255)
        y = np.clip(np.floor(xy[:, 1]).astype(int), 0, 191)
        pixel = y*256+x
        order = np.argsort(depth)
        unique, first = np.unique(pixel[order], return_index=True)
        anchor_m.flat[unique] = depth[order[first]]*scale_lidar
        valid = anchor_m[anchor_m > 0]
        if len(valid) < 20:
            rows.append({'scene': scene, 'frame': frame, 'status': 'rejected', 'matches': len(valid)})
            continue
        anchor_scale_m = float(np.median(valid))
        target = output / scene / f'frame_{frame:06d}.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, anchor_depth_m=anchor_m, anchor_scale_m=anchor_scale_m,
                            points=int(len(valid)))
        rows.append({'scene': scene, 'frame': frame, 'status': 'usable',
                     'anchor_pixels': int(len(valid)), 'anchor_scale_m': anchor_scale_m,
                     'lidar_scale_m': scale_lidar,
                     'scale_ratio': anchor_scale_m/scale_lidar, 'target': str(target.resolve())})
    ratios = [r['scale_ratio'] for r in rows if r['status'] == 'usable']
    report = {'groups': len(rows), 'usable': len(ratios),
              'median_scale_ratio': float(np.median(ratios)),
              'p10_scale_ratio': float(np.percentile(ratios, 10)),
              'p90_scale_ratio': float(np.percentile(ratios, 90)),
              'rows': rows,
              'note': 'Scale from RGB feature triangulation with released poses; no LiDAR depth used to choose the new normalization denominator.'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/depth_student_diagnosis/direct_sfm_anchors'))
    args = parser.parse_args()
    run(args.output)
