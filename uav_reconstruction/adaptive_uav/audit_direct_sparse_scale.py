"""Compare RGB-only triangulated anchor scale with direct-group LiDAR scale."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .direct_lidar_dataset import DirectLidarDataset


def triangulated_depths(rgb0, rgb1, k0, k1, p0, p1):
    sift = cv2.SIFT_create(nfeatures=1200)
    gray0 = cv2.cvtColor((rgb0.transpose(1, 2, 0)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor((rgb1.transpose(1, 2, 0)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    points0, desc0 = sift.detectAndCompute(gray0, None)
    points1, desc1 = sift.detectAndCompute(gray1, None)
    if desc0 is None or desc1 is None or len(points0) < 20 or len(points1) < 20:
        return np.empty((0, 2)), np.empty(0)
    pairs = cv2.BFMatcher().knnMatch(desc0, desc1, k=2)
    good = [m for m, n in pairs if m.distance < .7*n.distance]
    if len(good) < 16:
        return np.empty((0, 2)), np.empty(0)
    xy0 = np.float32([points0[m.queryIdx].pt for m in good])
    xy1 = np.float32([points1[m.trainIdx].pt for m in good])
    essential, inliers = cv2.findEssentialMat(xy0, xy1, k0, method=cv2.RANSAC,
                                              prob=.999, threshold=1.5)
    if inliers is None or int(inliers.sum()) < 12:
        return np.empty((0, 2)), np.empty(0)
    xy0, xy1 = xy0[inliers.ravel() > 0], xy1[inliers.ravel() > 0]
    proj0, proj1 = k0@p0[:3], k1@p1[:3]
    homogeneous = cv2.triangulatePoints(proj0, proj1, xy0.T, xy1.T)
    xyz = (homogeneous[:3] / homogeneous[3]).T
    camera0 = xyz@p0[:3, :3].T+p0[:3, 3]
    camera1 = xyz@p1[:3, :3].T+p1[:3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv0 = (camera0@k0.T)[:, :2]/(camera0@k0.T)[:, 2:]
        uv1 = (camera1@k1.T)[:, :2]/(camera1@k1.T)[:, 2:]
    error = np.linalg.norm(uv0-xy0, axis=1)+np.linalg.norm(uv1-xy1, axis=1)
    valid = ((camera0[:, 2] > .25) & (camera0[:, 2] < 4) & (camera1[:, 2] > .25) &
             (error < 3) & np.isfinite(camera0).all(1))
    return xy0[valid], camera0[valid, 2]


def run(output: Path, samples: int):
    dataset = DirectLidarDataset(Path('runs/uavscenes_multiflight'),
                                 Path('runs/uavscenes_multiflight_measured'),
                                 max_normalized_baseline=.16, augment=False)
    rows = []
    for index in np.linspace(0, len(dataset)-1, min(samples, len(dataset))).astype(int):
        scene, frame, sources, _ = dataset.groups[index]
        item = dataset[index]
        rgb, k, p = (item[name].numpy() for name in ('rgb', 'K', 'P'))
        depths = [triangulated_depths(rgb[0], rgb[j], k[0], k[j], p[0], p[j]) for j in (1, 2)]
        valid = np.concatenate([depth for _, depth in depths])
        row = {'scene': scene, 'frame': frame, 'rgb_triangulated_points': int(len(valid)),
               'lidar_scale_m': float(item['metric_per_normalized']),
               'rgb_anchor_to_lidar_scale_ratio': float(np.median(valid)) if len(valid) >= 12 else None}
        rows.append(row)
    ratios = [r['rgb_anchor_to_lidar_scale_ratio'] for r in rows if r['rgb_anchor_to_lidar_scale_ratio'] is not None]
    report = {'sampled_groups': len(rows), 'usable_groups': len(ratios),
              'median_rgb_anchor_to_lidar_scale_ratio': float(np.median(ratios)) if ratios else None,
              'p10_ratio': float(np.percentile(ratios, 10)) if ratios else None,
              'p90_ratio': float(np.percentile(ratios, 90)) if ratios else None,
              'rows': rows,
              'note': 'RGB features and released reference poses only for triangulation; LiDAR median is denominator of input pose. Ratio 1 means comparable scale, but does not verify pose or LiDAR pixel alignment.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/depth_student_diagnosis/direct_sparse_scale_audit.json'))
    parser.add_argument('--samples', type=int, default=60)
    args = parser.parse_args()
    run(args.output, args.samples)
