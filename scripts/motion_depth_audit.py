"""Test whether tracked pixel motion adds trustworthy depth to missing regions."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'uav_reconstruction'))
from adaptive_uav.hybrid_dense import camera_matrix, read_array


def track_pair(gray, other, seed, projection, other_projection, center, other_center):
    if seed is None or not len(seed):
        return np.zeros(0, bool), np.zeros(0, np.float32)
    seed = seed.reshape(-1, 1, 2).astype(np.float32)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(gray, other, seed, None, winSize=(21, 21),
                                                maxLevel=3)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(other, gray, moved, None,
                                                 winSize=(21, 21), maxLevel=3)
    good = (status[:, 0] > 0) & (reverse[:, 0] > 0) & (
        np.linalg.norm(back[:, 0] - seed[:, 0], axis=1) < 1.5)
    indices = np.flatnonzero(good)
    accepted = np.zeros(len(seed), bool)
    depth = np.zeros(len(seed), np.float32)
    if not len(indices):
        return accepted, depth
    points1 = np.ascontiguousarray(seed[indices, 0].T, dtype=np.float64)
    points2 = np.ascontiguousarray(moved[indices, 0].T, dtype=np.float64)
    points = cv2.triangulatePoints(projection, other_projection, points1, points2)
    with np.errstate(divide='ignore', invalid='ignore'):
        xyz = (points[:3] / points[3]).T
    world_h = np.c_[xyz, np.ones(len(xyz))]
    u1 = (projection @ world_h.T).T
    u2 = (other_projection @ world_h.T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        q1, q2 = u1[:, :2] / u1[:, 2:3], u2[:, :2] / u2[:, 2:3]
    ray1, ray2 = xyz - center, xyz - other_center
    norm = np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1)
    cosine = np.sum(ray1 * ray2, axis=1) / np.maximum(norm, 1e-9)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    good3d = (np.isfinite(xyz).all(axis=1) & (u1[:, 2] > 0) & (u2[:, 2] > 0) &
              (angle >= 3) & (angle <= 65) &
              (np.linalg.norm(q1 - seed[indices, 0], axis=1) < 2) &
              (np.linalg.norm(q2 - moved[indices, 0], axis=1) < 2))
    accepted[indices[good3d]] = True
    # The third coordinate of the first projective camera is camera-space Z.
    depth[indices[good3d]] = u1[good3d, 2]
    return accepted, depth


def main():
    import pycolmap
    workspace = ROOT / 'runs/session_20261011_202810_corrected/dense_hybrid_trial'
    plan = json.loads((workspace / 'dense_plan.json').read_text())['plan']
    reconstruction = pycolmap.Reconstruction(workspace / 'sparse')
    images = {image.name: image for image in reconstruction.images.values()}
    rows = []
    for row in plan[::16]:
        name = row['reference']
        image = images[name]
        gray = cv2.imread(str(workspace / 'images' / name), cv2.IMREAD_GRAYSCALE)
        measured = read_array(workspace / 'stereo/depth_maps' / (name + '.geometric.bin'))
        if gray.shape != measured.shape:
            gray = cv2.resize(gray, (measured.shape[1], measured.shape[0]))
        K = camera_matrix(reconstruction.cameras[image.camera_id], gray.shape)
        pose = image.cam_from_world().matrix()
        P = K @ pose
        center = image.cam_from_world().inverse().matrix()[:3, 3]
        candidates = {}
        for label, mask in (('missing', measured <= 0), ('measured', measured > 0)):
            corners = cv2.goodFeaturesToTrack(gray, 2500, .01, 5,
                                               mask=(mask.astype(np.uint8) * 255))
            candidates[label] = corners[:, 0] if corners is not None else np.zeros((0, 2), np.float32)
        accepted = {label: np.zeros(len(seed), bool) for label, seed in candidates.items()}
        estimated = {label: np.zeros(len(seed), np.float32) for label, seed in candidates.items()}
        for source in row['sources'][:3]:
            other_image = images[source['name']]
            other = cv2.imread(str(workspace / 'images' / source['name']), cv2.IMREAD_GRAYSCALE)
            if other.shape != gray.shape:
                other = cv2.resize(other, (gray.shape[1], gray.shape[0]))
            Ko = camera_matrix(reconstruction.cameras[other_image.camera_id], other.shape)
            Po = Ko @ other_image.cam_from_world().matrix()
            other_center = other_image.cam_from_world().inverse().matrix()[:3, 3]
            for label, seed in candidates.items():
                good, depth = track_pair(gray, other, seed, P, Po, center, other_center)
                new = good & ~accepted[label]
                accepted[label] |= good
                estimated[label][new] = depth[new]
        check = candidates['measured']
        checked = accepted['measured']
        x = np.rint(check[checked, 0]).astype(int)
        y = np.rint(check[checked, 1]).astype(int)
        valid = (x >= 0) & (x < measured.shape[1]) & (y >= 0) & (y < measured.shape[0])
        error = np.abs(estimated['measured'][checked][valid] - measured[y[valid], x[valid]]) / np.maximum(measured[y[valid], x[valid]], 1e-6)
        rows.append({'frame': name,
                     'missing_fraction': float((measured <= 0).mean()),
                     'missing_corners': len(candidates['missing']),
                     'missing_triangulated': int(accepted['missing'].sum()),
                     'measured_corners': len(candidates['measured']),
                     'measured_triangulated': int(accepted['measured'].sum()),
                     'measured_median_relative_error': float(np.median(error)) if len(error) else None,
                     'measured_p90_relative_error': float(np.percentile(error, 90)) if len(error) else None})
    summary = {'frames': len(rows),
               'missing_triangulated_mean': float(np.mean([r['missing_triangulated'] for r in rows])),
               'measured_triangulated_mean': float(np.mean([r['measured_triangulated'] for r in rows])),
               'measured_median_relative_error_median': float(np.median([r['measured_median_relative_error'] for r in rows if r['measured_median_relative_error'] is not None])),
               'rows': rows}
    (workspace / 'motion_depth_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
