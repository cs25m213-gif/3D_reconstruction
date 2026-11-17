"""Conservative rotation-only prediction from tracked image features.

The residual is an image-space novelty cue, not a translation or depth
measurement. Reject unstable fits instead of reporting false parallax.
"""
import cv2
import numpy as np


def rotation_compensated_motion(previous, current, K):
    """Return median residual pixels and fit diagnostics, or (None, reason)."""
    a = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    if len(a) != len(b) or len(a) < 30 or K.shape != (3, 3):
        return None, {'reason': 'insufficient_tracks'}
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(K).all():
        return None, {'reason': 'nonfinite_tracks_or_intrinsics'}
    try:
        H, inliers = cv2.findHomography(a.astype(np.float32), b.astype(np.float32),
                                        cv2.RANSAC, 2.0)
        if H is not None and inliers is not None and inliers.mean() >= .55:
            calibrated = np.linalg.solve(K, H @ K)
            determinant = np.linalg.det(calibrated)
            if not np.isfinite(determinant) or determinant <= 0:
                return None, {'reason': 'invalid_homography'}
            U, singular, Vt = np.linalg.svd(calibrated / np.cbrt(determinant))
            if singular[-1] <= 0 or singular[0] / singular[-1] > 1.7:
                return None, {'reason': 'homography_not_rotation_like'}
            flip = np.diag([1., 1., np.linalg.det(U @ Vt)])
            R = U @ flip @ Vt
            method = 'homography'
            inlier_fraction = float(inliers.mean())
        else:
            # A changing viewpoint of a nonplanar scene cannot fit one
            # homography. Essential-matrix geometry separates its rotation.
            E, inliers = cv2.findEssentialMat(a, b, K, cv2.RANSAC, .999, 1.)
            if E is None or inliers is None or inliers.mean() < .55:
                return None, {'reason': 'no_consistent_camera_motion'}
            count, R, _, pose_inliers = cv2.recoverPose(E[:3, :3], a, b, K, mask=inliers)
            if count < .5 * len(a) or pose_inliers is None:
                return None, {'reason': 'weak_relative_pose'}
            method = 'essential_matrix'
            inlier_fraction = count / len(a)
        rays = np.column_stack((a, np.ones(len(a)))) @ np.linalg.inv(K).T
        rotated = rays @ R.T @ K.T
        z = rotated[:, 2]
        valid = z > 1e-6
        if valid.mean() < .9:
            return None, {'reason': 'rotation_projects_outside_image'}
        predicted = rotated[valid, :2] / z[valid, None]
        residual = np.linalg.norm(b[valid] - predicted, axis=1)
        if not np.isfinite(residual).all():
            return None, {'reason': 'nonfinite_rotation_residual'}
        return float(np.median(residual)), {'reason': 'ok',
            'method': method, 'inlier_fraction': inlier_fraction,
            'rotation_angle_deg': float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))}
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return None, {'reason': 'rotation_fit_failed'}
