import unittest

import cv2
import numpy as np

from adaptive_uav.build_cached_lidar_targets import downsample_sparse_min
from adaptive_uav.lidar_depth_supervision import extrinsics_for_scene, project_returns
from adaptive_uav import validate_reference_geometry as geometry


class SyntheticPlane:
    """Exact-truth nadir camera pair over a textured plane at known depth."""

    def __init__(self, depth=100., baseline=15., size=(2048, 2448), seed=3):
        self.size = size
        h, w = size
        self.K = np.array([[1467., 0, 1172.5], [0, 1467., 1048.2], [0, 0, 1]])
        self.poses = [np.eye(4), np.eye(4)]
        self.poses[1][0, 3] = baseline
        texture = np.random.default_rng(seed).random((h // 4, w // 4), dtype=np.float32)
        texture = cv2.GaussianBlur(cv2.resize(texture, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 2.)
        self.images = [np.uint8(255 * (texture - texture.min()) / np.ptp(texture))]
        # A fronto-parallel plane shifts rigidly by f * baseline / depth.
        shift = np.float32([[1, 0, -self.K[0, 0] * baseline / depth], [0, 1, 0]])
        self.images.append(cv2.warpAffine(self.images[0], shift, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
        self.depth = depth

    def gray(self, index):
        return self.images[index]

    def pose(self, index):
        return self.poses[index]

    def k(self, index):
        return self.K


class GeometryContractTests(unittest.TestCase):
    def test_pose_inversion_round_trip_is_physical(self):
        rotation = cv2.Rodrigues(np.array([.2, -.1, .3]))[0]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, [1, 2, 3]
        point = np.array([4., 5., 6., 1.])
        np.testing.assert_allclose(pose @ (np.linalg.inv(pose) @ point), point, atol=1e-12)

    def test_projection_uses_camera_z_not_range(self):
        r, t = extrinsics_for_scene('HKisland03')
        camera = np.array([[3., 4., 12.]])
        lidar = (camera - t) @ np.linalg.inv(r.T)
        info = {'OriginalSceneName': 'HKisland03', 'Width': 400, 'Height': 300, 'T4x4': np.eye(4).tolist(),
                'P3x3': [[100., 0, 200.], [0, 100., 150.], [0, 0, 1]]}
        depth, stats = project_returns(lidar, info)
        self.assertAlmostEqual(float(depth[depth > 0][0]), 12., places=3)
        self.assertAlmostEqual(stats['mean_range_minus_z_m'], 13. - 12., places=3)

    def test_behind_camera_points_are_rejected(self):
        r, t = extrinsics_for_scene('HKisland03')
        lidar = (np.array([[0., 0., -5.], [0., 0., 10.]]) - t) @ np.linalg.inv(r.T)
        info = {'OriginalSceneName': 'HKisland03', 'Width': 100, 'Height': 80, 'T4x4': np.eye(4).tolist(),
                'P3x3': [[100., 0, 50.], [0, 100., 40.], [0, 0, 1]]}
        self.assertEqual(project_returns(lidar, info)[1]['valid_pixels'], 1)

    def test_sparse_downsampling_keeps_nearest_and_never_interpolates(self):
        native = np.zeros((4, 4), np.float32)
        native[0, 0], native[1, 1] = 12., 8.
        small = downsample_sparse_min(native, (2, 2))
        self.assertEqual(small[0, 0], 8.)
        self.assertEqual(int((small > 0).sum()), 1)

    def test_pixel_chain_stays_within_half_a_grid_pixel_of_rgb(self):
        self.assertLessEqual(max(geometry.pixel_chain()['target_minus_rgb_grid_px_max_abs']), 1.)

    def test_constants_gate_finds_a_substituted_calibration(self):
        rows = geometry.check_constants(1e-6, 1e-3)
        self.assertEqual(rows['HKisland']['status'], 'PASS')
        self.assertFalse(rows['HKairport']['published_rotation_valid'])

    def test_decimal_fixed_rotation_is_orthonormal(self):
        fixed = geometry.rotation_candidates('HKairport')['published_decimal_fixed'][0]
        self.assertLess(geometry.rotation_error(fixed), 1e-3)

    def test_shift_peak_recovers_a_planted_offset(self):
        rng = np.random.default_rng(0)
        steps = np.arange(-geometry.SHIFT_RANGE, geometry.SHIFT_RANGE + 1, geometry.SHIFT_STEP)
        maps = []
        for _ in range(8):
            xx, yy = np.meshgrid(steps, steps)
            maps.append(np.exp(-((xx - 8) ** 2 + (yy + 6) ** 2) / 32.) + .02 * rng.standard_normal(xx.shape))
        result = geometry.peak(np.stack(maps))
        self.assertEqual((result['dx'], result['dy']), (8, -6))
        self.assertGreater(result['z'], 4)

    def sweep(self, assumed):
        scene = SyntheticPlane()
        rng = np.random.default_rng(1)
        h, w = scene.size
        uv = rng.uniform([200, 100], [w - 400, h - 100], (800, 2))
        rays = (uv - scene.K[[0, 1], [2, 2]]) / scene.K[[0, 1], [0, 1]]
        points = np.c_[rays * assumed, np.full(len(rays), assumed)]
        return geometry.sweep_frame(scene, 0, [1], points, rng), scene

    def test_plane_sweep_is_unbiased_on_exact_truth(self):
        result, scene = self.sweep(100.)
        self.assertGreater(len(result['scale']), 100)
        self.assertLess(abs(float(np.median(result['scale'])) - 1), .0026)

    def test_plane_sweep_recovers_a_known_depth_error(self):
        result, _ = self.sweep(100. / 1.03)
        self.assertLess(abs(float(np.median(result['scale'])) - 1.03), .0051)


if __name__ == '__main__':
    unittest.main()
