import unittest

import numpy as np

from adaptive_uav.hybrid_dense import (cross_view, edge_invalid, huber_fit, infer_frame)


class Args:
    tol = .035
    min_agree = 2
    extrapolation_px = 150
    plane_margin_fraction = .5


class FakeRigid:
    def __init__(self, matrix):
        self.matrix4 = matrix

    def matrix(self):
        return self.matrix4[:3, :4]

    def inverse(self):
        return FakeRigid(np.linalg.inv(self.matrix4))


class FakeCamera:
    width, height = 64, 48

    def calibration_matrix(self):
        return np.array([[60., 0, 32], [0, 60., 24], [0, 0, 1]])


class FakeImage:
    camera_id = 1

    def __init__(self, name, x):
        self.name = name
        matrix = np.eye(4)
        matrix[0, 3] = -x          # camera centre at +x
        self.pose = FakeRigid(matrix)

    def cam_from_world(self):
        return self.pose


class FakeModel:
    cameras = {1: FakeCamera()}


def plane_depth(image, depth=2.):
    """A fronto-parallel wall at world z=depth seen from a camera translated in x."""
    return np.full((48, 64), depth, np.float32)


class HybridDenseTests(unittest.TestCase):
    def test_huber_fit_ignores_outliers(self):
        rng = np.random.default_rng(0)
        d = rng.uniform(.5, 4, 4000)
        m = 1.8 * d + .3 + rng.normal(0, .02, 4000)
        m[:400] += rng.uniform(1, 5, 400)
        scale, shift = huber_fit(d, m)
        self.assertAlmostEqual(scale, 1.8, delta=.05)
        self.assertAlmostEqual(shift, .3, delta=.1)

    def test_depth_step_is_removed(self):
        depth = np.full((40, 40), 2., np.float32)
        depth[:, 20:] = 3.
        bad = edge_invalid(depth)
        self.assertTrue(bad[:, 19:21].all())
        self.assertFalse(bad[:, :10].any())

    def test_plane_extension_fills_hole_but_not_different_colour_object(self):
        K = np.array([[300., 0, 160], [0, 300., 120], [0, 0, 1]])
        depth = np.full((240, 320), 2., np.float32)
        depth[80:160, 100:220] = 0
        rgb = np.full((240, 320, 3), 200, np.uint8)
        base = {'K': K, 'P': np.eye(4)}
        normal = np.array([0, 0, 1.])
        u = np.array([1., 0, 0])
        plane = {'normal': normal, 'offset': -2., 'u': u, 'v': np.cross(normal, u),
                 'lo': np.array([-1.5, -1.5]), 'hi': np.array([1.5, 1.5]), 'inliers': 9999}
        plain = dict(base, depth=depth.copy(), tier=(depth > 0).astype(np.uint8), rgb=rgb)
        self.assertEqual(infer_frame(plain, [plane], Args, 2.), 9600)
        self.assertTrue(np.allclose(plain['depth'][80:160, 100:220], 2.))
        self.assertTrue((plain['tier'][80:160, 100:220] == 3).all())
        object_rgb = rgb.copy()
        object_rgb[100:140, 130:190] = (20, 60, 20)
        dark = dict(base, depth=depth.copy(), tier=(depth > 0).astype(np.uint8), rgb=object_rgb)
        self.assertLess(infer_frame(dark, [plane], Args, 2.), 9600)
        self.assertTrue((dark['tier'][105:135, 140:180] == 0).all())

    def test_cross_view_needs_separated_cameras(self):
        reference = FakeImage('frame_000001.jpg', 0.)
        K = FakeCamera().calibration_matrix()
        depth = plane_depth(reference)
        candidate = np.zeros(depth.shape, bool)
        candidate[10:30, 20:40] = True
        far = [FakeImage('frame_000002.jpg', .3), FakeImage('frame_000003.jpg', -.3)]
        same = [FakeImage('frame_000004.jpg', 0.), FakeImage('frame_000005.jpg', .001)]
        combined = {image.name: plane_depth(image) for image in far + same}
        kept = cross_view(candidate, depth, reference, K, far, combined, FakeModel(), Args)
        self.assertGreater(kept.sum(), .5 * candidate.sum())
        # near-identical viewpoints add no parallax evidence, so nothing is confirmed
        rejected = cross_view(candidate, depth, reference, K, same, combined, FakeModel(), Args)
        self.assertEqual(int(rejected.sum()), 0)


if __name__ == '__main__':
    unittest.main()
