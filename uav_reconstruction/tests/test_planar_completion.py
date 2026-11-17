import unittest

import numpy as np

from adaptive_uav.planar_completion import fit_plane, plane_depth


class PlanarCompletionTests(unittest.TestCase):
    def test_sloped_plane_recovers_missing_depth_from_scattered_anchors(self):
        y, x = np.indices((80, 100), dtype=np.float64)
        matrix = np.array([[120., 0, 50], [0, 120., 40], [0, 0, 1]])
        actual = {'normal': np.array([.12, -.09, 1.]), 'offset': -4.}
        depth = plane_depth(actual, matrix, x, y)
        rng = np.random.default_rng(12)
        anchored = rng.random(depth.shape) < .15
        z = depth[anchored]
        xyz = np.column_stack(((x[anchored]-50)/120*z,
                               (y[anchored]-40)/120*z, z))
        fitted = fit_plane(xyz)
        self.assertIsNotNone(fitted)
        recovered = plane_depth(fitted, matrix, x, y)
        self.assertLess(np.median(np.abs(recovered-depth)/depth), 1e-5)

    def test_two_disjoint_depth_planes_are_not_accepted_as_one_wall(self):
        y, x = np.indices((80, 100), dtype=np.float64)
        z = np.where(x < 50, 3., 6.)
        xyz = np.column_stack(((x.ravel()-50)/120*z.ravel(),
                               (y.ravel()-40)/120*z.ravel(), z.ravel()))
        self.assertIsNone(fit_plane(xyz))


if __name__ == '__main__':
    unittest.main()
