import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_uav.dense_coverage import choose_sources, spatial_weights
from adaptive_uav.depth_normals import fit_depth_normals, read_array, write_array


class AutomaticDenseTests(unittest.TestCase):
    def test_source_selection_covers_regions_instead_of_repeating_one_region(self):
        xy = np.array([[10, 10]] * 6 + [[90, 90]] * 2)
        weights = spatial_weights(xy, 100, 100)
        dense = np.array([True] * 6 + [False] * 2)
        rare = ~dense
        masks = {f'a{i}': dense for i in range(4)}
        masks.update(b1=rare, b2=rare)
        chosen, supported = choose_sources(masks, weights, 4)
        self.assertTrue({'b1', 'b2'}.issubset(chosen))
        self.assertAlmostEqual(supported, 1.)

    def test_inverse_depth_plane_recovers_tilted_normal_without_moving_depth(self):
        y, x = np.indices((64, 64), dtype=float)
        depth = 1/(.25 + .0002*x - .0001*y)
        saved = depth.copy()
        original = np.zeros((64, 64, 3), np.float32)
        original[:, :, 2] = -1
        k = np.array([[80., 0, 32], [0, 80., 32], [0, 0, 1]])
        fitted, report = fit_depth_normals(depth, original, k)
        expected = -np.array([.0002*80, -.0001*80, .25+.0002*32-.0001*32])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(fitted[32, 32], expected, atol=1e-5)
        np.testing.assert_array_equal(depth, saved)
        self.assertGreater(report['fraction_of_valid_depth'], .9)

    def test_depth_boundary_does_not_become_a_sloped_bridge(self):
        depth = np.full((64, 64), 3.)
        depth[:, 32:] = 6.
        original = np.zeros((64, 64, 3), np.float32)
        original[:, :, 1] = 1
        k = np.array([[80., 0, 32], [0, 80., 32], [0, 0, 1]])
        fitted, _ = fit_depth_normals(depth, original, k)
        np.testing.assert_array_equal(fitted[32, 31:33], original[32, 31:33])

    def test_colmap_array_roundtrip_preserves_channel_layout(self):
        values = np.arange(7*11*3, dtype=np.float32).reshape(7, 11, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'normals.bin'
            write_array(path, values)
            np.testing.assert_array_equal(read_array(path), values)


if __name__ == '__main__':
    unittest.main()
