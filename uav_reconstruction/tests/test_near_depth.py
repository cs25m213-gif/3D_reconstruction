import unittest
import numpy as np
from adaptive_uav.multiview_dense import restrict_depth


class NearDepthTests(unittest.TestCase):
    def frame(self):
        return dict(depth=np.array([[1., 2., 10.]], dtype=np.float32),
                    measured=np.array([[1., 2., 10.]], dtype=np.float32),
                    tier=np.array([[1, 2, 3]], dtype=np.uint8), depth_limit=30.)

    def test_all_evidence_tiers_are_filtered_without_changing_anchors(self):
        f = self.frame()
        restrict_depth(f, max_depth=1.5)
        np.testing.assert_array_equal(f['tier'], [[1, 0, 0]])
        np.testing.assert_array_equal(f['depth'], [[1, 0, 0]])
        np.testing.assert_array_equal(f['measured'], [[1, 2, 10]])

    def test_relative_limit_and_absolute_cap(self):
        f = self.frame()
        report = restrict_depth(f, near_percentile=50)
        self.assertEqual(report['limit_visual_units'], 2.)
        self.assertEqual(report['removed_pixels'], 1)
        f = self.frame()
        self.assertEqual(restrict_depth(f, 1.5, 50)['limit_visual_units'], 1.5)

    def test_invalid_limits_rejected(self):
        for value in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                restrict_depth(self.frame(), max_depth=value)
        with self.assertRaises(ValueError):
            restrict_depth(self.frame(), near_percentile=101)
