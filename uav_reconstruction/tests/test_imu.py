import json
import unittest

import numpy as np

from adaptive_uav.imu import resample_imu, rotation_homography, summarize_imu


def sample(timestamp, gyro=(0.0, 0.0, 1.0), accel=(0.0, 9.81, 0.0)):
    return {"timestamp_s": timestamp, "accel": list(accel), "gyro": list(gyro)}


class IMUWindowTests(unittest.TestCase):
    def test_integrates_measured_gyro_and_retains_gravity(self):
        samples = [sample(float(t)) for t in np.linspace(10.0, 10.5, 101)]
        features = summarize_imu(samples, 10.0, 10.5)
        self.assertTrue(features["valid"])
        self.assertTrue(features["rotation_valid"])
        self.assertAlmostEqual(features["effective_rate_hz"], 200)
        np.testing.assert_allclose(features["gyro_integrated_rad"], [0, 0, .5])
        self.assertAlmostEqual(features["angular_path_rad"], .5)
        self.assertAlmostEqual(features["gyro_rms_rad_s"], 1)
        self.assertAlmostEqual(features["accel_norm_mean_mps2"], 9.81)
        self.assertTrue(features["acceleration_includes_gravity"])
        json.dumps(features, allow_nan=False)

    def test_direction_changes_preserve_angular_path(self):
        samples = [sample(float(t), gyro=(0, 0, 2 * t - 1)) for t in np.linspace(0, 1, 101)]
        features = summarize_imu(samples, 0, 1)
        self.assertAlmostEqual(features["angular_displacement_rad"], 0, places=10)
        self.assertAlmostEqual(features["angular_path_rad"], .5)

    def test_missing_and_duplicate_samples_do_not_fabricate_rate(self):
        samples = [sample(.5), sample(0), sample(1), sample(.5), sample(-1),
                   {"timestamp_s": .7, "accel": [0, 0, 0]}, sample(float("nan"))]
        features = summarize_imu(samples, 0, 1)
        self.assertEqual(features["count"], 3)
        self.assertEqual(features["duplicate_samples"], 1)
        self.assertEqual(features["invalid_samples"], 2)
        self.assertEqual(features["outside_window_samples"], 1)
        self.assertFalse(features["valid"])
        self.assertFalse(features["rotation_valid"])
        self.assertAlmostEqual(features["effective_rate_hz"], 2)
        np.testing.assert_allclose(features["gyro_integrated_rad"], [0, 0, 1])

    def test_low_rate_api_snapshots_are_explicitly_unfit_for_compensation(self):
        samples = [sample(float(t)) for t in np.linspace(0, 1, 11)]
        resampled, metadata = resample_imu(samples, 0, 1)
        self.assertTrue(metadata["valid"])
        self.assertFalse(metadata["rotation_valid"])
        self.assertTrue(metadata["low_sample_rate"])
        self.assertEqual(resampled.shape, (64, 6))
        self.assertAlmostEqual(metadata["effective_rate_hz"], 10)
        features = summarize_imu(samples, 0, 1)
        self.assertIsNone(rotation_homography(features, np.eye(3), np.eye(3)))

    def test_resampling_is_sorted_and_linearly_interpolated(self):
        samples = [sample(float(t), gyro=(2 * t, 0, 0), accel=(t, 0, 9.81))
                   for t in np.linspace(0, .1, 11)[::-1]]
        resampled, metadata = resample_imu(samples, 0, .1, fixed_samples=7)
        targets = np.linspace(0, .1, 7)
        np.testing.assert_allclose(resampled[:, 0], targets, atol=1e-7)
        np.testing.assert_allclose(resampled[:, 3], 2 * targets, atol=1e-7)
        self.assertTrue(metadata["rotation_valid"])
        self.assertEqual(metadata["count"], 11)

    def test_missing_coverage_and_long_gaps_are_invalid(self):
        partial = [sample(float(t)) for t in np.linspace(.3, .7, 81)]
        features = summarize_imu(partial, 0, 1)
        self.assertFalse(features["valid"])
        self.assertAlmostEqual(features["coverage_fraction"], .4)
        self.assertAlmostEqual(features["integrated_duration_s"], .4)
        self.assertAlmostEqual(features["gyro_integrated_rad"][2], .4)
        with_gap = [sample(float(t)) for t in np.concatenate((np.linspace(0, .2, 41),
                                                              np.linspace(.8, 1, 41)))]
        self.assertFalse(summarize_imu(with_gap, 0, 1)["valid"])

    def test_empty_or_single_sample_has_fixed_shape_but_is_invalid(self):
        empty, metadata = resample_imu([], 0, .5)
        self.assertEqual(empty.shape, (64, 6))
        self.assertFalse(metadata["valid"])
        self.assertTrue(np.all(empty == 0))
        values, metadata = resample_imu([sample(.25)], 0, .5)
        self.assertFalse(metadata["valid"])
        self.assertTrue(np.all(values[:, 4] == 0))
        json.dumps(summarize_imu([], 0, .5), allow_nan=False)
        with self.assertRaises(ValueError):
            resample_imu([], 1, 1)
        with self.assertRaises(ValueError):
            resample_imu([], 0, 1, fixed_samples=1)

    def test_rotation_homography_requires_calibrated_proper_extrinsic(self):
        features = summarize_imu([sample(float(t)) for t in np.linspace(0, .1, 21)], 0, .1)
        intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
        self.assertIsNone(rotation_homography(features, intrinsic, None))
        self.assertIsNone(rotation_homography(features, intrinsic, np.diag([1, 1, -1])))
        self.assertIsNone(rotation_homography(features, intrinsic, np.eye(3) * 2))
        homography = rotation_homography(features, intrinsic, np.eye(3))
        # Positive camera rotation about optical Z moves the static scene clockwise.
        target = homography @ np.array([420, 240, 1])
        target /= target[2]
        self.assertAlmostEqual(target[0], 320 + 100 * np.cos(.1))
        self.assertAlmostEqual(target[1], 240 - 100 * np.sin(.1))


if __name__ == "__main__":
    unittest.main()
