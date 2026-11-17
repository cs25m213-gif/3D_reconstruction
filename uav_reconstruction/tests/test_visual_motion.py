import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from adaptive_uav.visual_motion import rotation_compensated_motion


class VisualMotionTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[360., 0., 180.], [0., 360., 100.], [0., 0., 1.]])
        rng = np.random.default_rng(7)
        self.points = np.column_stack((rng.uniform(-2, 2, 240),
                                       rng.uniform(-1, 1, 240),
                                       rng.uniform(5, 15, 240)))

    def project(self, points):
        p = points @ self.K.T
        return p[:, :2] / p[:, 2, None]

    def test_rotation_only_has_near_zero_residual(self):
        R = Rotation.from_euler('y', 8, degrees=True).as_matrix()
        a = self.project(self.points)
        b = self.project(self.points @ R.T)
        error, fit = rotation_compensated_motion(a, b, self.K)
        self.assertEqual(fit['reason'], 'ok')
        self.assertLess(error, .05)
        self.assertGreater(float(np.median(np.linalg.norm(b-a,axis=1))), 40)

    def test_simultaneous_rotation_and_translation_retains_parallax(self):
        R = Rotation.from_euler('y', 8, degrees=True).as_matrix()
        a = self.project(self.points)
        b = self.project(self.points @ R.T + [0.8, 0., 0.])
        error, fit = rotation_compensated_motion(a, b, self.K)
        self.assertEqual(fit['reason'], 'ok')
        self.assertGreater(error, 2.)
        self.assertLess(error, float(np.median(np.linalg.norm(b-a,axis=1))))

    def test_no_tracks_does_not_claim_translation(self):
        error, fit = rotation_compensated_motion(np.zeros((3, 2)), np.ones((3, 2)), self.K)
        self.assertIsNone(error)
        self.assertEqual(fit['reason'], 'insufficient_tracks')


if __name__ == '__main__':
    unittest.main()
