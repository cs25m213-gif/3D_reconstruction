import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from adaptive_uav.sensor_audit import compare_rotations


class SensorAuditTests(unittest.TestCase):
    def test_frames_outside_sensor_extent_are_excluded_not_clamped(self):
        timestamps=np.arange(20,dtype=float);times=np.arange(.5,18.501,.01)
        camera=Rotation.from_rotvec(np.column_stack((np.zeros(20),np.zeros(20),.1*timestamps)))
        gyro=Rotation.from_rotvec(np.column_stack((np.zeros(len(times)),np.zeros(len(times)),.1*times)))
        report=compare_rotations(timestamps,camera,times,gyro,[0.])
        np.testing.assert_array_equal(report['indices'],np.arange(1,19))
        self.assertAlmostEqual(report['sensor_overlap_fraction'],.9)
        self.assertLess(report['median_rotation_error_deg'],.1)

    def test_insufficient_sensor_overlap_cannot_pass_audit(self):
        timestamps=np.arange(20,dtype=float);times=np.arange(5.,10.,.01)
        self.assertIsNone(compare_rotations(timestamps,Rotation.identity(20),times,Rotation.identity(len(times)),[0.]))

    def test_recorded_offset_reproduces_nonconstant_rotation(self):
        timestamps=np.arange(1.,12.);times=np.arange(0.,15.,.005)
        def poses(t):
            return Rotation.from_rotvec(np.column_stack((np.zeros(len(t)),np.zeros(len(t)),.04*t*t)))
        report=compare_rotations(timestamps,poses(timestamps+1.25),times,poses(times),[1.25])
        self.assertLess(report['median_rotation_error_deg'],.2)
        self.assertEqual(report['diagnostic_video_offset_seconds'],1.25)


if __name__=='__main__':unittest.main()
