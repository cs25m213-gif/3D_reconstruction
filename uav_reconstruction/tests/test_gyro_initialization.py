import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptive_uav.gyro_initialization import rotation_difference, choose_initial_pair


class GyroInitializationTests(unittest.TestCase):
    def test_rotation_guidance_matches_known_motion(self):
        t=np.linspace(0,10,1001)
        poses=Rotation.from_rotvec(np.column_stack((np.zeros_like(t),np.zeros_like(t),.2*t)))
        error,offset=rotation_difference(t,poses,2.,3.,np.degrees(.2))
        self.assertLess(error,.2)
        wrong,_=rotation_difference(t,poses,2.,3.,80.)
        self.assertGreater(wrong,60.)

    def test_no_sensor_log_returns_explicit_no_guidance(self):
        with tempfile.TemporaryDirectory() as d:
            result=choose_initial_pair(Path(d)/'database.db',Path(d),30.,None)
            self.assertFalse(result['used'])

    def test_outside_sensor_extent_is_unusable(self):
        t=np.array([0.,1.])
        self.assertIsNone(rotation_difference(t,Rotation.identity(2),10.,11.,0.))


if __name__=='__main__':unittest.main()
