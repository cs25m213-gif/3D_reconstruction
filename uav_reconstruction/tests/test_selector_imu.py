import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adaptive_uav.keyframe_policy import KeyframePolicy
from adaptive_uav.runtime_types import FramePacket, PoseEstimate
from adaptive_uav.selector_imu import GyroHistory


class SelectorImuTests(unittest.TestCase):
    def test_exact_timing_and_gap_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            metadata={'session_start_elapsed_ns':1_000_000_000,
                      'video_start_elapsed_ns':1_100_000_000}
            (path/'metadata.json').write_text(json.dumps(metadata))
            rows=['elapsed_ns,type,x,y,z']
            rows += [f'{1_000_000_000+i*20_000_000},4,0,0,1' for i in range(40)]
            (path/'sensors.csv').write_text('\n'.join(rows))
            history,reason=GyroHistory.from_session(path)
            self.assertIsNotNone(history)
            self.assertIn('exact',reason)
            self.assertAlmostEqual(history.motion(.1,.4)['angular_speed_rad_s'],1.)
            self.assertIsNone(history.motion(.7,.9))
            metadata.pop('video_start_elapsed_ns')
            (path/'metadata.json').write_text(json.dumps(metadata))
            self.assertIsNone(GyroHistory.from_session(path)[0])

    def test_bridge_keeps_translating_view_but_rejects_pure_rotation(self):
        rgb=np.random.default_rng(1).integers(0,255,(80,80,3),dtype=np.uint8)
        k=np.array([[80.,0,40],[0,80.,40],[0,0,1.]])
        old=FramePacket('a',0,0.,rgb,k)
        new=FramePacket('b',1,.3,rgb,k)
        points=np.tile(np.array([[[30.,30.]]],np.float32),(25,1,1))
        moved=points+np.array([5.,0.],np.float32)
        start=PoseEstimate('a',0.,np.eye(4),'test')
        translated=np.eye(4);translated[0,3]=.1
        end=PoseEstimate('b',.3,translated,'test')
        gyro={'angular_speed_rad_s':1.,'angular_motion_rad':.3}
        with patch('adaptive_uav.keyframe_policy.cv2.goodFeaturesToTrack',return_value=points), \
             patch('adaptive_uav.keyframe_policy.cv2.calcOpticalFlowPyrLK',return_value=(moved,np.ones((25,1),np.uint8),None)):
            baseline=KeyframePolicy(minimum_sharpness=0.)
            baseline.decide(old,start)
            self.assertFalse(baseline.decide(new,end)[0])
            adaptive=KeyframePolicy(minimum_sharpness=0.)
            adaptive.decide(old,start)
            keep,stats=adaptive.decide(new,end,gyro_motion=gyro)
            self.assertTrue(keep)
            self.assertEqual(stats['reason'],'rotation_bridge_with_geometric_content')
            stationary=KeyframePolicy(minimum_sharpness=0.)
            stationary.decide(old,start)
            self.assertFalse(stationary.decide(new,start,gyro_motion=gyro)[0])


if __name__=='__main__':
    unittest.main()
