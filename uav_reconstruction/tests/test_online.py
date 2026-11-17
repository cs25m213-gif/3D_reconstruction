import unittest
import tempfile
import subprocess
import sys
from pathlib import Path
from adaptive_uav.online import past_references, model_is_valid
from adaptive_uav.phone_source import RecordedVideoSource
from adaptive_uav.session_reconstruct import SessionReconstructor
import cv2
import numpy as np


class OnlineTests(unittest.TestCase):
    def test_saved_video_is_sampled_causally_with_a_time_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'video.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,(80,60))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(12):
                    writer.write(np.full((60,80,3),index*15,dtype=np.uint8))
            finally:
                writer.release()
            source=RecordedVideoSource(video,sample_fps=2,max_seconds=1)
            frames=list(source.frames())
            self.assertEqual([frame.index for frame in frames],[0,3])
            self.assertLessEqual(max(frame.timestamp_s for frame in frames),1)
            self.assertEqual(source.signature['max_video_seconds'],1)

    def test_tool_timeout_stops_a_stalled_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            runner=SessionReconstructor(root,root/'output',{'threads':1},Path('/unused/colmap'),root/'mvs')
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run_command('timeout_test',[sys.executable,'-c','import time; time.sleep(2)'],timeout=.05)

    def test_references_are_causal_even_if_future_descriptors_exist(self):
        desc=np.eye(10)
        refs=past_references(desc,4,temporal=2,visual=1)
        self.assertTrue(all(i<4 for i in refs))
        self.assertIn(3,refs)
        self.assertEqual(past_references(desc,0),[])

    def test_model_losing_existing_camera_is_rejected(self):
        c={'min_registered_images':2,'min_sparse_points':2,'max_median_reprojection_error_px':2.}
        poses=[{'image':'b'},{'image':'c'}]
        points=[{'error_px':.2},{'error_px':.3}]
        valid,reason=model_is_valid(['a','b'],poses,points,c)
        self.assertFalse(valid)
        self.assertEqual(reason,'previous registered cameras lost')


if __name__=='__main__':
    unittest.main()
