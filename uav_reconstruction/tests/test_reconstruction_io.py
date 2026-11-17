"""Frame access and cache compatibility checks, without GPU or training."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from adaptive_uav.adaptive_sfm import read_profile_frame
from adaptive_uav.depth_cache_io import save_depth_cache


class Capture:
    def __init__(self):
        self.position=0; self.seeks=[]; self.grabs=0
    def get(self, prop):
        return self.position
    def set(self, prop, index):
        self.position=index; self.seeks.append(index); return True
    def grab(self):
        self.position+=1; self.grabs+=1; return True
    def read(self):
        frame=self.position; self.position+=1; return True,frame


class ReconstructionIOTests(unittest.TestCase):
    def test_nearby_frame_reads_avoid_repeated_seek(self):
        cap=Capture()
        frames=[read_profile_frame(cap,i,8)[1] for i in (0,2,5,7)]
        self.assertEqual(frames,[0,2,5,7]);self.assertEqual(cap.seeks,[])

    def test_backwards_and_distant_frames_still_seek(self):
        cap=Capture()
        for i in (0,100,50):
            self.assertEqual(read_profile_frame(cap,i,8),(True,i))
        self.assertEqual(cap.seeks,[100,50])

    def test_grab_failure_is_reported(self):
        cap=Capture();cap.grab=lambda:False
        self.assertEqual(read_profile_frame(cap,2,8),(False,None))

    def test_fast_cache_preserves_all_array_types_and_values(self):
        arrays={'depth':np.array([[1.,np.nan]],dtype=np.float32),
                'rgb':np.arange(12,dtype=np.uint8).reshape(2,2,3),
                'context':np.array(['a.jpg','b.jpg']), 'scale':1.25,
                'validation':'{"valid":true}','mask':np.array([True,False])}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.npz';save_depth_cache(path,**arrays)
            with np.load(path,allow_pickle=False) as data:
                self.assertEqual(set(data.files),set(arrays))
                for key,value in arrays.items():
                    expected=np.asarray(value)
                    self.assertEqual(data[key].dtype,expected.dtype)
                    self.assertEqual(data[key].shape,expected.shape)
                    self.assertEqual(data[key].tobytes(),expected.tobytes())

    def test_bad_compression_setting_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.dict(os.environ,{'GEOSTREAM_CACHE_COMPRESSION':'20'}):
            path=Path(tmp)/'cache.npz'
            with self.assertRaisesRegex(ValueError,'COMPRESSION'):
                save_depth_cache(path,depth=np.ones(2))
            self.assertFalse(path.exists())


if __name__=='__main__':
    unittest.main()
