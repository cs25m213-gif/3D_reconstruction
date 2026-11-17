import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from adaptive_uav.session_inputs import inspect_session,input_hashes,sensor_summary


class SessionInputTests(unittest.TestCase):
    def create_session(self,root):
        video=cv2.VideoWriter(str(root/'video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10.,(320,180))
        self.assertTrue(video.isOpened())
        for _ in range(8):video.write(np.full((180,320,3),140,np.uint8))
        video.release()
        metadata={'complete':True,'session_start_elapsed_ns':1_000_000_000,'duration_seconds':1.,
            'camera_intrinsics':{'lens_intrinsic_calibration':[2943.75,2943.75,0.,0.,0.],'pixel_array_size':[4000,3000]}}
        (root/'metadata.json').write_text(json.dumps(metadata))
        (root/'camera_intrinsics.json').write_text(json.dumps({'is_available':False,'intrinsic_calibration':[],
                                                            'pixel_array_width':4000,'pixel_array_height':3000}))
        (root/'stabilization_profile.json').write_text(json.dumps({'electronic_eis':True,'imu_rate_hz':0,'gyroflow_sync_ready':True}))
        with (root/'sensors.csv').open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['elapsed_ns','type','x','y','z'])
            for i in range(120):writer.writerow([2_000_000_000+i*2_500_000,4,0.,0.,.1])
        with (root/'gps.csv').open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['elapsed_ns','latitude','longitude','accuracy_m','altitude_m'])
            for i in range(5):writer.writerow([2_000_000_000+i*100_000_000,13.+i*.00001,79.,3.8,40.])

    def test_unavailable_calibration_and_eis_do_not_freeze_metadata_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.create_session(root)
            report,config=inspect_session(root)
            self.assertIsNone(config['camera_intrinsics'])
            self.assertFalse(config['freeze_guided_intrinsics'])
            self.assertTrue(config['refine_intrinsics'])
            self.assertFalse(config['calibration_exact'])
            self.assertAlmostEqual(config['focal_ratio'],.7359375)
            self.assertAlmostEqual(report['imu']['types']['4']['median_rate_hz'],400.)
            self.assertFalse(report['timing']['exact_start_timestamp_recorded'])
            self.assertFalse(report['gps']['metric_scale_used'])
            self.assertTrue(report['files_stable_during_inspection'])

    def test_explicit_video_calibration_is_used_when_stabilization_is_off(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.create_session(root)
            raw={'is_available':True,'calibration_exact':True,'width':320,'height':180,
                 'fx':250.,'fy':251.,'cx':160.,'cy':90.,'distortion':[0.]*5}
            (root/'camera_intrinsics.json').write_text(json.dumps(raw))
            (root/'stabilization_profile.json').write_text(json.dumps({'electronic_eis':False}))
            report,config=inspect_session(root)
            self.assertEqual(config['camera_intrinsics']['fy'],251.)
            self.assertTrue(config['calibration_exact'])
            self.assertFalse(config['refine_intrinsics'])
            self.assertTrue(config['freeze_guided_intrinsics'])

    def test_sensor_array_calibration_is_not_silently_scaled_into_video_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.create_session(root)
            raw={'is_available':True,'pixel_array_width':4000,'pixel_array_height':3000,
                 'intrinsic_calibration':[2943.,2943.,2000.,1500.,0.]}
            (root/'camera_intrinsics.json').write_text(json.dumps(raw))
            _,config=inspect_session(root)
            self.assertIsNone(config['camera_intrinsics'])
            self.assertFalse(config['calibration_exact'])

    def test_active_recording_is_not_processed_as_a_completed_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.create_session(root)
            metadata=json.loads((root/'metadata.json').read_text());metadata['complete']=False
            (root/'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'still recording'):inspect_session(root)

    def test_new_sidecars_are_included_in_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.create_session(root)
            before=input_hashes(root)
            self.assertIn('camera_intrinsics.json',before)
            self.assertIn('stabilization_profile.json',before)
            (root/'stabilization_profile.json').write_text(json.dumps({'electronic_eis':False}))
            after=input_hashes(root)
            self.assertNotEqual(before['stabilization_profile.json'],after['stabilization_profile.json'])

    def test_invalid_sensor_samples_are_reported_without_fabricating_a_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sensors.csv'
            path.write_text('elapsed_ns,type,x,y,z\n100,4,nan,0,0\n200,4,0,0,0\n')
            report=sensor_summary(path)
            self.assertEqual(report['invalid_rows'],1)
            self.assertIsNone(report['types']['4']['median_rate_hz'])


if __name__=='__main__':unittest.main()
