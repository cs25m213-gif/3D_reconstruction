import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pycolmap

from adaptive_uav.combined_reconstruct import (frame_index, fusion_rank, learned_depth_options,
                                               run_parallel, select_camera_model, plausible_focal_lengths)
from adaptive_uav.camera_coverage import capture_coverage, result_status
from adaptive_uav.final_reconstruct import continue_checked_segment


class CombinedTests(unittest.TestCase):
    def test_shared_focal_model_respects_available_calibration(self):
        from adaptive_uav.pycolmap_backend import _camera_parameters
        config = {'camera_model_policy': 'shared-focal', 'camera_intrinsics':
                  {'width': 100, 'height': 80, 'fx': 80, 'fy': 80, 'cx': 50, 'cy': 40}}
        model, params = _camera_parameters(config, 200, 160)
        self.assertEqual(model, 'SIMPLE_PINHOLE')
        self.assertEqual(params, [160., 100., 80.])
        config['camera_intrinsics']['fy'] = 60
        with self.assertRaises(ValueError):
            _camera_parameters(config, 200, 160)
        config['camera_intrinsics']['fy'] = 80
        config['camera_intrinsics']['distortion'] = [.1]
        with self.assertRaises(ValueError):
            _camera_parameters(config, 200, 160)

    def test_focal_quality_checks_all_camera_focal_parameters(self):
        for model, params, expected in [
            ('SIMPLE_RADIAL', [80., 50., 40., 0.], True),
            ('PINHOLE', [80., 85., 50., 40.], True),
            ('PINHOLE', [80., 20., 50., 40.], False),
            ('OPENCV', [80., 85., 50., 40., 0., 0., 0., 0.], True),
        ]:
            camera = pycolmap.Camera(model=model, width=100, height=80, params=params)
            self.assertEqual(plausible_focal_lengths(camera), expected, model)

    def test_consistency_precedes_coverage(self):
        bad = {'quality_status': 'inconsistent_geometry',
               'mean_measured_depth_agreement_under_5pct': .5,
               'surface_audit': {'mean_ray_hit': {'completed': .99}}}
        good = {'quality_status': 'incomplete',
                'mean_measured_depth_agreement_under_5pct': .9,
                'surface_audit': {'mean_ray_hit': {'completed': .6}}}
        self.assertGreater(fusion_rank(good), fusion_rank(bad))

    def test_unknown_quality_does_not_win(self):
        self.assertLess(fusion_rank({'quality_status': 'unknown'}),
                        fusion_rank({'quality_status': 'incomplete'}))
        self.assertEqual(frame_index('frame_001533.jpg'), 1533)

    def test_full_coverage_uses_every_registered_camera(self):
        for fast in (False, True):
            with self.subTest(fast=fast):
                options = learned_depth_options('ws', 'assets', fast=fast, coverage='full')
                self.assertIn('--include-unmeasured', options)
                if fast:
                    # Subsampling would discard stereo-measured views among all cameras.
                    self.assertEqual(options[options.index('--max-frames')+1], 0)

    def test_measured_coverage_keeps_previous_policy(self):
        self.assertNotIn('--include-unmeasured', learned_depth_options('ws', 'assets', fast=False, coverage='measured'))
        options = learned_depth_options('ws', 'assets', fast=True, coverage='measured')
        self.assertNotIn('--include-unmeasured', options)
        self.assertEqual(options[options.index('--max-frames')+1], 20)


class ParallelStageTests(unittest.TestCase):
    def test_failure_stops_the_other_stage(self):
        slow = [sys.executable, '-c', 'import time; time.sleep(60)']
        failing = [sys.executable, '-c', 'raise SystemExit(3)']
        started = time.monotonic()
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            run_parallel([slow, failing], Path.cwd(), poll_seconds=.05)
        self.assertEqual(caught.exception.returncode, 3)
        self.assertLess(time.monotonic()-started, 30)

    def test_all_stages_succeed(self):
        quick = [sys.executable, '-c', 'pass']
        run_parallel([quick, quick], Path.cwd(), poll_seconds=.05)


class PartialCameraTests(unittest.TestCase):
    def make_segment(self, root, *, error=.8, focal=80., points=150, cameras=12):
        source=root/'sparse_run';folder=source/'online';images=folder/'images'
        images.mkdir(parents=True)
        for index in range(16):
            (images/f'frame_{index:06d}.jpg').touch()
        rec=pycolmap.Reconstruction()
        camera=pycolmap.Camera(camera_id=1,model='SIMPLE_PINHOLE',width=100,height=80,
                              params=[focal,50.,40.])
        rec.add_camera_with_trivial_rig(camera)
        xyz=np.column_stack((np.linspace(-1.,1.,points),np.zeros(points),np.full(points,5.)))
        indices=list(range(cameras-1))+[15]
        for index in indices:
            translation=np.array([-.1*index,0.,0.])
            observations=pycolmap.Point2DList([pycolmap.Point2D(camera.img_from_cam(p+translation)) for p in xyz])
            image=pycolmap.Image(image_id=index+1,camera_id=1,name=f'frame_{index:06d}.jpg',points2D=observations)
            rec.add_image_with_trivial_frame(image,pycolmap.Rigid3d(pycolmap.Rotation3d(),translation))
        for index,point in enumerate(xyz):
            track=pycolmap.Track([pycolmap.TrackElement(i+1,index) for i in indices])
            point_id=rec.add_point3D(point,track)
            rec.points3D[point_id].error=error
        model=folder/'blocks/0000/sparse/0';model.mkdir(parents=True);rec.write(model)
        (folder/'current_map.json').write_text(json.dumps({'binary_model':str(model)}))
        output=root/'output';output.mkdir()
        session=root/'session';session.mkdir();(session/'sensors.csv').touch()
        return source,session,output,model

    def test_valid_partial_segment_produces_output_but_cannot_claim_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            source,session,output,model=self.make_segment(Path(directory))
            with patch('adaptive_uav.combined_reconstruct.audit_gyro',return_value={'usable':True,'passed':True}):
                with self.assertRaisesRegex(RuntimeError,'interval_registration'):
                    select_camera_model(source,session,output,30.)
                row,_=select_camera_model(source,session,output,30.,allow_partial=True)
            self.assertTrue(row['geometry_passed'])
            self.assertTrue(row['partial_interval_accepted'])
            self.assertEqual(row['failed_checks'],['interval_registration'])
            self.assertEqual(row['registered_frames'],12)
            coverage=capture_coverage(row,1.)
            self.assertFalse(coverage['passed'])
            self.assertEqual(result_status(coverage,'internally_consistent_experimental'),'incomplete')
            self.assertEqual(pycolmap.Reconstruction(model).num_reg_images(),12)

    def test_partial_fallback_cannot_bypass_bad_gyro(self):
        with tempfile.TemporaryDirectory() as directory:
            source,session,output,_=self.make_segment(Path(directory))
            with patch('adaptive_uav.combined_reconstruct.audit_gyro',return_value={'usable':True,'passed':False}):
                with self.assertRaisesRegex(RuntimeError,'gyro_agreement'):
                    select_camera_model(source,session,output,30.,allow_partial=True)

    def test_startup_exclusion_cannot_bypass_minimum_camera_count(self):
        with tempfile.TemporaryDirectory() as directory:
            source,session,output,_=self.make_segment(Path(directory))
            audits=[{'usable':True,'passed':False},{'usable':True,'passed':True}]
            with patch('adaptive_uav.combined_reconstruct.audit_gyro',side_effect=audits):
                with self.assertRaisesRegex(RuntimeError,'minimum_camera_count'):
                    select_camera_model(source,session,output,30.,allow_partial=True)

    def test_partial_fallback_cannot_bypass_geometric_quality(self):
        cases=({'error':3.},'reprojection_error'),({'error':float('nan')},'reprojection_error'),({'focal':20.},'intrinsics'),({'points':149},'minimum_sparse_points'),({'cameras':11},'minimum_camera_count')
        for options,reason in cases:
            with self.subTest(reason=reason),tempfile.TemporaryDirectory() as directory:
                source,session,output,_=self.make_segment(Path(directory),**options)
                with patch('adaptive_uav.combined_reconstruct.audit_gyro',return_value={'usable':False}):
                    with self.assertRaisesRegex(RuntimeError,reason):
                        select_camera_model(source,session,output,30.,allow_partial=True)

    def test_optional_continuation_failure_keeps_the_checked_seed(self):
        source=Path('/saved/sparse_run');camera={'model':'/checked/segment'}
        with patch('adaptive_uav.final_reconstruct.continue_mapping',side_effect=RuntimeError('native mapper failed')):
            fallback,report=continue_checked_segment(camera,source,Path('/new/output'),90.)
        self.assertEqual(fallback,source)
        self.assertEqual(report['status'],'failed')
        self.assertTrue(report['fallback_geometry_checked'])
        self.assertEqual(report['fallback_model'],camera['model'])


if __name__ == '__main__':
    unittest.main()
