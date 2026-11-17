import sqlite3
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from adaptive_uav.adaptive_sfm import (PAIR_BASE, choose_settings, match_graph,
                                      profile_video, quality_passes, recovery_settings)


class AdaptiveSfMTests(unittest.TestCase):
    def test_overlap_decay_controls_sampling(self):
        profile = {'matchability': {'.2': .5, '.5': .3, '1.0': .1, '2.0': .02},
                   'median_sift_features': 900, 'median_flow_px': 10, 'median_blur': 80}
        settings = choose_settings(profile)
        self.assertAlmostEqual(settings['max_keyframe_gap_seconds'], .4)
        self.assertGreaterEqual(settings['sample_fps'], 5)
        self.assertEqual(settings['min_motion_pixels'], 5)
        profile['matchability']['2.0'] = .3
        self.assertGreater(choose_settings(profile)['max_keyframe_gap_seconds'], .4)

    def test_graph_includes_isolates_and_locates_breaks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'database.db'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE images (image_id INTEGER, name TEXT)')
                db.execute('CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)')
                db.executemany('INSERT INTO images VALUES (?, ?)', [(1, 'frame_001.jpg'), (2, 'frame_002.jpg'), (3, 'frame_003.jpg')])
                db.executemany('INSERT INTO two_view_geometries VALUES (?, ?)', [(PAIR_BASE+2, 70), (2*PAIR_BASE+3, 4)])
            graph = match_graph(path)
            self.assertEqual(graph['component_count'], 2)
            self.assertEqual(graph['temporal_breaks'], [['frame_002.jpg', 'frame_003.jpg']])
            metrics = {'registration_fraction': .9, 'median_reprojection_error_px': .7,
                       'registered_frames': 10, 'sparse_points': 1000}
            self.assertFalse(quality_passes(metrics, graph))
            self.assertTrue(quality_passes(metrics, {'available': True, 'component_count': 1}))
            metrics['median_reprojection_error_px'] = float('nan')
            self.assertFalse(quality_passes(metrics, {'available': True, 'component_count': 1}))
            self.assertIsNone(recovery_settings({}, metrics))

    def test_recovery_is_bounded_and_preserves_calibration(self):
        config = {'max_keyframe_gap_seconds': .4, 'min_motion_pixels': 5.,
                  'sample_fps': 5., 'temporal_references': 12,
                  'retrieval_references': 4, 'max_features': 8000, 'focal_ratio': .8}
        for _ in range(10):
            config = recovery_settings(config, {'median_reprojection_error_px': .7})
        self.assertEqual(config['max_features'], 16000)
        self.assertEqual(config['temporal_references'], 36)
        self.assertEqual(config['retrieval_references'], 16)
        self.assertEqual(config['sample_fps'], 15)
        self.assertEqual(config['focal_ratio'], .8)
        self.assertIsNone(recovery_settings(config, {'median_reprojection_error_px': 3.}))

    def test_profiler_on_generated_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'video.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(17)
            image = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
            for i in range(30):
                writer.write(np.roll(image, i, axis=1))
            writer.release()
            result = profile_video(path, analysis_width=160, samples=10)
            self.assertGreater(result['sampled_windows'], 0)
            self.assertIsNotNone(result['median_flow_px'])
            self.assertGreater(result['matchability']['0.2'], .15)
            self.assertFalse(result['gyro_used'])


if __name__ == '__main__':
    unittest.main()
