import unittest
from pathlib import Path
import tempfile

from adaptive_uav.camera_coverage import capture_coverage, mapping_timeout, result_status, link_cached_candidates


class CameraCoverageTests(unittest.TestCase):
    def test_previous_coverage_model_survives_folder_name_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'previous';target=root/'new'
            previous=source/'online_coverage';previous.mkdir(parents=True)
            (previous/'current_map.json').write_text('{}')
            reserved=target/'online_coverage';reserved.mkdir(parents=True)
            (reserved/'new.txt').write_text('keep new mapping')
            link_cached_candidates(source,target)
            self.assertEqual((target/'online_cached_online_coverage').resolve(),previous.resolve())
            self.assertEqual((reserved/'new.txt').read_text(),'keep new mapping')

    def test_previous_session_segment_cannot_pass_capture_gate(self):
        camera = {'start_seconds': 18.915, 'end_seconds': 40.727,
                  'registered_frames': 51, 'full_run_selected_frames': 118}
        coverage = capture_coverage(camera, 54.27)
        self.assertFalse(coverage['passed'])
        self.assertEqual(result_status(coverage, 'internally_consistent_experimental'), 'incomplete')

    def test_endpoints_alone_do_not_prove_video_coverage(self):
        camera = {'start_seconds': .1, 'end_seconds': 59.9,
                  'registered_frames': 12, 'full_run_selected_frames': 120}
        self.assertFalse(capture_coverage(camera, 60)['passed'])

    def test_capture_wide_cameras_can_pass_but_incomplete_mesh_cannot(self):
        camera = {'start_seconds': .4, 'end_seconds': 59.4,
                  'registered_frames': 110, 'full_run_selected_frames': 120}
        coverage = capture_coverage(camera, 60)
        self.assertTrue(coverage['passed'])
        self.assertEqual(result_status(coverage, 'internally_consistent_experimental'), 'complete')
        self.assertEqual(result_status(coverage, 'incomplete'), 'incomplete')
        self.assertEqual(result_status(coverage, 'inconsistent_depth'), 'incomplete')
        self.assertEqual(result_status(coverage, None), 'incomplete')

    def test_guidance_does_not_silently_limit_90_seconds_to_30(self):
        self.assertEqual(mapping_timeout({'mapping_timeout_seconds': 90}, guided=True), 90)

    def test_explicit_guided_limit_and_unlimited_policy_are_honored(self):
        self.assertEqual(mapping_timeout({'mapping_timeout_seconds': 90,
                                         'guided_mapping_timeout_seconds': 45}, True), 45)
        self.assertEqual(mapping_timeout({'mapping_timeout_seconds': 0}, True), 0)

    def test_nonfinite_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            mapping_timeout({'mapping_timeout_seconds': float('nan')}, True)
        with self.assertRaises(ValueError):
            mapping_timeout({'guided_mapping_timeout_seconds': float('nan')}, True)


if __name__ == '__main__':
    unittest.main()
