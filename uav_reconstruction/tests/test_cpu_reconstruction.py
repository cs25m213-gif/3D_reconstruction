import tempfile
import unittest
import importlib.util
from pathlib import Path
import numpy as np
from adaptive_uav.geometry import weighted_umeyama, residuals
from adaptive_uav.session_reconstruct import read_model, retrieve_pairs, budget_keyframes


class GeometryTests(unittest.TestCase):
    def test_frame_budget_spans_input_without_duplicates(self):
        records=[{'blur':100.,'motion_pixels':float(i%20)} for i in range(1000)]
        indices=budget_keyframes(records,300)
        self.assertEqual(len(indices),300)
        self.assertEqual(indices,sorted(set(indices)))
        self.assertEqual(indices[0],0)
        self.assertEqual(indices[-1],999)
        self.assertLessEqual(max(np.diff(indices)),8)

    @unittest.skipUnless(importlib.util.find_spec('open3d'), 'optional dense Open3D runtime is not installed')
    def test_open3d_cpu_load_does_not_import_gpu_probing_package(self):
        import sys
        from adaptive_uav.open3d_cpu import cpu_bindings
        module = cpu_bindings()
        self.assertEqual(len(module.geometry.PointCloud().points), 0)
        self.assertNotIn('open3d', sys.modules)
        self.assertFalse(any(name.startswith('open3d.cuda') for name in sys.modules))

    def test_weighted_similarity_recovers_rotation_scale_translation(self):
        rng = np.random.default_rng(17)
        source = rng.normal(size=(30, 3))
        r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = 2.3 * source @ r.T + [7., -3., 1.]
        result = weighted_umeyama(source, target, np.linspace(.1, 1., 30))
        self.assertLess(residuals(source, target, result).max(), 1e-10)
        self.assertAlmostEqual(np.linalg.det(result[1]), 1.)

    def test_bounded_retrieval_has_no_self_or_duplicate_pairs(self):
        names = [f'frame_{i:03}.jpg' for i in range(50)]
        desc = np.random.default_rng(4).normal(size=(50, 10))
        desc /= np.linalg.norm(desc, axis=1, keepdims=True)
        pairs = retrieve_pairs(names, desc, temporal=3, retrieval=2)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(a != b for a, b in pairs))
        self.assertLessEqual(len(pairs), 50 * 5)
        self.assertIn((names[0], names[1]), pairs)

    def test_colmap_parser_handles_blank_point_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'cameras.txt').write_text('# cameras\n1 SIMPLE_RADIAL 640 480 500 320 240 0\n')
            (p / 'images.txt').write_text('# images\n1 1 0 0 0 0 0 0 1 a.jpg\n\n2 1 0 0 0 1 0 0 1 b.jpg\n10 10 1\n')
            (p / 'points3D.txt').write_text('# points\n1 0 0 1 255 0 0 0.2 1 0 2 0\n')
            cameras, poses, points = read_model(p)
            self.assertEqual(len(poses), 2)
            self.assertEqual(points[0]['observations'], 2)
            self.assertEqual(cameras[0]['parameters'][0], 500)

    def test_selective_tsdf_does_not_select_reliable_blocks(self):
        from adaptive_uav.selective_tsdf import uncertain_blocks
        points = [{'xyz': [float(i), 0., 0.], 'uncertainty': u}
                  for i,u in [(0,.8),(0,.8),(0,.8),(3,.1),(3,.1),(3,.1)]]
        blocks = uncertain_blocks(points, 1., .65, 1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][1], (0,0,0))


if __name__ == '__main__':
    unittest.main()
