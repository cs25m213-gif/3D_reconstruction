import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from adaptive_uav.dense_views import pair_quality


class DenseViewTests(unittest.TestCase):
    def test_pure_rotation_has_no_triangulation_support(self):
        # Even abundant feature overlap cannot recover depth at one center.
        points = np.array([[0., 0., 5.], [1., .2, 5.], [-1., .3, 6.]])
        self.assertAlmostEqual(pair_quality(np.zeros(3), np.zeros(3), points), 0., places=5)

    def test_sideways_baseline_has_useful_triangulation(self):
        points = np.array([[0., 0., 5.], [.2, .1, 5.], [-.2, 0., 5.]])
        angle = pair_quality(np.zeros(3), np.array([1., 0., 0.]), points)
        self.assertGreater(angle, 10.)
        self.assertLess(angle, 12.)

    def test_very_distant_scene_is_poor_depth_even_with_baseline(self):
        angle = pair_quality(np.zeros(3), np.array([1., 0., 0.]), np.array([[0., 0., 1000.]]))
        self.assertLess(angle, .8)

    def test_metadata_initialization_keeps_crop_uncertainty_explicit(self):
        from adaptive_uav.fast_session import camera_initialization
        metadata = {'camera_intrinsics': {'lens_intrinsic_calibration': [3025, 3025, 1632, 1224, 0],
                                         'pixel_array_size': [4096, 3072]}}
        result = camera_initialization(metadata, 720, 1280)
        self.assertFalse(result['calibration_exact'])
        self.assertAlmostEqual(result['initial_focal_px'], 3025 * 1280 / 3264)

    def test_mesh_filter_rejects_triangles_bridging_unobserved_space(self):
        from adaptive_uav.supported_mesh import support_mask
        x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        measured = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [100., 0., 0.], [0., 100., 0.]])
        cloud = np.vstack([measured, vertices[3:]])
        keep = support_mask(vertices, np.array([[0, 1, 2], [0, 3, 4]]), cloud, .2)
        self.assertEqual(keep.tolist(), [True, False])

    def test_gyro_audit_catches_accumulated_pose_drift(self):
        from scipy.spatial.transform import Rotation
        from adaptive_uav.sensor_audit import audit_gyro
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata.json').write_text(json.dumps({'session_start_elapsed_ns': 0}))
            rows = ['elapsed_ns,wall_ms,sensor,type,x,y,z,accuracy']
            for j in range(1201):
                rows.append(f'{j * 10000000},0,gyro,4,0,0,{np.deg2rad(10)},3')
            (root / 'sensors.csv').write_text('\n'.join(rows))
            def model(rate):
                images = {}
                for j in range(6):
                    rotation = Rotation.from_euler('z', rate * j * 2, degrees=True).as_matrix()
                    pose = SimpleNamespace(rotation=SimpleNamespace(matrix=lambda r=rotation: r))
                    images[j] = SimpleNamespace(name=f'frame_{j * 60:06d}.jpg', has_pose=True,
                                                cam_from_world=lambda p=pose: p)
                return SimpleNamespace(images=images)
            with patch('pycolmap.Reconstruction', return_value=model(10)):
                self.assertTrue(audit_gyro(root, '/unused', 30)['passed'])
            with patch('pycolmap.Reconstruction', return_value=model(40)):
                self.assertFalse(audit_gyro(root, '/unused', 30)['passed'])

    def test_viewer_recovers_missing_mesh_counts_from_artifact(self):
        import trimesh
        from adaptive_uav.mesh_viewer import write_mesh_viewer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / 'assets'
            assets.mkdir()
            for name in ['three.module.js', 'PLYLoader.js', 'OrbitControls.js',
                         'BufferGeometryUtils.js', 'GLTFLoader_local.js']:
                (assets / name).write_text('// fixture')
            trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                            faces=[[0, 1, 2]], process=False).export(root / 'scene_mesh.ply')
            (root / 'dense_metrics.json').write_text(json.dumps({'dense_points': 3}))
            viewer = write_mesh_viewer(root, assets)
            metrics = json.loads((root / 'dense_metrics.json').read_text())
            self.assertEqual(metrics['mesh_faces'], 1)
            self.assertEqual(metrics['mesh_vertices'], 3)
            self.assertTrue(viewer.is_file())
