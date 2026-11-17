import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('video_mesh_loop', ROOT / 'scripts/finetune_video_mesh.py')
loop = importlib.util.module_from_spec(spec); spec.loader.exec_module(loop)


class MeshLoopTests(unittest.TestCase):
    def poses(self):
        poses = np.repeat(np.eye(4)[None], 10, axis=0)
        poses[:, :3, 3] = np.array([[i, i % 3, i * .1] for i in range(10)])
        return poses

    def test_camera_alignment_recovers_scale_rotation_and_translation(self):
        poses = self.poses(); truth = poses.copy()
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        truth[:, :3, :3] = rotation
        truth[:, :3, 3] = 3.2 * (poses[:, :3, 3] @ rotation.T) + [5, -2, 8]
        scale, r, t, report = loop.fit_camera_alignment(poses, truth)
        self.assertAlmostEqual(scale, 3.2)
        np.testing.assert_allclose(r, rotation, atol=1e-8)
        np.testing.assert_allclose(t, [5, -2, 8])
        self.assertLess(report['heldout_position_median_m'], 1e-7)

    def test_bad_heldout_cameras_cannot_be_hidden_by_fit(self):
        poses = self.poses(); truth = poses.copy(); truth[1::2, :3, 3] += 20
        with self.assertRaises(ValueError): loop.fit_camera_alignment(poses, truth)

    def test_stationary_camera_scale_is_not_invented(self):
        poses = np.repeat(np.eye(4)[None], 10, axis=0)
        with self.assertRaises(ValueError): loop.fit_camera_alignment(poses, poses)

    def test_subset_uses_original_video_frames_without_reference_pose_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; (source / 'session').mkdir(parents=True)
            writer = cv2.VideoWriter(str(source / 'session/video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 2., (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(8): writer.write(np.full((48, 64, 3), i * 25, np.uint8))
            writer.release()
            info = {'P3x3': [[50, 0, 32], [0, 50, 24], [0, 0, 1]], 'Width': 64, 'Height': 48, 'T4x4': np.eye(4).tolist()}
            (source / 'selected_infos.json').write_text(json.dumps([info] * 8))
            chosen = [SimpleNamespace(index=i) for i in [1, 4, 7]]
            out = root / 'rollout/session'; selected = loop.make_session(source, chosen, out)
            self.assertEqual(len(selected), 3)
            self.assertNotIn('T4x4', (out / 'metadata.json').read_text())
            self.assertEqual(json.loads((out.parent / 'selection.json').read_text())['indices'], [1, 4, 7])
            capture = cv2.VideoCapture(str(out / 'video.mp4')); values = []
            while True:
                ok, frame = capture.read()
                if not ok: break
                values.append(float(frame.mean()))
            capture.release()
            np.testing.assert_allclose(values, [25, 100, 175], atol=10)


if __name__ == '__main__': unittest.main()
