"""Geometry and split-isolation checks for the remote dataset adapters."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from prepare_uavlight import images, rotation
from prepare_jb3d_metadata import corrected_text


class GeometryTests(unittest.TestCase):
    def test_translation_preserves_camera_coordinates(self):
        q = np.array([.9, .1, .2, -.3]); q /= np.linalg.norm(q)
        r = rotation(q)
        t, point, delta = np.array([2.,3.,4.]), np.array([5.,6.,7.]), np.array([.2,-.4,.3])
        text = '1 ' + ' '.join(map(str, q)) + ' 2 3 4 1 image.png\n\n'
        row = next(images(corrected_text(text, delta, True)))
        np.testing.assert_allclose(row['P'][:3,:3] @ (point+delta) + row['P'][:3,3], r @ point + t)
        points = corrected_text('9 5 6 7 10 20 30 0.5 1 0 2 0\n', delta, False).split()
        np.testing.assert_allclose(np.array(points[1:4], float), point+delta)
        self.assertEqual(points[4:], ['10','20','30','0.5','1','0','2','0'])

    def test_empty_observations_do_not_swallow_next_pose(self):
        text = '# header\n1 1 0 0 0 0 0 0 1 a.png\n\n2 1 0 0 0 1 0 0 1 b.png\n2 3 9\n'
        rows = list(images(text))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['obs'], [])
        self.assertEqual(rows[1]['obs'], [(2., 3., 9)])

    def test_manifest_never_reads_held_out_cache(self):
        from uav_reconstruction.adaptive_uav.student_dataset import auxiliary_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/'train'; cache.mkdir()
            for i in range(3):
                p = np.eye(4); p[0,3] = i*.1
                np.savez(cache/f'frame_{i:06d}.jpg.npz', P=p, anchors=np.ones((8,8)))
            rows = [dict(name='train',scene='tile',role='train',cache='train',frames=3,supervision='sfm_anchors_and_photometric'),
                    dict(name='test',scene='heldout',role='test',cache='does_not_exist',frames=3)]
            file = root/'manifest.json'
            file.write_text(json.dumps(dict(schema='geostreamfusion.auxiliary.v1', clips=rows,failures=[])))
            status = root/'status.json'; status.write_text('{"state":"complete"}')
            self.assertEqual([c.name for c in auxiliary_manifest(file)], ['train'])
            status.write_text('{"state":"preparing"}')
            with self.assertRaises(ValueError): auxiliary_manifest(file)


if __name__ == '__main__':
    unittest.main()
