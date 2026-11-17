"""Scientific-contract checks for newly admitted selector geometry."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from uav_reconstruction.adaptive_uav.prepare_multisource_selector import cloud, emit, partition, project


class MultisourceGeometryTests(unittest.TestCase):
    def test_projection_uses_world_to_camera_and_nearest_surface(self):
        K=np.array([[100.,0,128],[0,100,96],[0,0,1]])
        W=np.eye(4);W[:3,3]=[10,20,30]
        # Two points along the optical axis: z-buffer must retain five metres.
        points=np.array([[10.,20.,35.],[10.,20.,38.], [10.,20.,29.]])
        depth=project(points,np.linalg.inv(W),K)
        self.assertEqual(np.count_nonzero(depth),1)
        self.assertEqual(depth[96,128],5.)
        np.testing.assert_allclose(cloud(depth,K,W),points[:1])

    def test_repeated_world_cells_cannot_cross_anchor_reference_boundary(self):
        points=np.array([[1.001,2.001,3.001],[1.002,2.002,3.002],[1.011,2.001,3.001]])
        groups=partition(points)
        self.assertEqual(groups[0],groups[1])
        self.assertNotEqual(groups[0],groups[2])

    def test_short_tail_is_retained_without_reusing_scored_cells_as_anchors(self):
        K=np.array([[100.,0,128],[0,100,96],[0,0,1]])
        depth=np.zeros((192,256));depth[20:170:4,30:230:4]=5.
        points=cloud(depth,K,np.eye(4))
        rows=[dict(rgb=np.zeros((192,256,3),np.uint8),K=K,P=np.eye(4),points=points,index=i) for i in range(35)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'geometry').mkdir()
            paths=emit(root,'test','synthetic','train',rows,'test_only')
            indices=[i for p in paths for i in json.loads(Path(p).with_suffix('.json').read_text())['source_indices']]
            self.assertEqual(indices,list(range(35)))
            for p in paths:
                with np.load(p) as z:self.assertTrue(np.all(~partition(z['reference_points'])))


if __name__=='__main__':unittest.main()
