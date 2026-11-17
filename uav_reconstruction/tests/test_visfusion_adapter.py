import unittest
import numpy as np
from adaptive_uav.visfusion_infer import normalization, fragment_indices


class VisFusionAdapterTests(unittest.TestCase):
    def test_similarity_preserves_pixel_projection_and_camera_rotation(self):
        poses=np.repeat(np.eye(4)[None],2,axis=0)
        poses[0,:3,3]=[2.,3.,4.]
        poses[1,:3,3]=[2.5,3.,4.]
        transform,report=normalization(poses,4.)
        world=np.array([2.,3.,7.,1.])
        for pose in poses:
            converted=np.eye(4)
            converted[:3,:3]=transform[:3,:3]@pose[:3,:3]/report['scale']
            converted[:3,3]=(transform@np.r_[pose[:3,3],1.])[:3]
            original=np.linalg.inv(pose)@world
            new=np.linalg.inv(converted)@transform@world
            np.testing.assert_allclose(original[:2]/original[2],new[:2]/new[2],atol=1e-12)
            np.testing.assert_allclose(converted[:3,:3].T@converted[:3,:3],np.eye(3),atol=1e-12)
        self.assertFalse(report['metric_scale_validated'])
        np.testing.assert_allclose((transform@np.r_[poses[0,:3,3],1.])[:3],[0,0,1.5])

    def test_overlap_windows_include_last_view(self):
        windows=fragment_indices(36)
        self.assertTrue(all(len(w)==9 for w in windows))
        self.assertEqual(windows[-1][-1],35)
        self.assertEqual(set.union(*map(set,windows)),set(range(36)))
        with self.assertRaises(ValueError):fragment_indices(8)

    def test_invalid_depth_and_scale_rejected(self):
        poses=np.eye(4)[None]
        for depth in (0.,float('nan')):
            with self.assertRaises(ValueError):normalization(poses,depth)
        with self.assertRaises(ValueError):normalization(poses,2.,-1.)


if __name__=='__main__':unittest.main()
