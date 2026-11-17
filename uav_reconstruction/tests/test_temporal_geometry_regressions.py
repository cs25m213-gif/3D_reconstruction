import unittest
import torch
from adaptive_uav.temporal_depth_student import warp_prior,warp_hidden

class TemporalGeometryRegressions(unittest.TestCase):
    def test_forward_warp_uses_previous_intrinsics_for_unprojection(self):
        depth=torch.zeros(1,1,3,6);depth[0,0,1,2]=2.
        previous=torch.eye(3)[None];current=previous.clone();current[:,0,0]=2.
        result,_,valid=warp_prior(depth,torch.ones_like(depth),current,torch.eye(4)[None],previous_k=previous)
        self.assertEqual(result[0,0,1,4],2.);self.assertEqual(float(valid.sum()),1.)

    def test_colliding_points_use_nearest_depth_and_its_confidence(self):
        depth=torch.zeros(1,1,2,4);depth[0,0,0,0]=4.;depth[0,0,0,1]=2.
        confidence=torch.zeros_like(depth);confidence[0,0,0,0]=.3;confidence[0,0,0,1]=.8
        previous=torch.eye(3)[None];current=previous.clone();current[:,0,0]=.1
        result,weight,valid=warp_prior(depth,confidence,current,torch.eye(4)[None],previous_k=previous)
        self.assertEqual(result[0,0,0,0],2.);self.assertAlmostEqual(float(weight[0,0,0,0]),.8,places=6)
        self.assertEqual(float(valid.sum()),1.)

    def test_equal_depth_collision_has_stable_source_confidence(self):
        depth=torch.zeros(1,1,2,4);depth[0,0,0,:2]=2.
        confidence=torch.zeros_like(depth);confidence[0,0,0,0]=.3;confidence[0,0,0,1]=.8
        previous=torch.eye(3)[None];current=previous.clone();current[:,0,0]=.1
        _,weight,_=warp_prior(depth,confidence,current,torch.eye(4)[None],previous_k=previous)
        self.assertAlmostEqual(float(weight[0,0,0,0]),.3,places=6)

    def test_hidden_warp_projects_into_previous_camera_and_masks_holes(self):
        hidden=torch.arange(6,dtype=torch.float32).reshape(1,1,1,6).expand(1,1,3,6)
        current=torch.eye(3)[None];current[:,0,0]=2.;previous=torch.eye(3)[None]
        depth=torch.zeros(1,1,3,6);depth[0,0,1,4]=2.
        result=warp_hidden(hidden,depth,current,torch.eye(4)[None],previous_k=previous)
        self.assertAlmostEqual(float(result[0,0,1,4]),2.,places=5)
        self.assertEqual(int(torch.count_nonzero(result)),1)

if __name__=='__main__':unittest.main()
