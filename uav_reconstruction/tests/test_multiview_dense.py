import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import pycolmap
from adaptive_uav.multiview_dense import remap_camera, fit_group_scale, evidence_masks, correct_local_scale, fuse_scale_groups, validation_anchors, spatial_holdout


class MultiViewTests(unittest.TestCase):
    def test_added_learned_views_cannot_overpower_strong_stereo_weight(self):
        def inputs():
            frames=[]
            for depth_value,tier in ((2.,1),(2.1,3),(2.1,3),(2.1,3)):
                depth=np.full((48,64),depth_value,np.float32)
                measured=depth.copy() if tier==1 else np.zeros_like(depth)
                frames.append({'depth':depth,'measured':measured,'anchors':depth.copy(),
                    'tier':np.full(depth.shape,tier,np.uint8),'rgb':np.full((48,64,3),150,np.uint8),
                    'P':np.eye(4),'K':np.array([[60.,0,32],[0,60.,24],[0,0,1]]),'depth_limit':6.})
            return frames
        weak,_=fuse_scale_groups(inputs(),voxel_fraction=.02,measured_repeat=2)
        strong,_=fuse_scale_groups(inputs(),voxel_fraction=.02,measured_repeat=32)
        weak_error=abs(np.median(np.asarray(weak['completed'].vertices)[:,2])-2.)
        strong_error=abs(np.median(np.asarray(strong['completed'].vertices)[:,2])-2.)
        self.assertLess(strong_error,weak_error/2)

    def test_sparse_feature_anchors_never_become_stereo_measured_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            camera=pycolmap.Camera(camera_id=1,model='PINHOLE',width=100,height=80,
                                   params=[60.,60.,50.,40.])
            observation=SimpleNamespace(point3D_id=7,xy=np.array([50.,40.]),has_point3D=lambda:True)
            image=SimpleNamespace(camera_id=1,points2D=[observation],cam_from_world=lambda:pycolmap.Rigid3d())
            landmark=SimpleNamespace(xyz=np.array([0.,0.,3.]),error=.5,track=SimpleNamespace(length=lambda:3))
            model=SimpleNamespace(cameras={1:camera},points3D={7:landmark})
            measured,anchors,source=validation_anchors(Path(directory),'frame.jpg',model,image,camera.calibration_matrix(),(80,100))
            self.assertFalse(measured.any())
            self.assertEqual(source,'triangulated_sfm_tracks')
            self.assertEqual(np.count_nonzero(anchors),1)
            self.assertEqual(anchors[40,50],3.)

    def test_sparse_scale_validation_uses_held_out_feature_pixels(self):
        depth=np.ones((80,80),np.float32);anchors=np.zeros_like(depth)
        hold=spatial_holdout(depth.shape)
        train=np.flatnonzero(~hold);test=np.flatnonzero(hold)
        anchors.ravel()[train[::max(1,len(train)//40)][:40]]=3.
        anchors.ravel()[test[::max(1,len(test)//15)][:15]]=6.
        scale,checks=fit_group_scale([depth],[anchors],sparse=True)
        self.assertAlmostEqual(scale,3.)
        self.assertEqual(checks[0]['pixels'],15)
        self.assertAlmostEqual(checks[0]['median'],.5)
        self.assertTrue(checks[0]['scale_fit_valid'])

    def test_fusion_retains_near_and_far_depth_ranges(self):
        frames=[]
        for z in (1.,100.):
            depth=np.full((48,64),z,np.float32)
            frames.append({'depth':depth,'measured':depth,'tier':np.ones(depth.shape,np.uint8),
                'rgb':np.full((48,64,3),180,np.uint8),'P':np.eye(4),
                'K':np.array([[60.,0,32],[0,60.,24],[0,0,1]]),'depth_limit':3*z})
        meshes,levels=fuse_scale_groups(frames)
        self.assertEqual(len(levels),2)
        vertices=np.asarray(meshes['completed'].vertices)
        self.assertGreater(vertices[:,2].max(),95.)
        self.assertLess(vertices[:,2].min(),1.1)

    def test_local_correction_recovers_smooth_scale_distortion(self):
        y,x=np.indices((160,120))
        true=np.full((160,120),3.,np.float32)
        distorted=(true*(.7+.6*x/120)).astype(np.float32)
        corrected,report=correct_local_scale(distorted,true)
        self.assertTrue(report['used'])
        self.assertLess(report['after_median'],.03)
        self.assertLess(report['after_p90'],report['before_p90'])

    def test_local_correction_does_not_invent_missing_measurements(self):
        prior=np.full((50,50),2.,np.float32)
        corrected,report=correct_local_scale(prior,np.zeros_like(prior))
        self.assertFalse(report['used'])
        np.testing.assert_array_equal(corrected,prior)

    def test_camera_remap_preserves_crop_offset(self):
        image = np.arange(100, dtype=np.float32).reshape(10,10)
        source = np.array([[5.,0,5],[0,5,5],[0,0,1]])
        target = source.copy()
        target[0,2] -= 2
        result = remap_camera(image, source, target, (10,8))
        np.testing.assert_array_equal(result, image[:,2:])

    def test_common_scale_keeps_relative_depths(self):
        depths = [np.ones((80,80),np.float32), np.full((80,80),2.,np.float32)]
        measured = [d*3 for d in depths]
        measured[0][:4] = 300 # minority outliers
        scale, errors = fit_group_scale(depths,measured)
        self.assertAlmostEqual(scale,3.)
        self.assertAlmostEqual(errors[1]['median'],0.)

    def test_occluded_surface_is_not_free_space_conflict(self):
        k=np.array([[50.,0,32],[0,50,24],[0,0,1]])
        p=np.eye(4)
        ref={'prior':np.full((48,64),2.),'K':k,'P':p}
        q=np.eye(4); q[0,3]=-.2
        other={'prior':np.full((48,64),1.),'measured':np.full((48,64),1.),'K':k,'P':q}
        supported, conflict=evidence_masks(ref,[other])
        self.assertFalse(conflict.any())
        self.assertFalse(supported.any())
        other['measured'][:]=3
        _, conflict=evidence_masks(ref,[other])
        self.assertTrue(conflict[10:30,20:40].all())

    def test_measured_source_can_support_plane(self):
        k=np.array([[50.,0,32],[0,50,24],[0,0,1]])
        ref={'prior':np.full((48,64),2.),'K':k,'P':np.eye(4)}
        p=np.eye(4);p[0,3]=-.2
        other={'prior':np.full((48,64),2.),'measured':np.full((48,64),2.),'K':k,'P':p}
        supported, conflict=evidence_masks(ref,[other])
        self.assertTrue(supported[10:30,20:40].all())
        self.assertFalse(conflict.any())


if __name__=='__main__': unittest.main()
