from types import SimpleNamespace
import unittest

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from adaptive_uav.learned_camera import crop_camera, similarity, connected_to_existing,rectified_pixel,opencv_distortion


class LearnedCameraTests(unittest.TestCase):
    def test_fitted_radial_lens_rectifies_real_features_into_neural_coordinates(self):
        camera=pycolmap.Camera(camera_id=1,model='SIMPLE_RADIAL',width=100,height=80,params=[80.,50.,40.,.1])
        point=np.array([.3,.2,1.]);distorted=camera.img_from_cam(point)
        rectified=rectified_pixel(camera,distorted)
        expected=camera.calibration_matrix()@point
        np.testing.assert_allclose(rectified,expected[:2],atol=1e-6)
        np.testing.assert_allclose(opencv_distortion(camera),[.1,0,0,0,0])
        intrinsic,affine,axis=crop_camera(camera,90)
        projected=intrinsic@(axis@point)
        np.testing.assert_allclose((affine@np.r_[rectified,1.])[:2],projected[:2]/projected[2],atol=1e-6)

    def test_rotated_crop_preserves_pixel_ray_correspondence(self):
        camera=pycolmap.Camera(camera_id=1,model='PINHOLE',width=1280,height=720,
                              params=[942.,948.,640.,360.])
        point=np.array([.4,-.2,3.])
        original=camera.img_from_cam(point)
        for angle in (0,90,180,270):
            intrinsic,affine,axis=crop_camera(camera,angle)
            projected=intrinsic@(axis@point);projected=projected[:2]/projected[2]
            np.testing.assert_allclose(projected,(affine@np.r_[original,1.])[:2],atol=1e-8)
        intrinsic,affine,_=crop_camera(camera,90)
        np.testing.assert_allclose(affine@np.array([640.,360.,1.]),[111.68888888888888,112.,1.])

    def test_similarity_uses_proper_rotation_and_visual_scale(self):
        rng=np.random.default_rng(42);points=rng.normal(size=(100,3))
        rotation=Rotation.from_euler('xyz',[.2,-.5,.6]).as_matrix()
        target=3.7*points@rotation.T+np.array([20.,-4.,8.])
        scale,fitted,translation=similarity(points,target)
        self.assertAlmostEqual(scale,3.7)
        np.testing.assert_allclose(fitted,rotation,atol=1e-8)
        np.testing.assert_allclose(scale*points@fitted.T+translation,target,atol=1e-8)
        _,reflection,_=similarity(points,points*np.array([-1.,1.,1.]))
        self.assertAlmostEqual(np.linalg.det(reflection),1.)

    def test_learned_track_components_need_a_link_to_existing_map(self):
        images={i:SimpleNamespace(has_pose=True) for i in range(1,6)}
        points={n:SimpleNamespace(track=SimpleNamespace(elements=[SimpleNamespace(image_id=i) for i in ids]))
                for n,ids in enumerate(((1,2),(2,3),(4,5)))}
        rec=SimpleNamespace(images=images,points3D=points)
        self.assertEqual(connected_to_existing(rec,{1}),{1,2,3})


if __name__=='__main__':
    unittest.main()
