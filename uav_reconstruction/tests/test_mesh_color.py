import unittest

import numpy as np
import open3d as o3d

from adaptive_uav.mesh_color import project_visible


class MeshColorTests(unittest.TestCase):
    def test_camera_colors_cannot_pass_through_foreground_wall(self):
        # Four interior samples on two parallel surfaces; the back one is hidden.
        vertices = np.array([[-1., -1., 2.], [1., -1., 2.], [1., 1., 2.], [-1., 1., 2.],
                             [-1., -1., 4.], [1., -1., 4.], [1., 1., 4.], [-1., 1., 4.]], np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.uint32)
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
        samples = np.array([[.3, .2, 2.], [-.3, -.2, 2.], [.3, .2, 4.], [-.3, -.2, 4.]])
        intrinsic = np.array([[80., 0., 50.], [0., 80., 50.], [0., 0., 1.]])
        indices, pixels, depths = project_visible(scene, samples, np.eye(4), intrinsic, (100, 100))
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_allclose(depths, 2.)
        np.testing.assert_allclose(pixels, [[62., 58.], [38., 42.]])

    def test_visibility_uses_camera_axis_depth_with_translated_camera(self):
        vertices = np.array([[-3., -3., 4.], [3., -3., 4.], [3., 3., 4.], [-3., 3., 4.]], np.float32)
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))
        pose = np.eye(4)
        pose[:3, 3] = [-1., 0., -1.]
        intrinsic = np.array([[80., 0., 50.], [0., 80., 50.], [0., 0., 1.]])
        samples = np.array([[2., .3, 4.], [2., .3, 6.], [100., 0., 4.], [1., 0., 0.]])
        indices, _, depths = project_visible(scene, samples, pose, intrinsic, (100, 100))
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_allclose(depths, 3.)


if __name__ == '__main__':
    unittest.main()
