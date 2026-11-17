import tempfile
import json
from pathlib import Path
import unittest

import cv2
import numpy as np
import pycolmap

from adaptive_uav.dense_resolution import prepare_dense_inputs, matching_camera_resolution, validate_cached_workspace


class DenseResolutionTests(unittest.TestCase):
    def make_cache_pair(self,root):
        cached=root/'old_run/dense_recovery';current=root/'current'
        rec=pycolmap.Reconstruction()
        rec.add_camera_with_trivial_rig(pycolmap.Camera(camera_id=1,model='PINHOLE',width=320,height=180,
                                                      params=[200.,200.,160.,90.]))
        names=[]
        for i in range(8):
            name=f'frame_{i:06d}.jpg';names.append(name)
            rec.add_image_with_trivial_frame(pycolmap.Image(image_id=i+1,camera_id=1,name=name),
                                            pycolmap.Rigid3d(pycolmap.Rotation3d(),np.array([-.1*i,0.,0.])))
        for path in (cached,current):
            (path/'sparse').mkdir(parents=True);(path/'images').mkdir()
            rec.write(path/'sparse')
            for name in names:(path/'images'/name).write_bytes(b'captured RGB bytes')
        (cached/'stereo/depth_maps').mkdir(parents=True)
        for name in names:
            with (cached/'stereo/depth_maps'/(name+'.geometric.bin')).open('wb') as stream:
                stream.write(b'320&180&1&');stream.write(np.full((320,180,1),3.,np.float32).tobytes(order='F'))
        configuration='same reference/source cameras\n'
        (cached/'stereo/patch-match.cfg').write_text(configuration)
        identity={'path':'video.mp4','size_bytes':100,'mtime_ns':1}
        (cached.parent/'pipeline_manifest.json').write_text(json.dumps({'video':identity}))
        (cached/'depth_settings.json').write_text(json.dumps({'max_image_size':320,'iterations':2,
                'filter_min_triangulation_angle':3.,'configuration':configuration,'references':names}))
        (cached/'dense_plan.json').write_text(json.dumps({'max_reference_views':32,'max_source_views':4}))
        return cached,current,identity,{'dense_size':320,'stereo_iterations':2},rec

    def test_stereo_reuse_checks_pixels_even_when_poses_match(self):
        with tempfile.TemporaryDirectory() as directory:
            cached,current,identity,detail,_=self.make_cache_pair(Path(directory))
            report=validate_cached_workspace(cached,current,identity,detail,32)
            self.assertEqual(report['references'],8)
            (current/'images/frame_000000.jpg').write_bytes(b'different captured pixels')
            with self.assertRaisesRegex(ValueError,'RGB differs'):
                validate_cached_workspace(cached,current,identity,detail,32)

    def test_stereo_reuse_rejects_camera_pose_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            cached,current,identity,detail,rec=self.make_cache_pair(Path(directory))
            rec.frames[1].set_cam_from_world(1,pycolmap.Rigid3d(pycolmap.Rotation3d(),np.array([1.,0.,0.])))
            rec.write(current/'sparse')
            with self.assertRaisesRegex(ValueError,'pose differs'):
                validate_cached_workspace(cached,current,identity,detail,32)

    def test_original_pixels_and_calibration_are_restored_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            cache = root/'cached'; cache.mkdir()
            video = root/'source.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 180))
            self.assertTrue(writer.isOpened())
            color = np.full((180, 320, 3), [30, 140, 220], np.uint8)
            for _ in range(8):
                writer.write(color)
            writer.release()
            cv2.imwrite(str(cache/'frame_000003.jpg'), np.zeros((45, 80, 3), np.uint8))
            rec = pycolmap.Reconstruction()
            camera = pycolmap.Camera(camera_id=1, model='PINHOLE', width=80, height=45,
                                     params=[50., 60., 40., 22.5])
            rec.add_camera_with_trivial_rig(camera)
            pixel = camera.img_from_cam(np.array([1., .2, 4.]))
            image = pycolmap.Image(image_id=1, camera_id=1, name='frame_000003.jpg',
                                  points2D=pycolmap.Point2DList([pycolmap.Point2D(pixel)]))
            rec.add_image_with_trivial_frame(image, pycolmap.Rigid3d())
            rec.write(source)
            model, rgb_dir, report = prepare_dense_inputs(source, cache, video, root/'restored', 320)
            restored = pycolmap.Reconstruction(model)
            im = restored.images[1]; cam = restored.cameras[1]
            self.assertEqual((cam.width, cam.height), (320, 180))
            np.testing.assert_allclose(im.points2D[0].xy, pixel*4)
            np.testing.assert_allclose(cam.img_from_cam(np.array([1., .2, 4.])), im.points2D[0].xy)
            np.testing.assert_allclose(im.cam_from_world().matrix(), rec.images[1].cam_from_world().matrix())
            pixels = cv2.imread(str(rgb_dir/im.name))
            self.assertEqual(pixels.shape[:2], (180, 320))
            np.testing.assert_allclose(pixels.mean(axis=(0,1)), [30,140,220], atol=5)
            self.assertTrue(report['restored_original_pixels'])
            self.assertFalse(matching_camera_resolution(rec, restored, [im.name]))
            self.assertTrue(matching_camera_resolution(restored, restored, [im.name]))

    def test_request_never_exceeds_original_video_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model=root/'model';model.mkdir();images=root/'images';images.mkdir()
            video=root/'source.avi'
            writer=cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'),10.,(320,180))
            writer.write(np.zeros((180,320,3),np.uint8));writer.release()
            rec=pycolmap.Reconstruction()
            rec.add_camera_with_trivial_rig(pycolmap.Camera(camera_id=1,model='PINHOLE',width=320,height=180,
                                                          params=[200.,200.,160.,90.]))
            rec.add_image_with_trivial_frame(pycolmap.Image(image_id=1,camera_id=1,name='frame_000000.jpg'),pycolmap.Rigid3d())
            rec.write(model)
            prepared,rgb,report=prepare_dense_inputs(model,images,video,root/'output',1920)
            self.assertEqual(prepared,model)
            self.assertEqual(rgb,images)
            self.assertFalse(report['restored_original_pixels'])


if __name__=='__main__':
    unittest.main()
