import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adaptive_uav.semantic_scene import rotate_image, remap_labels, object_position, filter_frame, write_observations


class SemanticSceneTests(unittest.TestCase):
    def frame(self):
        depth=np.full((12,16),4.,np.float32)
        return {'name':'frame_000030.jpg','prior':depth.copy(),'measured':depth.copy(),
                'anchors':depth.copy(),'K':np.array([[8.,0,7.5],[0,8.,5.5],[0,0,1]]),
                'P':np.eye(4),'validation':{'median':.03}}

    def test_orientation_preserves_original_camera_pixels(self):
        image=np.arange(60).reshape(4,5,3)
        for angle in (0,90,180,270):
            np.testing.assert_array_equal(rotate_image(rotate_image(image,angle),(-angle)%360),image)

    def test_crop_and_resize_masks_use_intrinsics_and_discrete_ids(self):
        labels=np.arange(100,dtype=np.uint16).reshape(10,10)
        source=np.array([[5.,0,5],[0,5.,5],[0,0,1]])
        cropped=source.copy();cropped[0,2]-=2
        np.testing.assert_array_equal(remap_labels(labels,source,cropped,(10,8)),labels[:,2:])
        scaled=source.copy();scaled[:2]*=.5
        np.testing.assert_array_equal(remap_labels(labels,source,scaled,(5,5)),labels[::2,::2])

    def test_object_point_uses_inverse_world_to_camera_pose(self):
        frame=self.frame()
        frame['P'][:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        frame['P'][:3,3]=[1,2,3]
        result=object_position(frame,np.ones((12,16),bool))
        np.testing.assert_allclose(result['position'],[-2,1,1])
        self.assertEqual(result['position_status'],'stereo_estimate')

    def test_missing_object_depth_does_not_borrow_background(self):
        frame=self.frame();mask=np.zeros((12,16),bool);mask[3:9,5:11]=True
        frame['measured'][mask]=0;frame['prior'][mask]=0
        self.assertIsNone(object_position(frame,mask)['position'])
        frame['prior'][mask]=7;frame['validation']['median']=.4
        self.assertIsNone(object_position(frame,mask)['position'])
        frame['validation']['median']=float('nan')
        self.assertIsNone(object_position(frame,mask)['position'])

    def test_learned_point_requires_frame_depth_validation(self):
        frame=self.frame();frame['measured'][:]=0
        self.assertEqual(object_position(frame,np.ones((12,16),bool))['position_status'],'learned_depth_estimate')
        frame['validation']['median']=None
        self.assertIsNone(object_position(frame,np.ones((12,16),bool))['position'])

    def test_exclusion_preserves_marker_evidence_and_neighbouring_surface(self):
        # Location is estimated before the depth used for fusion is erased.
        frame=self.frame();classes=np.zeros((12,16),np.uint8);classes[:2]=1;classes[3:9,5:11]=2
        ids=np.where(classes==2,1,0).astype(np.uint16)
        detection={'instance_id':1,'class':'person','confidence':.9}
        # Scale-fitting measurements may already be masked; marker evidence stays separate.
        frame['object_measured']=frame['measured'].copy();frame['measured'][classes>0]=0
        frame['validation']['median']=None
        with patch('adaptive_uav.semantic_scene.load_frame',return_value=(classes,ids,[detection])):
            rows=filter_frame(frame,Path('/unused'),30.)
        self.assertEqual(rows[0]['timestamp_seconds'],1.)
        self.assertEqual(rows[0]['position_status'],'stereo_estimate')
        for key in ('prior','measured','anchors'):
            self.assertTrue((frame[key][classes>0]==0).all())
            self.assertTrue((frame[key][classes==0]==4).all())

    def test_tiny_detection_is_kept_without_inventing_a_learned_position(self):
        frame=self.frame();frame['measured'][:]=0
        mask=np.zeros((12,16),bool);mask[3:9,4:10]=True
        result=object_position(frame,mask)
        self.assertIsNone(result['position'])
        self.assertEqual(result['position_status'],'object_too_small_for_learned_depth')

    def test_generic_filename_has_unknown_time_and_no_assumed_identity(self):
        frame=self.frame();frame['name']='street.jpg'
        classes=np.full((12,16),3,np.uint8);ids=np.ones((12,16),np.uint16)
        with patch('adaptive_uav.semantic_scene.load_frame',return_value=(classes,ids,[{'instance_id':1,'class':'dog','confidence':.8}])):
            rows=filter_frame(frame,Path('/unused'),30.)
        self.assertIsNone(rows[0]['timestamp_seconds'])
        with tempfile.TemporaryDirectory() as folder:
            write_observations(Path(folder),rows)
            report=json.loads((Path(folder)/'object_observations.json').read_text())
            self.assertFalse(report['identity_tracking'])
            self.assertEqual(report['positioned_observations'],1)
            self.assertIn('element vertex 1',(Path(folder)/'object_markers.ply').read_text())

    def test_ray_coverage_excludes_sky_but_reports_sky_geometry_separately(self):
        import open3d as o3d
        from adaptive_uav.multiview_dense import ray_audit
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector([[-20,-20,4],[20,-20,4],[20,20,4],[-20,20,4]])
        mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
        mesh.paint_uniform_color([.5,.5,.5])
        frame=self.frame();frame['rgb']=np.zeros((12,16,3),np.uint8)
        classes=np.zeros((12,16),np.uint8);classes[:6]=1
        with tempfile.TemporaryDirectory() as folder, patch('adaptive_uav.semantic_scene.load_frame',return_value=(classes,np.zeros_like(classes),[])):
            report=ray_audit({'completed':mesh},[frame],Path(folder),semantic_directory=Path('/unused'))
        values=report['sampled_views'][0]['meshes']['completed']
        self.assertEqual(values['evaluated_pixels'],96)
        self.assertEqual(values['measured_pixels'],96)
        self.assertEqual(values['sky_ray_hit_fraction'],1.)
        self.assertIn('static',report['coverage_domain'])


if __name__=='__main__':
    unittest.main()
