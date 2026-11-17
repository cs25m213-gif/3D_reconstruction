import json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2,numpy as np
from adaptive_uav.runtime_types import FramePacket,PoseEstimate,DepthFrame,transform
from adaptive_uav.temporal_depth import warp_depth
from adaptive_uav.submaps import SubmapStore,save_depth,load_depth
from adaptive_uav.keyframe_policy import KeyframePolicy
from adaptive_uav.stereo_policy import StereoPolicy,evaluation_frames
from adaptive_uav.frame_stream import video_frames,imu_interval


def frame(name='frame_000000.jpg',time=0.,x=0.,context=False):
    h,w=48,64;k=np.array([[60.,0,32],[0,60.,24],[0,0,1.]])
    pose=np.eye(4);pose[0,3]=x
    rgb=np.full((h,w,3),150,np.uint8);depth=np.full((h,w),2.,np.float32)
    return DepthFrame(name,time,depth,rgb,k,pose,name,context_ids=(name,),
        context_poses={name:pose.tolist()} if context else {},source='sensor_measured')


class ContractTests(unittest.TestCase):
    def test_image_quarter_turn_preserves_projection_and_camera_pose(self):
        from adaptive_uav.camera_orientation import rotate_calibration
        k=np.array([[100.,0,25.],[0,80.,15.],[0,0,1.]])
        point=np.array([.1,.05,2.,1.]);pixel=k@point[:3];pixel=pixel[:2]/pixel[2]
        for angle in (0,90,180,270):
            new_k,q,(w,h)=rotate_calibration(k,64,48,angle)
            new_point=q@point;projected=new_k@new_point[:3];projected=projected[:2]/projected[2]
            expected={0:pixel,90:np.array([47-pixel[1],pixel[0]]),180:np.array([63-pixel[0],47-pixel[1]]),270:np.array([pixel[1],63-pixel[0]])}[angle]
            np.testing.assert_allclose(projected,expected)
            restored,_,_=rotate_calibration(new_k,w,h,(-angle)%360);np.testing.assert_allclose(restored,k)
            world_from_new=np.eye(4);world_from_new[:3,3]=[1,2,3]
            np.testing.assert_allclose(world_from_new@new_point,(world_from_new@q)@point)

    def test_two_view_depth_scale_does_not_require_sim3_rotation_fit(self):
        from adaptive_uav.depth_backend import baseline_scale
        expected=np.stack([np.eye(4),np.eye(4)]);expected[1,0,3]=-2
        predicted=expected.copy();predicted[:,0,3]*=3
        self.assertAlmostEqual(baseline_scale(expected,predicted),1/3)
        with self.assertRaises(ValueError):baseline_scale(np.stack([np.eye(4)]*2),predicted)

    def test_pose_rejects_nonrotation_or_nonfinite(self):
        matrix=np.eye(4);matrix[0,0]=2
        with self.assertRaises(ValueError):transform(matrix)
        matrix=np.eye(4);matrix[0,3]=float('nan')
        with self.assertRaises(ValueError):transform(matrix)

    def test_pose_inverse_round_trip(self):
        matrix=np.eye(4);matrix[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];matrix[:3,3]=[3,4,5]
        pose=PoseEstimate('a',0,matrix,'test')
        point=np.array([2.,3,4,1]);np.testing.assert_allclose(pose.cam_from_world@matrix@point,point)

    def test_invalid_depth_and_masks_do_not_become_geometry(self):
        f=frame();f.depth[0,:3]=[np.nan,np.inf,-3];f.excluded[1,:]=True
        f=replace(f)
        self.assertFalse(f.valid[0,:3].any());self.assertFalse(f.valid[1].any())

    def test_checkpoint_round_trip_preserves_evidence(self):
        f=frame(context=True)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'f.npz';save_depth(path,f);g=load_depth(path)
        np.testing.assert_array_equal(g.depth,f.depth);self.assertEqual(g.observation_id,f.observation_id)
        self.assertEqual(g.context_poses,f.context_poses)


class TemporalTests(unittest.TestCase):
    def target(self,f,x=0.):
        matrix=f.T_world_from_camera.copy();matrix[0,3]=x
        packet=FramePacket('frame_000001.jpg',1,1.,f.rgb,f.K)
        return packet,PoseEstimate(packet.frame_id,1.,matrix,'test')

    def test_stationary_warp_preserves_depth_and_evidence_identity(self):
        f=frame();packet,pose=self.target(f)
        result,stats=warp_depth(f,packet,pose)
        self.assertIsNotNone(result);np.testing.assert_allclose(result.depth,f.depth)
        self.assertEqual(result.observation_id,f.observation_id);self.assertEqual(result.source,'warped_previous_estimate')

    def test_disocclusion_requires_new_inference(self):
        f=frame();packet,pose=self.target(f,1.)
        result,stats=warp_depth(f,packet,pose,min_coverage=.95)
        self.assertIsNone(result);self.assertLess(stats['coverage'],.95)

    def test_occlusion_uses_nearest_positive_camera_z(self):
        f=frame();f.K=np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
        f.depth[:]=0;f.depth[0,0]=1;f.depth[0,1]=2
        packet,pose=self.target(f);packet.K=np.array([[.1,0,0],[0,.1,0],[0,0,1.]])
        result,_=warp_depth(f,packet,pose,min_coverage=0.)
        self.assertEqual(result.depth[0,0],1.)

    def test_reuse_age_and_tracking_loss_are_rejected(self):
        f=frame();packet,pose=self.target(f)
        self.assertIsNone(warp_depth(replace(f,age=2),packet,pose,max_age=2)[0])
        pose.state='lost';self.assertIsNone(warp_depth(f,packet,pose)[0])

    def test_dynamic_target_is_excluded(self):
        f=frame();packet,pose=self.target(f);mask=np.zeros(f.depth.shape,bool);mask[:,20:30]=True
        result,_=warp_depth(f,packet,pose,excluded=mask)
        self.assertTrue((result.depth[mask]==0).all())


class SubmapTests(unittest.TestCase):
    def test_reestimated_depth_resets_voxel_scale(self):
        with tempfile.TemporaryDirectory() as d:
            store=SubmapStore(d);f=frame();store.put(f)
            new=replace(f,depth=f.depth*20,pose_version=1);store.put(new,replace=True)
            self.assertAlmostEqual(store.meta[store.index[f.frame_id]]['depth_scale'],40.)

    def vertices(self,store,path):
        return np.asarray(store.export(path).vertices)

    def test_repeat_and_warp_do_not_inflate_observations(self):
        with tempfile.TemporaryDirectory() as d:
            store=SubmapStore(d);f=frame();store.put(f);count=store.sequence
            store.put(f);self.assertEqual(store.sequence,count)
            store.put(replace(f,frame_id='b',source='warped_previous_estimate'))
            self.assertEqual(store.sequence,count);self.assertNotIn('b',store.index)

    def test_active_map_budget_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            store=SubmapStore(d,max_active=1,max_frames=1)
            store.put(frame());store.put(frame('b',1,.1))
            self.assertEqual(len(store.volumes),1)
            restored=SubmapStore(d,max_active=1,max_frames=1)
            self.assertEqual(restored.index,store.index)
            self.assertGreater(len(self.vertices(restored,Path(d)/'mesh.ply')),0)

    def test_pose_correction_matches_clean_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);store=SubmapStore(d/'original');f=frame();store.put(f)
            corrected=f.T_world_from_camera.copy();corrected[0,3]=.4
            store.correct({f.frame_id:corrected},1)
            clean=SubmapStore(d/'clean');clean.put(replace(f,T_world_from_camera=corrected,pose_version=1))
            a=self.vertices(store,d/'a.ply');b=self.vertices(clean,d/'b.ply')
            np.testing.assert_allclose(np.sort(a,axis=0),np.sort(b,axis=0),atol=1e-5)

    def test_changed_relative_context_marks_learned_depth_stale(self):
        with tempfile.TemporaryDirectory() as d:
            f=frame();f.source='learned_prediction';f.context_poses={'other':np.eye(4).tolist()}
            store=SubmapStore(d);store.put(f)
            other=np.eye(4);other[0,3]=.2;update=store.correct({'other':other},1)
            self.assertIn(f.frame_id,store.stale_frames());self.assertTrue(update.rebuilding)
            fresh=replace(f,context_poses={'other':other.tolist()},pose_version=1)
            store.put(fresh,replace=True);self.assertFalse(store.stale_frames())

    def test_no_stereo_plane_is_exportable_but_not_measured(self):
        with tempfile.TemporaryDirectory() as d:
            f=replace(frame(),source='learned_prediction');store=SubmapStore(d);store.put(f)
            self.assertGreater(len(store.export(Path(d)/'mesh.ply').triangles),0)
            self.assertEqual(load_depth(store._frame_path(f.frame_id)).source,'learned_prediction')


class PolicyTests(unittest.TestCase):
    def test_pose_planner_keeps_connected_sources_inside_the_budget(self):
        from adaptive_uav.stereo_policy import plan_pose_stereo
        records={};poses={};depths={}
        for i,x in enumerate([0.,.3,.6,.9,1.2,1.5]):
            name=f'{i}.jpg';f=frame(name,float(i),x)
            records[name]={'K':f.K.tolist(),'width':64,'height':48}
            poses[name]=f.T_world_from_camera;depths[name]=f
        with tempfile.TemporaryDirectory() as d:
            plan=plan_pose_stereo(Path(d),records,poses,depths,3)
        self.assertGreaterEqual(plan['reference_views'],3);self.assertLessEqual(plan['reference_views'],3)
        selected={r['reference'] for r in plan['plan']}
        for row in plan['plan']:
            self.assertGreaterEqual(len(row['sources']),2)
            self.assertTrue({s['name'] for s in row['sources']}<=selected)

    def test_stereo_refinement_fits_scale_but_rejects_bad_holdout(self):
        from adaptive_uav.stereo_refinement import refine_depth
        from adaptive_uav.multiview_dense import spatial_holdout
        f=replace(frame(),source='learned_prediction');f.depth*=3
        measured=np.full(f.depth.shape,2.,np.float32)
        corrected,report=refine_depth(f,measured)
        self.assertTrue(report['learned_completion_qualified']);np.testing.assert_allclose(corrected.depth,2.)
        measured[spatial_holdout(measured.shape)]=20
        corrected,report=refine_depth(f,measured)
        self.assertFalse(report['learned_completion_qualified']);self.assertTrue((corrected.evidence==1).all())

    def test_cli_dispatch_accepts_equals_syntax(self):
        from adaptive_uav.runtime_cli import requested_mode
        self.assertEqual(requested_mode(['session','--mode=balanced']),'balanced')
        self.assertEqual(requested_mode(['session']),'legacy')
    def test_stationary_heartbeat_does_not_insert_mapping_frames(self):
        f=frame();rng=np.random.default_rng(4);f.rgb=rng.integers(0,255,f.rgb.shape,dtype=np.uint8)
        selector=KeyframePolicy(minimum_sharpness=0)
        a=FramePacket('a',0,0.,f.rgb,f.K);p=PoseEstimate('a',0,np.eye(4),'test')
        self.assertTrue(selector.decide(a,p)[0])
        b=FramePacket('b',1,5.,f.rgb,f.K);p=PoseEstimate('b',5,np.eye(4),'test')
        self.assertFalse(selector.decide(b,p)[0])

    def test_stereo_budget_and_pure_rotation(self):
        policy=StereoPolicy(max_references=1,budget_seconds=2)
        self.assertFalse(policy.request('a',parallax_degrees=0,scale_uncertain=True)[0])
        self.assertTrue(policy.request('a',parallax_degrees=5,scale_uncertain=True)[0])
        self.assertFalse(policy.request('a',parallax_degrees=5,scale_uncertain=True)[0])
        self.assertFalse(policy.request('b',parallax_degrees=5,scale_uncertain=True)[0])
        policy.record(3);self.assertEqual(policy.elapsed_seconds,3)

    def test_fixed_evaluation_includes_endpoints(self):
        names=[str(i) for i in range(100)];selected=evaluation_frames(names,12)
        self.assertEqual(len(selected),12);self.assertEqual(selected[0],'0');self.assertEqual(selected[-1],'99')

    def test_imu_interpolation_never_extrapolates(self):
        times=np.array([1.,2.,3.]);data=np.column_stack((times,times*2,times*3))
        self.assertIsNone(imu_interval(times,data,0,2))
        stamps,values=imu_interval(times,data,1.5,2.5)
        np.testing.assert_allclose(stamps,[1.5,2,2.5]);np.testing.assert_allclose(values[:,0],stamps)

    def test_vfr_decoder_pts_are_preserved(self):
        image=np.zeros((32,48,3),np.uint8)
        class Capture:
            index=-1
            def isOpened(self):return True
            def read(self):
                self.index+=1;return (True,image.copy()) if self.index<3 else (False,None)
            def get(self,key):return [0.,41.,115.][self.index] if key==cv2.CAP_PROP_POS_MSEC else 30.
            def release(self):pass
        with patch('adaptive_uav.frame_stream.cv2.VideoCapture',return_value=Capture()):
            frames=list(video_frames('test',np.eye(3),48,0))
        np.testing.assert_allclose([f.timestamp for f in frames],[0,.041,.115])

    def test_pose_only_workspace_has_no_fabricated_tracks(self):
        from adaptive_uav.pose_workspace import write_workspace
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);f=frame();path=d/'input.jpg';cv2.imwrite(str(path),f.rgb)
            frames={'a.jpg':{'path':str(path),'K':f.K.tolist(),'width':64,'height':48}}
            model=write_workspace(d/'workspace',frames,{'a.jpg':f.T_world_from_camera})
            self.assertEqual(model.num_reg_images(),1);self.assertEqual(model.num_points3D(),0)
            self.assertEqual(next(iter(model.images.values())).num_points3D,0)


if __name__=='__main__':unittest.main()
