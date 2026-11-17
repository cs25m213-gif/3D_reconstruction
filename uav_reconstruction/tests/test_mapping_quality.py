from types import SimpleNamespace
from contextlib import nullcontext
from pathlib import Path
import tempfile
import threading
import queue
import unittest
from unittest import mock

import cv2
import numpy as np

from adaptive_uav.mapping_quality import MappingQualityGate,image_evidence


class MappingQualityTests(unittest.TestCase):
    def frame(self,t=0.,rgb=None):
        if rgb is None:rgb=np.random.default_rng(4).integers(0,256,(192,256,3),dtype=np.uint8)
        return SimpleNamespace(timestamp=t,rgb=rgb,frame_id=str(t))

    def evidence(self,focus=1.,features=100,texture=100.):
        return {'observable':True,'reason':None,'focus':focus,'orb_features':features,
                'texture':texture,'denoised_laplacian':focus*texture,'raw_laplacian':1000.}

    def initialized(self,**kwargs):
        gate=MappingQualityGate(**kwargs)
        gate.reference.extend([{'focus':1.,'orb_features':100}]*4)
        gate.last_timestamp=0.
        return gate

    def test_real_blur_loses_denoised_focus_and_feature_support(self):
        base=self.frame().rgb
        blurred=cv2.GaussianBlur(base,(19,19),4)
        a,b=image_evidence(base),image_evidence(blurred)
        self.assertLess(b['focus'],a['focus'])
        self.assertLess(b['orb_features'],a['orb_features'])

    def test_relative_blur_is_deferred_but_does_not_change_reference(self):
        gate=self.initialized();before=list(gate.reference)
        with mock.patch('adaptive_uav.mapping_quality.image_evidence',return_value=self.evidence(.1,10)):
            keep,row=gate.propose(self.frame(.1),SimpleNamespace(state='tracking'),{},True)
        self.assertFalse(keep);self.assertTrue(row['mapping_quality']['suspected_blur'])
        self.assertEqual(list(gate.reference),before)

    def test_mapping_bridge_is_preserved_and_bad_frame_does_not_poison_baseline(self):
        gate=self.initialized();f=self.frame(.1)
        with mock.patch('adaptive_uav.mapping_quality.image_evidence',return_value=self.evidence(.1,10)):
            keep,row=gate.propose(f,SimpleNamespace(state='tracking'),{'reason':'rotation_bridge_with_geometric_content'},True)
        self.assertTrue(keep);gate.commit(f,row)
        self.assertEqual(len(gate.reference),4)
        self.assertEqual(row['mapping_quality']['protected_reason'],'geometric_bridge')
        self.assertFalse(row['mapping_quality']['dense_eligible'])

    def test_rejection_does_not_override_geometric_rule(self):
        gate=self.initialized()
        with mock.patch('adaptive_uav.mapping_quality.image_evidence',side_effect=AssertionError('unnecessary work')):
            keep,_=gate.propose(self.frame(),SimpleNamespace(state='tracking'),{},False)
        self.assertFalse(keep)

    def test_low_texture_is_not_automatically_blur(self):
        gate=self.initialized()
        with mock.patch('adaptive_uav.mapping_quality.image_evidence',return_value=self.evidence(.1,1,texture=.1)):
            keep,_=gate.propose(self.frame(.1),SimpleNamespace(state='tracking'),{},True)
        self.assertTrue(keep)

    def test_proposals_only_enter_baseline_after_final_commit(self):
        gate=MappingQualityGate();f=self.frame()
        keep,row=gate.propose(f,SimpleNamespace(state='tracking'),{},True)
        self.assertTrue(keep);self.assertFalse(gate.reference)
        gate.commit(f,row);self.assertEqual(len(gate.reference),1)

    def test_deferral_limit_prevents_permanent_mapping_starvation(self):
        gate=self.initialized(max_deferrals=2,max_gap_seconds=100)
        with mock.patch('adaptive_uav.mapping_quality.image_evidence',return_value=self.evidence(.1,10)):
            results=[gate.propose(self.frame(.1+i*.1),SimpleNamespace(state='tracking'),{},True)[0] for i in range(3)]
        self.assertEqual(results,[False,False,True])

    def test_gap_limit_is_explicit(self):
        gate=self.initialized(max_gap_seconds=1.)
        with mock.patch('adaptive_uav.mapping_quality.image_evidence',return_value=self.evidence(.1,10)):
            keep,row=gate.propose(self.frame(1.1),SimpleNamespace(state='tracking'),{},True)
        self.assertTrue(keep);self.assertEqual(row['mapping_quality']['protected_reason'],'mapping_gap_limit')

    def test_blank_rgb_never_becomes_optional_dense_observation(self):
        gate=self.initialized()
        keep,row=gate.propose(self.frame(10,np.zeros((16,16,3),np.uint8)),SimpleNamespace(state='tracking'),{},True)
        self.assertFalse(keep);self.assertEqual(row['mapping_quality']['reason'],'blank_rgb')

    def test_noisy_low_feature_frame_cannot_use_raw_laplacian_alone(self):
        gate=self.initialized()
        metrics=self.evidence(.1,2);metrics['raw_laplacian']=999999.
        with mock.patch('adaptive_uav.mapping_quality.image_evidence',return_value=metrics):
            keep,_=gate.propose(self.frame(.1),SimpleNamespace(state='tracking'),{},True)
        self.assertFalse(keep)

    def test_tracking_loss_does_not_call_quality_model(self):
        gate=self.initialized()
        with mock.patch('adaptive_uav.mapping_quality.image_evidence',side_effect=AssertionError('unnecessary work')):
            keep,_=gate.propose(self.frame(),SimpleNamespace(state='lost'),{},True)
        self.assertFalse(keep)

    def test_runtime_tracks_all_frames_and_commits_only_final_mapping_keeps(self):
        from adaptive_uav.runtime import StreamRunner
        from adaptive_uav.runtime_types import FramePacket,PoseEstimate
        frames=[FramePacket(str(i),i,i*.5,self.frame().rgb,np.eye(3)) for i in range(4)]
        prior=[]
        class Geometric:
            def __init__(self,*args):self.last=None
            def decide(self,frame,pose,*args):
                prior.append(self.last[0].frame_id if self.last else None)
                self.last=(frame,pose)
                return True,{'reason':'new_geometric_content'}
        gate=mock.Mock()
        gate.propose.side_effect=lambda f,p,s,k:(f.index!=1,{'mapping_quality':{'checked':True}})
        utility=mock.Mock()
        utility.decide.side_effect=lambda f,p,s,k:(k and f.index!=3,{})
        runner=StreamRunner.__new__(StreamRunner)
        runner.args=SimpleNamespace(live_url=None,replay_paced=False)
        runner.config={'device':'cpu','mapping_min_seconds':.25,'mapping_max_seconds':1.,'mapping_parallax_pixels':8.,'settle_tracking_frames':0}
        runner.stop=threading.Event();runner.producer_done=threading.Event()
        runner.frames=lambda:iter(frames)
        runner.metrics=mock.Mock();runner.metrics.scope.side_effect=lambda *a,**k:nullcontext()
        runner.gpu_lock=threading.RLock();runner.state_lock=threading.RLock()
        runner.backend=mock.Mock()
        runner.backend.track.side_effect=lambda f:PoseEstimate(f.frame_id,f.timestamp,np.eye(4),'test')
        runner.backend.get_pose_updates.return_value=[]
        runner.pose_version=0;runner.report={};runner.duration=2.
        runner.tracking_rows=[];runner.eval_bins=set();runner.eval_records={};runner.eval_poses={}
        runner.records={};runner.poses={};runner.selected=[];runner.decisions=[];runner.errors=[]
        runner.record=lambda f,p:{}
        runner.selector_gyro=None;runner.learned_selector=None;runner.utility_policy=utility
        runner.mapping_quality=gate;runner.queue=queue.Queue(maxsize=4);runner.enqueue=mock.Mock()
        with tempfile.TemporaryDirectory() as d,mock.patch('adaptive_uav.runtime.KeyframePolicy',Geometric):
            runner.out=Path(d);runner.producer()
        self.assertFalse(runner.errors)
        self.assertEqual(runner.backend.track.call_count,4)
        self.assertEqual(runner.selected,['0','2'])
        self.assertEqual(gate.commit.call_count,2)
        self.assertEqual(prior,[None,'0','0','2'])
        self.assertEqual([r['selected'] for r in runner.decisions],[True,False,True,False])

    def test_poor_bridge_is_excluded_from_depth_sources_and_fusion(self):
        from adaptive_uav.runtime import StreamRunner
        runner=StreamRunner.__new__(StreamRunner)
        runner.state_lock=threading.RLock()
        runner.records={'a':{},'blurred_bridge':{'dense_eligible':False},'c':{}}
        runner.poses={name:np.eye(4) for name in runner.records}
        runner.config={'depth_views':3};runner.pose_version=0;runner.utility_policy=None
        runner.producer_done=threading.Event();runner.metrics=mock.Mock()
        records,poses=runner.context('c')
        self.assertEqual(set(records),{'a','c'})
        self.assertEqual(runner.context('blurred_bridge'),({},{}))
        with mock.patch.object(runner,'ensure_models',side_effect=AssertionError('Bad RGB must not trigger depth')):
            runner.process('blurred_bridge')
        runner.metrics.event.assert_called_with('depth_skipped',frame='blurred_bridge',reason='mapping_quality_bridge_only')


if __name__=='__main__':unittest.main()
