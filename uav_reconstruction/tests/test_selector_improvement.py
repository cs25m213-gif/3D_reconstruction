"""Small CPU software checks; no dataset, GPU, downloads or alignment claims."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from adaptive_uav.counterfactual_replay import View
from adaptive_uav.graph_selector import MemoryUtilitySelector, UtilityRuntime
from adaptive_uav.selector_policy import UtilityPolicy
from adaptive_uav import selector_improvement as imp
from adaptive_uav.train_selector_improvement import oracle_loss, load_groups, atomic_torch


def frame(i):
    rgb=np.full((16,16,3),i,np.uint8);pose=np.eye(4);pose[0,3]=i
    k=np.array([[20.,0,8],[0,20.,8],[0,0,1]])
    depth=np.full((16,16),10.,np.float32)
    return View(f'c/{i}',i,float(i),rgb,depth,k,pose,pose,k,.5,rgb,depth,k,anchors_m=depth)


class ImprovementTests(unittest.TestCase):
    def test_full_window_includes_tail_and_start(self):
        for n in range(1,41):
            got=imp.candidate_positions(np.arange(10,10+n))
            self.assertEqual(got[0],10);self.assertEqual(got[-1],9+n)
            self.assertEqual(len(got),len(set(got)));self.assertLessEqual(len(got),5)
        self.assertIn(6,imp.candidate_positions(np.arange(7)))

    def test_invalid_candidate_limit_rejected(self):
        with self.assertRaises(ValueError):imp.candidate_positions([1,2],1)

    def rows(self,tracks):
        result=[]
        for t in tracks:
            q=np.zeros(17,np.float32);q[imp.SHARED]=t/500
            result.append({'query':q})
        return result

    def test_overlap_rejects_unsupported_when_verified_available(self):
        mask,fallback=imp.overlap_eligibility(self.rows([2,20,40]),True)
        self.assertEqual(mask.tolist(),[False,True,True]);self.assertFalse(fallback)

    def test_no_supported_edge_has_explicit_deterministic_fallback(self):
        mask,fallback=imp.overlap_eligibility(self.rows([2,9,5]),True)
        self.assertEqual(mask.tolist(),[False,True,False]);self.assertTrue(fallback)
        mask,fallback=imp.overlap_eligibility(self.rows([0,0]),False)
        self.assertTrue(mask.all());self.assertFalse(fallback)

    def test_oracle_never_picks_ineligible_high_score(self):
        self.assertEqual(imp.oracle_index([100,.2,.3],[False,True,True]),2)
        with self.assertRaises(ValueError):imp.oracle_index([float('nan')],[True])

    def test_validation_and_test_cannot_supply_oracle_training_labels(self):
        imp.require_training_role('train')
        for role in ['validation','test','calibration','final_test',None]:
            with self.assertRaises(ValueError):imp.require_training_role(role)

    def test_new_checkpoints_reject_legacy_and_online_dispatch(self):
        data={'selection_protocol_id':imp.PROTOCOL,'decision_head':2}
        with self.assertRaises(ValueError):UtilityPolicy(None,data,'cpu')
        with mock.patch('adaptive_uav.graph_selector.load_utility_checkpoint',return_value=(None,data)):
            with self.assertRaises(ValueError):UtilityRuntime('unused',device='cpu')

    def test_context_scoring_changes_sources_and_is_bounded(self):
        calls=[]
        def provider(v,p,history):
            sources=tuple(x.frame_id for x,_ in history[-2:])
            calls.append((v.frame_id,sources))
            return replace(v,source_ids=sources),0.,False
        metrics={'f_score':.5,'accuracy_m':.2,'supported_coverage':.5}
        scorer=imp.ContextScorer(provider,object(),.5,max_scores=1)
        with mock.patch.object(imp,'fuse',return_value=np.zeros((1,3))),mock.patch.object(imp,'surface_metrics',return_value=metrics):
            scorer.score([frame(0),frame(2),frame(3)])
            self.assertEqual(calls[-1],('c/3',('c/0','c/2')))
            scorer.score([frame(0),frame(1),frame(3)])
            self.assertEqual(calls[-1],('c/3',('c/0','c/1')))
            n=len(calls);scorer.score([frame(0),frame(1),frame(3)])
            self.assertEqual(n,len(calls));self.assertEqual(len(scorer.cache),1)
        with self.assertRaises(ValueError):scorer.score([frame(1),frame(1)])
        with self.assertRaises(ValueError):scorer.score([frame(2),frame(1)])

    def test_whole_selection_score_includes_every_selected_observation(self):
        provider=lambda v,p,h:(v,0.,False)
        scorer=imp.ContextScorer(provider,object(),.5)
        with mock.patch.object(imp,'fuse',return_value=np.zeros((1,3))) as fuse, mock.patch.object(imp,'surface_metrics',return_value={'f_score':.5,'accuracy_m':0.,'supported_coverage':.5}):
            scorer.score([frame(0),frame(1),frame(2)])
            self.assertEqual([v.index for v,_ in fuse.call_args.args[0]],[0,1,2])

    def test_candidate_features_need_no_reference(self):
        class Candidate:
            feature_pose=np.eye(4);index=0;rgb=np.zeros((8,8,3),np.uint8)
            frame_id='x';timestamp=0
            @property
            def reference_pose(self):raise AssertionError('Reference feature leak')
        state=mock.Mock();descriptors=mock.Mock()
        imp.describe_options(state,SimpleNamespace(views=[Candidate()],name='c'),[0],descriptors,12)
        state.features.assert_called_once()

    def test_oracle_loss_has_real_memory_and_gru_gradients(self):
        torch.manual_seed(2)
        model=MemoryUtilitySelector(visual_dim=8,hidden=16,heads=4,layers=1)
        rng=np.random.default_rng(1)
        tokens=rng.normal(size=(3,2,11)).astype(np.float32);tokens[:,:,-1]=1
        row={'visual':rng.normal(size=(3,8)).astype(np.float32),'query':rng.normal(size=(3,17)).astype(np.float32),
             'token_visual':rng.normal(size=(3,2,8)).astype(np.float32),'tokens':tokens,
             'quality':np.float32([.1,.3,.2]),'eligible':np.array([True,True,True])}
        loss=oracle_loss(model,row,2,.05);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for prefix in ['query','token','memory','gru','utility']:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(prefix)),prefix)

    def test_atomic_checkpoints_leave_no_partial_file(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'latest.pt';atomic_torch({'model':torch.ones(1)},p)
            self.assertTrue(p.exists());self.assertFalse(p.with_suffix('.partial.pt').exists())


if __name__=='__main__':unittest.main()
