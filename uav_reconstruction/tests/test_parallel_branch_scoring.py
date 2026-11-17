"""Remote scoring must use the branch's estimated poses, never replace them with truth."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from adaptive_uav import sequential_replay as replay
from adaptive_uav.reconstruction_quality import DEFAULT_WEIGHTS

class BranchScoringTests(unittest.TestCase):
    def test_training_plateau_keeps_warmup_and_respects_disable(self):
        from adaptive_uav.finetune_selector import training_plateau
        self.assertFalse(training_plateau(999,1,100))
        self.assertFalse(training_plateau(2000,1500,1000))
        self.assertTrue(training_plateau(2500,1500,1000))
        self.assertFalse(training_plateau(6500,1,0))
    def test_estimated_branch_poses_and_pose_penalty_survive_batching(self):
        truth=np.eye(4);estimated=truth.copy();estimated[0,3]=.4
        view=SimpleNamespace(fusion_depth=np.ones((2,2)),fusion_rgb=np.zeros((2,2,3)),fusion_k=np.eye(3),reference_pose=truth)
        branch=SimpleNamespace(observations=[(view,estimated)],poses={0:estimated})
        metrics={'f_score':.5,'supported_coverage':.5,'accuracy_m':.2}
        with patch.object(replay,'geometry_metrics_batch',return_value=[metrics]) as remote:
            result=replay.score_branches({'NEW_KEYFRAME':branch},None,.5,{0:truth},DEFAULT_WEIGHTS)
        sent=remote.call_args.args[0][0][0]
        np.testing.assert_array_equal(sent.reference_pose,estimated)
        np.testing.assert_array_equal(view.reference_pose,truth)
        self.assertAlmostEqual(result['NEW_KEYFRAME']['quality'],.5-.25*.2-.25*.4)

    def test_local_fallback_scores_every_branch(self):
        branches={name:SimpleNamespace(observations=[],poses={}) for name in replay.ACTIONS}
        metrics={'f_score':0.,'supported_coverage':0.,'accuracy_m':None}
        with patch.object(replay,'geometry_metrics_batch',side_effect=lambda choices,reference,voxel,local:local()),patch.object(replay,'score',return_value={'metrics':metrics}) as score:
            result=replay.score_branches(branches,None,.5,{},DEFAULT_WEIGHTS)
        self.assertEqual(score.call_count,3)
        self.assertEqual(set(result),set(replay.ACTIONS))
        self.assertTrue(all(r['quality']==-.25 for r in result.values()))

if __name__=='__main__':unittest.main()
