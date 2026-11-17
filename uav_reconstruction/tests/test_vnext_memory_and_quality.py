import unittest

import numpy as np
import torch

from adaptive_uav.graph_selector import MemoryUtilitySelector
from adaptive_uav.keyframe_policy import KeyframePolicy
from adaptive_uav.memory_graph import CausalMemory
from adaptive_uav.reconstruction_quality import ReferencePopulation, quality, surface_metrics, utility, DEFAULT_WEIGHTS
from adaptive_uav.runtime_types import FramePacket, PoseEstimate
from adaptive_uav import utility_contracts as contracts


def plane(n=400, seed=0):
    return np.random.default_rng(seed).uniform([-10, -10, 0], [10, 10, 0], (n, 3))


class MemoryTests(unittest.TestCase):
    def memory(self, count, **limits):
        memory = CausalMemory(limits)
        for i in range(count):
            memory.add(str(i), float(i), np.eye(32)[i % 32], np.eye(4), payload=i, component=0)
        return memory

    def test_retrieval_is_bounded_and_causal(self):
        memory = self.memory(40)
        found = memory.retrieve(np.eye(32)[3], 20.)
        self.assertLessEqual(len(found), 16)
        self.assertTrue(all(float(name) <= 20 for name in found))
        self.assertIn('3', found)
        self.assertGreater(memory.counters.rejected_future, 0)

    def test_past_timestamp_cannot_enter(self):
        memory = self.memory(3)
        with self.assertRaises(ValueError):
            memory.add('late', 0., np.ones(32), np.eye(4))

    def test_noncausal_evidence_is_rejected(self):
        memory = self.memory(2)
        with self.assertRaises(ValueError):
            memory.connect('0', '1', 50, 5., 1., causal=False)

    def test_components_merge_only_with_alignment(self):
        memory = CausalMemory()
        memory.add('a', 0., np.ones(4), np.eye(4))
        memory.add('b', 1., np.ones(4), np.eye(4))
        self.assertFalse(memory.connect('a', 'b', 80, 5., 1.))
        self.assertEqual(memory.components(), 2)
        self.assertTrue(memory.connect('a', 'b', 80, 5., 1., aligned=True))
        self.assertEqual(memory.components(), 1)

    def test_weak_geometry_makes_no_edge(self):
        memory = self.memory(2)
        self.assertFalse(memory.connect('0', '1', 5, 5., 1.))

    def test_bridge_survives_eviction(self):
        memory = CausalMemory({'active_keyframes': 2})
        memory.add('left', 0., np.ones(4), np.eye(4), payload=1, component=0)
        memory.add('bridge', 1., np.ones(4), np.eye(4), payload=1, component=0)
        memory.connect('left', 'bridge', 50, 5., 1.)
        memory.add('right', 2., np.ones(4), np.eye(4), payload=1, component=0)
        memory.connect('bridge', 'right', 50, 5., 2.)
        self.assertEqual(memory.bridges(), {'bridge'})
        memory.add('new', 3., np.ones(4), np.eye(4), payload=1, component=0)
        self.assertIn('bridge', memory.residency)
        self.assertIn('left', memory.nodes)  # Identity stays after its image is evicted.

    def test_state_changes_with_budget(self):
        memory = self.memory(2)
        self.assertNotEqual(memory.snapshot({'seconds': 10})['state_id'], memory.snapshot({'seconds': 5})['state_id'])


class ProposalTests(unittest.TestCase):
    def test_propose_leaves_policy_state_alone(self):
        policy = KeyframePolicy()
        frame = FramePacket('a', 0, 0., np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8), np.eye(3))
        pose = PoseEstimate('a', 0., np.eye(4), 'test')
        keep, _ = policy.propose(frame, pose)
        self.assertTrue(keep)
        self.assertIsNone(policy.last)
        policy.commit(frame, pose)
        self.assertIs(policy.last[0], frame)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.reference = ReferencePopulation(plane(), .25, 'TEST')

    def test_perfect_surface_scores_one(self):
        metrics = surface_metrics(self.reference.points, self.reference)
        self.assertAlmostEqual(metrics['f_score'], 1.)

    def test_missing_half_lowers_recall_not_precision(self):
        half = self.reference.points[self.reference.points[:, 0] < 0]
        metrics = surface_metrics(half, self.reference)
        self.assertLess(metrics['recall'], .7)
        self.assertAlmostEqual(metrics['precision'], 1.)

    def test_empty_branch_scores_zero_not_missing(self):
        metrics = surface_metrics(np.empty((0, 3)), self.reference)
        self.assertEqual(metrics['f_score'], 0.)
        self.assertEqual(quality(metrics, None, DEFAULT_WEIGHTS), -DEFAULT_WEIGHTS['surface_error'])

    def test_empty_map_never_beats_a_poor_one(self):
        poor = surface_metrics(self.reference.points[:20] + [0, 0, .9], self.reference)
        empty = surface_metrics(np.empty((0, 3)), self.reference)
        self.assertGreaterEqual(quality(poor, None, DEFAULT_WEIGHTS), quality(empty, None, DEFAULT_WEIGHTS))

    def test_floaters_over_the_scene_lower_precision(self):
        floaters = self.reference.points[:200] + [0, 0, 30.]
        metrics = surface_metrics(np.vstack((self.reference.points, floaters)), self.reference)
        self.assertLess(metrics['precision'], .75)

    def test_worse_error_never_scores_better_than_smaller_error(self):
        small = surface_metrics(self.reference.points + [0, 0, 2.], self.reference)
        large = surface_metrics(self.reference.points + [0, 0, 8.], self.reference)
        self.assertGreaterEqual(quality(small, None, DEFAULT_WEIGHTS), quality(large, None, DEFAULT_WEIGHTS))

    def test_cost_is_bounded_when_budget_is_gone(self):
        self.assertTrue(np.isfinite(utility(.5, .6, 0., 10., 0., DEFAULT_WEIGHTS)))


class ContractTests(unittest.TestCase):
    def test_unknown_feature_is_an_error(self):
        with self.assertRaises(ValueError):
            contracts.feature_vector({'reference_error': 1.}, contracts.QUERY_FEATURES)

    def test_whitelists_hold_no_reference_fields(self):
        contracts.assert_causal_names(contracts.QUERY_FEATURES + contracts.TOKEN_FEATURES)
        with self.assertRaises(ValueError):
            contracts.assert_causal_names(['future_pose'])

    def test_budget_never_refunds(self):
        budget = contracts.Budget(10., 2, 5)
        budget.charge(seconds=4.)
        with self.assertRaises(ValueError):
            budget.charge(seconds=-1.)
        self.assertEqual(budget.remaining('seconds'), 6.)
        self.assertFalse(budget.feasible(depth_calls=3))


class SelectorTests(unittest.TestCase):
    def inputs(self):
        tokens = torch.randn(1, 2, 16, 11)
        tokens[..., -1] = 1
        return (torch.randn(1, 2, 384), torch.randn(1, 2, 17), torch.randn(1, 2, 16, 384), tokens,
                torch.ones(1, 2, 16, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.long))

    def test_gradients_reach_attention_gru_and_heads(self):
        model = MemoryUtilitySelector()
        out = model(*self.inputs(), pairs=True)
        (out['utility'].sum() + out['pair'].sum() + out['cost'].sum() + out['log_variance'].sum() + out['action'].sum()).backward()
        for name in ('memory.layers.0.self_attn.out_proj.weight', 'gru.weight_hh_l0', 'utility.weight', 'pair.0.weight', 'token.0.weight',
                     'query.0.weight', 'question.weight', 'log_variance.weight', 'action.weight'):
            self.assertGreater(float(dict(model.named_parameters())[name].grad.abs().sum()), 0, name)

    def test_pair_score_is_symmetric(self):
        out = MemoryUtilitySelector().eval()(*self.inputs(), pairs=True)
        self.assertTrue(torch.allclose(out['pair'], out['pair'].transpose(-1, -2)))

    def test_masked_tokens_do_not_change_output(self):
        model = MemoryUtilitySelector().eval()
        tokens, visual = torch.randn(1, 1, 16, 11), torch.randn(1, 1, 16, 384)
        mask = torch.zeros(1, 1, 16, dtype=torch.bool)
        mask[..., :4] = True
        args = (torch.randn(1, 1, 384), torch.randn(1, 1, 17))
        question = torch.zeros(1, 1, dtype=torch.long)
        changed = visual.clone()
        changed[..., 8:, :] = 9.
        with torch.no_grad():
            a = model(*args, visual, tokens, mask, question)['utility']
            b = model(*args, changed, tokens, mask, question)['utility']
        self.assertTrue(torch.allclose(a, b, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
