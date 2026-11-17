"""Causal masking, replacement symmetry and gradient contracts of the new head."""
import unittest

import torch

from uav_reconstruction.adaptive_uav.reconstruction_action_selector import (
    ReconstructionActionSelector, ReconstructionSelectorConfig,
    FRAME_FEATURES, MAP_FEATURES, MEMORY_FEATURES,
)


class ReconstructionSelectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        torch.set_num_threads(2)
        self.model = ReconstructionActionSelector(ReconstructionSelectorConfig(visual_dim=16, width=32, max_memory=4)).eval()
        self.inputs = dict(candidate=torch.randn(2,16), occupant=torch.randn(2,16),
            has_occupant=torch.tensor([True,True]), frame_features=torch.randn(2,2,len(FRAME_FEATURES)),
            query=torch.randn(2,len(MAP_FEATURES)), memory_visual=torch.randn(2,3,16),
            memory_features=torch.randn(2,3,len(MEMORY_FEATURES)),
            memory_valid=torch.tensor([[True,True,False],[False,False,False]]))
        self.inputs['frame_features'][...,8] = 0

    def test_reversing_same_map_swap_negates_gain(self):
        a = self.inputs
        original = self.model(**a)['gain']
        reversed_inputs = dict(a, candidate=a['occupant'], occupant=a['candidate'],
                               frame_features=a['frame_features'].flip(1))
        torch.testing.assert_close(original, -self.model(**reversed_inputs)['gain'], atol=1e-6, rtol=1e-5)

    def test_same_frame_cannot_gain_by_replacing_itself(self):
        a = self.inputs
        same = dict(a, occupant=a['candidate'], frame_features=a['frame_features'][:,:1].expand(-1,2,-1))
        torch.testing.assert_close(self.model(**same)['gain'], torch.zeros(2), atol=1e-6, rtol=0)

    def test_padding_and_missing_depth_do_not_change_decision(self):
        a = self.inputs
        before = self.model(**a)['gain']
        b = {k:v.clone() for k,v in a.items()}
        b['memory_visual'][~b['memory_valid']] = float('nan')
        b['memory_features'][~b['memory_valid']] = float('nan')
        b['frame_features'][...,9:] = float('nan')
        torch.testing.assert_close(before, self.model(**b)['gain'])

    def test_absent_occupant_and_empty_memory_are_safe(self):
        a = self.inputs
        a = dict(a, has_occupant=torch.zeros(2,dtype=torch.bool), memory_visual=a['memory_visual'][:,:0],
                 memory_features=a['memory_features'][:,:0],memory_valid=a['memory_valid'][:,:0])
        before = self.model(**a)['gain']
        a['occupant'] = torch.full_like(a['occupant'],float('nan'))
        a['frame_features'] = a['frame_features'].clone();a['frame_features'][:,1] = float('nan')
        result = self.model(**a)
        torch.testing.assert_close(before, result['gain'])
        self.assertTrue(torch.isfinite(result['gain']).all())
        torch.testing.assert_close(result['swap_gain'], torch.zeros(2))

    def test_memory_order_does_not_change_decision(self):
        a = self.inputs
        b = dict(a, memory_visual=a['memory_visual'].flip(1), memory_features=a['memory_features'].flip(1),
                 memory_valid=a['memory_valid'].flip(1))
        torch.testing.assert_close(self.model(**a)['gain'],self.model(**b)['gain'],atol=1e-6,rtol=1e-5)

    def test_gradients_reach_both_frame_embeddings_and_heads(self):
        a = dict(self.inputs)
        a['candidate'] = a['candidate'].requires_grad_();a['occupant'] = a['occupant'].requires_grad_()
        a['has_occupant'] = torch.tensor([False,True])
        result = self.model(**a)['gain']
        (result-torch.tensor([.05,-.03])).square().mean().backward()
        for grad in (a['candidate'].grad, a['occupant'].grad, self.model.add[-1].weight.grad,
                     self.model.frame_value[-1].weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()),0)

    def test_memory_budget_is_enforced(self):
        a = dict(self.inputs, memory_visual=torch.randn(2,5,16),
                 memory_features=torch.randn(2,5,len(MEMORY_FEATURES)),memory_valid=torch.ones(2,5,dtype=torch.bool))
        with self.assertRaisesRegex(ValueError,'bounded memory'):
            self.model(**a)


if __name__ == '__main__':
    unittest.main()
