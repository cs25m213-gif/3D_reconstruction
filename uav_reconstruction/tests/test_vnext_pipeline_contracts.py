import unittest

import numpy as np
import torch

from adaptive_uav.build_utility_dataset import audit, build, group_key, ranking_pairs
from adaptive_uav.temporal_depth_student import TemporalDepthStudent, TemporalState, warp_hidden, warp_prior
from adaptive_uav.train_utility_selector import split_groups
from adaptive_uav.trusted_sequences import PolicyInputLoader, validate_splits
from adaptive_uav import counterfactual_replay as replay


def row(name, state, value, fidelity='F_DEPTH', status='complete', candidate=1):
    return {'experiment_id': name, 'state_id': state, 'budget_id': 'tight', 'fidelity': fidelity, 'candidate': candidate, 'clip': 'c',
            'evaluation_protocol_id': 'p', 'execution_status': status, 'targets': {'utility': value, 'delta_surface': value},
            'intervention': 'add', 'flight': 'f', 'prefix': [1], 'split': 'train', 'reference_status': 'QUALIFIED', 'control': {}, 'intervened': {}, 'cost': {}, 'features': 'x'}


class LabelTests(unittest.TestCase):
    def test_pairs_ignore_ties_below_noise(self):
        rows = [row('a', 's', .1), row('b', 's', .1001), row('c', 's', -.1)]
        self.assertEqual(sorted(map(tuple, ranking_pairs(rows, .01))), [('a', 'c'), ('b', 'c')])

    def test_groups_never_cross_state_or_fidelity(self):
        table, _ = build([row('a', 's', .1), row('b', 't', .2), row('c', 's', .3, 'P_LOCAL')], .01)
        self.assertEqual(len(table), 3)
        self.assertTrue(all(not g['pairs'] for g in table))

    def test_censored_runs_are_not_negative_labels(self):
        table, _ = build([row('a', 's', .1), row('b', 's', -9., status='censored')], .01)
        self.assertEqual(table[0]['members'], ['a'])

    def test_failed_registration_stays_in_the_denominator(self):
        table, report = build([row('a', 's', .1), row('b', 's', -.1, status='registration_failed')], .01)
        self.assertEqual(len(table[0]['members']), 2)
        self.assertEqual(report['outcomes']['registration_failed'], 1)

    def test_proxy_audit_reports_disagreement(self):
        rows = []
        for i in range(6):
            rows += [row(f'd{i}', 's', float(i), candidate=i), row(f'p{i}', 's', float(-i), 'P_LOCAL', candidate=i)]
        self.assertLess(audit(rows)['proxy_fidelity']['P_LOCAL']['spearman'], -.9)

    def test_held_out_clips_are_whole(self):
        samples = {f'{c}{i}': {'flight': c} for c in 'abcd' for i in range(2)}
        groups = [{'members': [f'{c}{i}'], 'split': 'train'} for c in 'abcd' for i in range(2)]
        train, held, _ = split_groups(groups, samples)
        self.assertTrue(held)
        self.assertFalse({g['members'][0][0] for g in train} & {g['members'][0][0] for g in held})

    def test_labelled_validation_flights_are_preferred(self):
        samples = {'a': {'flight': 'x'}, 'b': {'flight': 'y'}}
        groups = [{'members': ['a'], 'split': 'train'}, {'members': ['b'], 'split': 'validation'}]
        train, held, name = split_groups(groups, samples)
        self.assertEqual((len(train), len(held), name), (1, 1, 'validation flights'))

    def test_unresolved_reference_is_refused_without_diagnostic(self):
        self.assertIsNone(replay.eligible({'flights': {'a': {'status': 'UNRESOLVED'}}}, 'a', False))
        self.assertEqual(replay.eligible({'flights': {'a': {'status': 'UNRESOLVED'}}}, 'a', True), 'UNRESOLVED')
        self.assertEqual(replay.eligible({'flights': {'a': {'status': 'QUALIFIED'}}}, 'a', False), 'QUALIFIED')


class PlanTests(unittest.TestCase):
    def test_one_frame_is_tried_from_two_maps(self):
        class Fake:
            views = list(range(60))
            keyframes = list(range(0, 60, 4))

            @staticmethod
            def candidates_after(position, ratios):
                return {position + r for r in ratios if position + r < 60}
        settings = {'prefix_keyframes': [6, 12], 'candidate_baseline_ratios': [1, 4, 8], 'rollout_frames': 8, 'budgets': {'a': {}}}
        seen = {}
        for prefix, candidate, _ in replay.plan(Fake, settings):
            seen.setdefault(candidate, set()).add(len(prefix))
        self.assertIn({6, 12}, seen.values())

    def test_map_size_is_what_defines_state_dependence(self):
        rows = []
        for name, prefix, value in (('a', (1,), .1), ('b', (1,), .1), ('c', (1, 2), .5)):
            item = row(name, name, value)
            item['prefix'] = prefix
            rows.append(item)
        self.assertAlmostEqual(audit(rows)['state_dependence_median_spread'], .4)


class SplitTests(unittest.TestCase):
    def test_flight_in_two_roles_is_leakage(self):
        rows = [{'frame_id': '1', 'flight_id': 'f', 'location_id': 'l', 'split': 'train'},
                {'frame_id': '2', 'flight_id': 'f', 'location_id': 'l', 'split': 'validation'}]
        self.assertTrue(validate_splits(rows))

    def test_policy_loader_refuses_reference(self):
        loader = PolicyInputLoader({'a': {'frame_id': 'a', 'measured_depth': 1, 'reference_geometry_ref': 2, 'time': 3}})
        self.assertNotIn('measured_depth', loader['a'])
        with self.assertRaises(PermissionError):
            loader.reference('a')


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, rgb, k, p):
        return torch.full((1, 1, 8, 8), 2.), torch.ones(1, 1, 8, 8)


class TemporalTests(unittest.TestCase):
    def test_untrained_branch_matches_student(self):
        model = TemporalDepthStudent(Student())
        prior = torch.full((1, 1, 8, 8), 3.)
        depth, _, visibility, hidden = model(None, None, None, prior, torch.ones_like(prior), torch.ones_like(prior))
        self.assertTrue(torch.allclose(depth, torch.full((1, 1, 8, 8), 2.)))
        self.assertEqual(hidden.shape[1], 16)
        self.assertEqual(visibility.shape, depth.shape)

    def test_student_stays_frozen(self):
        model = TemporalDepthStudent(Student())
        self.assertFalse(any(p.requires_grad for p in model.student.parameters()))

    def test_identity_hidden_warp_keeps_memory(self):
        hidden = torch.rand(1, 4, 6, 6)
        k = torch.tensor([[[10., 0, 3], [0, 10., 3], [0, 0, 1]]])
        warped = warp_hidden(hidden, torch.full((1, 1, 6, 6), 5.), k, torch.eye(4)[None])
        self.assertTrue(torch.allclose(warped, hidden, atol=1e-5))

    def test_identity_warp_keeps_depth(self):
        depth = torch.full((1, 1, 6, 6), 5.)
        out, _, valid = warp_prior(depth, torch.ones_like(depth), torch.tensor([[[10., 0, 3], [0, 10., 3], [0, 0, 1]]]), torch.eye(4)[None])
        self.assertTrue(torch.allclose(out, depth))
        self.assertEqual(float(valid.mean()), 1.)

    def test_reset_clears_prior_and_rejects_unknown_reason(self):
        state = TemporalState()
        state.update(torch.ones(1), torch.ones(1), torch.eye(4))
        state.reset('tracking_lost')
        self.assertIsNone(state.depth)
        with self.assertRaises(ValueError):
            state.reset('because')


if __name__ == '__main__':
    unittest.main()
