import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from adaptive_uav.student_losses import supervised_masks
from adaptive_uav.student_dataset import ClipRecord, discover
from adaptive_uav.train_student_update import data_identities
from adaptive_uav.selector_policy import sampling_distribution
from adaptive_uav.train_utility_selector import evaluate, training_identity, training_sources
from adaptive_uav.utility_contracts import DECISION_HEAD, QUERY_FEATURES


class Corrections(unittest.TestCase):
    def test_teacher_does_not_compete_with_sparse_anchor(self):
        empty = torch.zeros(1, 1, 1, 2, dtype=torch.bool)
        batch = dict(measured_valid=empty, neighbour_valid=empty, map_valid=empty,
                     anchor=torch.tensor([[[[1., 0.]]]]), teacher_valid=~empty)
        masks = supervised_masks(batch)
        self.assertEqual(masks['anchor'].flatten().tolist(), [True, False])
        self.assertEqual(masks['teacher'].flatten().tolist(), [False, True])

    def test_two_candidate_policy_gradient_does_not_cancel(self):
        scores = torch.tensor([0., 1.], requires_grad=True)
        distribution = sampling_distribution(scores, 1.)
        distribution.log_prob(torch.tensor(1)).backward()
        self.assertGreater(float(scores.grad.abs().sum()), .1)
        expected = torch.distributions.Categorical(logits=(scores.detach() - scores.detach().mean()) / (scores.detach().std() + 1e-6))
        torch.testing.assert_close(distribution.probs, expected.probs)

    def test_auxiliary_cache_changes_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / 'frame_000000.jpg.npz'; p.write_bytes(b'first')
            clip = ClipRecord('aux', 'unlabelled', 'unlabelled', None, root, None, None, indices=[0])
            before = data_identities([clip]); p.write_bytes(b'changed content')
            self.assertNotEqual(before, data_identities([clip]))

    def test_unqualified_validation_does_not_supply_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'cache/B_1').mkdir(parents=True)
            (root / 'cache/B_1/selected_infos.json').write_text('[]')
            (root / 'scales.json').write_text(json.dumps({'B_1': {'usable': True, 'scene': 'B', 'scale_m_per_visual_unit': 1}}))
            (root / 'gate.json').write_text(json.dumps({'flights': {'B': {'status': 'FAILED', 'registration_outcome': 'offset'}}}))
            contract = {'data': {'cache_root': 'cache', 'scale_calibration': 'scales.json', 'sidecars': 'targets'},
                        'splits': {'train': ['A'], 'validation': ['B']}, 'geometry_gate': 'gate.json'}
            with mock.patch('adaptive_uav.student_dataset.index_clip', lambda clip: clip):
                self.assertEqual(discover(root, contract)['validation'], [])

    def test_coverage_ties_do_not_count_as_baseline_defeats(self):
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(()))
            def forward(self, *args): return {'utility': torch.tensor([[[0., 0., 0., 1.]], [[0., 0., 0., 0.]]])}
        row = dict(question=DECISION_HEAD, query=np.zeros(len(QUERY_FEATURES)), visual=np.zeros(384),
                   tokens=np.zeros((1, 11)), token_visual=np.zeros((1, 384)), targets=np.zeros(4))
        report = evaluate(Model(), {'a': row, 'b': row}, [{'members': ['a', 'b'], 'pairs': [['a', 'b']]}])
        self.assertEqual(report['pair_accuracy'], 1.)
        self.assertIsNone(report['least_covered_rule_accuracy'])
        self.assertEqual(report['least_covered_rule_pairs'], 0)

    def test_fallback_holdout_also_excludes_source_pair_labels(self):
        sources = [{'split': 'train', 'flight': 'A'}, {'split': 'train', 'flight': 'B'}]
        self.assertEqual(training_sources(sources, [{'members': ['a']}], {'a': {'flight': 'A'}}), sources[:1])

    def test_selector_identity_covers_features_sources_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'features').mkdir()
            feature = root / 'features/x.npz'; feature.write_bytes(b'one')
            contract = {'selector': {'seed': 1}}
            before = training_identity(root, contract, 'dino')
            feature.write_bytes(b'two')
            self.assertNotEqual(before, training_identity(root, contract, 'dino'))
            before = training_identity(root, contract, 'dino')
            (root / 'source_labels.jsonl').write_text('{}\n')
            self.assertNotEqual(before, training_identity(root, contract, 'dino'))
            self.assertNotEqual(training_identity(root, contract, 'dino'), training_identity(root, {'selector': {'seed': 2}}, 'dino'))


if __name__ == '__main__': unittest.main()
