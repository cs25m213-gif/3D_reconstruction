import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

from adaptive_uav import counterfactual_replay as replay
from adaptive_uav import build_calibrated_targets as targets
from adaptive_uav import register_lidar_to_map as registration
from adaptive_uav.build_calibrated_targets import nearest, static_support
from adaptive_uav.build_utility_dataset import read_labels
from adaptive_uav.embedding_cache import Descriptors
from adaptive_uav.evaluate_vnext import promotion
from adaptive_uav.graph_selector import UtilityRuntime, load_utility_checkpoint
from adaptive_uav.reconstruction_quality import cluster_interval
from adaptive_uav.runtime_types import FramePacket, PoseEstimate
from adaptive_uav.selector_features import CoverageMap, MapState, as_pose
from adaptive_uav.selector_policy import UtilityPolicy, coverage_gain, random_policy, rule, uniform
from adaptive_uav.student_dataset import ClipRecord, UpdateDataset, choose_sources, discover, index_clip
from adaptive_uav.student_losses import DEFAULT_WEIGHTS, supervised_masks, student_loss
from adaptive_uav.train_student_update import evaluate, forward_sequence
from adaptive_uav.train_utility_selector import train as train_selector
from adaptive_uav.utility_contracts import DECISION_HEAD, QUERY_FEATURES, Budget, head_index
from adaptive_uav.depth_student import DepthStudent
from adaptive_uav import sfm_dedome
from adaptive_uav.validate_reference_geometry import interpolate_pose
from adaptive_uav.build_uavscenes_targets import pose_at_time

K = np.array([[200., 0, 128.], [0, 200., 96.], [0, 0, 1.]])


def texture(seed=0, size=(192, 256)):
    noise = np.random.default_rng(seed).random((size[0] // 4, size[1] // 4, 3), dtype=np.float32)
    return np.uint8(255 * cv2.resize(noise, size[::-1], interpolation=cv2.INTER_CUBIC).clip(0, 1))


def nadir(x):
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = [x, 0., 100.]
    return pose


def view(index, x):
    depth = np.full((192, 256), 100., np.float32)
    rgb = texture(index)
    return replay.View(f'c/{index}', index, index / 2., rgb, depth, K.copy(), nadir(x), nadir(x), K.copy(), .5, rgb, depth, K.copy())


class FakeClip(replay.Clip):
    def __init__(self, count=24, spacing=8.):
        self.name, self.scene = 'c', 'AMtown01'
        self.views = [view(i, i * spacing) for i in range(count)]


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.map = CoverageMap()
        self.map.observe(np.full((192, 256), 100., np.float32), K, nadir(0.))

    def test_seen_area_reads_as_covered_and_new_area_does_not(self):
        here = self.map.view_stats(K, nadir(0.), 256, 192)[0]
        away = self.map.view_stats(K, nadir(1000.), 256, 192)[0]
        self.assertGreater(here, .8)
        self.assertEqual(away, 0.)

    def test_covered_area_shrinks_as_the_camera_moves_away(self):
        self.assertGreater(self.map.view_stats(K, nadir(30.), 256, 192)[0], self.map.view_stats(K, nadir(90.), 256, 192)[0])

    def test_seeing_twice_raises_redundancy(self):
        once = self.map.view_stats(K, nadir(0.), 256, 192)[1]
        self.map.observe(np.full((192, 256), 100., np.float32), K, nadir(0.))
        self.assertGreater(self.map.view_stats(K, nadir(0.), 256, 192)[1], once)

    def test_coverage_is_scale_free(self):
        pose = nadir(3.)
        pose[2, 3] = 10.
        reference = self.map.view_stats(K, nadir(30.), 256, 192)[0]
        start = nadir(0.)
        start[2, 3] = 10.
        other = CoverageMap()
        other.observe(np.full((192, 256), 10., np.float32), K, start)
        self.assertAlmostEqual(other.view_stats(K, pose, 256, 192)[0], reference, delta=.1)


class CalibrationScaleTests(unittest.TestCase):
    def test_every_flight_takes_the_median_of_its_area(self):
        ratios = {'AMtown01_0': 1.02, 'AMtown02_0': .99, 'AMtown03_0': .98, 'HKisland01_0': 1.1, 'AMtown01_1': 5.}

        def clip(root, contract, name):
            return mock.Mock(scene=name.split('_')[0], lidar_ratio=ratios[name])

        with mock.patch.object(replay, 'Clip', clip):
            scales = replay.calibration_scales(None, {'geometry': {'calibration_batch_suffix': '_0'}}, list(ratios))
        self.assertEqual(scales, {'AMtown01': .99, 'AMtown02': .99, 'AMtown03': .99, 'HKisland01': 1.1})


class KeyframeTests(unittest.TestCase):
    def spaced(self, depth, spacing=2.):
        clip = FakeClip(30, spacing)
        for v in clip.views:
            v.depth_m = np.full((192, 256), depth, np.float32)
        return clip

    def test_spacing_follows_altitude_not_frame_count(self):
        low, high = self.spaced(20.).choose_keyframes(.12), self.spaced(100.).choose_keyframes(.12)
        self.assertGreater(len(low), len(high))
        self.assertEqual(low[:3], [0, 2, 4])
        self.assertEqual(high[:3], [0, 6, 12])

    def test_candidates_reach_the_same_overlap_at_any_altitude(self):
        self.assertEqual(self.spaced(20.).candidates_after(0, [.1, .2, .4]), {1, 2, 4})
        self.assertEqual(self.spaced(100.).candidates_after(0, [.1, .2, .4]), {5, 10, 20})

    def test_candidates_stop_at_the_end_of_the_clip(self):
        self.assertEqual(self.spaced(100.).candidates_after(25, [.1, .4]), set())

    def test_orientation_settles_the_roll_positions_cannot(self):
        clip, angle = FakeClip(12, 5.), np.radians(15.)
        roll = np.eye(4)
        roll[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        roll[:3, 3] = [7., -3., 40.]
        for v in clip.views:  # All centres lie on the x axis, which this rotation leaves in place.
            v.feature_pose = roll @ v.reference_pose
        clip.align_to_reference()
        for v in clip.views:
            self.assertTrue(np.allclose(v.aligned_pose, v.reference_pose))

    def test_map_only_clips_ignore_lidar(self):
        data = UpdateDataset.targets
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            lidar, dense = np.zeros((420, 504), np.float32), np.full((420, 504), 80., np.float32)
            lidar[10, 10] = 80.
            np.savez(target / 'frame_000000.npz', measured_depth_z_m=lidar, neighbour_depth_z_m=lidar, map_depth_z_m=dense)
            clip = ClipRecord('c', 's', 'train', 2., target, target, None, True)
            row = data(None, clip, 0, 40.)
        self.assertFalse(np.asarray(row['measured']).any())
        self.assertTrue(np.asarray(row['map']).any())


class TargetResumeTests(unittest.TestCase):
    def build(self, root, contract):
        built = []

        def clip(row, *_):
            built.append(row['name'])
            return {'clip': row['name']}

        rows = [{'name': 'a', 'area': 'A'}, {'name': 'b', 'area': 'A'}]
        with mock.patch.object(targets, 'load_contract', return_value=contract), mock.patch.object(targets, 'clips', return_value=rows), \
                mock.patch.object(targets, 'write_scales'), mock.patch.object(targets, 'build_clip', clip), \
                mock.patch('sys.argv', ['x', '--config', 'c', '--output', str(root / 't'), '--root', str(root), '--no-map']):
            targets.main()
        return built

    def test_finished_clips_are_kept_until_the_calibration_changes(self):
        contract = {'data': {}, 'geometry': {'exclude_label_ids': [], 'map_registration': {}}, 'geometry_gate': 'gate.json',
                    'geometry_calibration': 'calibration.json', 'provisional_geometry_gate': 'old.json'}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'calibration.json').write_text('{"areas": {}}')
            (root / 'old.json').write_text('{"flights": {}}')
            self.assertEqual(self.build(root, contract), ['a', 'b'])
            self.assertEqual(self.build(root, contract), [])
            self.assertEqual(len(json.loads((root / 'gate.json').read_text())['target_reports']), 2)
            (root / 'calibration.json').write_text('{"areas": {}, "changed": 1}')
            self.assertEqual(self.build(root, contract), ['a', 'b'])


class MapOnlyTests(unittest.TestCase):
    def test_only_clean_offset_flights_in_training_lose_their_lidar(self):
        flights = {'A1': {'status': 'QUALIFIED', 'registration_outcome': 'aligned', 'blocking_reasons': []},
                   'A2': {'status': 'FAILED', 'registration_outcome': 'offset', 'blocking_reasons': []},
                   'A3': {'status': 'FAILED', 'registration_outcome': 'offset', 'blocking_reasons': ['pose gap']},
                   'B1': {'status': 'QUALIFIED', 'registration_outcome': 'aligned', 'blocking_reasons': []}}
        contract = {'data': {'scale_calibration': 'scales.json', 'cache_root': 'cache', 'sidecars': 'targets'},
                    'splits': {'train': ['A'], 'validation': ['B']}, 'geometry_gate': 'gate.json'}
        with tempfile.TemporaryDirectory() as folder:
            root, scales = Path(folder), {}
            for scene in flights:
                (root / 'cache' / f'{scene}_1').mkdir(parents=True)
                (root / 'cache' / f'{scene}_1' / 'selected_infos.json').write_text('[]')
                scales[f'{scene}_1'] = {'usable': True, 'scene': scene, 'scale_m_per_visual_unit': 1.}
            (root / 'scales.json').write_text(json.dumps(scales))
            (root / 'gate.json').write_text(json.dumps({'flights': flights}))
            with mock.patch('adaptive_uav.student_dataset.index_clip', lambda clip: clip):
                roles = discover(root, contract)
        self.assertEqual({c.scene: c.map_only for c in roles['train']}, {'A1': False, 'A2': True})
        self.assertEqual([c.map_only for c in roles['validation']], [False])


class RuleSearchTests(unittest.TestCase):
    def test_search_finds_the_tightest_threshold_that_fits_in_few_passes(self):
        clip, passes = FakeClip(40, 8.), []

        def selection(current, threshold):
            passes.append(threshold)
            return current.views[:int(400 / threshold)]

        with mock.patch('adaptive_uav.selector_policy.rule_selection', selection):
            chosen = rule(clip, 10)
        tightest = next(t for t in np.geomspace(2., 600., 48) if int(400 / t) <= 10)
        self.assertEqual(len(chosen), 10)
        self.assertIn(tightest, passes)
        self.assertLessEqual(len(passes), 6)


class RigOffsetTests(unittest.TestCase):
    def test_every_area_shares_the_median_of_the_flight_fits(self):
        fits = {'A': {'A1': .10, 'A2': .12}, 'B': {'B1': .14}}
        rows = [{'area': 'A'}, {'area': 'B'}]
        used = {}

        def calibrated(area, rows, terra, count, rules, offset):
            used[area] = offset
            return {'rotation_source': 'x', 'time_offset_s': offset, 'validation': {'outcome': 'aligned'}}

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(registration, 'load_contract', return_value={'geometry': {'map_registration': {}}, 'data': {'terra': 't'}}), \
                mock.patch.object(registration, 'clips', return_value=rows), mock.patch.object(registration, 'sha256_file', return_value='x'), \
                mock.patch.object(registration, 'flight_offsets', lambda area, *_: fits[area]), mock.patch.object(registration, 'calibrate_area', calibrated), \
                mock.patch('sys.argv', ['x', '--config', 'c', '--output', folder]):
            registration.main()
            report = json.loads((Path(folder) / 'calibration.json').read_text())
        self.assertEqual(used, {'A': .12, 'B': .12})
        self.assertEqual(report['rig_time_offset_s'], .12)
        self.assertEqual(report['calibration_flight_offsets_s'], {'A1': .10, 'A2': .12, 'B1': .14})


class FeatureTests(unittest.TestCase):
    def state(self):
        clip, state = FakeClip(6), MapState()
        for v in clip.views[:4]:
            state.commit(v, as_pose(v, v.feature_pose), np.ones(384, np.float32))
            state.observe(v.frame_id, v.depth_m, v.K, v.feature_pose, v.confidence)
        return clip, state

    def test_policy_can_see_budget_and_map(self):
        clip, state = self.state()
        v = clip.views[4]
        names = QUERY_FEATURES
        tight = state.features(v, as_pose(v, v.feature_pose), Budget(20., 2, 16).state(), np.ones(384, np.float32))['query']
        loose = state.features(v, as_pose(v, v.feature_pose), Budget(120., 16, 64).state(), np.ones(384, np.float32))['query']
        self.assertNotEqual(tight[names.index('remaining_depth_calls')], loose[names.index('remaining_depth_calls')])
        self.assertNotEqual(tight[names.index('remaining_seconds')], loose[names.index('remaining_seconds')])
        self.assertGreater(tight[names.index('map_coverage')], 0.)
        self.assertGreater(tight[names.index('sharpness')], 0.)

    def test_features_do_not_depend_on_units(self):
        clip, state = self.state()
        v = clip.views[4]
        value = state.features(v, as_pose(v, v.feature_pose), Budget(20., 2, 16).state(), np.ones(384, np.float32))['query'][QUERY_FEATURES.index('baseline_over_depth')]
        self.assertAlmostEqual(float(value), 8. / 100., delta=.01)


class PolicyTests(unittest.TestCase):
    def test_every_policy_spends_exactly_the_budget(self):
        clip = FakeClip()
        descriptors = Descriptors()
        for policy in (rule, uniform, random_policy, coverage_gain):
            self.assertEqual(len(policy(clip, 6, descriptors)), 6, policy.__name__)

    def test_coverage_gain_avoids_what_the_map_already_holds(self):
        clip = FakeClip(12, 2.)
        chosen = coverage_gain(clip, 3, Descriptors())
        self.assertGreaterEqual(chosen[1].index, 4)

    def test_rollout_budget_is_a_real_opportunity_cost(self):
        clip = FakeClip()
        start = [(clip.views[0], nadir(0.))]
        with mock.patch.object(replay, 'register', return_value={'ok': True, 'pose': nadir(0.), 'inliers': 99, 'matches': 1}):
            full, _, _ = replay.rollout(start, clip.views[1:9], 3, 30)
            spent, _, _ = replay.rollout(start, clip.views[1:9], 2, 30)
            none, _, _ = replay.rollout(start, clip.views[1:9], 0, 30)
        self.assertEqual((len(full), len(spent), len(none)), (4, 3, 1))

    def test_budget_never_lets_the_map_grow_past_its_allowance(self):
        self.assertEqual(Budget(1., 1, 1).state()['depth_calls'], 1)

    def promotion_rows(self, flights, gain, status='QUALIFIED', keyframes=12):
        rows = []
        for i in range(flights):
            for policy, quality in (('rule', .5), ('coverage_gain', .5), ('utility', .5 + gain + .001 * i)):
                rows.append({'clip': f'f{i}_1', 'scene': f'f{i}', 'policy': policy, 'quality': quality, 'reference_status': status,
                             'keyframes': keyframes - (policy == 'rule' and keyframes == 1)})
        return rows

    def test_promotion_requires_gain_flights_and_a_trusted_reference(self):
        policies = ['rule', 'coverage_gain', 'utility']
        result=promotion(self.promotion_rows(3, .02), policies, .005)
        self.assertTrue(result['development_candidate'])
        self.assertFalse(result['promote'])  # Offline evidence never activates the streaming model.
        self.assertFalse(promotion(self.promotion_rows(2, .02), policies, .005)['promote'])
        self.assertFalse(promotion(self.promotion_rows(3, .001), policies, .005)['promote'])
        self.assertFalse(promotion(self.promotion_rows(3, .02, 'UNRESOLVED'), policies, .005)['promote'])

    def test_promotion_refuses_mismatched_budgets(self):
        rows = self.promotion_rows(3, .02)
        for row in rows:
            if row['policy'] == 'rule':
                row['keyframes'] = 5
        self.assertFalse(promotion(rows, ['rule', 'coverage_gain', 'utility'], .005)['promote'])

    def test_flights_not_clips_are_the_unit_of_evidence(self):
        self.assertEqual(cluster_interval({'a': [.1, .1, .1], 'b': [.3]})['mean'], .2)


class TimeOffsetTests(unittest.TestCase):
    def test_a_planted_clock_error_is_recovered(self):
        h, w = 2048, 2448
        ys, xs = np.mgrid[0:h:registration.RENDER_STEP, 0:w:registration.RENDER_STEP]
        surface = (100 + 6 * np.sin(xs / 40.) * np.cos(ys / 55.)).astype(np.float32)
        rng = np.random.default_rng(0)
        true_offset, speed_px = .12, 300.

        class Fake:
            def in_camera(self, scan, xyz, rotation, translation, offset):
                # A wrong offset slides returns along the motion direction.
                moved = xyz.copy()
                moved[:, 0] += (offset - true_offset) * speed_px * moved[:, 2] / 1450.
                return moved

            def k(self, i):
                return np.array([[1450., 0, 1224.], [0, 1450., 1024.], [0, 0, 1.]])

        fake = Fake()
        u, v = rng.uniform(100, w - 100, 4000), rng.uniform(100, h - 100, 4000)
        z = surface[(v / registration.RENDER_STEP).astype(int), (u / registration.RENDER_STEP).astype(int)]
        xyz = np.c_[(u - 1224.) / 1450. * z, (v - 1024.) / 1450. * z, z]
        frames = [{'clip': fake, 'image_index': 0, 'scan_index': 0, 'same': True, 'xyz': xyz, 'uv': None, 'z': None, 'centre': np.array([1224., 1024.]), 'depth': surface}]
        found = registration.fit_time(frames)
        self.assertAlmostEqual(found['time_offset_s'], true_offset, delta=.02)
        self.assertLess(found['spread'], found['spread_at_zero'])

    def test_identity_warp_changes_nothing(self):
        uv = np.array([[10., 20.], [300., 400.]])
        np.testing.assert_allclose(registration.warp(uv, np.array([100., 100.]), *registration.IDENTITY), uv)

    def test_cached_pose_lookup_matches_the_original(self):
        infos = []
        for i in range(5):
            pose = np.eye(4)
            pose[0, 3] = i
            infos.append({'OriginalImageName': f'{100 + i * .1:.3f}.jpg', 'T4x4': pose.tolist()})
        times = np.array([float(Path(i['OriginalImageName']).stem) for i in infos])
        np.testing.assert_allclose(interpolate_pose(times, infos, 100.25), pose_at_time(infos, 100.25))


class TargetTests(unittest.TestCase):
    def test_a_moving_return_loses_static_support(self):
        ground = np.c_[np.random.default_rng(0).uniform(0, 20, (500, 2)), np.zeros(500)]
        here = np.array([[10., 10., .1], [10., 10., 3.], [500., 500., 0.]])
        self.assertEqual(static_support(here, ground).tolist(), [1, 3, 2])

    def test_nearest_return_wins_and_conflicts_are_marked(self):
        y, x = np.array([5, 5, 7]), np.array([5, 5, 7])
        depth, reason = nearest(y, x, np.float32([100., 50., 80.]), np.uint8([1, 1, 1]))
        self.assertEqual(float(depth[5, 5]), 50.)
        self.assertEqual(int(reason[5, 5]), 4)
        self.assertEqual(int(reason[7, 7]), 1)


class LossTests(unittest.TestCase):
    def batch(self, measured=True):
        h, w = 192, 256
        ones = torch.ones(1, 1, h, w)
        batch = {'rgb': torch.rand(1, 3, 3, h, w), 'K': torch.as_tensor(K, dtype=torch.float32)[None, None].repeat(1, 3, 1, 1), 'P': torch.eye(4)[None, None].repeat(1, 3, 1, 1),
                 'teacher': ones * 1.3, 'teacher_valid': ones > 0, 'anchor': ones * 0, 'measured': ones, 'neighbour': ones, 'map': ones}
        batch['P'][:, 1, 0, 3] = .05
        batch['P'][:, 2, 0, 3] = -.05
        batch['measured_valid'] = torch.zeros_like(ones, dtype=torch.bool)
        batch['neighbour_valid'] = torch.zeros_like(ones, dtype=torch.bool)
        batch['map_valid'] = torch.zeros_like(ones, dtype=torch.bool)
        if measured:
            batch['measured_valid'][..., :50, :] = True
        return batch

    def test_empty_targets_stay_finite(self):
        depth = torch.ones(1, 1, 192, 256, requires_grad=True)
        total, parts = student_loss(self.batch(False), depth, torch.full_like(depth, .5), DEFAULT_WEIGHTS)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(parts['pixels_measured'], 0)

    def test_teacher_never_applies_where_a_measurement_exists(self):
        batch = self.batch()
        masks = supervised_masks(batch)
        self.assertFalse((masks['teacher'] & batch['measured_valid']).any())
        self.assertTrue(masks['teacher'].any())

    def test_trust_order_holds_in_the_default_weights(self):
        w = DEFAULT_WEIGHTS
        self.assertGreater(w['measured'], w['map'])
        self.assertGreater(w['map'], w['multiview'])
        self.assertGreater(w['multiview'], w['photometric'])
        self.assertGreater(w['photometric'], 0)
        self.assertLessEqual(w['teacher'], w['photometric'])

    def test_hiding_hard_pixels_cannot_lower_the_depth_term(self):
        batch = self.batch()
        depth = torch.full((1, 1, 192, 256), 2.)
        _, sure = student_loss(batch, depth, torch.full_like(depth, .9), DEFAULT_WEIGHTS)
        _, unsure = student_loss(batch, depth, torch.full_like(depth, .1), DEFAULT_WEIGHTS)
        self.assertEqual(sure['measured'], unsure['measured'])


class StudentDataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.cache, self.targets = root / 'cache', root / 'targets'
        self.cache.mkdir()
        self.targets.mkdir()
        k = np.array([[120., 0, 126.], [0, 120., 105.], [0, 0, 1.]], np.float32)
        for i in range(8):
            pose = np.eye(4)
            pose[0, 3] = -i * 2.  # World-to-camera, in visual units.
            anchors = np.zeros((210, 252), np.float32)
            anchors[::3, ::3] = 40.
            np.savez_compressed(self.cache / f'frame_{i:06d}.jpg.npz', rgb=texture(i, (210, 252)), K=k, P=pose, depth=np.full((210, 252), 40., np.float32),
                                confidence=np.ones((210, 252), np.float32), anchors=anchors, context=np.array(['x']))
            measured = np.zeros((210, 252), np.float32)
            measured[100, 100:140] = 80.
            np.savez_compressed(self.targets / f'frame_{i:06d}.npz', measured_depth_z_m=measured, neighbour_depth_z_m=measured * 0, map_depth_z_m=measured * 0)
        self.clip = index_clip(ClipRecord('c', 'AMtown01', 'train', 2., self.cache, self.targets, None))

    def tearDown(self):
        self.folder.cleanup()

    def test_causal_sources_are_always_earlier(self):
        for position in range(1, 8):
            index = self.clip.indices[position]
            self.assertTrue(all(i < index for i in choose_sources(self.clip, position, True)))

    def test_normalization_matches_the_student_contract(self):
        item = UpdateDataset([self.clip], 1, True)[0]
        target = item['measured'][0][item['measured_valid'][0]]
        self.assertAlmostEqual(float(target.mean()), 80. / (2. * 40.), places=3)
        self.assertAlmostEqual(float(item['metric_per_normalized']), 80., places=3)
        self.assertAlmostEqual(float(item['teacher'][0].mean()), 1., places=3)

    def test_two_frame_update_step_trains_the_real_student(self):
        batch = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in UpdateDataset([self.clip], 2, True)[0].items()}
        model = DepthStudent(48, 24)
        total, parts = forward_sequence(model, batch, DEFAULT_WEIGHTS, lambda: torch.autocast('cpu', enabled=False), False)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(parts['pixels_measured'], 0)
        self.assertGreater(float(sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)), 0)


class ValidationTests(unittest.TestCase):
    def test_score_uses_the_anchor_scaling_deployment_applies(self):
        class Doubled(torch.nn.Module):
            def forward(self, rgb, k, p):
                return torch.full((1, 1, 8, 8), 2.), torch.ones((1, 1, 8, 8))

        field = lambda value: torch.full((1, 1, 1, 8, 8), value)
        batch = {'rgb': torch.zeros(1, 1, 3, 8, 8), 'K': torch.zeros(1, 1, 3, 3, 3), 'P': torch.zeros(1, 1, 3, 4, 4), 'measured': field(1.),
                 'measured_valid': field(1.).bool(), 'anchor': field(1.), 'teacher': field(1.), 'metric_per_normalized': torch.tensor([10.]), 'scene': ['s']}
        result = evaluate(Doubled(), [batch], torch.device('cpu'), lambda: torch.autocast('cpu', enabled=False))['all']
        self.assertAlmostEqual(result['student_absrel'], 0.)
        self.assertAlmostEqual(result['raw_student_absrel'], 1.)


class DedomeTests(unittest.TestCase):
    def test_every_camera_model_keeps_its_focal_and_centre(self):
        class Camera:
            def __init__(self, name, params):
                self.model_name, self.params = name, params
        self.assertEqual(sfm_dedome.as_opencv(Camera('PINHOLE', [10., 11., 3., 4.])), [10., 11., 3., 4., 0., 0., 0., 0.])
        self.assertEqual(sfm_dedome.as_opencv(Camera('SIMPLE_RADIAL', [10., 3., 4., .1])), [10., 10., 3., 4., .1, 0., 0., 0.])
        with self.assertRaises(ValueError):
            sfm_dedome.as_opencv(Camera('FISHEYE', [1.]))


    def test_refined_model_survives_undistortion(self):
        import pycolmap
        focal, grid = 300., [(x, y, .2 * np.sin(x + y)) for x in np.linspace(-6, 26, 17) for y in np.linspace(-6, 6, 9)]
        pixels = {}
        for image in range(6):
            for point, (x, y, z) in enumerate(grid):
                depth = 20. - z
                u, v = focal * (x - image * 4.) / depth + 160., -focal * y / depth + 120.
                if 0 <= u < 320 and 0 <= v < 240:
                    pixels.setdefault(point, []).append((image, u, v))
        shown = {point: seen for point, seen in pixels.items() if len(seen) >= 2}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'model').mkdir()
            (root / 'images').mkdir()
            (root / 'model/cameras.txt').write_text(f'1 PINHOLE 320 240 {focal} {focal} 160 120\n')
            lines, tracks = [], {point: [] for point in shown}
            for image in range(6):
                rows = [(point, u, v) for point, seen in shown.items() for i, u, v in seen if i == image]
                for slot, (point, _, _) in enumerate(rows):
                    tracks[point].append(f'{image + 1} {slot}')
                lines += [f'{image + 1} 0 1 0 0 {-image * 4.} 0 20 1 frame_{image:06d}.jpg', ' '.join(f'{u} {v} {point + 1}' for point, u, v in rows)]
                cv2.imwrite(str(root / 'images' / f'frame_{image:06d}.jpg'), texture(image, (240, 320)))
            (root / 'model/images.txt').write_text('\n'.join(lines) + '\n')
            (root / 'model/points3D.txt').write_text(''.join(f'{p + 1} {grid[p][0]} {grid[p][1]} {grid[p][2]} 128 128 128 0 ' + ' '.join(track) + '\n' for p, track in tracks.items()))
            model, report = sfm_dedome.refine_model(root / 'model', root / 'refined')
            json.dumps(report)
            self.assertTrue((root / 'refined/cameras.bin').is_file())
            camera = next(iter(pycolmap.Reconstruction(model).cameras.values()))
            self.assertEqual(camera.model_name, 'OPENCV')
            self.assertAlmostEqual(camera.params[0], focal)
            pycolmap.undistort_images(root / 'dense', model, root / 'images', num_threads=1, undistort_options=pycolmap.UndistortCameraOptions(max_image_size=320))
            dense = pycolmap.Reconstruction(root / 'dense/sparse')
            self.assertEqual(next(iter(dense.cameras.values())).model_name, 'PINHOLE')
            self.assertEqual(len(dense.images), 6)


class SelectorTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        labels = cls.root / 'labels'
        (labels / 'features').mkdir(parents=True)
        rng = np.random.default_rng(0)
        rows, groups = [], {}
        for split, flight, count in (('train', 'AMtown01', 16), ('validation', 'AMvalley01', 6)):
            for state in range(count):
                for candidate in range(3):
                    coverage = rng.random()
                    query = rng.random(len(QUERY_FEATURES)).astype(np.float32)
                    query[QUERY_FEATURES.index('map_coverage')] = coverage
                    tokens = np.zeros((16, 11), np.float32)
                    tokens[:4, -1] = 1
                    name = f'{split}{state}_{candidate}'
                    np.savez_compressed(labels / 'features' / f'{name}.npz', query=query, visual=rng.random(384).astype(np.float32), tokens=tokens, token_visual=np.zeros((16, 384), np.float32))
                    value = float(.5 - coverage)  # Less covered means more useful, so ranking can be learned.
                    rows.append({'experiment_id': name, 'fidelity': 'R_ROLLOUT', 'intervention': 'add', 'execution_status': 'complete', 'split': split, 'clip': f'{flight}_1', 'flight': flight,
                                 'features': name + '.npz', 'targets': {'delta_coverage': value, 'delta_surface': value, 'delta_connectivity': 0., 'utility': value, 'delta_cost': .3}})
                    groups.setdefault((split, state), []).append((name, value))
        (labels / 'labels_a.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        with (labels / 'ranking_groups.jsonl').open('w') as stream:
            for (split, state), members in groups.items():
                pairs = [[a, b] for a, x in members for b, y in members if x > y]
                stream.write(json.dumps({'group': f'{split}{state}', 'members': [m for m, _ in members], 'pairs': pairs, 'split': split, 'fidelity': 'R_ROLLOUT', 'intervention': 'add'}) + '\n')
        (labels / 'quality_audit.json').write_text(json.dumps({'trusted_reference': True}))
        cls.labels = labels
        cls.contract = {'selector': {'seed': 42, 'hidden_dim': 64, 'heads': 4, 'layers': 1, 'learning_rate': 1e-3, 'weight_decay': 1e-4, 'gradient_clip': 1.,
                                     'epochs': 5, 'patience': 5, 'loss_weights': {'rank': 1., 'regression': .5, 'cost': .1, 'uncertainty': .1, 'action': .1, 'source': .5}}}
        cls.output = cls.root / 'out'
        cls.report = train_selector(cls.contract, cls.labels, cls.output, torch.device('cpu'), None, False, False, 'thumbnail384')

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_read_labels_finds_per_clip_files(self):
        self.assertEqual(len(read_labels(self.labels)), 66)

    def test_diagnostic_labels_cannot_train_by_accident(self):
        audit = self.labels / 'quality_audit.json'
        audit.write_text(json.dumps({'trusted_reference': False}))
        try:
            with self.assertRaises(SystemExit):
                train_selector(self.contract, self.labels, self.root / 'blocked', torch.device('cpu'), 1, False, False, 'thumbnail384')
        finally:
            audit.write_text(json.dumps({'trusted_reference': True}))

    def test_training_learns(self):
        self.assertEqual(self.report['skipped_updates'], 0)
        self.assertGreater(self.report['curve'][-1]['train']['pair_accuracy'], .7)
        for head in ('gru', 'memory', 'utility', 'action', 'cost', 'log_variance'):
            self.assertIn(head, self.report['gradient_reached'])

    def test_runtime_loads_the_checkpoint_and_acts(self):
        _, data = load_utility_checkpoint(self.output / 'best.pt')
        self.assertEqual(data['decision_head'], DECISION_HEAD)
        self.assertEqual(head_index('R_ROLLOUT', 'add'), DECISION_HEAD)
        runtime = UtilityRuntime(self.output / 'best.pt', device='cpu')
        kept, reasons = 0, []
        for i in range(8):
            frame = FramePacket(f'f{i}', i, float(i), texture(i), K)
            pose = PoseEstimate(f'f{i}', float(i), nadir(i * 6.), 'test')
            keep, row = runtime.decide(frame, pose, {'reason': 'new_geometric_content', 'tracked_features': 100, 'rotation_compensated_flow_pixels': 20.}, True)
            kept += keep
            reasons.append(row['fallback'])
            if keep:
                runtime.observe(frame.frame_id, np.full((192, 256), 100., np.float32), K, pose.T_world_from_camera, .5)
        self.assertGreaterEqual(kept, 2)
        self.assertLess(sum(r is not None for r in reasons), 8, 'The policy must not fall back on every frame')

    def test_rule_rejection_is_never_overruled(self):
        runtime = UtilityRuntime(self.output / 'best.pt', device='cpu')
        frame = FramePacket('f', 0, 0., texture(0), K)
        self.assertFalse(runtime.decide(frame, PoseEstimate('f', 0., nadir(0.), 'test'), {}, False)[0])

    def test_lost_tracking_keeps_the_rule_decision(self):
        runtime = UtilityRuntime(self.output / 'best.pt', device='cpu')
        frame = FramePacket('f', 0, 0., texture(0), K)
        keep, report = runtime.decide(frame, PoseEstimate('f', 0., nadir(0.), 'test', state='lost'), {}, True)
        self.assertTrue(keep)
        self.assertIn('lost', report['fallback'])

    def test_bridge_frames_are_never_deferred(self):
        runtime = UtilityRuntime(self.output / 'best.pt', device='cpu')
        for i in range(3):
            frame = FramePacket(f'f{i}', i, float(i), texture(i), K)
            keep, row = runtime.decide(frame, PoseEstimate(f'f{i}', float(i), nadir(i * 6.), 'test'), {'reason': 'rotation_bridge_with_geometric_content'}, True)
            self.assertTrue(keep)

    def test_sampling_policy_gives_a_trainable_log_probability(self):
        model, data = load_utility_checkpoint(self.output / 'best.pt')
        model.gru.train()
        policy = UtilityPolicy(model, data, torch.device('cpu'))
        chosen, logs, _ = policy.run(FakeClip(16), 3, Descriptors(), 1.)
        self.assertEqual(len(chosen), 3)
        self.assertTrue(logs)
        (-torch.stack(logs).sum()).backward()
        self.assertGreater(float(model.utility.weight.grad.abs().sum()), 0)

    def test_source_ranking_trains_the_pair_head(self):
        from adaptive_uav.train_utility_selector import source_loss
        from adaptive_uav.graph_selector import MemoryUtilitySelector
        model = MemoryUtilitySelector(hidden=32, layers=1)
        tokens = np.zeros((16, 11), np.float32)
        tokens[:4, -1] = 1
        row = {'query': np.zeros(17, np.float32), 'visual': np.zeros(384, np.float32), 'tokens': tokens, 'token_visual': np.zeros((16, 384), np.float32),
               'question': DECISION_HEAD, 'pairs': [[0, 1, .05], [0, 2, .09], [1, 3, .2]]}
        source_loss(model, row).backward()
        self.assertGreater(float(model.pair[0].weight.grad.abs().sum()), 0)


class ScriptTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[2]

    def test_step_list_runs_calibration_before_any_training(self):
        result = subprocess.run(['bash', str(self.root / 'run_complete_training.sh'), '--list'], capture_output=True, text=True, check=True)
        steps = result.stdout.split()
        self.assertLess(steps.index('calibrate'), steps.index('student'))
        self.assertLess(steps.index('targets'), steps.index('student'))
        self.assertLess(steps.index('student'), steps.index('labels'))
        self.assertLess(steps.index('labels'), steps.index('selector'))
        self.assertLess(steps.index('selector'), steps.index('finetune'))
        self.assertEqual(steps[-1], 'evaluate')

    def test_unknown_step_is_refused(self):
        result = subprocess.run(['bash', str(self.root / 'run_complete_training.sh'), '--to', 'nothing'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
