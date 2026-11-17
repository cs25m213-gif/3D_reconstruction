"""Stage 4: measure what a frame adds to this reconstruction, in this state, on this budget.

Each decision compares isolated branches built from one prefix and scored on one fixed reference
population. Three fidelities stay separate. F_DEPTH holds poses fixed and asks about adding,
removing or replacing an observation. P_LOCAL registers the candidate by PnP against prefix-only
landmarks. R_ROLLOUT continues both branches through the same future frames, where the budget is a
real limit: a depth call spent now is one fewer later. Cost is the work the frame would cost the
pipeline, never the time this evaluator takes.

Keyframes and candidates are spaced by baseline over depth, so a clip near takeoff and one at
altitude get the same geometry. Prefix poses are the de-domed SfM trajectory aligned to the reference
frame, standing in for the running tracker; chaining PnP instead drifts by metres and loses
low-texture clips. The candidate itself only ever sees prefix landmarks. No bundle adjustment runs;
records say so.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .embedding_cache import Descriptors
from .hybrid_dense import integrate, make_volume
from .reconstruction_quality import DEFAULT_WEIGHTS, ReferencePopulation, pose_error, quality, surface_metrics, utility
from .selector_features import MapState, as_pose
from .student_inference import anchor_scale, load_student, predict
from .utility_contracts import THREADS, Budget, load_contract, sha256_file, state_identity, validate_record, write_json, write_status
from .validate_reference_geometry import area_of

cv2.setNumThreads(THREADS)


@dataclass(eq=False)
class View:
    frame_id: str
    index: int
    timestamp: float
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    reference_pose: np.ndarray  # Evaluation and the named fixed-pose diagnostic only.
    feature_pose: np.ndarray
    feature_k: np.ndarray
    confidence: float
    fusion_rgb: np.ndarray
    fusion_depth: np.ndarray
    fusion_k: np.ndarray
    aligned_pose: np.ndarray | None = None
    anchors_m: np.ndarray | None = None
    source_ids: tuple = ()


def role_of(scene: str, contract: dict) -> str | None:
    return next((role for role, areas in contract['splits'].items() if any(scene.startswith(a) for a in areas)), None)


class Clip:
    def __init__(self, root: Path, contract: dict, name: str):
        data, folder = contract['data'], root / contract['data']['cache_root'] / name
        self.name = name
        infos = json.loads((folder / 'selected_infos.json').read_text())
        self.scene = infos[0]['OriginalSceneName']
        alpha = json.loads((root / data['scale_calibration']).read_text())[name]['scale_m_per_visual_unit']
        overlay = root / data['overlay'] / name if data.get('overlay') else None
        self.depth_scale, self.views, points, ratios, map_points = 1., [], [], [], []
        for file in sorted((folder / 'reconstruction/multiview/depth_cache').glob('frame_*.jpg.npz')):
            index = int(file.name.split('_')[1].split('.')[0])
            with np.load(file) as cached:
                rgb, depth, k, p, confidence = (cached[key] for key in ('rgb', 'depth', 'K', 'P', 'confidence'))
                anchors = cached['anchors'].copy() if 'anchors' in cached else None
            k, p, feature_k, teacher = k.astype(np.float64), p.astype(np.float64), k.astype(np.float64), 1.
            extra = overlay / f'frame_{index:06d}.npz' if overlay else None
            if extra is not None and extra.exists():
                with np.load(extra) as refined:
                    p, feature_k = refined['P'].astype(np.float64), refined['K'].astype(np.float64)
                    if 'anchors' in refined:
                        anchors = refined['anchors'].copy()
                    value = float(refined['teacher_scale'])
                    teacher = value if np.isfinite(value) else 1.
            if not np.isfinite(p).all():
                continue
            pose = np.asarray(infos[index]['T4x4'], dtype=np.float64)
            feature = np.linalg.inv(p)
            feature[:3, 3] *= alpha
            metric = (depth * teacher * alpha).astype(np.float32)
            self.views.append(View(f'{name}/{index}', index, index / 2., rgb, metric, k, pose, feature, feature_k,
                                   float(np.median(confidence)), rgb, metric, k,
                                   anchors_m=None if anchors is None else anchors.astype(np.float32) * alpha))
            target = root / data['sidecars'] / name / f'frame_{index:06d}.npz'
            if target.exists():
                with np.load(target) as measured:
                    lidar = measured['measured_depth_z_m'].copy()
                    if 'reliable' in measured:
                        lidar = lidar * measured['reliable']
                    dense = measured['map_depth_z_m'] if 'map_depth_z_m' in measured else None
                ys, xs = np.nonzero(lidar)
                if len(ys):
                    z = lidar[ys, xs].astype(np.float64)
                    rays = np.c_[(xs + .5 - k[0, 2]) / k[0, 0], (ys + .5 - k[1, 2]) / k[1, 1], np.ones(len(z))] * z[:, None]
                    points.append(rays @ pose[:3, :3].T + pose[:3, 3])
                    ratios.append(float(np.median(z / np.maximum(metric[ys, xs], 1e-3))))
                if dense is not None and dense.any():
                    ys, xs = np.nonzero(dense)
                    z = dense[ys, xs].astype(np.float64)
                    rays = np.c_[(xs + .5 - k[0, 2]) / k[0, 0], (ys + .5 - k[1, 2]) / k[1, 1], np.ones(len(z))] * z[:, None]
                    map_points.append((rays @ pose[:3, :3].T + pose[:3, 3])[::7])
        self.points = np.concatenate(points)[::5] if points else np.empty((0, 3))
        self.map_points = np.concatenate(map_points) if map_points else np.empty((0, 3))
        self.lidar_ratio = float(np.nanmedian(ratios)) if ratios else None
        self.keyframes: list = []
        if self.views:
            self.align_to_reference()

    def align_to_reference(self) -> None:
        """Rotation from camera orientations: positions along a straight run leave the roll about
        that line free, which tilted whole maps by up to 19 degrees. The calibrated scale stays."""
        blend = sum(v.reference_pose[:3, :3] @ v.feature_pose[:3, :3].T for v in self.views)
        left, _, right = np.linalg.svd(blend)
        rotation = left @ np.diag([1., 1., np.linalg.det(left @ right)]) @ right
        centres = np.array([v.feature_pose[:3, 3] for v in self.views]) @ rotation.T
        shift = np.mean([v.reference_pose[:3, 3] for v in self.views], axis=0) - centres.mean(0)
        for view, centre in zip(self.views, centres):
            view.aligned_pose = np.eye(4)
            view.aligned_pose[:3, :3] = rotation @ view.feature_pose[:3, :3]
            view.aligned_pose[:3, 3] = centre + shift

    def spacing(self, start: int, later: int) -> float:
        """Baseline over the start frame's depth: what sets overlap, at any altitude or speed."""
        first, second = self.views[start], self.views[later]
        values = first.anchors_m if first.anchors_m is not None else first.depth_m
        valid = values[np.isfinite(values) & (values > 0)]
        return float(np.linalg.norm(second.feature_pose[:3, 3] - first.feature_pose[:3, 3]) / np.median(valid)) if len(valid) else 0.

    def choose_keyframes(self, ratio: float) -> list:
        chosen = [0]
        for position in range(1, len(self.views)):
            if self.spacing(chosen[-1], position) >= ratio:
                chosen.append(position)
        self.keyframes = chosen
        return chosen

    def candidates_after(self, position: int, ratios: list) -> set:
        """The first frame past each spacing, so candidates reach as far at every altitude."""
        found = {next((later for later in range(position + 1, len(self.views)) if self.spacing(position, later) >= ratio), None) for ratio in ratios}
        return found - {None}

    def apply_scale(self, scale: float) -> None:
        """One calibration-derived scale for both branches; never fitted per branch."""
        self.depth_scale = scale
        for view in self.views:
            view.depth_m = view.depth_m * scale
            view.fusion_depth = view.fusion_depth * scale

    def use_student(self, model, device) -> None:
        """Replace teacher depth with past-only student depth, so labels describe the deployed backend."""
        for position, view in enumerate(self.views):
            earlier = [self.views[p] for p in self.keyframes if p < position][-2:]
            depth, _ = predict(model, [(v.rgb, v.feature_k, v.feature_pose) for v in [view] + earlier[::-1]],
                               anchor_scale(view.anchors_m), device, anchors=view.anchors_m)
            view.fusion_depth, view.fusion_k = depth.astype(np.float32), view.K.copy()
            scale_x, scale_y = depth.shape[1] / view.rgb.shape[1], depth.shape[0] / view.rgb.shape[0]
            view.fusion_k[0] *= scale_x
            view.fusion_k[1] *= scale_y
            view.fusion_rgb = cv2.resize(view.rgb, depth.shape[::-1], interpolation=cv2.INTER_AREA)
            view.depth_m = cv2.resize(view.fusion_depth, view.rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)


def fuse(observations: list, voxel: float) -> np.ndarray:
    """A fresh volume per branch keeps branches isolated."""
    volume = make_volume(voxel)
    for view, pose in observations:
        integrate(volume, [{'depth': view.fusion_depth, 'tier': (view.fusion_depth > 0).astype(np.uint8), 'rgb': view.fusion_rgb,
                            'K': view.fusion_k, 'P': np.linalg.inv(pose)}], [1], 250.)
    return np.asarray(volume.extract_point_cloud().points)


def register(view: View, keyframes: list, min_inliers: int) -> dict:
    """PnP against landmarks from committed keyframes only, so no future track can help."""
    orb = cv2.ORB_create(1500)
    points, descriptors = orb.detectAndCompute(cv2.cvtColor(view.rgb, cv2.COLOR_RGB2GRAY), None)
    if descriptors is None:
        return {'ok': False, 'inliers': 0, 'pose': None, 'matches': 0}
    world, pixels, calls = [], [], 0
    for key, pose in keyframes[-3:]:
        older, old_descriptors = orb.detectAndCompute(cv2.cvtColor(key.rgb, cv2.COLOR_RGB2GRAY), None)
        calls += 1
        if old_descriptors is None:
            continue
        for match in cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(old_descriptors, descriptors):
            u, v = older[match.queryIdx].pt
            z = float(key.depth_m[int(v), int(u)])
            if z <= 0:
                continue
            camera = np.array([(u - key.K[0, 2]) / key.K[0, 0] * z, (v - key.K[1, 2]) / key.K[1, 1] * z, z])
            world.append(pose[:3, :3] @ camera + pose[:3, 3])
            pixels.append(points[match.trainIdx].pt)
    if len(world) < 8:
        return {'ok': False, 'inliers': 0, 'pose': None, 'matches': calls}
    cv2.setRNGSeed(0)
    world, pixels = np.float64(world), np.float64(pixels)
    found, rotation, translation, inliers = cv2.solvePnPRansac(world, pixels, view.K, None, reprojectionError=3.,
                                                               iterationsCount=200, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    count = 0 if inliers is None else len(inliers)
    if not found or count < min_inliers:
        return {'ok': False, 'inliers': count, 'pose': None, 'matches': calls}
    rotation, translation = cv2.solvePnPRefineLM(world[inliers[:, 0]], pixels[inliers[:, 0]], view.K, None, rotation, translation)
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(rotation)[0]
    matrix[:3, 3] = translation.ravel()
    return {'ok': True, 'inliers': count, 'pose': np.linalg.inv(matrix), 'matches': calls}


def score(observations: list, reference, voxel: float, poses, truth, weights: dict) -> dict:
    metrics = surface_metrics(fuse(observations, voxel), reference)
    pose = pose_error(poses, truth) if poses else None
    return {'metrics': metrics, 'pose': pose, 'quality': quality(metrics, pose, weights)}


def rollout(start: list, future: list, allowance: int, min_inliers: int) -> tuple:
    """Frozen downstream policy: spread the remaining depth calls evenly over the same frames."""
    observations, estimated, failures = list(start), {}, 0
    if allowance <= 0 or not future:
        return observations, estimated, failures
    for position in np.unique(np.linspace(0, len(future) - 1, allowance).astype(int)):
        view = future[int(position)]
        found = register(view, observations, min_inliers)
        if not found['ok']:
            failures += 1
            continue
        observations.append((view, found['pose']))
        estimated[view.index] = found['pose']
    return observations, estimated, failures


def integration_seconds(view: View, pose: np.ndarray, voxel: float) -> float:
    started = time.monotonic()
    fuse([(view, pose)], voxel)
    return time.monotonic() - started


def plan(clip, settings: dict):
    """Fixed, seedless order. Frames after the largest map are tried from every map, so one
    frame is measured against several map states."""
    views, chosen, ratios = clip.views, clip.keyframes, settings['candidate_baseline_ratios']
    sizes = [size for size in settings['prefix_keyframes'] if size <= len(chosen)]
    lasts = [chosen[size - 1] for size in sizes]
    for size, last in zip(sizes, lasts):
        prefix = [views[p] for p in chosen[:size]]
        for where in sorted(clip.candidates_after(last, ratios) | clip.candidates_after(lasts[-1], ratios)):
            yield prefix, views[where], views[where + 1:where + 1 + settings['rollout_frames']]


def eligible(gate: dict | None, flight: str, diagnostic: bool) -> str | None:
    status = (((gate or {}).get('flights') or {}).get(flight) or {}).get('status', 'UNRESOLVED')
    return status if status in ('VERIFIED', 'QUALIFIED') or diagnostic else None


class Recorder:
    """Shared fields and bookkeeping for one candidate, written once per budget."""

    def __init__(self, stream, clip, reference, status: str, settings: dict, weights: dict, backend: dict, done: set):
        self.stream, self.clip, self.reference, self.status = stream, clip, reference, status
        self.settings, self.weights, self.backend, self.done = settings, weights, backend, done
        self.written = 0

    def write(self, state: str, prefix: list, candidate: View, fidelity: str, intervention: str, budget_id: str, control: dict, intervened: dict,
              connectivity: float, cost: float, execution: str, detail: dict, features: str) -> None:
        identity = state_identity(state=state, candidate=candidate.index, fidelity=fidelity, intervention=intervention, budget=budget_id)
        if identity in self.done:
            return
        remaining = float(self.settings['budgets'][budget_id]['seconds'])
        if execution == 'complete' and cost > remaining:
            execution = 'over_budget'
        record = {'schema_version': 2, 'experiment_id': identity, 'state_id': state, 'candidate': candidate.index, 'action': 'NEW_KEYFRAME',
                  'intervention': intervention, 'fidelity': fidelity, 'control_action': 'TRACK_ONLY', 'budget_id': budget_id, 'clip': self.clip.name,
                  'flight': self.clip.scene, 'split': self.settings['_role'], 'evaluation_protocol_id': self.reference.protocol,
                  'reference_identity': self.reference.identity(), 'reference_status': self.status, 'execution_status': execution, 'control': control,
                  'intervened': intervened, 'cost': {'incremental_seconds': cost, 'remaining_seconds': remaining},
                  'targets': {'delta_coverage': intervened['metrics']['recall'] - control['metrics']['recall'],
                              'delta_surface': intervened['quality'] - control['quality'], 'delta_connectivity': connectivity, 'delta_cost': cost,
                              'utility': utility(control['quality'], intervened['quality'], connectivity, cost, remaining, self.weights)},
                  'scalarization': self.weights['version'], 'detail': {**self.backend, 'local_bundle_adjustment': False, **detail},
                  'features': features, 'prefix': [v.index for v in prefix]}
        validate_record(record)
        self.stream.write(json.dumps(record) + '\n')
        self.stream.flush()
        self.written += 1


def run_clip(clip, settings: dict, reference, recorder: Recorder, descriptors, output: Path, limit: int) -> dict:
    voxel, inliers, costs = settings['voxel_m'], settings['registration_min_inliers'], settings['costs']
    budgets, report = settings['budgets'], {'states': 0, 'decisions': 0}
    current, tracked, state, fixed_base, tracked_base, active = None, None, None, None, None, None

    def removal_labels(prefix, name) -> None:
        """What would dropping this stored observation lose? Past work is spent, so no refund."""
        for victim in (len(prefix) // 2, len(prefix) - 1):
            view = prefix[victim]
            rest = [(v, v.reference_pose) for j, v in enumerate(prefix) if j != victim]
            after = score(rest, reference, voxel, None, None, recorder.weights)
            for budget_id, limits in budgets.items():
                row = state.features(view, as_pose(view, view.aligned_pose), Budget(**limits).state(), descriptors.get(clip.name, view.index, view.rgb))
                file = output / 'features' / f'{name}_{view.index}_{budget_id}_remove.npz'
                np.savez_compressed(file, **{k: v for k, v in row.items() if k != 'token_ids'}, token_ids=np.array(row['token_ids']))
                recorder.write(name, prefix, view, 'F_DEPTH', 'remove', budget_id, fixed_base, after, 0., 0., 'complete', {'semantics': 'fusion observation only'}, str(file))

    for prefix, candidate, future in plan(clip, settings):
        if report['decisions'] >= limit:
            break
        key = tuple(v.index for v in prefix)
        if key != current:
            if active is not None:
                removal_labels(*active)
                active = None
            current, tracked = key, [(v, v.aligned_pose) for v in prefix]
            report['states'] += 1
            state = MapState()
            for view, pose in tracked:
                frame = as_pose(view, pose)
                state.commit(view, frame, descriptors.get(clip.name, view.index, view.rgb))
                state.observe(view.frame_id, view.depth_m, view.K, pose, view.confidence)
            fixed = [(v, v.reference_pose) for v in prefix]
            fixed_base, tracked_base = score(fixed, reference, voxel, None, None, recorder.weights), score(tracked, reference, voxel, None, None, recorder.weights)
        name = state_identity(clip=clip.name, prefix=key, policy='rule', backend=recorder.backend['depth_backend'])
        active = (prefix, name)
        started = time.monotonic()
        found = register(candidate, tracked, inliers)
        seconds = time.monotonic() - started
        estimate = found['pose'] if found['ok'] else tracked[-1][1]
        work = integration_seconds(candidate, candidate.reference_pose, voxel) + costs['depth_call_seconds']
        files = {}
        for budget_id, limits in budgets.items():
            budget = Budget(**limits).state()
            row = state.features(candidate, as_pose(candidate, estimate), budget, descriptors.get(clip.name, candidate.index, candidate.rgb))
            file = output / 'features' / f'{name}_{candidate.index}_{budget_id}.npz'
            np.savez_compressed(file, **{k: v for k, v in row.items() if k != 'token_ids'}, token_ids=np.array(row['token_ids']))
            files[budget_id] = str(file)
            report['decisions'] += 1
        add_fixed = score(fixed + [(candidate, candidate.reference_pose)], reference, voxel, None, None, recorder.weights)
        swap_fixed = score(fixed[:-1] + [(candidate, candidate.reference_pose)], reference, voxel, None, None, recorder.weights)
        branch = tracked + [(candidate, found['pose'])] if found['ok'] else tracked
        add_local = score(branch, reference, voxel, None, None, recorder.weights)
        detail = {'inliers': found['inliers'], 'replaced': prefix[-1].index}
        if found['ok']:
            detail['candidate_pose_error'] = pose_error({candidate.index: found['pose']}, {candidate.index: candidate.reference_pose})
        execution = 'complete' if found['ok'] else 'registration_failed'
        truth = {v.index: v.reference_pose for v in future}
        for budget_id, limits in budgets.items():
            recorder.write(name, prefix, candidate, 'F_DEPTH', 'add', budget_id, fixed_base, add_fixed, 0., work, 'complete', {}, files[budget_id])
            recorder.write(name, prefix, candidate, 'F_DEPTH', 'replace', budget_id, fixed_base, swap_fixed, 0., work, 'complete', detail, files[budget_id])
            recorder.write(name, prefix, candidate, 'P_LOCAL', 'add', budget_id, tracked_base, add_local, float(found['ok']), seconds + (work if found['ok'] else 0.),
                           execution, detail, files[budget_id])
            allowance = int(limits['depth_calls'])
            control_end, control_poses, control_failed = rollout(tracked, future, allowance, inliers)
            branch_end, branch_poses, branch_failed = rollout(branch, future, allowance - int(found['ok']), inliers)
            shared = set(control_poses) & set(branch_poses)
            before = score(control_end, reference, voxel, {i: control_poses[i] for i in shared}, truth, recorder.weights)
            after = score(branch_end, reference, voxel, {i: branch_poses[i] for i in shared}, truth, recorder.weights)
            attempts = max(allowance, 1)
            connected = float(found['ok']) * .5 + (control_failed / attempts - branch_failed / max(allowance - int(found['ok']), 1) if allowance > int(found['ok']) else 0.)
            recorder.write(name, prefix, candidate, 'R_ROLLOUT', 'add', budget_id, before, after, connected, seconds + (work if found['ok'] else 0.), execution,
                           {**detail, 'control_failures': control_failed, 'intervention_failures': branch_failed, 'allowance': allowance}, files[budget_id])
    if active is not None:
        removal_labels(*active)
    return report


def calibration_scales(root: Path, contract: dict, names: list) -> dict:
    """LiDAR/depth ratio per flight: the median over its area's calibration clips.

    After de-doming one clip's ratio is mostly that clip's own noise, about 2%, which is over a
    metre at 60 m. On six qualified clips the area median gave F 0.62, against 0.57 for the
    flight's own calibration clip and 0.58 for no ratio at all.
    """
    ratios, suffix = {}, contract['geometry']['calibration_batch_suffix']
    for name in names:
        if name.endswith(suffix):
            clip = Clip(root, contract, name)
            if clip.lidar_ratio:
                ratios.setdefault(area_of(clip.scene), {})[clip.scene] = clip.lidar_ratio
    return {scene: float(np.median(list(flights.values()))) for flights in ratios.values() for scene in flights}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--clips', nargs='*')
    parser.add_argument('--roles', nargs='*', default=['train', 'validation'])
    parser.add_argument('--decisions', type=int, help='Limit per clip')
    parser.add_argument('--total-decisions',type=int,help='Global sequential pilot cap; default is utility.pilot_decisions')
    parser.add_argument('--all-decisions', action='store_true', help='Exhaust every planned state/candidate/budget in every eligible clip; no pilot caps')
    parser.add_argument('--student-checkpoint', type=Path, help='Past-only student depth instead of cached teacher depth')
    parser.add_argument('--embeddings', type=Path, help='Folder from embedding_cache; thumbnails otherwise')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--replay-mode', choices=('sequential', 'cached'), default='sequential',
                        help='Sequential three-action branches, or historical cached-depth diagnostics')
    parser.add_argument('--diagnostic', action='store_true', help='Allow unresolved references; labels are marked and cannot promote')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(v is not None and v<1 for v in (args.decisions,args.total_decisions)):
        parser.error('Decision limits must be positive')
    if args.all_decisions and (args.decisions is not None or args.total_decisions is not None):
        parser.error('--all-decisions cannot be combined with decision caps')
    contract = load_contract(args.config, ('utility.quality_protocol', 'utility.scalarization'))
    root, output, settings = args.root.resolve(), args.output.resolve(), dict(contract['utility'])
    gate_path = root / contract['geometry_gate']
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    cache_root = root / contract['data']['cache_root']
    every = sorted(p.name for p in cache_root.iterdir() if (p / 'selected_infos.json').exists() and 'GNSS' not in p.name)
    names = args.clips or [n for n in every if role_of(n, contract) in args.roles]
    if args.dry_run:
        print(json.dumps({'clips': names, 'decisions_per_clip': None if args.all_decisions else args.decisions or settings['pilot_decisions'],
                          'replay_mode':args.replay_mode,'total_decisions':None if args.all_decisions else args.total_decisions or settings['pilot_decisions'], 'all_decisions':args.all_decisions}))
        return
    if args.replay_mode == 'sequential' and not args.student_checkpoint:
        raise SystemExit('Sequential replay requires --student-checkpoint; cached teacher depths cannot be shared across changing contexts')
    (output / 'features').mkdir(parents=True, exist_ok=True)
    write_status(output, 'stage4_labels', 'running', None, None)
    scales = calibration_scales(root, contract, every)
    model = load_student(args.student_checkpoint, args.device) if args.student_checkpoint else None
    backend = {'depth_backend': 'student_branch_context' if model and args.replay_mode=='sequential' else 'student_past_only' if model else 'cached_da3_large',
               'depth_causal':False, 'source_images_past_only':model is not None, 'causal_online_evidence':False,
               'candidate_pose_source': 'PnP against prefix landmarks', 'prefix_pose_source': 'de-domed full-clip SfM aligned to the reference frame',
               'student_checkpoint': str(args.student_checkpoint) if model else None,
               'checkpoint_sha256':sha256_file(args.student_checkpoint) if model else None,
               'config_sha256':sha256_file(args.config), 'replay_mode':args.replay_mode,
               'diagnostic_labels':args.diagnostic,
               'normalization':'sparse_anchor_median_then_anchor_fit' if model else 'teacher_cache'}
    descriptors = Descriptors(args.embeddings)
    report = {'clips': {}, 'skipped': {}, 'backend': backend, 'descriptor': descriptors.name, 'diagnostic_labels': args.diagnostic, 'calibration_scales': scales}
    total_limit=float('inf') if args.all_decisions else args.total_decisions or settings['pilot_decisions']
    if total_limit<1 or (args.decisions is not None and args.decisions<1):
        raise SystemExit('Decision limits must be positive')
    scale_rows=json.loads((root/contract['data']['scale_calibration']).read_text())
    candidates=[n for n in names if not n.endswith(contract['geometry']['calibration_batch_suffix'])
                and n in scale_rows and (cache_root/n/'reconstruction/multiview/depth_cache').is_dir()
                and eligible(gate,scale_rows[n]['scene'],args.diagnostic) is not None]
    per_clip_limit=float('inf') if args.all_decisions else min(args.decisions or settings['pilot_decisions'],max(1,total_limit//max(1,len(candidates))))
    used=0
    report['sequential_pilot']={'total_cap':None if args.all_decisions else total_limit,'per_clip_cap':None if args.all_decisions else per_clip_limit,'decisions':0,'all_decisions':args.all_decisions}
    for name in names:
        if args.replay_mode=='sequential' and used>=total_limit:
            break
        if name.endswith(contract['geometry']['calibration_batch_suffix']):
            continue  # Calibration clips fit the scale, so they cannot also score it.
        if name not in scale_rows or not (cache_root/name/'reconstruction/multiview/depth_cache').is_dir():
            report['skipped'][name]='No prepared camera/depth cache or scale; reconstruct and calibrate this clip first'
            continue
        clip = Clip(root, contract, name)
        status = eligible(gate, clip.scene, args.diagnostic)
        if status is None or clip.scene not in scales or not len(clip.points):
            report['skipped'][name] = 'reference not eligible' if status is None else 'no calibration or reference'
            continue
        clip.apply_scale(scales[clip.scene])
        clip.choose_keyframes(settings['keyframe_baseline_ratio'])
        if model is not None and args.replay_mode=='cached':
            clip.use_student(model, args.device)
        points = clip.map_points if settings.get('reference') == 'map' and len(clip.map_points) else clip.points
        reference = ReferencePopulation(points, settings['tolerance_m'], status, settings['quality_protocol'])
        labels = output / f'labels_{name}.jsonl'
        done = {json.loads(line)['experiment_id'] for line in labels.read_text().splitlines()} if labels.exists() else set()
        with labels.open('a') as stream:
            recorder = Recorder(stream, clip, reference, status, {**settings, '_role': role_of(clip.scene, contract)}, DEFAULT_WEIGHTS, backend, done)
            if args.replay_mode=='sequential':
                from .sequential_replay import StudentDepthProvider, run_sequential_clip
                report['clips'][name] = run_sequential_clip(clip, {**settings,'_role':role_of(clip.scene,contract)}, reference,
                    recorder, descriptors, output, min(per_clip_limit,total_limit-used), StudentDepthProvider(model,args.device))
                used+=report['clips'][name]['decisions']
                report['sequential_pilot']['decisions']=used
            else:
                report['clips'][name] = run_clip(clip, settings, reference, recorder, descriptors, output, per_clip_limit if args.all_decisions else args.decisions or settings['pilot_decisions'])
            report['clips'][name]['records_written'] = recorder.written
        print(name, json.dumps(report['clips'][name]), flush=True)
    write_json(output / 'generation_report.json', report)
    gate_name = 'DIAGNOSTIC' if args.diagnostic else 'LABELS_WRITTEN' if report['clips'] else 'BLOCKED'
    write_status(output, 'stage4_labels', 'complete', gate_name, 'stage4_dataset' if report['clips'] else None, json.dumps(report['skipped']))
    print(gate_name)
    raise SystemExit(0 if report['clips'] else 3)


if __name__ == '__main__':
    main()
