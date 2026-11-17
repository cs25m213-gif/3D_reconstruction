"""Offline video -> fresh SfM/depth -> mesh reward -> selector policy gradient.

Cached full-video RGB/SfM features propose a frame set. Each reward executes the
real reconstruction on that subset, with newly estimated cameras. Reference
camera poses align coordinates for scoring only; measured points never fit the
alignment. This is not an online-causal or end-to-end differentiable pipeline.
"""
import argparse
import copy
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from uav_reconstruction.adaptive_uav.counterfactual_replay import Clip, role_of
from uav_reconstruction.adaptive_uav.embedding_cache import Descriptors
from uav_reconstruction.adaptive_uav.graph_selector import load_utility_checkpoint
from uav_reconstruction.adaptive_uav.open3d_cpu import cpu_bindings
from uav_reconstruction.adaptive_uav.reconstruct_training_clips import is_descendant
from uav_reconstruction.adaptive_uav.reconstruction_quality import ReferencePopulation, surface_metrics, quality, DEFAULT_WEIGHTS
from uav_reconstruction.adaptive_uav.selector_policy import UtilityPolicy, rule
from uav_reconstruction.adaptive_uav.utility_contracts import sha256_file, write_json


def fit_camera_alignment(estimated, reference):
    """Fit even cameras only, reject unstable alignment on held-out odd cameras."""
    if len(estimated) < 8: raise ValueError('Fewer than eight reconstructed cameras')
    a, b = np.asarray(estimated), np.asarray(reference)
    fit = np.arange(0, len(a), 2); held = np.arange(1, len(a), 2)
    blend = sum(b[i, :3, :3] @ a[i, :3, :3].T for i in fit)
    u, _, vt = np.linalg.svd(blend); rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    x, y = a[fit, :3, 3] @ rotation.T, b[fit, :3, 3]
    xc, yc = x - x.mean(0), y - y.mean(0)
    denominator = float((xc * xc).sum())
    if denominator < 1e-8: raise ValueError('Insufficient camera motion to establish scale')
    scale = float((xc * yc).sum() / denominator)
    if not np.isfinite(scale) or not 0 < scale < 1000: raise ValueError('Invalid camera-derived scale')
    translation = y.mean(0) - scale * x.mean(0)
    errors = np.linalg.norm(scale * (a[held, :3, 3] @ rotation.T) + translation - b[held, :3, 3], axis=1)
    if not np.isfinite(errors).all() or np.median(errors) >= 2.: raise ValueError('Held-out camera alignment error exceeds 2 m')
    return scale, rotation, translation, {'fitted_cameras': len(fit), 'heldout_cameras': len(held), 'heldout_position_median_m': float(np.median(errors))}


def wait_for_other_gpu_jobs():
    while True:
        pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()
        if all(is_descendant(int(pid), os.getpid()) for pid in pids): return
        time.sleep(15)


def make_session(source, chosen, destination):
    """Read original RGB frames; no reference trajectory enters reconstruction."""
    destination.mkdir(parents=True, exist_ok=False)
    video = source / 'session/video.mp4'
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 2.) > .01:
            raise ValueError('This version requires prepared 2 Hz UAVScenes videos')
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        infos = json.loads((source / 'selected_infos.json').read_text())
        selected = [infos[v.index] for v in chosen]
        k = np.asarray(selected[0]['P3x3'])
        if (width, height) != (selected[0]['Width'], selected[0]['Height']): raise ValueError('RGB/calibration dimensions differ')
        camera = {'is_available': True, 'calibration_exact': True, 'width': width, 'height': height,
                  'fx': float(k[0, 0]), 'fy': float(k[1, 1]), 'cx': float(k[0, 2]), 'cy': float(k[1, 2]), 'distortion': [0, 0, 0, 0]}
        write_json(destination / 'camera_intrinsics.json', camera)
        write_json(destination / 'metadata.json', {'complete': True, 'video_width': width, 'video_height': height, 'video_fps': 2,
                   'duration_seconds': len(chosen) / 2, 'source': str(video), 'timing': 'subset playback; original frame indices retained separately',
                   'camera_intrinsics': {'lens_intrinsic_calibration': [camera['fx'], camera['fy'], camera['cx'], camera['cy'], 0], 'pixel_array_size': [width, height]}})
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', '2', '-i', '-', '-c:v', 'libx264', '-threads', '2', '-crf', '17', '-pix_fmt', 'yuv420p', str(destination / 'video.mp4')]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for view in chosen:
                capture.set(cv2.CAP_PROP_POS_FRAMES, view.index)
                ok, frame = capture.read()
                if not ok: raise ValueError(f'Cannot decode original frame {view.index}')
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close(); code = process.wait()
        if code: raise RuntimeError('Selected-video encoding failed')
        write_json(destination.parent / 'selection.json', {'indices': [v.index for v in chosen], 'source': str(video)})
        return selected
    finally: capture.release()


def execute(clip, chosen, backend, args, name):
    output = args.output / 'rollouts' / name
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    queue_seconds = 0.
    result = {'clip': clip.name, 'backend': backend, 'indices': [v.index for v in chosen], 'status': 'failed', 'reward': -10., 'reference_kind': 'measured_lidar'}
    try:
        source = ROOT / args.contract['data']['cache_root'] / clip.name
        selected = make_session(source, chosen, output / 'session')
        waiting = time.monotonic(); wait_for_other_gpu_jobs(); queue_seconds = time.monotonic() - waiting
        command = ['bash', './reconstruct_final.sh', str(output / 'session'), '--output', str(output / 'reconstruction'), '--mode', 'legacy', '--fresh',
                   '--detail', 'balanced', '--da3-model', 'large', '--depth-backend', backend, '--max-views', str(len(chosen)),
                   '--stereo-policy', 'skip', '--semantic-filter', 'off', '--dense-recovery-views', '0', '--mapping-seconds', '90']
        if backend == 'student': command += ['--student-checkpoint', str(args.student_checkpoint)]
        write_json(output / 'command.json', command)
        with (output / 'reconstruction.log').open('x') as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        result['returncode'] = code
        reconstruction = output / 'reconstruction'
        report_path = reconstruction / 'run_report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if code or not report.get('result_ready'): raise ValueError('Reconstruction did not produce a ready result')
        estimated, reference, depth_backends = [], [], Counter()
        for file in sorted((reconstruction / 'multiview/depth_cache').glob('frame_*.jpg.npz')):
            index = int(file.name.split('_')[1].split('.')[0])
            if index >= len(selected): raise ValueError('Reconstructed frame identity is outside the selected video')
            with np.load(file) as cached:
                estimated.append(np.linalg.inv(cached['P'].astype(float)))
                depth_backends[str(cached['depth_backend'].item()) if 'depth_backend' in cached else 'da3'] += 1
            reference.append(np.asarray(selected[index]['T4x4'], dtype=float))
        scale, rotation, translation, alignment = fit_camera_alignment(estimated, reference)
        o3d = cpu_bindings()
        mesh_path = reconstruction / 'multiview/scene_mesh_colored.ply'
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not len(mesh.triangles): raise ValueError('Empty mesh')
        o3d.utility.random.seed(42)
        points = np.asarray(mesh.sample_points_uniformly(number_of_points=50000).points)
        points = scale * (points @ rotation.T) + translation
        population = ReferencePopulation(clip.points, args.contract['utility']['tolerance_m'], 'QUALIFIED', 'fresh_video_mesh_v1')
        metrics = surface_metrics(points, population)
        score = float(np.clip(quality(metrics, None, DEFAULT_WEIGHTS), -10., 1.))
        result.update(status='complete', mesh=str(mesh_path), metrics=metrics, alignment=alignment, depth_backend_views=dict(depth_backends),
                      student_fallback_views=depth_backends['da3'] if backend == 'student' else 0, reference_identity=population.identity(),
                      capture_status=report.get('capture_status'), reconstruction_status=report.get('status'), surface_quality=score, reward=score)
    except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        result['error'] = str(error)
    result['seconds'] = time.monotonic() - started - queue_seconds
    result['excluded_gpu_queue_seconds'] = queue_seconds
    result['reward'] -= args.cost_weight * min(result['seconds'] / 300., 10.)
    write_json(output / 'reward.json', result)
    return result


def select_clips(contract, cohort, role):
    """One eligible video per physical flight for an explicitly bounded pilot."""
    chosen = {}
    for name in cohort[role]:
        if name.rsplit('_', 1)[0] in chosen: continue
        folder = ROOT / contract['data']['cache_root'] / name
        if not (folder / 'session/video.mp4').is_file(): continue
        clip = Clip(ROOT, contract, name)
        if role_of(clip.scene, contract) != role or len(clip.views) < 24 or not len(clip.points): continue
        if clip.scene not in chosen: chosen[clip.scene] = clip
        if len(chosen) == 3: break
    return list(chosen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--student-checkpoint', type=Path, required=True)
    parser.add_argument('--selector-checkpoint', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rollouts', type=int, default=6)
    parser.add_argument('--cost-weight', type=float, default=.01)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve()
    args.contract = json.loads(args.config.read_text()); cohort = json.loads(args.cohort.read_text())['selector_clips']
    if args.rollouts < 1 or args.cost_weight < 0: parser.error('Need positive rollouts and nonnegative cost weight')
    for path in (args.student_checkpoint, args.selector_checkpoint):
        if not path.is_file(): raise FileNotFoundError(path)
    if not args.embeddings.is_dir(): raise FileNotFoundError(args.embeddings)
    gate = json.loads((ROOT / args.contract['geometry_gate']).read_text())['flights']
    if args.dry_run:
        eligible = {}
        for role in ('train', 'validation'):
            flights = set()
            for name in cohort[role]:
                folder = ROOT / args.contract['data']['cache_root'] / name
                infos = json.loads((folder / 'selected_infos.json').read_text())
                scene = infos[0]['OriginalSceneName']
                if ((folder / 'session/video.mp4').is_file() and
                    len(list((folder / 'reconstruction/multiview/depth_cache').glob('frame_*.jpg.npz'))) >= 24 and
                    role_of(scene, args.contract) == role and gate[scene]['status'] in ('QUALIFIED', 'VERIFIED')):
                    flights.add(scene)
            if len(flights) < 3: raise ValueError(f'Need three eligible {role} flights')
            eligible[role] = sorted(flights)
        print(json.dumps({'train_candidates': len(cohort['train']), 'validation_candidates': len(cohort['validation']), 'rollouts': args.rollouts,
                          'eligible_flights': eligible, 'fresh_reconstruction': True, 'causal_online_evidence': False, 'automatic_promotion': False})); return
    if args.output.exists(): raise SystemExit('Use a fresh mesh-loop output directory')
    args.output.mkdir(parents=True)
    torch.manual_seed(42); np.random.seed(42); cv2.setNumThreads(1)
    train, held = select_clips(args.contract, cohort, 'train'), select_clips(args.contract, cohort, 'validation')
    if len(train) < 3 or len(held) < 3: raise ValueError('Need three training and three validation flights with eligible videos')
    if any(gate[c.scene]['status'] not in ('QUALIFIED', 'VERIFIED') for c in train + held): raise ValueError('Unqualified reference')
    model, data = load_utility_checkpoint(args.selector_checkpoint, 'cuda')
    if data.get('diagnostic_labels', True) or data.get('descriptor') != 'dinov2_vits14': raise ValueError('Require qualified DINO selector')
    model.eval(); model.gru.train()
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.split('.')[0] in ('gru', 'utility'))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    policy, descriptors = UtilityPolicy(model, data, 'cuda'), Descriptors(args.embeddings)
    baseline, teacher = {}, {}
    # Fixed clip/reference populations before sampling; never drop failed rollouts.
    write_json(args.output / 'protocol.json', {'train': [c.name for c in train], 'validation': [c.name for c in held],
               'student_sha256': sha256_file(args.student_checkpoint), 'selector_sha256': sha256_file(args.selector_checkpoint),
               'frames_per_video': 24, 'reward': 'measured LiDAR surface quality minus bounded runtime penalty; -10 for failure',
               'reference_alignment': 'camera poses only, even fit / odd check', 'teacher': 'DA3-LARGE paired control, not ground truth',
               'causal_online_evidence': False, 'scope': 'Fresh reconstructed selected video, proposed using full-video cached SfM features', 'automatic_promotion': False})
    for clip in train + held:
        chosen = rule(clip, 24)
        teacher[clip.name] = execute(clip, chosen, 'da3', args, clip.name + '_teacher_rule')
        baseline[clip.name] = execute(clip, chosen, 'student', args, clip.name + '_student_rule')
    starting = copy.deepcopy(model.state_dict())
    steps = 0
    with (args.output / 'optimizer_steps.jsonl').open('x') as log:
        for step in range(args.rollouts):
            clip = train[step % len(train)]
            selected, probabilities, entropies = policy.run(clip, 24, descriptors, 1.)
            result = execute(clip, selected, 'student', args, f'{clip.name}_sample_{step:04d}')
            advantage = result['reward'] - baseline[clip.name]['reward']
            updated = False
            if probabilities:
                loss = -(advantage * torch.stack(probabilities).sum() + .01 * torch.stack(entropies).sum())
                optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if torch.isfinite(norm): optimizer.step(); steps += 1; updated = True
            log.write(json.dumps({'step': step, 'clip': clip.name, 'advantage': advantage, 'reward': result['reward'], 'status': result['status'], 'updated': updated}) + '\n'); log.flush()
    trained = copy.deepcopy(model.state_dict()); validation = []
    for variant, weights in [('before', starting), ('after', trained)]:
        model.load_state_dict(weights); model.eval()
        for clip in held:
            with torch.no_grad(): selected, _, _ = policy.run(clip, 24, descriptors)
            result = execute(clip, selected, 'student', args, f'{clip.name}_validation_{variant}')
            validation.append({'variant': variant, **result})
    torch.save({**data, 'model': trained, 'mesh_reward_finetuned_from': str(args.selector_checkpoint), 'causal_online_evidence': False}, args.output / 'candidate.pt')
    gains = [next(r['reward'] for r in validation if r['clip'] == c.name and r['variant'] == 'after') -
             next(r['reward'] for r in validation if r['clip'] == c.name and r['variant'] == 'before') for c in held]
    paired = {}
    for name, student in baseline.items():
        teacher_result = teacher[name]
        both_ready = student['status'] == teacher_result['status'] == 'complete'
        paired[name] = {'student_status': student['status'], 'teacher_status': teacher_result['status'],
                        'student_fallback_views': student.get('student_fallback_views'),
                        'pure_student_comparison': both_ready and student.get('student_fallback_views') == 0,
                        'student_minus_teacher_surface_quality': student['surface_quality'] - teacher_result['surface_quality'] if both_ready else None}
    write_json(args.output / 'report.json', {'optimizer_steps': steps, 'rollout_attempts': args.rollouts, 'teacher_controls': teacher,
               'student_rule_controls': baseline, 'paired_student_teacher': paired, 'validation': validation, 'mean_validation_gain': float(np.mean(gains)),
               'validation_gain_by_flight': dict(zip([c.scene for c in held], gains)), 'promote': False,
               'scope': 'Bounded fresh video-to-mesh reward pilot; no online-causal or final-test claim'})


if __name__ == '__main__': main()
