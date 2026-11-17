"""Reproducible CPU SfM/MVS reconstruction, adaptive frame/pair selection.

This is a classical geometric backend, not a feed-forward neural 3R model.
Run via the existing drone3d environment; no installs or GPU imports required.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def frame_descriptor(gray):
    small = cv2.resize(gray, (32, 24)).astype(np.float32).ravel()
    small -= small.mean()
    return small / max(float(np.linalg.norm(small)), 1e-8)


def budget_keyframes(records, limit):
    """Bound expensive work while sampling across the full temporal extent.

    Preserve endpoints and choose a sharp, novel candidate in each interior
    time-order bin. This is a budget heuristic; SfM must verify view overlap.
    """
    if limit < 5:
        raise ValueError('A keyframe budget must be at least five')
    if len(records) <= limit:
        return list(range(len(records)))
    indices = [0, len(records)-1]
    for group in np.array_split(np.arange(1, len(records)-1), limit-2):
        indices.append(max(map(int, group), key=lambda i:
            math.log1p(records[i]['blur']) * (1. + min(records[i]['motion_pixels'], 40.) / 40.)))
    return sorted(indices)


def select_frames(candidates, images, config):
    """Cheap sharpness/exposure check plus tracked pixel displacement.

    Optical flow detects visual change; it does not certify 3D parallax.
    The SfM backend validates that later through triangulation/reprojection.
    """
    records, previous, last_selected = [], None, -100.0
    descriptors, names = [], []
    for index, path in enumerate(sorted(candidates.glob('*.jpg'))):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode {path}')
        width = config['analysis_width']
        gray = cv2.cvtColor(cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1]))), cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        exposure = float(np.mean((gray < 4) | (gray > 251)))
        stamp = (index + 0.5) / config['sample_fps']
        motion, tracks = 0.0, 0
        if previous is not None:
            corners = cv2.goodFeaturesToTrack(previous, 350, .015, 8)
            if corners is not None:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, corners, None)
                if tracked is not None:
                    good = status.ravel() == 1
                    tracks = int(good.sum())
                    if tracks:
                        motion = float(np.median(np.linalg.norm(tracked[good] - corners[good], axis=2)))
        quality_ok = blur >= config['min_blur_score'] and exposure < .8
        novel = previous is None or tracks < 20 or motion >= config['min_motion_pixels']
        keep = quality_ok and (novel or stamp - last_selected >= config['max_keyframe_gap_seconds'])
        record = {'name': path.name, 'timestamp_s': stamp, 'blur': blur,
                  'exposure_fraction': exposure, 'motion_pixels': motion,
                  'tracks': tracks, 'selected': keep,
                  'decision': 'local_reconstruction' if keep else 'skip',
                  'reason': 'visual_change' if novel else 'coverage_interval',
                  'neural_3r_called': False}
        if not quality_ok:
            record['reason'] = 'poor_quality'
        if keep:
            previous, last_selected = gray, stamp
            descriptors.append(frame_descriptor(gray))
            names.append(path.name)
        records.append(record)
    if len(names) < 5:
        raise RuntimeError(f'Only {len(names)} usable frames; insufficient for multi-view reconstruction')
    if config.get('max_keyframes'):
        proposed = [record for record in records if record['selected']]
        indices = budget_keyframes(proposed, config['max_keyframes'])
        names = [names[i] for i in indices]
        descriptors = [descriptors[i] for i in indices]
        chosen = set(names)
        for record in records:
            if record['selected'] and record['name'] not in chosen:
                record.update(selected=False, decision='skip', reason='compute_budget')
    for name in names:
        shutil.copy2(candidates / name, images / name)
    return records, names, np.asarray(descriptors)


def retrieve_pairs(names, descriptors, temporal=10, retrieval=5):
    """Bounded temporal context plus non-local visual retrieval.

    Descriptors only propose pairs. COLMAP geometrically verifies all matches.
    """
    pairs = set()
    for i, name in enumerate(names):
        for j in range(i + 1, min(len(names), i + temporal + 1)):
            pairs.add((name, names[j]))
        scores = descriptors @ descriptors[i]
        candidates = [j for j in np.argsort(-scores) if abs(int(j) - i) > temporal]
        for j in candidates[:retrieval]:
            pairs.add(tuple(sorted((name, names[int(j)]))))
    return sorted(pairs)


def read_model(model):
    """Read COLMAP text cameras/poses/point quality without pycolmap."""
    cameras = []
    for line in (model / 'cameras.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            values = line.split()
            cameras.append({'id': int(values[0]), 'model': values[1],
                            'width': int(values[2]), 'height': int(values[3]),
                            'parameters': list(map(float, values[4:]))})
    poses, want_pose = [], True
    for line in (model / 'images.txt').read_text().splitlines():
        if line.startswith('#'):
            continue
        if want_pose and line.strip():
            values = line.split()
            poses.append({'id': int(values[0]), 'quaternion_wxyz': list(map(float, values[1:5])),
                          'translation_world_to_camera': list(map(float, values[5:8])),
                          'camera_id': int(values[8]), 'image': values[9]})
            want_pose = False
        elif not want_pose:
            want_pose = True
    points = []
    for line in (model / 'points3D.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            v = line.split()
            points.append({'id': int(v[0]), 'xyz': list(map(float, v[1:4])),
                           'rgb': list(map(int, v[4:7])),
                           'error_px': float(v[7]), 'observations': (len(v) - 8) // 2})
    return cameras, poses, points


class SessionReconstructor:
    def __init__(self, session, output, config, colmap, openmvs, resume=False):
        self.session, self.output = session.resolve(), output.resolve()
        self.config, self.colmap, self.openmvs = config, str(colmap), openmvs.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs = self.output / 'logs'
        self.logs.mkdir(exist_ok=True)
        self.state_path = self.output / 'checkpoint.json'
        if self.state_path.exists() and not resume:
            raise RuntimeError('Output already has a checkpoint: use --resume or a new --output')
        self.state = json.loads(self.state_path.read_text()) if resume and self.state_path.exists() else {'stages': {}}
        if 'config' in self.state and self.state['config'] != config:
            raise RuntimeError('Resume configuration differs; use a separate output directory')
        self.state['config'] = config
        self.env = os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(config['threads']),
                        OPENBLAS_NUM_THREADS=str(config['threads']), MKL_NUM_THREADS=str(config['threads']),
                        QT_QPA_PLATFORM='offscreen')
        cv2.setNumThreads(min(config['threads'], 8))

    def run_command(self, label, command, cwd=None, timeout=None):
        log = self.logs / (label + '.log')
        print(f'[{label}] running; log: {log}', flush=True)
        with log.open('a') as stream:
            stream.write(json.dumps(list(map(str, command))) + '\n')
            stream.flush()
            environment = self.env.copy()
            if Path(str(command[0])).parent == self.openmvs:
                environment.pop('LD_LIBRARY_PATH', None)
            subprocess.run(list(map(str, command)), cwd=cwd or self.output,
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           timeout=timeout)

    def stage(self, name, action):
        if self.state['stages'].get(name, {}).get('status') == 'complete':
            print(f'[{name}] checkpoint reused', flush=True)
            return
        start = time.monotonic()
        try:
            action()
        except Exception as error:
            self.state['stages'][name] = {'status': 'failed', 'seconds': time.monotonic() - start, 'error': str(error)}
            save_json(self.state_path, self.state)
            raise
        self.state['stages'][name] = {'status': 'complete', 'seconds': time.monotonic() - start}
        save_json(self.state_path, self.state)

    def reconstruct(self):
        c, out = self.config, self.output
        help_text = subprocess.check_output([self.colmap, 'help'], stderr=subprocess.STDOUT, text=True)
        if 'without CUDA' not in help_text:
            raise RuntimeError('This command requires a verified CPU-only COLMAP build')
        links = subprocess.check_output(['ldd', str(self.openmvs / 'DensifyPointCloud')], text=True)
        if any('libcuda.' in line or 'libcudart.' in line or 'libcublas.' in line for line in links.splitlines()):
            raise RuntimeError('A CUDA-linked OpenMVS binary is not permitted in this CPU runner')
        save_json(out / 'hardware.json', {
            'cpu_count_logical': os.cpu_count(), 'cpu_affinity_count': len(os.sched_getaffinity(0)),
            'configured_threads': c['threads'], 'colmap': help_text.splitlines()[0],
            'colmap_without_cuda': True, 'openmvs_cuda_linked': False,
            'python': sys.version, 'execution': 'cpu'})
        self.state['backend'] = help_text.splitlines()[0]
        save_json(out / 'config.json', c)
        candidates, images = out / 'candidates', out / 'images'
        candidates.mkdir(exist_ok=True)
        images.mkdir(exist_ok=True)

        def decode():
            video = self.session / 'video.mp4'
            if not video.exists():
                raise FileNotFoundError(video)
            self.run_command('decode', ['ffmpeg', '-v', 'warning', '-y', '-threads', c['threads'],
                '-i', video, '-vf', f"fps={c['sample_fps']},scale={c['image_width']}:-2:flags=lanczos",
                '-q:v', '2', candidates / 'frame_%06d.jpg'])
        self.stage('decode', decode)

        def select():
            records, names, descriptors = select_frames(candidates, images, c)
            save_json(out / 'frame_decisions.json', records)
            save_json(out / 'keyframes.json', names)
            np.save(out / 'descriptors.npy', descriptors)
        self.stage('selection', select)
        names = json.loads((out / 'keyframes.json').read_text())

        def retrieval():
            desc = np.load(out / 'descriptors.npy')
            pairs = retrieve_pairs(names, desc, c['temporal_references'], c['retrieval_references'])
            (out / 'pairs.txt').write_text(''.join(f'{a} {b}\n' for a, b in pairs))
        self.stage('retrieval', retrieval)
        db = out / 'database.db'
        def features():
            example = cv2.imread(str(images / names[0]))
            h, w = example.shape[:2]
            self.run_command('features', [self.colmap, 'feature_extractor', '--database_path', db,
                '--image_path', images, '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'SIMPLE_RADIAL',
                '--ImageReader.camera_params', f"{c['focal_ratio'] * w},{w / 2},{h / 2},0",
                '--SiftExtraction.use_gpu', '0', '--SiftExtraction.num_threads', c['threads'],
                '--SiftExtraction.max_num_features', c['max_features']])
        self.stage('features', features)
        self.stage('matching', lambda: self.run_command('matching', [self.colmap, 'matches_importer',
            '--database_path', db, '--match_list_path', out / 'pairs.txt', '--match_type', 'pairs',
            '--SiftMatching.use_gpu', '0', '--SiftMatching.num_threads', c['threads'],
            '--SiftMatching.guided_matching', int(c.get('guided_matching', True))]))

        def mapping():
            choices = []
            for attempt, fixed in enumerate((True, False)):
                sparse = out / f'sparse_attempt_{attempt}'
                sparse.mkdir(exist_ok=True)
                self.run_command(f'mapping_{attempt}', [self.colmap, 'mapper', '--database_path', db,
                    '--image_path', images, '--output_path', sparse, '--Mapper.num_threads', c['threads'],
                    '--Mapper.ba_refine_focal_length', int(not fixed),
                    '--Mapper.ba_refine_extra_params', int(not fixed),
                    '--Mapper.min_focal_length_ratio', '.4', '--Mapper.max_focal_length_ratio', '1.5',
                    '--Mapper.init_min_tri_angle', '4'])
                for model in sorted(sparse.iterdir()):
                    if not (model / 'images.bin').exists():
                        continue
                    text = out / f'model_{attempt}_{model.name}'
                    text.mkdir(exist_ok=True)
                    self.run_command(f'convert_{attempt}_{model.name}', [self.colmap, 'model_converter',
                        '--input_path', model, '--output_path', text, '--output_type', 'TXT'])
                    cameras, poses, points = read_model(text)
                    err = float(np.median([p['error_px'] for p in points])) if points else float('inf')
                    choices.append((len(poses), -err, len(points), model, text))
                if choices and max(x[0] for x in choices) >= max(c['min_registered_images'], .85 * len(names)):
                    break
            if not choices:
                raise RuntimeError('SfM found no valid reconstruction')
            registered, neg_err, npoints, model, text = max(choices, key=lambda x: x[:3])
            cameras, poses, points = read_model(text)
            save_json(out / 'camera_poses.json', {'convention': 'COLMAP world-to-camera; arbitrary scale',
                                                  'cameras': cameras, 'poses': poses})
            confidences = [{'point_id': p['id'], 'xyz': p['xyz'], 'observations': p['observations'],
                'error_px': p['error_px'], 'confidence': min(1., p['observations'] / 4.) * math.exp(-p['error_px'] / 2.)}
                for p in points]
            save_json(out / 'confidence_map.json', confidences)
            save_json(out / 'uncertainty_map.json', [{'point_id': p['point_id'], 'xyz': p['xyz'],
                'uncertainty': 1 - p['confidence']} for p in confidences])
            metrics = {'selected_frames': len(names), 'registered_frames': registered,
                'registration_fraction': registered / len(names), 'sparse_points': npoints,
                'median_reprojection_error_px': -neg_err, 'scale': 'arbitrary',
                'ground_truth_available': False, 'neural_3r_calls': 0,
                'attempts': [{'registered': a, 'median_error_px': -b, 'points': d, 'model': str(e)}
                              for a,b,d,e,_ in choices]}
            save_json(out / 'reconstruction_metrics.json', metrics)
            if (registered < c['min_registered_images'] or npoints < c['min_sparse_points']
                    or registered / len(names) < c.get('min_registration_fraction', 0.0)):
                raise RuntimeError(f'SfM validation failed: {registered} cameras, {npoints} points')
            save_json(out / 'best_model.json', {'binary': str(model), 'text': str(text)})
        self.stage('mapping', mapping)

        dense = out / 'dense'
        dense.mkdir(exist_ok=True)
        best = json.loads((out / 'best_model.json').read_text())
        self.stage('undistortion', lambda: self.run_command('undistortion', [self.colmap, 'image_undistorter',
            '--image_path', images, '--input_path', best['binary'], '--output_path', dense, '--output_type', 'COLMAP']))
        def pinhole_compatibility():
            sparse = dense / 'sparse'
            text = dense / 'pinhole_text'
            text.mkdir(exist_ok=True)
            self.run_command('pinhole_to_text', [self.colmap, 'model_converter',
                '--input_path', sparse, '--output_path', text, '--output_type', 'TXT'])
            lines = []
            for line in (text / 'cameras.txt').read_text().splitlines():
                values = line.split()
                if len(values) >= 7 and values[1] in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL'):
                    if values[1] == 'SIMPLE_RADIAL' and abs(float(values[7])) > 1e-12:
                        raise RuntimeError('Undistorted camera retains nonzero radial distortion')
                    values = values[:1] + ['PINHOLE'] + values[2:4] + [values[4], values[4]] + values[5:7]
                    line = ' '.join(values)
                lines.append(line)
            (text / 'cameras.txt').write_text('\n'.join(lines) + '\n')
            self.run_command('pinhole_to_binary', [self.colmap, 'model_converter',
                '--input_path', text, '--output_path', sparse, '--output_type', 'BIN'])
        self.stage('pinhole_compatibility_v2', pinhole_compatibility)
        self.stage('mvs_import', lambda: self.run_command('mvs_import', [self.openmvs / 'InterfaceCOLMAP',
            '-i', '.', '-o', 'scene.mvs'], dense))

        def densify():
            self.run_command('densify', [self.openmvs / 'DensifyPointCloud', '-i', 'scene.mvs', '-o', 'scene_dense.mvs',
                '--max-threads', c['threads'], '--resolution-level', c['dense_resolution_level'],
                '--min-resolution', c['dense_min_resolution'], '--number-views', c['dense_views'],
                '--number-views-fuse', '3', '--estimate-roi', '0', '--crop-to-roi', '0', '--tower-mode', '0'], dense)
            from .open3d_cpu import cpu_bindings
            o3d = cpu_bindings()
            cloud = o3d.io.read_point_cloud(str(dense / 'scene_dense.ply'))
            if len(cloud.points) < c['min_dense_points']:
                raise RuntimeError(f'Dense validation failed: {len(cloud.points)} points')
            shutil.copy2(dense / 'scene_dense.ply', out / 'global_point_cloud.ply')
            m = json.loads((out / 'reconstruction_metrics.json').read_text())
            m['dense_points'] = len(cloud.points)
            save_json(out / 'reconstruction_metrics.json', m)
        self.stage('densify', densify)
        self.stage('mesh', lambda: self.run_command('mesh', [self.openmvs / 'ReconstructMesh', '-i', 'scene_dense.mvs',
            '-o', 'scene_mesh.mvs', '--max-threads', c['threads']], dense))
        def texture():
            import trimesh
            mesh = trimesh.load(dense / 'scene_mesh.ply', force='mesh', process=False)
            if len(mesh.faces) < c['min_mesh_faces']:
                raise RuntimeError(f'Mesh validation failed: {len(mesh.faces)} faces')
            shutil.copy2(dense / 'scene_mesh.ply', out / 'global_mesh.ply')
            self.run_command('texture_fullres', [self.openmvs / 'TextureMesh', '-i', 'scene.mvs', '-m', 'scene_mesh.ply',
                '-o', 'scene_textured_clean.obj', '--export-type', 'obj', '--max-threads', c['threads'],
                '--resolution-level', '0', '--global-seam-leveling', '0', '--local-seam-leveling', '0'], dense)
        self.stage('texture_fullres', texture)

        def export():
            import trimesh
            scene = trimesh.load(dense / 'scene_textured_clean.obj', force='scene', process=False)
            vertices = sum(len(g.vertices) for g in scene.geometry.values())
            faces = sum(len(g.faces) for g in scene.geometry.values())
            if faces < c['min_mesh_faces']:
                raise RuntimeError('Textured export contains insufficient faces')
            # Center without claiming a georeference or metric scale.
            transform = np.eye(4)
            transform[:3, 3] = -scene.bounds.mean(axis=0)
            scene.apply_transform(transform)
            glb = out / 'reconstruction.glb'
            glb.write_bytes(scene.export(file_type='glb'))
            restored = trimesh.load(glb, force='scene', process=False)
            if sum(len(g.faces) for g in restored.geometry.values()) != faces:
                raise RuntimeError('GLB round-trip changed triangle count')
            if not all(np.isfinite(g.vertices).all() for g in restored.geometry.values()):
                raise RuntimeError('GLB has non-finite geometry')
            m = json.loads((out / 'reconstruction_metrics.json').read_text())
            m.update(vertices=vertices, mesh_faces=faces, glb_bytes=glb.stat().st_size,
                     export_transform=transform.tolist(), glb_roundtrip_validated=True)
            save_json(out / 'reconstruction_metrics.json', m)
        self.stage('export_fullres', export)
        self.state['resource_measurement'] = {
            'max_child_peak_rss_kib_current_invocation': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            'runner_peak_rss_kib_current_invocation': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope': 'Linux process high-water RSS; resuming measures only processes of this invocation'}
        save_json(out / 'runtime_metrics.json', self.state)
        print(f"Reconstruction complete: {out / 'reconstruction.glb'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--profile', choices=['balanced', 'fast'], default='balanced')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--colmap', type=Path, default=Path('/home/vishnu/miniconda3/envs/drone3d/bin/colmap'))
    parser.add_argument('--openmvs', type=Path, default=Path('/home/vishnu/eswar/apps/openmvs/bin/OpenMVS'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config_path = args.config or Path(__file__).resolve().parent.parent / 'configs' / f'cpu_{args.profile}.json'
    c = json.loads(config_path.read_text())
    c['threads'] = min(c['threads'], len(os.sched_getaffinity(0)))
    default_name = 'reconstruction_adaptive_cpu_fast' if c['profile'] == 'fast' else 'reconstruction_adaptive_cpu'
    pipeline = SessionReconstructor(args.session, args.output or args.session / default_name,
                                    c, args.colmap, args.openmvs, args.resume)
    pipeline.reconstruct()


if __name__ == '__main__':
    main()
