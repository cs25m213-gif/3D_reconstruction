"""Adaptive camera selection, measured depth, learned depth and visibility fusion.

Uses DA3, not the pretrained VisFusion network. Retains partial-capture status
and arbitrary visual units. Original sparse and dense artifacts are preserved.
"""
import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import pycolmap

from .camera_coverage import capture_coverage, result_status
from .sensor_audit import audit_gyro


def plausible_focal_lengths(camera):
    """Check every focal parameter for both single- and two-focal models."""
    params = np.asarray(camera.params)
    indices = camera.focal_length_idxs()
    return bool(indices) and all(
        np.isfinite(params[i]) and .5 <= params[i] / max(camera.width, camera.height) <= 2
        for i in indices)


def save_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def frame_index(name):
    return int(Path(name).stem.split('_')[-1])


def fusion_rank(report):
    """Prioritize consistency over coverage; a complete-looking bad mesh loses."""
    status = report.get('quality_status')
    consistent = status in ('incomplete', 'high_rendered_coverage_unvalidated')
    agreement = report.get('mean_measured_depth_agreement_under_5pct')
    coverage = report.get('surface_audit', {}).get('mean_ray_hit', {}).get('completed', 0.)
    return consistent, float(agreement or 0.), float(coverage or 0.)


def run_parallel(commands, cwd, poll_seconds=1.):
    """Run independent stages together; one failure stops the others."""
    processes = [subprocess.Popen(c, cwd=cwd) for c in commands]
    try:
        while True:
            # Poll every process each pass; any() would stop at the first running one.
            codes = [process.poll() for process in processes]
            for process, code in zip(processes, codes):
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
            if None not in codes:
                return
            time.sleep(poll_seconds)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


def learned_depth_options(workspace, assets, *, fast, coverage):
    """Shared multiview_dense arguments for the selected preset and coverage policy."""
    options = ['--workspace', workspace, '--three-assets', assets, '--threads', 8]
    if coverage == 'full':
        # Cameras without stereo maps were the main surface-coverage bottleneck.
        # Their learned depth is anchored by triangulated tracks, kept separate
        # from stereo measurements and still passes the visibility gates.
        options += ['--include-unmeasured']
    if fast:
        # Subsampling all registered cameras would drop stereo-measured views.
        options += ['--resolution', 336, '--views', 3, '--max-frames', 0 if coverage == 'full' else 20,
                    '--voxel-fraction', .02, '--audit-views', 8, '--visibility-references', 8]
    return options


def select_camera_model(source, session, output, fps, *, allow_partial=False):
    """Check pose quality separately from the fraction of recovered cameras.

    The final pipeline can use a geometrically checked partial segment for
    recovery and inspectable dense output. Capture-wide success still requires
    the independent coverage gate; sparse registration is not pose accuracy.
    """
    candidates = []
    diagnostics = []
    folders = sorted(source.glob('online*'))
    for folder in folders:
        path = folder/'current_map.json'
        if not path.is_file():
            continue
        model = Path(json.loads(path.read_text())['binary_model'])
        original = pycolmap.Reconstruction(model)
        ordered = sorted((i for i in original.images.values() if i.has_pose), key=lambda i: i.name)
        if len(ordered) < 12:
            diagnostics.append({'source': str(model), 'accepted': False,
                                'registered_frames': len(ordered),
                                'failed_checks': ['minimum_camera_count']})
            continue
        camera_ok = all(plausible_focal_lengths(c) and
                        (c.model_name != 'SIMPLE_RADIAL' or abs(c.params[-1]) <= .2)
                        for c in original.cameras.values())
        if not camera_ok:
            diagnostics.append({'source': str(model), 'accepted': False,
                                'reason': 'implausible intrinsics',
                                'failed_checks': ['intrinsics']})
            continue
        frames = [frame_index(p.name) for p in (folder/'images').glob('*.jpg')]
        # Test the whole model and a single-frame startup exclusion. Every
        # tested candidate and audit stays visible; no iterative outlier pruning.
        for trim in (0, 1):
            # The untrimmed model is the one already loaded; only the trimmed one is reread.
            reconstruction = pycolmap.Reconstruction(model) if trim else original
            if trim:
                reconstruction.deregister_frame(ordered[0].frame_id)
            chosen = ordered[trim:]
            low, high = frame_index(chosen[0].name), frame_index(chosen[-1].name)
            selected = sum(low <= index <= high for index in frames)
            errors = np.fromiter((p.error for p in reconstruction.points3D.values()), float)
            error = float(np.median(errors)) if errors.size else None
            if error is not None and not np.isfinite(error):
                error = None  # Reports are strict JSON; a non-finite error fails the check below.
            fraction = len(chosen)/max(1, selected)
            candidate = output/'camera_candidates'/f'{folder.name}_trim_{trim}'
            candidate.mkdir(parents=True, exist_ok=True)
            reconstruction.write(candidate)
            gyro = (audit_gyro(session, candidate, fps) if (session/'sensors.csv').is_file()
                    else {'usable': False, 'reason': 'no sensor log'})
            geometry_failures = []
            if len(chosen) < 12:
                geometry_failures.append('minimum_camera_count')
            if error is None or not 0 <= error <= 2:
                geometry_failures.append('reprojection_error')
            if reconstruction.num_points3D() < 150:
                geometry_failures.append('minimum_sparse_points')
            if gyro.get('usable') and not gyro.get('passed', False):
                geometry_failures.append('gyro_agreement')
            geometry_passed = not geometry_failures
            interval_passed = fraction >= .8
            passed = geometry_passed and (interval_passed or allow_partial)
            row = {'source': str(model), 'model': str(candidate), 'startup_frames_excluded': trim,
                   'registered_frames': len(chosen), 'selected_in_interval': selected,
                   'interval_registration_fraction': fraction, 'full_run_selected_frames': len(frames),
                   'median_reprojection_error_px': error, 'start_seconds': low/fps, 'end_seconds': high/fps,
                   'sparse_points': reconstruction.num_points3D(),
                   'gyro': gyro, 'geometry_passed': geometry_passed,
                   'interval_coverage_passed': interval_passed,
                   'failed_checks': geometry_failures + ([] if interval_passed else ['interval_registration']),
                   'accepted': bool(passed),
                   'partial_interval_accepted': bool(passed and not interval_passed),
                   'metric_scale_validated': False}
            diagnostics.append(row)
            if passed:
                candidates.append((len(chosen), -error, row, folder/'images'))
    save_json(output/'camera_selection.json', diagnostics)
    if not candidates:
        failed = sorted({check for row in diagnostics for check in row.get('failed_checks', [])})
        detail = ', '.join(failed) if failed else 'no saved model with enough cameras'
        raise RuntimeError(f'No usable camera segment: {detail}; see {output / "camera_selection.json"}')
    _, _, row, images = max(candidates, key=lambda c: c[:2])
    return row, images


def main():
    from .runtime_cli import requested_mode
    if requested_mode(sys.argv[1:])!='legacy':
        from .runtime import main as stream_main
        return stream_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('legacy','preview','balanced','quality'),default='legacy')
    parser.add_argument('session', type=Path)
    parser.add_argument('--sparse-run', type=Path, required=True, help='Adaptive run for this same session')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=('quality', 'fast'), default='quality')
    parser.add_argument('--semantic-filter', choices=('on','off'), default='on')
    parser.add_argument('--semantic-rotation', choices=('auto','0','90','180','270'), default='auto')
    parser.add_argument('--reuse-depth', type=Path, help='Snapshot completed maps from a combined run for this video')
    parser.add_argument('--dense-views', type=int, help='Default: 96 (quality) or 20 (fast)')
    parser.add_argument('--dense-size', type=int, help='Default: 960 (quality) or 640 (fast)')
    parser.add_argument('--coverage', choices=('full', 'measured'), default='full',
                        help='full: learned depth for every registered camera, anchored by stereo or '
                             'triangulated tracks; measured: only cameras with stereo maps (previous behaviour)')
    parser.add_argument('--strict-interval', action='store_true',
                        help='Fail instead of reconstructing a geometry-checked partial camera interval')
    args = parser.parse_args()
    fast = args.preset == 'fast'
    # Presets supply defaults only; explicit values are never silently replaced.
    if args.dense_views is None:
        args.dense_views = 20 if fast else 96
    if args.dense_size is None:
        args.dense_size = 640 if fast else 960
    if args.reuse_depth is not None and not fast:
        parser.error('--reuse-depth is supported by the fast preset')
    session, source, out = args.session.resolve(), args.sparse_run.resolve(), args.output.resolve()
    if out == source or out.is_relative_to(session):
        raise ValueError('Use a separate run output to preserve inputs and sparse results')
    video = session/'video.mp4'
    manifest = json.loads((source/'pipeline_manifest.json').read_text())
    identity = manifest['inputs']['video']
    stat = video.stat()
    if Path(identity['path']).resolve() != video.resolve() or identity['size_bytes'] != stat.st_size or identity['mtime_ns'] != stat.st_mtime_ns:
        raise ValueError('Sparse run does not match the input video')
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or count <= 0:
        raise ValueError('Invalid video')
    policy = {'video': identity, 'sparse_run': str(source), 'dense_views': args.dense_views,
              'dense_size': args.dense_size, 'preset': args.preset, 'coverage': args.coverage,
              'allow_partial_interval': not args.strict_interval,
              'semantic_filter': args.semantic_filter, 'semantic_rotation': args.semantic_rotation,
              'reuse_depth': str(args.reuse_depth.resolve()) if args.reuse_depth else None, 'algorithm': 'COLMAP + DA3 + bounded global visibility',
              'code': {n: hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in
                       ('combined_reconstruct.py', 'multiview_dense.py', 'semantic_scene.py', 'mesh_viewer.py', 'dense_cuda.py', 'global_visibility.py')}}
    policy_path = out/'pipeline_manifest.json'
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError('Output policy changed; choose a fresh output directory')
    save_json(policy_path, policy)
    started = time.monotonic()
    report = {'status': 'starting', 'source': str(session), 'method': policy['algorithm'],
              'pretrained_visfusion_used': False, 'geometry_accuracy_validated': False,
              'units': 'COLMAP visual units', 'video_seconds': count/fps,
              'preset': args.preset, 'target_seconds': 180 if fast else None,
              'runtime_scope': 'cached poses and depth' if args.reuse_depth else 'cached sparse poses'}

    def save(status):
        report.update(status=status, elapsed_seconds=time.monotonic()-started)
        save_json(out/'run_report.json', report)
        print(status, flush=True)

    project = Path(__file__).resolve().parent.parent
    tools = Path(os.environ.get('GEOSTREAM_TOOLS', str(project.parent/'tools')))

    def command(module, *values):
        return [sys.executable, '-u', '-m', module, *map(str, values)]

    def run(module, *values):
        subprocess.run(command(module, *values), cwd=project, check=True)

    def snapshot(source_dir, target):
        """Promote a complete copy only, so an interrupted run never leaves partial input."""
        staging = target.with_name(target.name + '.partial')
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(source_dir, staging)
        staging.rename(target)

    try:
        ws = out/'dense'
        ws.mkdir(exist_ok=True)
        if args.reuse_depth:
            old = args.reuse_depth.resolve()
            old_policy = json.loads((old.parent/'pipeline_manifest.json').read_text())
            if old_policy['video'] != identity:
                raise ValueError('Depth input belongs to a different video')
            previous = json.loads((old.parent/'run_report.json').read_text())
            camera = previous['camera_selection']
            save('snapshot_cached_depth')
            # Freeze completed maps; do not mix partial writes or altered poses.
            if not (ws/'sparse').exists():
                snapshot(old/'sparse', ws/'sparse')
            if not (ws/'images').exists():
                (ws/'images').symlink_to(old/'images', target_is_directory=True)
            maps = ws/'stereo/depth_maps'
            if maps.is_dir() and not any(maps.iterdir()):
                maps.rmdir()  # Empty folder left by an earlier interrupted run.
            if not maps.exists():
                available = sorted((old/'stereo/depth_maps').glob('*.geometric.bin'))
                if len(available) < 8:
                    raise ValueError('At least eight completed cached depth maps are needed')
                from .hybrid_dense import read_array
                staging = maps.with_name(maps.name + '.partial')
                shutil.rmtree(staging, ignore_errors=True)
                staging.mkdir(parents=True)
                for i in np.unique(np.linspace(0, len(available)-1, min(20, len(available)), dtype=int)):
                    # Validate bytes before promoting the immutable snapshot.
                    read_array(available[i])
                    shutil.copy2(available[i], staging/available[i].name)
                staging.rename(maps)
            report['depth_maps_reused'] = len(list(maps.glob('*.geometric.bin')))
            report['depth_cache_source'] = str(old)
            report['cached_depth_names'] = [p.name for p in sorted(maps.glob('*.geometric.bin'))]
        else:
            save('camera_selection')
            camera, images = select_camera_model(source, session, out, fps,
                                                 allow_partial=not args.strict_interval)
            save('dense_preparation')
            if not (ws/'sparse/cameras.bin').is_file():
                pycolmap.undistort_images(ws, Path(camera['model']), images, num_threads=-1,
                    undistort_options=pycolmap.UndistortCameraOptions(max_image_size=args.dense_size))
            save('measured_depth_running')
            run('adaptive_uav.dense_cuda', '--workspace', ws, '--openmvs', tools/'OpenMVS_2.4.0',
                '--runtime', tools/'noble_runtime', '--three-assets', tools/'three_0.169.0',
                '--max-image-size', args.dense_size, '--max-views', args.dense_views,
                '--source-views', 3 if fast else 6, '--iterations', 1 if fast else 3,
                '--threads', 4, '--fusion-min-pixels', 3, '--spatial-plan', '--depth-only')
        report['camera_selection'] = camera
        report['capture_fraction_in_selected_interval'] = (camera['end_seconds']-camera['start_seconds'])/(count/fps)
        # Span alone overstates capture when many frames inside it failed to register.
        report['capture_coverage'] = capture_coverage(camera, count/fps)
        report['capture_status'] = 'most_of_capture' if report['capture_coverage']['passed'] else 'partial'
        base, visible = out/'multiview', out/'multiview_visibility'
        common = learned_depth_options(ws, tools/'three_0.169.0', fast=fast, coverage=args.coverage)
        if args.semantic_filter=='on':
            save('semantic_masks_running')
            semantic_dir=out/'semantics'
            run('adaptive_uav.semantic_scene','--workspace',ws,'--output',semantic_dir,'--rotation',args.semantic_rotation)
            report['semantic_filter']={'directory':str(semantic_dir),'report':str(semantic_dir/'semantic_report.json')}
            common += ['--semantic-dir',semantic_dir,'--video-fps',fps]
        if fast:
            save('learned_depth_and_fusion_running')
            run('adaptive_uav.multiview_dense', *common, '--output', base, '--global-visibility')
            folders = [base]
        else:
            # Infer depth once on the GPU; both CPU-only fusions then read the
            # shared cache concurrently instead of one after the other.
            save('learned_depth_running')
            run('adaptive_uav.multiview_dense', *common, '--output', base, '--stage', 'infer')
            save('fusion_and_visibility_comparison_running')
            visible.mkdir(exist_ok=True)
            if not (visible/'depth_cache').exists():
                (visible/'depth_cache').symlink_to(base/'depth_cache', target_is_directory=True)
            run_parallel([
                command('adaptive_uav.multiview_dense', *common, '--output', base, '--stage', 'fuse'),
                command('adaptive_uav.multiview_dense', *common, '--output', visible, '--stage', 'fuse',
                        '--global-visibility')], project)
            folders = [base, visible]
        results = [(folder, json.loads((folder/'multiview_report.json').read_text())) for folder in folders]
        best, quality = max(results, key=lambda r: fusion_rank(r[1]))
        report['comparisons'] = {folder.name: {'quality_status': q.get('quality_status'),
            'depth_agreement': q.get('mean_measured_depth_agreement_under_5pct'),
            'surface_audit': q.get('surface_audit')} for folder, q in results}
        report.update(selected_result=str(best), quality_status=quality['quality_status'],
                      result_status=result_status(report['capture_coverage'], quality['quality_status']),
                      learned_depth_frames=quality.get('frames'),
                      mesh=str(best/'scene_completed.glb'), supported_mesh=str(best/'scene_supported.glb'))
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Combined reconstruction</title>'
            f'<p>Capture: {escape(report["capture_status"])}; quality: {escape(str(quality["quality_status"]))}. '
            'Geometry and metric scale remain unvalidated.</p>'
            f'<a href="{escape(best.name)}/mesh_viewer.html">Selected mesh viewer</a> '
            '<a href="run_report.json">Run report</a>')
        report['runtime_target_met'] = (time.monotonic()-started <= 180) if fast else None
        save('complete')
    except Exception as error:
        report['error'] = str(error)
        save('failed')
        raise


if __name__ == '__main__':
    main()
