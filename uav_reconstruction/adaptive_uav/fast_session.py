"""Buffered saved-session trial with GPU SfM and bounded dense reconstruction."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import pycolmap

from .adaptive_sfm import (profile_video, choose_settings, match_graph,
                           quality_passes, recovery_settings)


def camera_initialization(metadata, width, height):
    """Record ambiguity instead of declaring sensor intrinsics video-calibrated."""
    values = metadata.get('camera_intrinsics', {}).get('lens_intrinsic_calibration', [])
    sensor = metadata.get('camera_intrinsics', {}).get('pixel_array_size', [])
    focal = max(width, height) * .8
    reason = 'generic focal initialization; optimized by bundle adjustment'
    if len(values) >= 4 and len(sensor) >= 2 and values[0] > 0:
        # A centered principal point identifies the calibration reference size.
        # Recording crop/stabilization metadata is absent, so this is only a seed.
        reference_width = 2 * values[2] if values[2] > 0 else sensor[0]
        focal = float(values[0]) * max(width, height) / reference_width
        reason = 'metadata focal seed; missing recording crop and video start calibration'
    return {'initial_focal_px': focal, 'focal_ratio': focal / width,
            'calibration_exact': False, 'reason': reason, 'metadata_intrinsics': values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dense-views', type=int, default=96)
    p.add_argument('--dense-size', type=int, default=960)
    p.add_argument('--fusion-min-pixels', type=int, choices=range(3, 11), default=3,
                   help='Minimum agreeing geometric depth observations; use 5 for stricter filtering')
    p.add_argument('--rotation', choices=('cw', 'ccw', 'none'), default='cw')
    p.add_argument('--hybrid', action='store_true',
                   help='Use PatchMatch depth then hybrid TSDF; skip OpenMVS meshing')
    p.add_argument('--multiview', action='store_true',
                   help='Use pose-conditioned DA3 depth and export supported and completed meshes')
    p.add_argument('--automatic', action='store_true',
                   help='Use spatial selection, checked depth normals, and automatic quality reporting')
    args = p.parse_args()
    if args.hybrid and args.multiview:
        p.error('Choose either --hybrid or --multiview')
    session, output = args.session.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'run_report.json'
    previous_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    previous_elapsed = float(previous_report.get('elapsed_seconds', 0))
    started = time.monotonic()
    metadata = json.loads((session / 'metadata.json').read_text())
    cap = cv2.VideoCapture(str(session / 'video.mp4'))
    vw, vh, fps, count = (cap.get(k) for k in (3, 4, 5, 7))
    cap.release()
    if min(vw, vh, fps, count) <= 0:
        raise RuntimeError('Invalid input video')
    effective_width, effective_height = (vh, vw) if args.rotation in ('cw', 'ccw') else (vw, vh)
    width = round(1280 * effective_width / max(effective_width, effective_height))
    height = round(width * effective_height / effective_width)
    calibration = camera_initialization(metadata, width, height)
    config = json.loads((Path(__file__).resolve().parent.parent / 'configs/cpu_online.json').read_text())
    config.update(image_width=width, analysis_width=360, camera_intrinsics=None,
                  focal_ratio=calibration['focal_ratio'], calibration_exact=False,
                  refine_intrinsics=True, sfm_use_gpu=True, threads=8, max_features=5000,
                  min_blur_score=45, min_motion_pixels=18, max_keyframe_gap_seconds=1.5,
                  temporal_references=12, retrieval_references=4, mapping_timeout_seconds=900,
                  bootstrap_keyframes=10000, update_keyframes=10000)
    profile = None
    if args.automatic:
        print('Profiling video for adaptive sparse settings', flush=True)
        profile = profile_video(session / 'video.mp4', config['analysis_width'], args.rotation)
        config.update(choose_settings(profile))
        (output / 'video_profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    report = {'status': 'sparse_running', 'source': str(session), 'video_seconds': count / fps,
              'camera_initialization': calibration, 'dense_reference_limit': args.dense_views,
              'dense_source_limit': 6, 'dense_iterations': 3,
              'fusion_min_pixels': args.fusion_min_pixels,
              'imu_log_available': (session / 'sensors.csv').is_file(),
              'imu_pose_constraints_used': False,
              'imu_note': 'Video PTS has no recorded monotonic start epoch; sensor timing cannot be assumed exact.'}
    def save(status):
        report['status'] = status
        report['last_attempt_seconds'] = time.monotonic() - started
        report['elapsed_seconds'] = previous_elapsed + report['last_attempt_seconds']
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(status, flush=True)

    if not pycolmap.has_cuda:
        raise RuntimeError('Launch with the isolated CUDA PyCOLMAP package on PYTHONPATH')
    project = Path(__file__).resolve().parent.parent
    online = output / 'online'
    try:
        save('sparse_running')
        attempts = []
        accepted = False
        for attempt in range(3 if args.automatic else 1):
            online = output / ('online' if attempt == 0 else f'online_recovery_{attempt}')
            attempt_config = output / f'sparse_config_{attempt}.json'
            attempt_config.write_text(json.dumps(config, indent=2) + '\n')
            process_error = None
            attempt_started = time.monotonic()
            if not (online / 'current_map.json').exists():
                result = subprocess.run([sys.executable, '-u', '-m', 'adaptive_uav.online', str(session),
                                        '--config', str(attempt_config), '--output', str(online),
                                        '--sfm-backend', 'pycolmap', '--sparse-only',
                                        '--quality-buffer-seconds', str(min(.75, config['max_keyframe_gap_seconds']/2)) if args.automatic else '.75',
                                        '--rotation', args.rotation],
                                       cwd=project)
                if result.returncode:
                    process_error = f'Sparse process exited with code {result.returncode}'
            metrics_path = online / 'reconstruction_metrics.json'
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            graph = match_graph(online / 'database.db') if args.automatic else {'available': False}
            passed = (quality_passes(metrics, graph) if args.automatic else
                      metrics.get('registration_fraction', 0) >= .8 and
                      metrics.get('median_reprojection_error_px', float('inf')) <= 2)
            passed = passed and process_error is None and (online / 'current_map.json').is_file()
            attempts.append({'attempt': attempt, 'config': dict(config), 'metrics': metrics,
                             'graph': graph, 'passed': passed, 'error': process_error,
                             'runtime_seconds': time.monotonic()-attempt_started})
            report['adaptive_sfm'] = {'profile': profile, 'attempts': attempts,
                                      'selected_output': str(online) if passed else None}
            (output / 'adaptive_sfm_history.json').write_text(json.dumps(report['adaptive_sfm'], indent=2) + '\n')
            report['sparse_metrics'] = metrics
            save('sparse_running')
            if passed:
                accepted = True
                config_path.write_text(json.dumps(config, indent=2) + '\n')
                break
            next_config = recovery_settings(config, metrics)
            if next_config is None or not args.automatic:
                break
            config = next_config
        if not accepted:
            raise RuntimeError('Sparse quality gate failed after bounded recovery; see adaptive_sfm_history.json')
        report['sparse_output'] = str(online)
        if online.name != 'online':
            page_path = output / 'index.html'
            if page_path.exists():
                page_path.write_text(page_path.read_text().replace('online/viewer.html', online.name + '/viewer.html'))
        model = Path(json.loads((online / 'current_map.json').read_text())['binary_model'])
        if (session / 'sensors.csv').exists():
            from .sensor_audit import audit_gyro
            gyro_audit = audit_gyro(session, model, fps)
            report['gyro_audit'] = gyro_audit
            (output / 'gyro_audit.json').write_text(json.dumps(gyro_audit, indent=2) + '\n')
            if gyro_audit.get('usable') and not gyro_audit.get('passed'):
                raise RuntimeError('Camera rotation disagrees with recorded gyro; refusing dense reconstruction')
        reconstruction = pycolmap.Reconstruction(model)
        report['optimized_camera'] = [{'model': c.model_name, 'width': c.width, 'height': c.height,
                                      'params': c.params.tolist()}
                                     for c in reconstruction.cameras.values()]
        # Lens optimization cannot prove geometry, but implausible lenses should fail early.
        from .combined_reconstruct import plausible_focal_lengths
        for camera in reconstruction.cameras.values():
            if not plausible_focal_lengths(camera):
                raise RuntimeError('Optimized focal length is implausible; video calibration is required')
        save('dense_preparation')
        dense = output / 'dense'
        dense.mkdir(exist_ok=True)
        if not (dense / 'sparse/cameras.bin').exists():
            pycolmap.undistort_images(dense, model, online / 'images', num_threads=4,
                                     undistort_options=pycolmap.UndistortCameraOptions(
                                         max_image_size=args.dense_size))
        save('dense_running')
        import os
        tools = Path(os.environ.get('GEOSTREAM_TOOLS', str(project.parent.parent / 'tools')))
        subprocess.run([sys.executable, '-u', '-m', 'adaptive_uav.dense_cuda',
                        '--workspace', str(dense), '--openmvs', str(tools / 'OpenMVS_2.4.0'),
                        '--runtime', str(tools / 'noble_runtime'),
                        '--three-assets', str(tools / 'three_0.169.0'),
                        '--max-image-size', str(args.dense_size), '--max-views', str(args.dense_views),
                        '--spatial-plan',
                        '--source-views', '6', '--iterations', '3', '--threads', '4',
                        '--fusion-min-pixels', str(args.fusion_min_pixels), '--skip-texture'] +
                       (['--depth-only'] if args.hybrid or args.multiview else
                        ['--stable-normals', '--quality-report'] if args.automatic else []),
                       cwd=project, check=True)
        report['dense_metrics'] = json.loads((dense / 'dense_metrics.json').read_text())
        report['geometry_accuracy_validated'] = False
        report['quality_status'] = report['dense_metrics'].get('quality_status', 'unverified')
        if args.multiview:
            save('multiview_running')
            completed = output/'multiview'
            subprocess.run([sys.executable, '-u', '-m', 'adaptive_uav.multiview_dense',
                            '--workspace', str(dense), '--output', str(completed),
                            '--three-assets', str(tools/'three_0.169.0')], cwd=project, check=True)
            result = json.loads((completed/'multiview_report.json').read_text())
            report['multiview'] = {key: result[key] for key in ('coverage_mean', 'surface_audit', 'quality_status')}
            report['quality_status'] = result['quality_status']
        elif args.hybrid:
            save('hybrid_running')
            subprocess.run([sys.executable, '-u', '-m', 'adaptive_uav.hybrid_dense',
                            '--workspace', str(dense), '--stage', 'all'],
                           cwd=project, check=True)
            hybrid = json.loads((dense / 'hybrid_dense_report.json').read_text())
            report['hybrid'] = {key: hybrid[key] for key in (
                'coverage_mean', 'coverage_measured_plus_supported', 'coverage_with_inferred',
                'coverage_target', 'quality_status', 'frames_alignment_failed', 'capture_diagnostics')}
            report['quality_status'] = hybrid['quality_status']
            for name, source in (('scene_mesh.ply', 'scene_hybrid_mesh.ply'),
                                 ('scene_mesh_colored.ply', 'scene_hybrid_mesh.ply'),
                                 ('scene_colored.glb', 'scene_hybrid_mesh.glb')):
                link = dense / name
                if not link.exists():
                    link.symlink_to(source)
            report['dense_metrics']['pipeline'] = 'COLMAP PatchMatch + aligned prior + CPU TSDF'
            report['dense_metrics']['dense_points'] = 0
            report['dense_metrics']['quality_warning'] = (
                'Experimental hybrid; prior-consistent and inferred depth are not measured geometry.')
            (dense / 'dense_metrics.json').write_text(json.dumps(report['dense_metrics'], indent=2) + '\n')
            from .mesh_viewer import write_mesh_viewer
            write_mesh_viewer(dense, tools / 'three_0.169.0')
        save('complete')
    except Exception as error:
        report['error'] = str(error)
        save('failed')
        raise


if __name__ == '__main__':
    main()
