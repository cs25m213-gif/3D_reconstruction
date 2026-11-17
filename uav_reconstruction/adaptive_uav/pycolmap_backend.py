"""CPU PyCOLMAP adapter for the incremental online sparse mapper.

The optional dependency is imported only when this backend is selected. Dense
OpenMVS reconstruction still requires its external executables.
"""
from pathlib import Path
import json
import time

import cv2
import numpy as np

from .session_reconstruct import read_model, save_json


def _camera_parameters(config, width, height):
    """Build COLMAP camera parameters in the saved image's pixel coordinates."""
    intrinsics = config.get('camera_intrinsics')
    shared_focal = config.get('camera_model_policy', 'auto') == 'shared-focal'
    if not intrinsics:
        focal = float(config['focal_ratio']) * width
        if not np.isfinite(focal) or focal <= 0:
            raise ValueError('focal_ratio must produce a positive finite focal length')
        if shared_focal:
            return 'SIMPLE_PINHOLE', [focal, width / 2, height / 2]
        return 'SIMPLE_RADIAL', [focal, width / 2, height / 2, 0.]
    source_width = float(intrinsics.get('width', width))
    source_height = float(intrinsics.get('height', height))
    if not np.isfinite([source_width, source_height]).all() or min(source_width, source_height) <= 0:
        raise ValueError('camera_intrinsics dimensions must be positive and finite')
    sx, sy = width / source_width, height / source_height
    values = [float(intrinsics['fx']) * sx, float(intrinsics['fy']) * sy,
              float(intrinsics['cx']) * sx, float(intrinsics['cy']) * sy]
    if not np.isfinite(values).all() or min(values[:2]) <= 0:
        raise ValueError('camera_intrinsics must contain finite values and positive focal lengths')
    if not (0 <= values[2] <= width and 0 <= values[3] <= height):
        raise ValueError('camera_intrinsics principal point is outside the image')
    distortion = list(intrinsics.get('distortion') or [])
    if len(distortion) > 8 or not np.isfinite(distortion).all():
        raise ValueError('distortion must contain at most eight finite OpenCV coefficients')
    if shared_focal:
        if any(distortion) or abs(values[0] / values[1] - 1) > .01:
            raise ValueError('shared-focal requires near-equal focal lengths and no specified distortion')
        return 'SIMPLE_PINHOLE', [(values[0] + values[1]) / 2, values[2], values[3]]
    if not distortion or not any(distortion):
        return 'PINHOLE', values
    if len(distortion) <= 4:
        return 'OPENCV', values + distortion + [0.] * (4 - len(distortion))
    # FULL_OPENCV uses k1,k2,p1,p2,k3,k4,k5,k6, matching OpenCV's ordering.
    return 'FULL_OPENCV', values + distortion + [0.] * (8 - len(distortion))


def update_map_pycolmap(runner, event_stream):
    """Update ``runner.state['online']`` and publish only a validated candidate.

    Feature/database progress survives a rejected map. Accepted model files are
    immutable; subsequent mapping loads their directory into a new block.
    Cancellation is cooperative, so timeout may overrun an ongoing solver step.
    """
    import pycolmap
    from .online import model_is_valid

    c, out = runner.config, runner.output
    online = runner.state['online']
    index = online['updates']
    online['updates'] += 1
    block = out / 'blocks' / f'{index:04d}'
    block.mkdir(parents=True, exist_ok=True)
    names = online['keyframes']
    db = out / 'database.db'
    started = time.monotonic()
    command_times = {}
    record = {'event': 'map_update', 'update_index': index,
              'source_timestamp_s': online['last_timestamp_s'],
              'seen_keyframes': len(names), 'referenced_future_frames': False,
              'accepted': False, 'reason': 'no model produced', 'backend': 'pycolmap'}

    def command(label, action, settings):
        tick = time.monotonic()
        log = runner.logs / f'online_{index:04d}_{label}.log'
        device_label = ('GPU' if c.get('sfm_use_gpu') and label in ('features', 'matching') else
                        'CUDA-enabled BA + CPU registration/local solves' if label == 'mapping' and record.get('bundle_adjustment', {}).get('gpu_bundle_adjustment') else 'CPU')
        print(f'[online_{index:04d}_{label}] PyCOLMAP {device_label}; log: {log}', flush=True)
        try:
            with log.open('a') as stream:
                stream.write(json.dumps({'backend': 'pycolmap', 'version': pycolmap.__version__,
                                         'operation': label, 'settings': settings}, default=str) + '\n')
            return action()
        except Exception as error:
            with log.open('a') as stream:
                stream.write(json.dumps({'error': str(error), 'type': type(error).__name__}) + '\n')
            raise
        finally:
            command_times[label] = time.monotonic() - tick

    try:
        pending = names[online['extracted_count']:]
        if pending:
            example = cv2.imread(str(out / 'images' / names[0]))
            if example is None:
                raise ValueError(f'Cannot read first keyframe {names[0]}')
            height, width = example.shape[:2]
            reader = pycolmap.ImageReaderOptions()
            existing = []
            if db.exists():
                with pycolmap.Database.open(db) as database:
                    existing = database.read_all_cameras()
            if existing:
                camera = min(existing, key=lambda item: item.camera_id)
                if camera.width != width or camera.height != height:
                    raise ValueError('Saved image dimensions differ from the shared database camera')
                reader.existing_camera_id = camera.camera_id
            else:
                reader.camera_model, params = _camera_parameters(c, width, height)
                reader.camera_params = ','.join(map(str, params))
            extraction = pycolmap.FeatureExtractionOptions()
            extraction.num_threads = int(c['threads'])
            use_gpu = bool(c.get('sfm_use_gpu', False))
            if use_gpu and not pycolmap.has_cuda:
                raise RuntimeError('sfm_use_gpu requires CUDA PyCOLMAP')
            extraction.use_gpu = use_gpu
            extraction.sift.max_num_features = int(c['max_features'])
            command('features', lambda: pycolmap.extract_features(
                db, out / 'images', image_names=pending, camera_mode=pycolmap.CameraMode.SINGLE,
                reader_options=reader, extraction_options=extraction,
                device=pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu),
                {'images': pending, 'reader': reader.todict(), 'extraction': extraction.todict()})
            online['extracted_count'] = len(names)
            runner.checkpoint()

        pairs = online['pairs'][online['matched_count']:]
        if pairs:
            match_list = block / 'pairs.txt'
            match_list.write_text(''.join(f'{a} {b}\n' for a, b in pairs))
            matching = pycolmap.FeatureMatchingOptions()
            matching.num_threads = int(c['threads'])
            use_gpu = bool(c.get('sfm_use_gpu', False))
            matching.use_gpu = use_gpu
            matching.guided_matching = bool(c.get('guided_matching', False))
            pairing = pycolmap.ImportedPairingOptions()
            pairing.match_list_path = match_list
            command('matching', lambda: pycolmap.match_image_pairs(
                db, matching_options=matching, pairing_options=pairing,
                device=pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu),
                {'pairs': len(pairs), 'matching': matching.todict(), 'pair_file': str(match_list)})
            online['matched_count'] = len(online['pairs'])
            runner.checkpoint()

        model_dir = block / 'sparse'
        model_dir.mkdir(exist_ok=True)
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = int(c['threads'])
        options.mapper.num_threads = int(c['threads'])
        options.multiple_models = not bool(online.get('best_model'))
        options.init_num_trials = 30
        options.min_model_size = 5
        options.mapper.init_min_tri_angle = float(c.get('init_min_tri_angle_deg', 4.))
        refine_intrinsics = bool(c.get('refine_intrinsics', False))
        options.ba_refine_focal_length = refine_intrinsics
        # This adapter uses a single camera with a trivial rig. Its sensor
        # extrinsics are fixed; there is no multi-camera calibration to optimize.
        options.ba_refine_sensor_from_rig = False
        options.ba_refine_extra_params = refine_intrinsics
        options.mapper.abs_pose_refine_focal_length = refine_intrinsics
        options.mapper.abs_pose_refine_extra_params = refine_intrinsics
        from .gpu_sfm import configure_mapping_ba
        with pycolmap.Database.open(db) as database:
            camera_models = [camera.model_name for camera in database.read_all_cameras()]
        ba_policy = configure_mapping_ba(options, c, pycolmap, camera_models=camera_models)
        record['bundle_adjustment'] = ba_policy
        if ba_policy['fallback_reason']:
            print('[compute] ' + ba_policy['fallback_reason'] + '; selecting Ceres', flush=True)
        save_json(out/'compute_backend.json', {'features':'GPU' if c.get('sfm_use_gpu') else 'CPU',
            'matching':'GPU' if c.get('sfm_use_gpu') else 'CPU', 'bundle_adjustment':ba_policy})
        print(f"[compute] {ba_policy['global_backend']}: {ba_policy['solver_policy']}; "
              "camera registration on CPU", flush=True)
        guidance = {'used': False, 'reason': 'disabled or no recorded gyro'}
        if c.get('gyro_init_guidance', False) and not online.get('best_model'):
            from .gyro_initialization import choose_initial_pair
            fps = float(c.get('video_fps', 0))
            if fps > 0:
                guidance = choose_initial_pair(db, runner.session, fps, pycolmap)
                if guidance['used']:
                    selected = guidance['selected']
                    options.init_image_id1 = selected['image_id1']
                    options.init_image_id2 = selected['image_id2']
                    options.multiple_models = False
                    if c.get('freeze_guided_intrinsics', True):
                        options.ba_refine_focal_length = False
                        options.ba_refine_extra_params = False
                        options.mapper.abs_pose_refine_focal_length = False
                        options.mapper.abs_pose_refine_extra_params = False
                    options.ba_local_max_num_iterations = 12
                    options.ba_global_max_num_iterations = 25
                    options.ba_global_max_refinements = 2
                    print(f"[initialization] gyro-consistent pair {selected['images']}; "
                          f"mapping one connected model with {'fixed' if c.get('freeze_guided_intrinsics',True) else 'refinable'} intrinsics", flush=True)
        record['initialization_guidance'] = guidance
        save_json(out/'initialization_guidance.json', guidance)
        from .camera_coverage import mapping_timeout
        timeout = mapping_timeout(c, guidance['used'])
        token = pycolmap.CancellationToken()
        if timeout > 0:
            options.max_runtime_seconds = max(1, int(np.ceil(timeout)))
        # The mapper's own budget returns its usable partial model. An external
        # cancellation timer used to discard that saved model as an exception.
        models = command('mapping', lambda: pycolmap.incremental_mapping(
                db, out / 'images', model_dir, options=options,
                input_path=online.get('best_model') or '', cancellation_token=token),
                {'input_model': online.get('best_model'), 'options': options.todict(),
                 'timeout_seconds': timeout})
        if token.is_cancelled:
            raise TimeoutError(f'Mapping exceeded {timeout:g}s; candidate was not published')

        # Prefer the strongest valid connected model during bootstrap.
        candidates = sorted(models.items(), key=lambda item:
                            (item[1].num_reg_images(), item[1].num_points3D()), reverse=True)
        for model_id, reconstruction in candidates:
            binary = model_dir / str(model_id)
            binary.mkdir(exist_ok=True)
            reconstruction.write(binary)
            text = block / f'text_{model_id}'
            text.mkdir(exist_ok=True)
            command('model_conversion', lambda: reconstruction.write_text(text), {'output': str(text)})
            cameras, poses, points = read_model(text)
            valid, reason = model_is_valid(online.get('registered_names', []), poses, points, c)
            record.update(reason=reason, registered_frames=len(poses), sparse_points=len(points),
                          median_reprojection_error_px=float(np.median([p['error_px'] for p in points]))
                          if points else None)
            if not valid:
                continue
            cloud = block / 'sparse_map.ply'
            command('pointcloud', lambda: reconstruction.export_PLY(cloud), {'output': str(cloud)})
            save_json(out / 'camera_poses.json', {
                'convention': 'COLMAP world-to-camera; arbitrary scale', 'cameras': cameras, 'poses': poses})
            save_json(out / 'best_model.json', {'binary': str(binary), 'text': str(text)})
            save_json(out / 'current_map.json', {'update_index': index, 'pointcloud': str(cloud),
                'binary_model': str(binary), 'source_timestamp_s': online['last_timestamp_s']})
            online.update(best_model=str(binary), best_text=str(text),
                          registered_names=[p['image'] for p in poses], accepted_at_count=len(names))
            record['accepted'] = True
            print(f"[online map {index}] {len(poses)} cameras, {len(points)} points at source "
                  f"{online['last_timestamp_s']:.2f}s (PyCOLMAP {ba_policy['global_backend']})", flush=True)
            break
    except Exception as error:
        record['reason'] = f'PyCOLMAP update failed: {type(error).__name__}: {error}'
    finally:
        record.update(seconds=time.monotonic() - started, command_seconds=command_times,
                      elapsed_invocation_seconds=time.monotonic() - runner.invocation_started,
                      logged_unix_seconds=time.time())
        event_stream.write(json.dumps(record) + '\n')
        event_stream.flush()
        online['map_updates'].append(record)
        runner.checkpoint()
    return record
