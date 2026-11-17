"""Causal, block-incremental CPU SfM with separately timed final dense meshing.

No future frames are decoded for a map update or used in reference retrieval.
This is online sparse reconstruction, not a neural 3R or real-time guarantee.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

import cv2
import numpy as np

from .session_reconstruct import SessionReconstructor, frame_descriptor, read_model, save_json
from .phone_source import RecordedVideoSource
from .imu import summarize_imu, rotation_homography, resample_imu


def past_references(descriptors, index, temporal=8, visual=4):
    if index < 0 or index >= len(descriptors):
        raise ValueError('invalid frame index')
    local = list(range(max(0, index-temporal), index))
    historical = list(range(max(0, index-temporal)))
    if historical:
        scores = np.asarray(descriptors[:index-temporal]) @ descriptors[index]
        local += [historical[int(i)] for i in np.argsort(-scores)[:visual]]
    return sorted(set(local))


def model_is_valid(previous_names, poses, points, config):
    names = {p['image'] for p in poses}
    if len(names) < config['min_registered_images'] or len(points) < config['min_sparse_points']:
        return False, 'insufficient camera/landmark support'
    if not set(previous_names).issubset(names):
        return False, 'previous registered cameras lost'
    errors = np.asarray([p['error_px'] for p in points])
    if not np.isfinite(errors).all() or np.median(errors) > config['max_median_reprojection_error_px']:
        return False, 'reprojection validation failed'
    if np.quantile(errors,.95)>config.get('max_p95_reprojection_error_px',6.):
        return False, 'reprojection outlier tail too large'
    if any('xyz' in p and not np.isfinite(p['xyz']).all() for p in points):
        return False, 'non-finite geometry'
    return True, 'accepted'


class OnlineReconstructor(SessionReconstructor):
    def checkpoint(self):
        save_json(self.state_path, self.state)

    def update_map(self, event_stream):
        if self.config.get('sfm_backend') == 'pycolmap':
            from .pycolmap_backend import update_map_pycolmap
            return update_map_pycolmap(self, event_stream)
        c, out = self.config, self.output
        online = self.state['online']
        index = online['updates']
        online['updates'] += 1
        block = out / 'blocks' / f'{index:04d}'
        block.mkdir(parents=True, exist_ok=True)
        names = online['keyframes']
        db = out / 'database.db'
        pending = names[online['extracted_count']:]
        start = time.monotonic()
        command_times = {}

        def command(label, args, timeout=None):
            began = time.monotonic()
            try:
                self.run_command(f'online_{index:04d}_{label}', args, timeout=timeout)
            finally:
                command_times[label] = time.monotonic() - began

        if pending:
            image_list = block / 'new_images.txt'
            image_list.write_text('\n'.join(pending) + '\n')
            example = cv2.imread(str(out / 'images' / names[0]))
            h,w = example.shape[:2]
            camera_id = None
            if db.exists():
                with sqlite3.connect(db) as connection:
                    row = connection.execute('SELECT camera_id FROM cameras ORDER BY camera_id LIMIT 1').fetchone()
                    if row:
                        camera_id = row[0]
            args = [self.colmap, 'feature_extractor', '--database_path', db,
                '--image_path', out / 'images', '--image_list_path', image_list,
                '--ImageReader.single_camera', '1', '--SiftExtraction.use_gpu', '0',
                '--SiftExtraction.num_threads', c['threads'], '--SiftExtraction.max_num_features', c['max_features']]
            if camera_id is None:
                from .pycolmap_backend import _camera_parameters
                camera_model, camera_params = _camera_parameters(c, w, h)
                args += ['--ImageReader.camera_model', camera_model, '--ImageReader.camera_params',
                         ','.join(map(str, camera_params))]
            else:
                args += ['--ImageReader.existing_camera_id', camera_id]
            command('features', args)
            online['extracted_count'] = len(names)
            self.checkpoint()

        pairs = online['pairs'][online['matched_count']:]
        if pairs:
            match_list = block / 'pairs.txt'
            match_list.write_text(''.join(f'{a} {b}\n' for a,b in pairs))
            command('matching', [self.colmap, 'matches_importer', '--database_path', db,
                '--match_list_path', match_list, '--match_type', 'pairs',
                '--SiftMatching.use_gpu', '0', '--SiftMatching.num_threads', c['threads'],
                '--SiftMatching.guided_matching', int(c['guided_matching'])])
            online['matched_count'] = len(online['pairs'])
            self.checkpoint()

        model_dir = block / 'sparse'
        model_dir.mkdir(exist_ok=True)
        args = [self.colmap, 'mapper', '--database_path', db, '--image_path', out/'images',
                '--output_path', model_dir, '--Mapper.num_threads', c['threads'],
                '--Mapper.multiple_models', '0' if online.get('best_model') else '1',
                '--Mapper.init_num_trials', '30', '--Mapper.min_model_size', '5',
                '--Mapper.init_min_tri_angle', '4', '--Mapper.ba_refine_focal_length', '0',
                '--Mapper.ba_refine_extra_params', '0']
        if online.get('best_model'):
            args += ['--input_path', online['best_model']]
        try:
            command('mapping', args, timeout=c.get('mapping_timeout_seconds',120))
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
            record={'event':'map_update','update_index':index,'accepted':False,
                    'reason':f'mapping failed: {type(error).__name__}',
                    'source_timestamp_s':online['last_timestamp_s'],'seen_keyframes':len(names),
                    'seconds':time.monotonic()-start,'command_seconds':command_times,
                    'referenced_future_frames':False}
            event_stream.write(json.dumps(record)+'\n')
            event_stream.flush()
            online['map_updates'].append(record)
            self.checkpoint()
            return
        models = sorted(p.parent for p in model_dir.rglob('images.bin'))
        record = {'event': 'map_update', 'update_index': index,
                  'source_timestamp_s': online['last_timestamp_s'], 'seen_keyframes': len(names),
                  'referenced_future_frames': False, 'accepted': False, 'reason': 'no model produced'}
        for binary in models:
            text = block / ('text_' + binary.name)
            text.mkdir(exist_ok=True)
            command('model_conversion', [self.colmap, 'model_converter', '--input_path', binary,
                                         '--output_path', text, '--output_type', 'TXT'])
            cameras, poses, points = read_model(text)
            valid, reason = model_is_valid(online.get('registered_names', []), poses, points, c)
            record.update(reason=reason, registered_frames=len(poses), sparse_points=len(points),
                          median_reprojection_error_px=float(np.median([p['error_px'] for p in points])) if points else None)
            if not valid:
                continue
            online.update(best_model=str(binary), best_text=str(text),
                          registered_names=[p['image'] for p in poses], accepted_at_count=len(names))
            save_json(out/'camera_poses.json', {'convention': 'COLMAP world-to-camera; arbitrary scale',
                                               'cameras': cameras, 'poses': poses})
            save_json(out/'best_model.json', {'binary': str(binary), 'text': str(text)})
            command('pointcloud', [self.colmap, 'model_converter', '--input_path', binary,
                '--output_path', block/'sparse_map.ply', '--output_type', 'PLY'])
            # Snapshot plus atomic pointer: no overwrite of a previously accepted model.
            save_json(out/'current_map.json', {'update_index': index, 'pointcloud': str(block/'sparse_map.ply'),
                'binary_model': str(binary), 'source_timestamp_s': online['last_timestamp_s']})
            record['accepted'] = True
            print(f"[online map {index}] {len(poses)} cameras, {len(points)} points at video {online['last_timestamp_s']:.2f}s", flush=True)
            break
        record.update(seconds=time.monotonic()-start, command_seconds=command_times,
                      elapsed_invocation_seconds=time.monotonic()-self.invocation_started,
                      logged_unix_seconds=time.time())
        event_stream.write(json.dumps(record)+'\n')
        event_stream.flush()
        online['map_updates'].append(record)
        self.checkpoint()

    def run_online(self, finalize=True, frame_source=None):
        c, out = self.config, self.output
        if c.get('sfm_backend') == 'pycolmap':
            import pycolmap
            if tuple(int(part) for part in pycolmap.__version__.split('.')[:2]) < (4, 2):
                raise RuntimeError('PyCOLMAP 4.2 or later is required')
        else:
            banner = subprocess.check_output([self.colmap, 'help'], stderr=subprocess.STDOUT, text=True)
            if 'without CUDA' not in banner:
                raise RuntimeError('CPU-only COLMAP is required')
        online = self.state.setdefault('online', {'keyframes': [], 'pairs': [], 'updates': 0,
            'extracted_count': 0, 'matched_count': 0, 'accepted_at_count': 0,
            'last_source_index': -1, 'last_timestamp_s': 0., 'map_updates': [], 'stream_complete': False,
            'invocations': [], 'sampled_frames': 0, 'cheap_check_seconds': 0., 'decode_seconds': 0.})
        frame_source = frame_source or RecordedVideoSource(self.session/'video.mp4', c['sample_fps'])
        signature = frame_source.signature
        if online.get('input_signature', signature) != signature:
            raise RuntimeError('Input source changed: use a new output directory')
        online['input_signature'] = signature
        if signature.get('kind') == 'phone_api':
            online['stream_complete'] = False
        self.checkpoint()
        out.joinpath('images').mkdir(exist_ok=True)
        began = time.monotonic()
        self.invocation_started = began
        descriptors = []
        previous, previous_selected_time = None, -100.
        for name in online['keyframes']:
            image = cv2.imread(str(out/'images'/name), cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(image, (c['analysis_width'], round(image.shape[0]*c['analysis_width']/image.shape[1])))
            descriptors.append(frame_descriptor(resized))
            previous = resized
        if online['keyframes']:
            previous_selected_time = online.get('last_selected_time', 0.)
        imu_history = []
        previous_sensor_time = online.get('last_selected_sensor_time')
        events = out/'online_events.jsonl'
        stream_start = time.monotonic()
        with events.open('a') as stream:
            if not online['stream_complete']:
                frames = frame_source.frames(after_index=online['last_source_index'])
                try:
                    for source_frame in frames:
                        tick = time.monotonic()
                        source_index, stamp, image = source_frame.index, source_frame.timestamp_s, source_frame.image
                        width = c['image_width']
                        image = cv2.resize(image, (width, round(image.shape[0]*width/image.shape[1])))
                        decode = source_frame.decode_seconds + time.monotonic()-tick
                        online['decode_seconds'] += decode
                        check_start = time.monotonic()
                        small = cv2.resize(image,(c['analysis_width'],round(image.shape[0]*c['analysis_width']/image.shape[1])))
                        gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
                        blur = float(cv2.Laplacian(gray,cv2.CV_64F).var())
                        exposure = float(np.mean((gray<4)|(gray>251)))
                        imu_history.extend(source_frame.imu_samples)
                        imu_history = imu_history[-10000:]
                        sensor_end = source_frame.sensor_timestamp_s
                        sensor_start = previous_sensor_time if previous_sensor_time is not None else sensor_end
                        imu_features = (summarize_imu(imu_history, sensor_start, sensor_end)
                                        if sensor_start is not None and sensor_end is not None and sensor_end > sensor_start
                                        else {'valid': False, 'rotation_valid': False, 'reason': 'no timestamped IMU window', 'sample_count': 0})
                        motion, raw_motion, tracked_count = 0., 0., 0
                        visual_rotation = {'reason': 'imu_rotation_available' if source_frame.sensor_timestamp_s is not None else 'not_attempted'}
                        calibration = c.get('camera_intrinsics')
                        homography = None
                        if calibration and c.get('camera_from_imu') is not None:
                            sx, sy = gray.shape[1]/calibration['width'], gray.shape[0]/calibration['height']
                            k = np.array([[calibration['fx']*sx, 0., calibration['cx']*sx],
                                          [0., calibration['fy']*sy, calibration['cy']*sy], [0., 0., 1.]])
                            homography = rotation_homography(imu_features, k, np.asarray(c['camera_from_imu']))
                        if previous is not None:
                            corners = cv2.goodFeaturesToTrack(previous,350,.015,8)
                            if corners is not None:
                                tracked,status,_ = cv2.calcOpticalFlowPyrLK(previous,gray,corners,None)
                                if tracked is not None:
                                    good=status.ravel()==1
                                    tracked_count=int(good.sum())
                                    if tracked_count:
                                        motion=float(np.median(np.linalg.norm(tracked[good]-corners[good],axis=2)))
                                        raw_motion = motion
                                        if homography is not None:
                                            rotation_pixels = cv2.perspectiveTransform(corners[good], homography)
                                            motion = float(np.median(np.linalg.norm(tracked[good]-rotation_pixels, axis=2)))
                                        elif c.get('keyframe_motion_policy', 'raw') == 'rotation-aware':
                                            from .visual_motion import rotation_compensated_motion
                                            if calibration:
                                                sx, sy = gray.shape[1]/calibration['width'], gray.shape[0]/calibration['height']
                                                k = np.array([[calibration['fx']*sx, 0., calibration['cx']*sx],
                                                              [0., calibration['fy']*sy, calibration['cy']*sy], [0., 0., 1.]])
                                            else:
                                                focal = c['focal_ratio'] * gray.shape[1]
                                                k = np.array([[focal, 0., gray.shape[1]/2],
                                                              [0., focal, gray.shape[0]/2], [0., 0., 1.]])
                                            compensated, visual_rotation = rotation_compensated_motion(
                                                corners[good], tracked[good], k)
                                            if compensated is not None:
                                                motion = compensated
                        quality = blur>=c['min_blur_score'] and exposure<.8
                        rapid_rotation = (imu_features.get('valid', False) and
                            imu_features.get('gyro_max_rad_s', 0.) > c.get('imu_max_angular_speed_rad_s', 3.))
                        if rapid_rotation and blur < c.get('imu_fast_rotation_min_blur_score', 2*c['min_blur_score']):
                            quality = False
                        rotation_compensated = homography is not None or visual_rotation.get('reason') == 'ok'
                        # A turn can move every pixel without creating a useful
                        # stereo baseline. Require visible residual motion after
                        # rotation, including for an interval-forced keyframe.
                        motion_threshold = (max(c['min_motion_pixels'],
                                                8. * gray.shape[1] / 360.)
                                            if rotation_compensated else c['min_motion_pixels'])
                        novelty = (previous is None or
                                   not rotation_compensated and tracked_count < 20 or
                                   motion >= motion_threshold)
                        interval_ready = stamp-previous_selected_time >= c['max_keyframe_gap_seconds']
                        interval_has_geometry = not rotation_compensated or motion >= .5 * motion_threshold
                        keep = quality and (novelty or interval_ready and interval_has_geometry)
                        reason='poor_quality' if not quality else ('visual_change' if novelty else 'coverage_interval')
                        if not keep and quality:
                            reason='redundant_view'
                        check_seconds=time.monotonic()-check_start
                        online['cheap_check_seconds'] += check_seconds
                        online['sampled_frames'] += 1
                        record={'event':'frame','source_index':source_index,'source_timestamp_s':stamp,
                            'selected':bool(keep),'reason':reason,'blur':blur,'motion_pixels':motion,
                            'raw_motion_pixels':raw_motion,
                            'rotation_compensated':rotation_compensated,
                            'motion_threshold_pixels':motion_threshold,
                            'visual_rotation_compensation':visual_rotation,
                            'imu':imu_features, 'timing':source_frame.timing,
                            'gps':source_frame.telemetry.get('gps', {}),
                            'decode_seconds':decode,'cheap_check_seconds':check_seconds,'references':[]}
                        if rapid_rotation and not quality:
                            record['reason'] = 'rapid_rotation_blur'
                        if keep:
                            name=f'frame_{source_index:06d}.jpg'
                            if not cv2.imwrite(str(out/'images'/name),image,[cv2.IMWRITE_JPEG_QUALITY,95]):
                                raise RuntimeError(f'Cannot save {name}')
                            descriptors.append(frame_descriptor(gray))
                            current_index=len(descriptors)-1
                            refs=past_references(descriptors,current_index,c['temporal_references'],c['retrieval_references'])
                            record['references']=[online['keyframes'][i] for i in refs]
                            online['pairs'].extend((online['keyframes'][i],name) for i in refs)
                            online['keyframes'].append(name)
                            previous,previous_selected_time=gray,stamp
                            online['last_selected_time']=stamp
                            if sensor_end is not None:
                                previous_sensor_time = sensor_end
                                online['last_selected_sensor_time'] = sensor_end
                                if c.get('save_imu_windows') and imu_features.get('valid'):
                                    sequence, _ = resample_imu(imu_history, sensor_start, sensor_end)
                                    sensor_dir = out/'imu_windows'
                                    sensor_dir.mkdir(exist_ok=True)
                                    np.save(sensor_dir/(name+'.npy'), sequence)
                                imu_history = [s for s in imu_history if s['timestamp_s'] >= sensor_end-.2]
                        online['last_source_index'],online['last_timestamp_s']=source_index,stamp
                        self.checkpoint()
                        update_seconds=0.
                        needed=c['bootstrap_keyframes'] if not online.get('best_model') else c['update_keyframes']
                        pending=len(online['keyframes'])-online.get('last_update_count',0)
                        if keep and pending>=needed:
                            update_start=time.monotonic()
                            self.update_map(stream)
                            update_seconds=time.monotonic()-update_start
                            online['last_update_count']=len(online['keyframes'])
                            self.checkpoint()
                        record.update(map_update_seconds=update_seconds,frame_processing_seconds=time.monotonic()-tick)
                        stream.write(json.dumps(record)+'\n')
                        stream.flush()
                        progress = getattr(self, 'on_progress', None)
                        if progress:
                            progress(record)
                finally:
                    frames.close()
                online['stream_complete']=True
                self.checkpoint()
            if len(online['keyframes'])>online.get('last_update_count',0) and len(online['keyframes'])>=c['min_registered_images']:
                self.update_map(stream)
                online['last_update_count']=len(online['keyframes'])
                self.checkpoint()
        stream_seconds=time.monotonic()-stream_start
        final_start=time.monotonic()
        if not online.get('best_model'):
            save_json(out/'online_timing.json', {'status':'insufficient_geometry',
                'mode':'causal_online_sparse', 'sampled_frames':online['sampled_frames'],
                'selected_keyframes':len(online['keyframes']), 'map_updates':online['map_updates'],
                'total_seconds':time.monotonic()-began, 'input_saved':True})
            raise RuntimeError('No validated online sparse model; capture saved; see online_events.jsonl')
        cameras,poses,points=read_model(Path(online['best_text']))
        save_json(out/'keyframes.json',online['keyframes'])
        np.save(out/'descriptors.npy',np.asarray(descriptors))
        confidence=[{'point_id':p['id'],'xyz':p['xyz'],'observations':p['observations'],
                     'error_px':p['error_px'],'confidence':min(1.,p['observations']/4)*np.exp(-p['error_px']/2)} for p in points]
        save_json(out/'confidence_map.json',confidence)
        save_json(out/'uncertainty_map.json',[{'point_id':p['point_id'],'xyz':p['xyz'],'uncertainty':1-p['confidence']} for p in confidence])
        metrics_path=out/'reconstruction_metrics.json'
        metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics.update({'mode':'online_sparse_then_final_dense' if finalize else 'online_sparse',
            'selected_frames':len(online['keyframes']),'registered_frames':len(poses),
            'registration_fraction':len(poses)/len(online['keyframes']),'sparse_points':len(points),
            'median_reprojection_error_px':float(np.median([p['error_px'] for p in points])),
            'scale':'arbitrary','ground_truth_available':False,'neural_3r_calls':0})
        save_json(metrics_path,metrics)
        # Reuse the existing dense/export implementation without rerunning offline selection or SfM.
        for name in ('decode','selection','retrieval','features','matching','mapping'):
            self.state['stages'][name]={'status':'complete','seconds':0.,'provided_by':'causal_online_pipeline'}
        self.checkpoint()
        if finalize:
            self.reconstruct()
        final_seconds=time.monotonic()-final_start
        invocation={'online_seconds':stream_seconds,'final_dense_seconds':final_seconds,
                    'total_seconds':time.monotonic()-began,'finalize':finalize}
        online['invocations'].append(invocation)
        self.checkpoint()
        first=next((u for u in online['map_updates'] if u['accepted']),None)
        report={'status':'complete','mode':'causal_online_sparse_reconstruction_with_optional_final_dense',
                'latest_invocation':invocation,'invocations':online['invocations'],
                'sampled_frames':online['sampled_frames'],'selected_keyframes':len(online['keyframes']),
                'accepted_map_updates':sum(u['accepted'] for u in online['map_updates']),
                'rejected_map_updates':sum(not u['accepted'] for u in online['map_updates']),
                'first_map_source_timestamp_s':first['source_timestamp_s'] if first else None,
                'decode_seconds':online['decode_seconds'],'cheap_check_seconds':online['cheap_check_seconds'],
                'map_update_seconds':sum(u['seconds'] for u in online['map_updates']),
                'map_updates':online['map_updates'],'real_time_guaranteed':False}
        save_json(out/'online_timing.json',report)
        from .timing_report import report as write_time_report
        write_time_report(out)
        print(json.dumps({k:v for k,v in report.items() if k not in ('map_updates','invocations')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='Saved MP4 file or session directory containing video.mp4')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent.parent/'configs/cpu_online.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--sfm-backend',choices=('pycolmap','colmap'),default='colmap')
    parser.add_argument('--sample-fps',type=float,help='Video analysis rate; use 0 to analyze every decoded frame before keyframe selection')
    parser.add_argument('--quality-buffer-seconds',type=float,default=0.,
                        help='Score every frame and choose sharp, matching views from windows of this duration')
    parser.add_argument('--rotation',choices=('none','cw','ccw','180'),default='none',
                        help='Orient quality-buffered source frames before camera calibration')
    parser.add_argument('--image-width',type=int,help='Width of saved COLMAP keyframes')
    parser.add_argument('--max-video-seconds',type=float,help='Process only the first N seconds of the saved video')
    parser.add_argument('--colmap',type=Path,default=Path('/home/vishnu/miniconda3/envs/drone3d/bin/colmap'))
    parser.add_argument('--openmvs',type=Path,default=Path('/home/vishnu/eswar/apps/openmvs/bin/OpenMVS'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--sparse-only',action='store_true',help='Skip final dense reconstruction and GLB export')
    args=parser.parse_args()
    input_path=args.input.resolve()
    video=input_path if input_path.is_file() else input_path/'video.mp4'
    if not video.is_file():
        parser.error('input must be a saved video file or a directory with video.mp4')
    session=input_path.parent if input_path.is_file() else input_path
    config=json.loads(args.config.read_text())
    config['threads']=min(config['threads'],len(os.sched_getaffinity(0)))
    config['sfm_backend']=args.sfm_backend
    if args.sample_fps is not None:
        config['sample_fps']=args.sample_fps
    if args.image_width is not None:
        config['image_width']=args.image_width
    if config['sample_fps']<0 or config['image_width']<100:
        parser.error('sample-fps must be nonnegative and image-width must be at least 100')
    if args.max_video_seconds is not None and args.max_video_seconds<=0:
        parser.error('max-video-seconds must be positive')
    if args.quality_buffer_seconds:
        from .quality_video import BufferedQualityVideoSource
        source=BufferedQualityVideoSource(video,args.quality_buffer_seconds,args.max_video_seconds,
            args.rotation,config['image_width'],config['analysis_width'])
    else:
        source=RecordedVideoSource(video,config['sample_fps'],args.max_video_seconds)
    default_output=(session/'reconstruction_online_cpu' if input_path.is_dir()
                    else session/(video.stem+'_reconstruction_online_cpu'))
    runner=OnlineReconstructor(session,args.output or default_output,config,args.colmap,args.openmvs,args.resume)
    started=time.monotonic()
    try:
        runner.run_online(finalize=not args.sparse_only,frame_source=source)
        from .recorded_viewer import write_recorded_viewer
        print(write_recorded_viewer(runner.output),flush=True)
    except Exception as error:
        save_json(runner.output/'online_failure.json',{'status':'failed','error':str(error),
            'seconds_current_invocation':time.monotonic()-started,
            'map_updates':runner.state.get('online',{}).get('map_updates',[])})
        raise


if __name__=='__main__':
    main()
