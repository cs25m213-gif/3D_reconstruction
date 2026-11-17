"""Adaptive COLMAP poses + pretrained VisFusion + selective geometric validation.

The primary learned network replaces full-video dense stereo. DA3 provides an
explicit fallback, with sparse stereo anchors rather than an unconditional
second full reconstruction. Capture completeness and geometric quality are
reported separately; assumed scale is never labelled measured metric scale.
"""
import argparse
import hashlib
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

from .combined_reconstruct import select_camera_model, save_json
from .hybrid_dense import read_array, camera_matrix, pose4
from .visfusion_infer import normalization
from .camera_coverage import capture_coverage, continue_mapping, result_status


def matching_sparse_run(root, video, session_hashes=None):
    stat = video.stat()
    found = []
    for p in (root/'runs').glob('*/pipeline_manifest.json'):
        try:
            d = json.loads(p.read_text())
            identity = d.get('inputs', {}).get('video', d.get('video', {}))
            if (Path(identity.get('path','')).resolve() == video.resolve() and
                identity.get('size_bytes') == stat.st_size and identity.get('mtime_ns') == stat.st_mtime_ns):
                saved_hashes=d.get('inputs',{}).get('session_files',d.get('sensors',{}))
                if session_hashes and saved_hashes!=session_hashes:continue
                candidate=Path(d.get('sparse_source',str(p.parent)))
                # Failed geometry is never promoted to a reusable pose cache.
                status_path=p.parent/'run_report.json'
                if status_path.is_file() and json.loads(status_path.read_text()).get('status')=='failed':continue
                if candidate.is_dir() and any(candidate.glob('online*/current_map.json')):found.append(candidate)
        except (OSError, ValueError, TypeError):
            continue
    return max(found, key=lambda p:p.stat().st_mtime_ns) if found else None


def continue_checked_segment(camera, source, output, seconds, *, ba_backend='auto', refine_intrinsics=False):
    """Keep a valid seed available when optional coverage continuation fails."""
    try:
        return continue_mapping(camera, source, output, seconds,
                                ba_backend=ba_backend, refine_intrinsics=refine_intrinsics)
    except (RuntimeError, ValueError) as error:
        print(f'[coverage] Continuation failed; retaining checked segment: {error}', flush=True)
        return source, {'status': 'failed', 'error': str(error),
                        'fallback_model': camera['model'],
                        'fallback_geometry_checked': True}


def adapter(ws, output, limit, metres_per_unit):
    reconstruction = pycolmap.Reconstruction(ws/'sparse')
    images = sorted((i for i in reconstruction.images.values() if i.has_pose), key=lambda i:i.name)
    if len(images) < 9:
        raise ValueError('Nine registered views required for VisFusion')
    indices = np.unique(np.linspace(0,len(images)-1,min(limit,len(images)),dtype=int))
    images = [images[i] for i in indices]
    depths = []
    for image in images:
        p = pose4(image)
        points = [reconstruction.points3D[k.point3D_id].xyz for k in image.points2D
                  if k.has_point3D() and k.point3D_id in reconstruction.points3D]
        if points:
            z = (np.asarray(points)@p[:3,:3].T+p[:3,3])[:,2]
            depths.extend(z[np.isfinite(z)&(z>0)][::5].tolist())
    poses = np.array([np.linalg.inv(pose4(i)) for i in images])
    transform, scale = normalization(poses, float(np.median(depths)), metres_per_unit)
    views = []
    for image, pose in zip(images,poses):
        camera = reconstruction.cameras[image.camera_id]
        if camera.model_name not in ('PINHOLE','SIMPLE_PINHOLE'):
            raise ValueError('Adapter requires undistorted pinhole images')
        converted = np.eye(4)
        converted[:3,:3] = transform[:3,:3]@pose[:3,:3]/scale['scale']
        converted[:3,3] = (transform@np.r_[pose[:3,3],1.])[:3]
        views.append({'name':image.name,'image':str(ws/'images'/image.name),
                      'K':camera.calibration_matrix().tolist(),'pose':converted.tolist()})
    data = {'views':views,'world_to_network':transform.tolist(),'normalization':scale}
    save_json(output, data)
    return data


def validate_mesh(mesh_path, ws, output, max_views=8):
    """Check held-out rendered depth against stereo at 160-pixel long side."""
    import open3d as o3d
    reconstruction = pycolmap.Reconstruction(ws/'sparse')
    by_name = {i.name:i for i in reconstruction.images.values() if i.has_pose}
    files = sorted((ws/'stereo/depth_maps').glob('*.geometric.bin'))
    if len(files)<4:
        return {'passed':False,'reason':'fewer than four stereo check maps'}
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(mesh.triangles):
        return {'passed':False,'reason':'empty mesh'}
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rows = []
    for i in np.unique(np.linspace(0,len(files)-1,min(max_views,len(files)),dtype=int)):
        path = files[i]
        name = path.name.removesuffix('.geometric.bin')
        if name not in by_name:
            continue
        image = by_name[name]
        d = read_array(path).astype(np.float32)
        k = camera_matrix(reconstruction.cameras[image.camera_id],d.shape)
        scale = min(1.,160/max(d.shape))
        w,h = max(1,round(d.shape[1]*scale)),max(1,round(d.shape[0]*scale))
        k[0] *= w/d.shape[1]; k[1] *= h/d.shape[0]
        d = cv2.resize(d,(w,h),interpolation=cv2.INTER_NEAREST)
        p = pose4(image)
        y,x = np.indices(d.shape)
        directions = np.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones_like(x)),-1)@p[:3,:3]
        origins = np.broadcast_to(-p[:3,:3].T@p[:3,3],directions.shape)
        rays = o3d.core.Tensor(np.concatenate((origins,directions),-1).astype(np.float32))
        rendered = scene.cast_rays(rays)['t_hit'].numpy()
        measured = np.isfinite(d)&(d>0)
        overlap = measured&np.isfinite(rendered)&(rendered>0)
        agreement = float((np.abs(rendered[overlap]-d[overlap])/d[overlap] < .05).mean()) if overlap.any() else None
        rows.append({'frame':name,'measured_pixels':int(measured.sum()),'overlap_pixels':int(overlap.sum()),
                     'measured_coverage':float(overlap.sum()/max(1,measured.sum())), 'agreement_under_5pct':agreement})
    valid = [r for r in rows if r['overlap_pixels']>=100 and r['agreement_under_5pct'] is not None]
    agreement = float(np.mean([r['agreement_under_5pct'] for r in valid])) if valid else None
    coverage = float(np.mean([r['measured_coverage'] for r in rows])) if rows else 0.
    report = {'passed':len(valid)>=4 and agreement is not None and agreement>=.8 and coverage>=.5,
              'mean_depth_agreement':agreement,'mean_measured_coverage':coverage,'per_view':rows,
              'geometry_accuracy_validated':False,'note':'Stereo is correlated with COLMAP poses; internal consistency is not ground truth.'}
    save_json(output,report)
    return report


def main():
    # Keep the proven path as the default until cross-session promotion passes.
    from .runtime_cli import requested_mode
    if requested_mode(sys.argv[1:])!='legacy':
        from .runtime import main as stream_main
        return stream_main()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('legacy','preview','balanced','quality'),default='legacy',help='Incremental modes preview/balanced/quality are dispatched to adaptive_uav.runtime')
    p.add_argument('session',type=Path)
    p.add_argument('--sparse-run',type=Path)
    p.add_argument('--fresh',action='store_true',help='Benchmark from the input video without sparse/depth reuse')
    p.add_argument('--output',type=Path)
    p.add_argument('--metres-per-unit',type=float,help='Optional externally established scale')
    p.add_argument('--max-views',type=int,default=36)
    p.add_argument('--target-seconds',type=float,default=180,help='Runtime target, not a success guarantee')
    p.add_argument('--mapping-seconds',type=float,default=90,help='Camera-mapping budget per pass, also honored for gyro-guided mapping')
    p.add_argument('--min-capture-fraction',type=float,default=.9,help='Minimum video span for complete capture status; selected-frame registration must also reach 80%%')
    p.add_argument('--ba-backend',choices=('auto','ceres','caspar'),default='auto',help='Native camera bundle-adjustment backend; CASPAR requires the verified CUDA build')
    p.add_argument('--camera-model',choices=('auto','shared-focal'),default='auto',
                   help='shared-focal uses a single refinable focal length and no lens distortion; use only when appropriate for the input')
    p.add_argument('--keyframe-motion',choices=('raw','rotation-aware'),default='raw',
                   help='Experimental visual rotation compensation for keyframe selection; raw preserves the established selector')
    p.add_argument('--dense-recovery-views',type=int,default=32,help='Expanded stereo references for dense fallback; 0 keeps only the initial check maps')
    p.add_argument('--stereo-policy',choices=('adaptive','skip'),default='adaptive',
                   help='skip runs DA3 with sparse-track anchors only; faster but has no measured-depth validation')
    p.add_argument('--detail',choices=('fast','balanced','high'),default='balanced',
                   help='RGB/depth/mesh detail: fast=640/336, balanced=1280/504, high=1920/672. Higher detail takes longer')
    p.add_argument('--da3-model',choices=('base','large'),default='base',
                   help='Pose-conditioned depth model for the DA3 fallback; large selects DA3-LARGE-1.1 and needs more GPU memory')
    p.add_argument('--depth-backend', choices=('da3','student'), default='da3',
                   help='Optional trained depth student; failed geometric checks use DA3')
    p.add_argument('--student-checkpoint', type=Path,
                   default=Path(__file__).resolve().parents[2]/'runs/depth_student_20261028/best.pt')
    p.add_argument('--refine-distortion',action='store_true',
                   help='Refine lens distortion before dense stages; removes SfM doming on near-parallel flights')
    p.add_argument('--neural-fusion',action='store_true',
                   help='Experimental overlap bins and validated depth student; production defaults remain unchanged')
    p.add_argument('--recurrent-checkpoint',type=Path,
                   help='Optional experimental ConvGRU checkpoint, gated by held-out sparse anchors')
    p.add_argument('--max-depth-frames',type=int,default=0,
                   help='Optional bounded dense smoke test; 0 uses all registered views')
    p.add_argument('--camera-recovery',choices=('auto','off'),default='auto',
                   help='Use installed pretrained SLAM3R camera seeds when mapping remains partial; real-track and gyro checks still apply')
    p.add_argument('--semantic-filter',choices=('on','off'),default='on',help='Exclude sky, people and supported animal classes from static geometry; retain timed object observations')
    p.add_argument('--semantic-rotation',choices=('auto','0','90','180','270'),default='auto',help='Clockwise orientation for segmentation only; camera geometry is unchanged')
    p.add_argument('--reuse-workspace',type=Path,
                   help='Reuse a saved stereo workspace after checking video identity, every pose/intrinsic/RGB hash and matching depth settings')
    args=p.parse_args()
    if args.neural_fusion:args.depth_backend='student'
    if args.recurrent_checkpoint and not args.neural_fusion:
        p.error('--recurrent-checkpoint requires --neural-fusion')
    from .dense_resolution import DETAIL_POLICIES, prepare_dense_inputs, matching_camera_resolution
    detail=DETAIL_POLICIES[args.detail]
    da3_model={'base':'depth-anything/DA3-BASE','large':'depth-anything/DA3-LARGE-1.1'}[args.da3_model]
    if args.fresh and args.sparse_run:p.error('--fresh cannot be combined with --sparse-run')
    root=Path(__file__).resolve().parents[2]
    if not args.student_checkpoint.is_absolute():
        args.student_checkpoint=(root/args.student_checkpoint).resolve()
    if args.max_depth_frames<0:p.error('--max-depth-frames must be nonnegative')
    session=args.session.resolve()
    video=(session/'video.mp4').resolve()
    if args.metres_per_unit is not None and (not np.isfinite(args.metres_per_unit) or args.metres_per_unit<=0):
        p.error('--metres-per-unit must be finite and positive')
    if not video.is_file() or args.max_views<9 or not np.isfinite(args.target_seconds) or args.target_seconds<=0:
        p.error('A saved video, at least nine views and positive time target are required')
    if not np.isfinite(args.mapping_seconds) or args.mapping_seconds<=0 or not 0<args.min_capture_fraction<=1:
        p.error('A positive mapping budget and capture fraction in (0, 1] are required')
    if args.dense_recovery_views != 0 and args.dense_recovery_views < 12:
        p.error('Dense recovery needs at least 12 reference slots, or 0 to disable it')
    out=(args.output or root/'runs'/f'{session.name}_final').resolve()
    if out==session or out.is_relative_to(session):
        p.error('Output must be separate from input')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    report={'status':'starting','source':str(session),'runtime_target_seconds':args.target_seconds,
            'detail':{'preset':args.detail,**detail},
            'da3_model':da3_model,
            'geometry_accuracy_validated':False,'units':'COLMAP visual units','capture_status':'unverified',
            'pipeline':'adaptive cameras with verified learned recovery -> pretrained VisFusion -> selective stereo checks -> capture-wide DA3 fallback'}
    stage_started=started
    def save(status):
        nonlocal stage_started
        now=time.monotonic()
        if status != report['status']:
            times=report.setdefault('stage_seconds',{})
            previous=report['status']
            times[previous]=times.get(previous,0.)+now-stage_started
            stage_started=now
        report.update(status=status,elapsed_seconds=now-started)
        save_json(out/'run_report.json',report)
        print(status,flush=True)
    project=root/'uav_reconstruction'
    def run(module,*values,python=None):
        subprocess.run([str(python or sys.executable),'-u','-m',module,*map(str,values)],cwd=project,check=True)
    try:
        save('session_input_validation')
        from .session_inputs import inspect_session, input_hashes
        input_report,camera_overrides=inspect_session(session)
        session_hashes=input_report['input_hashes']
        save_json(out/'session_inputs_report.json',input_report)
        report['session_inputs']={'report':str(out/'session_inputs_report.json'),
            'camera':input_report['camera'],'stabilization':input_report['stabilization'],
            'timing':input_report['timing'],'notes':input_report['notes'],
            'gps_metric_scale_used':False,'imu_valid_samples':input_report['imu'].get('valid_samples',0)}
        source=None if args.fresh else args.sparse_run.resolve() if args.sparse_run else matching_sparse_run(root,video,session_hashes)
        fresh_sparse=source is None
        report['runtime_scope']='fresh profiling, features, matching and camera mapping' if fresh_sparse else 'cached sparse poses'
        cap=cv2.VideoCapture(str(video))
        fps,count,vw,vh=(cap.get(k) for k in (5,7,3,4));cap.release()
        if min(fps,count,vw,vh)<=0:raise ValueError('Invalid video metadata')
        if source is None:
            save('video_profiling')
            from .adaptive_sfm import profile_video,choose_settings,recovery_settings
            config=json.loads((project/'configs/cpu_online.json').read_text())
            width=round(1280*vw/max(vw,vh))
            profile=profile_video(video,analysis_width=min(360,width))
            save_json(out/'video_profile.json',profile)
            config.update(choose_settings(profile),sfm_use_gpu=True,sfm_ba_backend=args.ba_backend,refine_intrinsics=True,
                          gyro_init_guidance=True,
                          video_fps=fps,
                          threads=8,image_width=width,analysis_width=min(360,width),
                          mapping_timeout_seconds=args.mapping_seconds,
                          bootstrap_keyframes=10000,update_keyframes=10000)
            config.update(camera_overrides)
            # The faster verified CUDA solver retained trajectory accuracy in
            # calibrated RGB-only trials. Keep the established Ceres path when
            # calibration is uncertain or gyro guidance may affect mapping.
            config['prefer_caspar_auto']=(args.ba_backend=='auto' and
                bool(input_report['camera'].get('calibration_exact')) and
                int(input_report['imu'].get('valid_samples') or 0) == 0)
            config['camera_model_policy']=args.camera_model
            config['keyframe_motion_policy']=args.keyframe_motion
            # Start with stronger connectivity instead of always doing a second run.
            config['temporal_references']=min(26,config['temporal_references']+6)
            config['retrieval_references']=min(12,config['retrieval_references']+4)
            source=out/'sparse_run'; source.mkdir(exist_ok=True)
            identity={'path':str(video),'size_bytes':video.stat().st_size,'mtime_ns':video.stat().st_mtime_ns}
            save_json(source/'pipeline_manifest.json',{'inputs':{'video':identity,'session_files':session_hashes}})
            report['sparse_attempts']=[]
            for attempt in range(2):
                save(f'adaptive_sparse_attempt_{attempt}')
                tick=time.monotonic()
                cfg=source/f'config_{attempt}.json'; save_json(cfg,config)
                attempt_output=source/('online' if attempt==0 else 'online_recovery_1')
                try:
                    run('adaptive_uav.online',session,'--config',cfg,'--output',attempt_output,
                        '--sfm-backend','pycolmap','--sparse-only','--quality-buffer-seconds',min(.4,config['max_keyframe_gap_seconds']/2))
                    process_error=None
                except subprocess.CalledProcessError as exc:
                    process_error=str(exc)
                row={'index':attempt,'seconds':time.monotonic()-tick,'error':process_error}
                compute=attempt_output/'compute_backend.json'
                if compute.is_file():row['compute_backend']=json.loads(compute.read_text())
                report['sparse_attempts'].append(row)
                save(f'adaptive_sparse_attempt_{attempt}')
                try:
                    eligible,_=select_camera_model(source,session,out,fps,allow_partial=True)
                    coverage=capture_coverage(eligible,count/fps,args.min_capture_fraction)
                    report['capture_coverage']=coverage
                    if coverage['passed']:
                        report['sparse_retry_skipped']='Camera model passed capture-wide coverage and quality checks'
                        break
                    # Extend a checked partial model rather than re-extracting
                    # all features and starting again from the same middle pair.
                    if attempt==0:
                        save('camera_coverage_continuation')
                        source,continuation=continue_checked_segment(eligible,source,out,args.mapping_seconds,ba_backend=args.ba_backend,
                            refine_intrinsics=camera_overrides['refine_intrinsics'] and not camera_overrides['freeze_guided_intrinsics'])
                        report['camera_continuation']=continuation
                        break
                except RuntimeError:
                    pass
                remaining=args.target_seconds-(time.monotonic()-started)
                if remaining<60:
                    report['sparse_retry_skipped']='Insufficient remaining runtime budget for another sparse pass'
                    break
                config=recovery_settings(config,{})
                config['mapping_timeout_seconds']=args.mapping_seconds
        sparse_manifest=json.loads((source/'pipeline_manifest.json').read_text())
        identity=sparse_manifest['inputs']['video']
        saved_hashes=sparse_manifest['inputs'].get('session_files')
        if saved_hashes is not None and saved_hashes!=session_hashes:
            raise ValueError('Sparse cache was created from different sensor/calibration files; use --fresh')
        if saved_hashes is None and ((session/'camera_intrinsics.json').exists() or (session/'stabilization_profile.json').exists()):
            raise ValueError('Legacy sparse cache has no record of these calibration/stabilization files; use --fresh')
        if Path(identity['path']).resolve()!=video or identity['size_bytes']!=video.stat().st_size or identity['mtime_ns']!=video.stat().st_mtime_ns:
            raise ValueError('Sparse source does not match this video')
        # Cached poses must pass the same capture coverage requirement. Continue
        # partial caches using their existing matches, with no video extraction.
        candidate,_=select_camera_model(source,session,out,fps,allow_partial=True)
        if args.camera_model == 'shared-focal':
            selected_model = pycolmap.Reconstruction(candidate['model'])
            if any(camera.model_name != 'SIMPLE_PINHOLE' for camera in selected_model.cameras.values()):
                raise ValueError('Cached camera model differs from --camera-model shared-focal; use --fresh')
        initial_coverage=capture_coverage(candidate,count/fps,args.min_capture_fraction)
        if not initial_coverage['passed'] and not report.get('camera_continuation'):
            save('camera_coverage_continuation')
            source,continuation=continue_checked_segment(candidate,source,out,args.mapping_seconds,ba_backend=args.ba_backend,
                refine_intrinsics=camera_overrides['refine_intrinsics'] and not camera_overrides['freeze_guided_intrinsics'])
            report['camera_continuation']=continuation
        candidate,_=select_camera_model(source,session,out,fps,allow_partial=True)
        if not capture_coverage(candidate,count/fps,args.min_capture_fraction)['passed'] and args.camera_recovery=='auto':
            save('learned_camera_recovery')
            from .learned_camera import recover_camera_model
            source,recovery=recover_camera_model(candidate,source,out,session,count/fps,fps,args.min_capture_fraction)
            report['learned_camera_recovery']=recovery
        identity=json.loads((source/'pipeline_manifest.json').read_text())['inputs']['video']
        if Path(identity['path']).resolve()!=video or identity['size_bytes']!=video.stat().st_size or identity['mtime_ns']!=video.stat().st_mtime_ns:
            raise ValueError('Sparse source does not match this video')
        policy={'video':identity,'sparse_source':str(source),'max_views':args.max_views,'metres_per_unit':args.metres_per_unit,
                'requested_camera_model':args.camera_model,
                'keyframe_motion_policy':args.keyframe_motion,
                'mapping_seconds':args.mapping_seconds,'min_capture_fraction':args.min_capture_fraction,
                'ba_backend':args.ba_backend,'dense_recovery_views':args.dense_recovery_views,
                'stereo_policy':args.stereo_policy,
                'detail':{'preset':args.detail,**detail},
                'camera_recovery':args.camera_recovery,
                'semantic_filter':args.semantic_filter,'semantic_rotation':args.semantic_rotation,
                'reuse_workspace':str(args.reuse_workspace.resolve()) if args.reuse_workspace else None,
                'code':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in
                        ('final_reconstruct.py','visfusion_infer.py','combined_reconstruct.py','sensor_audit.py',
                         'multiview_dense.py','semantic_scene.py','mesh_viewer.py','dense_cuda.py','global_visibility.py','hybrid_dense.py',
                         'gpu_sfm.py','pycolmap_backend.py','gyro_initialization.py','mesh_color.py','camera_coverage.py','dense_coverage.py','dense_resolution.py','learned_camera.py','session_inputs.py')}}
        policy['sensors']=session_hashes
        if args.refine_distortion:policy['refine_distortion']=True
        if input_hashes(session)!=session_hashes:raise ValueError('Sensor/calibration files changed during reconstruction; use a stable saved session')
        if args.semantic_filter=='off':
            policy['visfusion_code']=hashlib.sha256(b''.join(p.read_bytes() for p in
                sorted((root/'VisFusion/models').glob('*.py')))).hexdigest()
            policy['checkpoint']=hashlib.sha256((root/'VisFusion/checkpoints/pretrained/model_000049.ckpt').read_bytes()).hexdigest()
        manifest=out/'pipeline_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=policy:
            raise ValueError('Output policy changed; use a fresh --output')
        save_json(manifest,policy)
        cap=cv2.VideoCapture(str(video));fps=cap.get(5);count=cap.get(7);cap.release()
        save('camera_selection')
        camera,images=select_camera_model(source,session,out,fps,allow_partial=True)
        report['camera_selection']=camera
        report['capture_coverage']=capture_coverage(camera,count/fps,args.min_capture_fraction)
        report['capture_status']='most_of_capture' if report['capture_coverage']['passed'] else 'partial'
        if report['capture_status']=='partial':
            report['capture_warning']='Dense output covers only the geometry-checked registered cameras; missing camera coverage is retained as incomplete status.'
        ws=out/'dense';ws.mkdir(exist_ok=True)
        if args.refine_distortion:
            if not (out/'sparse_dedomed/cameras.bin').is_file():
                from .sfm_dedome import refine_model
                save('distortion_refinement')
                _,report['distortion_refinement']=refine_model(Path(camera['model']),out/'sparse_dedomed')
            camera['model']=str(out/'sparse_dedomed')
        if not (ws/'sparse/cameras.bin').is_file():
            save('undistortion')
            prepared_model,prepared_images,resolution=prepare_dense_inputs(Path(camera['model']),images,video,
                out/'source_rgb',detail['dense_size'])
            report['source_resolution']=resolution
            pycolmap.undistort_images(ws,prepared_model,prepared_images,num_threads=4,
                undistort_options=pycolmap.UndistortCameraOptions(max_image_size=detail['dense_size']))
        # Reuse only caches with identical registered poses AND video identity.
        target=pycolmap.Reconstruction(ws/'sparse')
        target_poses={i.name:pose4(i) for i in target.images.values() if i.has_pose}
        maps=ws/'stereo/depth_maps';maps.mkdir(parents=True,exist_ok=True)
        if args.stereo_policy=='skip' and args.reuse_workspace:
            raise ValueError('--stereo-policy skip cannot reuse a stereo workspace')
        if args.stereo_policy!='skip' and not args.fresh and not list(maps.glob('*.geometric.bin')):
            for cached in sorted((root/'runs').glob('*/dense/sparse')):
                if cached.parent==ws:continue
                mp=cached.parent.parent/'pipeline_manifest.json'
                if not mp.is_file():continue
                old=json.loads(mp.read_text())
                if old.get('video')!=identity:continue
                r=pycolmap.Reconstruction(cached)
                old_poses={i.name:pose4(i) for i in r.images.values() if i.has_pose}
                if old_poses.keys()!=target_poses.keys() or any(not np.allclose(old_poses[n],target_poses[n],rtol=0,atol=1e-7) for n in target_poses):continue
                if not matching_camera_resolution(r,target,target_poses):continue
                if old.get('detail') != policy['detail']:continue
                old_images={i.name:i for i in r.images.values() if i.has_pose}
                current_images={i.name:i for i in target.images.values() if i.has_pose}
                def normalized_k(rec,image):
                    camera=rec.cameras[image.camera_id]
                    k=camera.calibration_matrix().copy()
                    k[0]/=camera.width;k[1]/=camera.height
                    return k
                if any(not np.allclose(normalized_k(r,old_images[n]), normalized_k(target,current_images[n]),rtol=0,atol=1e-6) for n in target_poses):continue
                available=sorted((cached.parent/'stereo/depth_maps').glob('*.geometric.bin'))
                if len(available)<8:continue
                for i in np.unique(np.linspace(0,len(available)-1,min(20,len(available)),dtype=int)):
                    read_array(available[i]);shutil.copy2(available[i],maps/available[i].name)
                report['depth_cache_source']=str(cached.parent);break
        report['stereo_checks_cached']=len(list(maps.glob('*.geometric.bin')))
        if args.reuse_workspace:
            from .dense_resolution import validate_cached_workspace
            report['stereo_workspace_reuse']=validate_cached_workspace(args.reuse_workspace,ws,identity,detail,args.dense_recovery_views)
            ws=args.reuse_workspace.resolve()
            report['stereo_checks_cached']=report['stereo_workspace_reuse']['references']
            report['runtime_scope']='cached camera poses and validated cached stereo workspace'
        report['depth_check_source']='cached stereo' if report['stereo_checks_cached'] else 'selective fresh stereo'
        semantic_dir=None
        vis=out/'visfusion'
        model_error=None
        if args.stereo_policy=='skip':
            report['pipeline']='adaptive cameras -> sparse-track anchored DA3 -> visibility-gated mesh; no stereo depth check'
        if args.depth_backend=='student':
            report['pipeline']='adaptive cameras -> measured stereo + depth student -> DA3 on failed views -> RGB mesh'
        if args.semantic_filter=='on':
            semantic_dir=out/'semantics'
            save('semantic_masks_running')
            run('adaptive_uav.semantic_scene','--workspace',ws,'--output',semantic_dir,'--rotation',args.semantic_rotation)
            semantic=json.loads((semantic_dir/'semantic_report.json').read_text())
            report['semantic_filter']={key:semantic[key] for key in ('status','rotation_clockwise','mean_sky_fraction','mean_dynamic_fraction','object_observations','elapsed_seconds')}
            report['semantic_filter']['directory']=str(semantic_dir)
            report['visfusion_skipped']='Semantic masks require depth-level exclusion before fusion; using pose-conditioned DA3.'
            report['pipeline']='adaptive cameras -> GPU semantic masks -> measured stereo + DA3 -> masked static mesh and timed object observations'
        elif args.stereo_policy!='skip' and args.depth_backend!='student':
            save('pretrained_visfusion_running')
            data=adapter(ws,out/'visfusion_input.json',args.max_views,args.metres_per_unit)
            report['normalization']=data['normalization']
            try:
                run('adaptive_uav.visfusion_infer','--input',out/'visfusion_input.json','--output',vis,
                    '--repository',root/'VisFusion',python=root/'.venv_visfusion/bin/python')
            except subprocess.CalledProcessError as exc:
                model_error=str(exc)
                report['visfusion_error']=model_error
        if args.stereo_policy!='skip' and report['stereo_checks_cached']<8 and (semantic_dir is None or not args.dense_recovery_views):
            save('selective_stereo_checks_running')
            run('adaptive_uav.dense_cuda','--workspace',ws,'--openmvs',root/'tools/OpenMVS_2.4.0',
                '--runtime',root/'tools/noble_runtime','--three-assets',root/'tools/three_0.169.0',
                '--max-image-size',detail['dense_size'],'--max-views',12,'--source-views',3,'--iterations',detail['stereo_iterations'],
                '--threads',4,'--spatial-plan','--depth-only')
        checked=None
        if args.stereo_policy!='skip' and semantic_dir is None and model_error is None and args.depth_backend!='student':
            save('visfusion_geometry_check')
            checked=validate_mesh(vis/'scene_visfusion.ply',ws,out/'visfusion_validation.json')
        report['visfusion_validation']=checked
        if checked and checked['passed']:
            best=vis;quality='internally_consistent_experimental'
            report['selected_backend']='pretrained_visfusion'
            save('rgb_coloring')
            from .mesh_color import color_mesh
            coloring=color_mesh(vis/'scene_visfusion.ply',ws,vis)
            mesh=vis/'scene_colored.glb'
            report['coloring']={k:v for k,v in coloring.items() if k!='per_view'}
            from .mesh_viewer import write_mesh_viewer
            link=vis/'scene_mesh.ply'
            if not link.exists():link.symlink_to('scene_visfusion.ply')
            save_json(vis/'dense_metrics.json', {'source_model':str(ws/'sparse'),
                'pipeline':'pretrained VisFusion', 'quality_status':quality,
                'geometry_accuracy_validated':False,'coloring':report['coloring'],
                'quality_warning':'Learned geometry; partial capture and unvalidated metric scale.'})
            write_mesh_viewer(vis,root/'tools/three_0.169.0')
        else:
            fallback_ws=ws
            if args.reuse_workspace:
                report['dense_recovery']={'workspace':str(ws),'references':report['stereo_checks_cached'],
                    'validated_workspace_reused':True,'parallax_threshold_relaxed':False}
            elif args.stereo_policy!='skip' and args.dense_recovery_views:
                save('dense_coverage_recovery')
                from .dense_coverage import recovery_workspace
                fallback_ws=out/'dense_recovery'
                expanded=recovery_workspace(ws,fallback_ws,args.dense_recovery_views,4)
                report['dense_recovery']={'workspace':str(fallback_ws),'references':expanded['reference_views'],
                    'parallax_threshold_relaxed':False,'registered_cameras':expanded['registered_cameras']}
                run('adaptive_uav.dense_cuda','--workspace',fallback_ws,'--openmvs',root/'tools/OpenMVS_2.4.0',
                    '--runtime',root/'tools/noble_runtime','--three-assets',root/'tools/three_0.169.0',
                    '--max-image-size',detail['dense_size'],'--iterations',detail['stereo_iterations'],'--threads',4,'--depth-only')
            save('masked_da3_running' if semantic_dir else 'da3_fallback_running')
            base=out/'multiview'
            semantic_options=['--semantic-dir',semantic_dir,'--video-fps',fps] if semantic_dir else []
            run('adaptive_uav.multiview_dense','--workspace',fallback_ws,'--output',base,'--resolution',detail['depth_resolution'],
                '--model',da3_model,'--depth-backend',args.depth_backend,
                '--student-checkpoint',args.student_checkpoint,
                '--views',3,'--include-unmeasured','--max-frames',args.max_depth_frames,'--voxel-fraction',detail['voxel_fraction'],'--visibility-references',16,
                '--audit-views',len(target_poses),'--global-visibility','--three-assets',root/'tools/three_0.169.0',
                *(['--overlap-bins'] if args.neural_fusion else []),
                *(['--recurrent-checkpoint',args.recurrent_checkpoint] if args.recurrent_checkpoint else []),
                *semantic_options)
            q=json.loads((base/'multiview_report.json').read_text())
            if args.neural_fusion:
                report['neural_fusion']={'overlap_bins':json.loads((base/'overlap_bins.json').read_text())['bins'],
                    'recurrent':json.loads((base/'recurrent_report.json').read_text())
                        if (base/'recurrent_report.json').exists() else None}
            best=base;quality=q['quality_status']
            if args.stereo_policy=='skip':
                report['stereo_policy_warning']='No stereo measurements were made; sparse-track anchors set local visual scale, but learned surface accuracy is unvalidated.'
            save('rgb_coloring')
            from .mesh_color import color_mesh
            coloring=color_mesh(base/'scene_completed.ply',fallback_ws,base,preserve_colors=True,semantic_dir=semantic_dir)
            mesh=base/'scene_colored.glb'
            report['coloring']={k:v for k,v in coloring.items() if k!='per_view'}
            from .mesh_viewer import write_mesh_viewer
            write_mesh_viewer(base,root/'tools/three_0.169.0')
            report['selected_backend']=('pose_conditioned_student_with_da3_fallback'
                if args.depth_backend=='student' else 'pose_conditioned_da3_with_visibility')
            if semantic_dir:
                report['object_observations']=q['semantic_filter']
                report['coverage_domain']=q['surface_audit']['coverage_domain']
        report.update(selected_result=str(best),mesh=str(mesh),quality_status=quality,
                      runtime_target_met=time.monotonic()-started<=args.target_seconds)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Final reconstruction</title>'
            f'<p>Capture: {report["capture_status"]}. Quality: {quality}. Scale and external accuracy are unvalidated.</p>'
            f'<a href="{best.name}/mesh_viewer.html">View selected mesh</a> '
            f'<a href="{mesh.relative_to(out)}">Download GLB mesh</a> <a href="run_report.json">Run report</a>')
        report['result_ready']=True
        save(result_status(report['capture_coverage'],quality))
    except Exception as error:
        report['error']=str(error);save('failed');raise


if __name__=='__main__':main()
