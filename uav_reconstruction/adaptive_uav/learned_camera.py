"""Use pretrained point maps as camera seeds, then verify with real image tracks.

The learned confidence and point-map reprojection do not establish camera
quality. Existing cameras/landmarks stay fixed. New cameras must have measured
tracks, a connection to the existing map, low reprojection error and gyro
agreement before they can enter the reconstruction.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_coverage import capture_coverage, link_cached_candidates
from .combined_reconstruct import save_json


def crop_camera(camera, clockwise=0):
    """Match the upstream 224-square resize/crop and camera-axis rotation."""
    if clockwise not in (0, 90, 180, 270):
        raise ValueError('Image rotation must be a multiple of 90 degrees')
    w, h = camera.width, camera.height
    transforms = {
        0: np.eye(3),
        90: np.array([[0., -1, h-1], [1, 0, 0], [0, 0, 1]]),
        180: np.array([[-1., 0, w-1], [0, -1, h-1], [0, 0, 1]]),
        270: np.array([[0., 1, 0], [-1, 0, w-1], [0, 0, 1]]),
    }
    if clockwise in (90, 270):
        w, h = h, w
    edge = round(224 * max(w/h, h/w))
    rw, rh = round(w*edge/max(w, h)), round(h*edge/max(w, h))
    cx, cy = rw//2, rh//2; half = min(cx, cy)
    resize_crop = np.array([[rw/w, 0., -(cx-half)], [0., rh/h, -(cy-half)], [0, 0, 1.]])
    affine = resize_crop @ transforms[clockwise]
    axis = Rotation.from_euler('z', clockwise, degrees=True).as_matrix()
    return affine @ camera.calibration_matrix() @ axis.T, affine, axis


def opencv_distortion(camera):
    """Translate supported fitted COLMAP lens models into OpenCV order."""
    name=camera.model_name;parameters=camera.params
    if name in ('PINHOLE','SIMPLE_PINHOLE'):return np.zeros(5)
    if name=='SIMPLE_RADIAL':return np.array([parameters[3],0.,0.,0.,0.])
    if name=='RADIAL':return np.array([parameters[3],parameters[4],0.,0.,0.])
    if name=='OPENCV':return np.r_[parameters[4:8],0.]
    if name=='FULL_OPENCV':return np.asarray(parameters[4:12])
    raise ValueError('Learned camera rectification does not support '+name)


def rectified_pixel(camera, pixel):
    """Keep measured features in the same pinhole coordinates as neural RGB."""
    ray=camera.cam_from_img(pixel)
    if ray is None or not np.isfinite(ray).all():return None
    point=camera.calibration_matrix()@np.r_[ray,1.]
    return point[:2]/point[2]


def similarity(a, b):
    """Fit a proper (non-reflecting) similarity between matched landmarks."""
    am, bm = a.mean(0), b.mean(0)
    aa, bb = a-am, b-bm
    u, singular, v = np.linalg.svd(bb.T @ aa / len(a))
    sign = np.eye(3); sign[-1, -1] = np.linalg.det(u @ v)
    rotation = u @ sign @ v
    variance = np.mean(np.sum(aa*aa, axis=1))
    if variance < 1e-12:
        raise ValueError('Degenerate landmark alignment')
    scale = np.sum(singular*np.diag(sign))/variance
    return scale, rotation, bm-scale*rotation@am


def connected_to_existing(rec, known):
    """Reject learned components with no measured-track link to the old map."""
    graph = defaultdict(set)
    for point in rec.points3D.values():
        ids = {e.image_id for e in point.track.elements
               if e.image_id in rec.images and rec.images[e.image_id].has_pose}
        for i in ids:
            graph[i].update(ids-{i})
    reached = set(known); queue = deque(known)
    while queue:
        for other in graph[queue.popleft()] - reached:
            reached.add(other); queue.append(other)
    return reached


def fit_seeds(rec, names, points, confidence, clockwise):
    from .hybrid_dense import pose4
    camera = next(iter(rec.cameras.values()))
    k, affine, axis = crop_camera(camera, clockwise)
    yy, xx = np.indices((224, 224))
    pixels = np.stack((xx, yy), -1).reshape(-1, 2).astype(np.float32)
    seeds = {}; fits = []
    for index, name in enumerate(names):
        xyz = points[index].reshape(-1, 3)
        good = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1)>1e-6) & (confidence[index].ravel()>12)
        ids = np.flatnonzero(good)[::3]
        row = {'frame': name, 'usable_predictions': len(ids)}
        fits.append(row)
        if len(ids)<100:
            continue
        ok, r, t, inliers = cv2.solvePnPRansac(xyz[ids], pixels[ids], k, None,
            iterationsCount=500, reprojectionError=2., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<100:
            continue
        selected=ids[inliers.ravel()]
        r,t=cv2.solvePnPRefineLM(xyz[selected],pixels[selected],k,None,r,t)
        projected=cv2.projectPoints(xyz[selected],r,t,k,None)[0].reshape(-1,2)
        row.update(inlier_fraction=len(inliers)/len(ids),
                   median_prediction_reprojection_px=float(np.median(np.linalg.norm(projected-pixels[selected],axis=1))))
        pose=np.eye(4);pose[:3,:3]=axis.T@cv2.Rodrigues(r)[0];pose[:3,3]=axis.T@t.ravel()
        seeds[name]=pose
    by_name={im.name:im for im in rec.images.values() if im.has_pose}
    source=[];target=[]
    for index,name in enumerate(names):
        if name not in by_name:
            continue
        for observation in by_name[name].points2D:
            if not observation.has_point3D() or observation.point3D_id not in rec.points3D:
                continue
            pixel=rectified_pixel(camera,observation.xy)
            if pixel is None:continue
            uv=affine@np.r_[pixel,1.]
            u,v=np.rint(uv[:2]).astype(int)
            if 1<=u<223 and 1<=v<223 and confidence[index,v,u]>12:
                point=points[index,v,u]
                if np.isfinite(point).all() and np.linalg.norm(point)>1e-6:
                    source.append(point);target.append(rec.points3D[observation.point3D_id].xyz)
    source,target=np.asarray(source),np.asarray(target)
    if len(source)<100:
        raise ValueError('Insufficient measured landmarks for learned-world alignment')
    rng=np.random.default_rng(42)
    if len(source)>12000:
        ids=rng.choice(len(source),12000,replace=False);source,target=source[ids],target[ids]
    extent=float(np.median(np.linalg.norm(target-np.median(target,axis=0),axis=1)))
    if extent<=1e-8:
        raise ValueError('Degenerate existing map extent')
    best=None
    for _ in range(300):
        ids=rng.choice(len(source),6,replace=False)
        try: scale,rotation,translation=similarity(source[ids],target[ids])
        except ValueError: continue
        if scale<=0:continue
        error=np.linalg.norm(scale*source@rotation.T+translation-target,axis=1)
        good=error<.08*extent
        if best is None or good.sum()>best.sum():best=good
    if best is None or best.sum()<100:
        raise ValueError('Learned world does not align with measured landmarks')
    good=best
    for _ in range(3):
        scale,rotation,translation=similarity(source[good],target[good])
        error=np.linalg.norm(scale*source@rotation.T+translation-target,axis=1)
        good=error<.08*extent
        if good.sum()<100:raise ValueError('Unstable learned-world alignment')
    aligned={};angles=[]
    for name,pose in seeds.items():
        inverse=np.linalg.inv(pose)
        inverse[:3,:3]=rotation@inverse[:3,:3]
        inverse[:3,3]=scale*rotation@inverse[:3,3]+translation
        aligned[name]=np.linalg.inv(inverse)
        if name in by_name:
            old=pose4(by_name[name])
            angles.append(float(Rotation.from_matrix(aligned[name][:3,:3]@old[:3,:3].T).magnitude()*180/np.pi))
    audit={'fitted_cameras':len(aligned),'alignment_landmarks':len(source),
           'alignment_inlier_fraction':float(good.mean()),'median_alignment_error_scene_fraction':float(np.median(error)/extent),
           'known_rotation_median_deg':float(np.median(angles)) if angles else None,
           'known_rotation_p90_deg':float(np.quantile(angles,.9)) if angles else None,
           'prediction_fits':fits,'note':'Learned confidence/reprojection is not measured-track validation.'}
    if len(angles)<9 or np.median(angles)>10 or np.quantile(angles,.9)>20 or np.median(error)/extent>.25:
        raise ValueError('Learned camera seeds fail agreement with existing checked cameras')
    return aligned,audit


def verify_tracks(model, source, seeds, output):
    import pycolmap
    from .hybrid_dense import pose4
    from .gpu_sfm import verified_capabilities
    rec=pycolmap.Reconstruction(model)
    known={im.image_id:pose4(im).copy() for im in rec.images.values() if im.has_pose}
    fixed_points={pid:p.xyz.copy() for pid,p in rec.points3D.items()}
    with pycolmap.Database.open(source/'database.db') as db:
        for image in db.read_all_images():
            if image.image_id in known or image.name not in seeds:continue
            image.points2D=pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in db.read_keypoints(image.image_id)[:,:2]])
            pose=seeds[image.name]
            rec.add_image_with_trivial_frame(image,pycolmap.Rigid3d(pycolmap.Rotation3d(pose[:3,:3]),pose[:3,3]))
    options=pycolmap.IncrementalPipelineOptions()
    options.num_threads=options.mapper.num_threads=4
    options.ba_global_max_num_iterations=1;options.ba_global_max_refinements=1
    options.ba_refine_focal_length=options.ba_refine_extra_params=options.ba_refine_sensor_from_rig=False
    options.fix_existing_frames=options.mapper.fix_existing_frames=True
    options.triangulation.ignore_two_view_tracks=False
    temp=output/'triangulated';temp.mkdir()
    rec=pycolmap.triangulate_points(rec,source/'database.db',source/'images',temp,
                                   clear_points=False,options=options,refine_intrinsics=False)
    for image_id,pose in known.items():
        im=rec.images[image_id]
        rec.frames[im.frame_id].set_cam_from_world(im.camera_id,pycolmap.Rigid3d(pycolmap.Rotation3d(pose[:3,:3]),pose[:3,3]))
    for pid,xyz in fixed_points.items():
        if pid in rec.points3D:rec.points3D[pid].xyz=xyz
    ba=pycolmap.BundleAdjustmentOptions()
    if not verified_capabilities(pycolmap.__file__,pycolmap.__version__).get('caspar_verified'):
        raise RuntimeError('Learned-camera recovery needs the verified CASPAR GPU build')
    ba.backend=pycolmap.BundleAdjustmentBackend.CASPAR;ba.caspar.solver_iter_max=80
    ba.refine_focal_length=ba.refine_extra_params=ba.refine_principal_point=ba.refine_sensor_from_rig=False
    config=pycolmap.BundleAdjustmentConfig()
    for im in rec.images.values():
        if im.has_pose:
            config.add_image(im.image_id)
            if im.image_id in known:config.set_constant_rig_from_world_pose(im.frame_id)
    for pid in fixed_points:
        if pid in rec.points3D:config.add_constant_point(pid)
    summary=pycolmap.create_default_bundle_adjuster(ba,config,rec).solve()
    pycolmap.ObservationManager(rec).filter_all_points3D(4.,1.5)
    connected=connected_to_existing(rec,known)
    rows=[]
    for image in list(rec.images.values()):
        if not image.has_pose:continue
        errors=[];pose=pose4(image);camera=rec.cameras[image.camera_id]
        for kp in image.points2D:
            if not kp.has_point3D() or kp.point3D_id not in rec.points3D:continue
            pc=pose[:3,:3]@rec.points3D[kp.point3D_id].xyz+pose[:3,3]
            if pc[2]>0:errors.append(float(np.linalg.norm(camera.img_from_cam(pc)-kp.xy)))
        median=float(np.median(errors)) if errors else None
        existing=image.image_id in known
        accepted=existing or (len(errors)>=30 and median is not None and median<=2 and image.image_id in connected)
        rows.append({'frame':image.name,'existing_camera':existing,'measured_tracks':len(errors),
                     'median_track_reprojection_px':median,'connected_to_existing_map':image.image_id in connected,
                     'accepted':bool(accepted)})
        if not accepted:rec.deregister_frame(image.frame_id)
    for image_id,pose in known.items():
        if not np.allclose(pose4(rec.images[image_id]),pose,rtol=0,atol=1e-8):
            raise RuntimeError('Recovery changed an existing checked camera')
    # Deleting weak frames must not disconnect retained learned components.
    connected=connected_to_existing(rec,known)
    for im in list(rec.images.values()):
        if im.has_pose and im.image_id not in connected:rec.deregister_frame(im.frame_id)
    return rec,{'solver':'verified CASPAR CUDA','termination':summary.termination_type.name,
                'existing_cameras':len(known),'new_measured_cameras':rec.num_reg_images()-len(known),
                'existing_camera_poses_unchanged':True,'per_frame':rows}


def recover_camera_model(camera, source, output, session, duration, fps, minimum=.9):
    """Run the isolated model worker only when camera coverage needs recovery."""
    root=Path(__file__).resolve().parents[2]
    python=root/'.venv_slam3r/bin/python'
    prerequisites=[python,root/'tools/SLAM3R/slam3r/models.py',root/'runs/slam3r_checkpoints.json']
    missing=[str(p) for p in prerequisites if not p.is_file()]
    if missing:
        return source,{'status':'unavailable','missing':missing}
    job=Path(output)/'learned_camera';job.mkdir()
    request={'camera':camera,'source':str(Path(source).resolve()),'output':str(Path(output).resolve()),
             'session':str(Path(session).resolve()),'duration':duration,'fps':fps,'minimum':minimum}
    save_json(job/'request.json',request)
    with (job/'worker.log').open('w') as log:
        result=subprocess.run([str(python),'-u','-m','adaptive_uav.learned_camera','--request',str(job/'request.json')],
                              stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        report={'status':'worker_failed','accepted':False,'exit_code':result.returncode,'log':str(job/'worker.log')}
        save_json(job/'recovery_report.json',report)
        return source,report
    report=json.loads((job/'recovery_report.json').read_text())
    return Path(report['sparse_source']) if report.get('accepted') else source,report


def worker(request):
    import pycolmap
    import torch
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'tools/SLAM3R'))
    from safetensors.torch import load_file
    from slam3r.models import Image2PointsModel,Local2WorldModel
    from slam3r.datasets.wild_seq import Seq_Data
    from slam3r.pipeline.recon_offline_pipeline import scene_recon_pipeline_offline
    from .combined_reconstruct import select_camera_model
    from .sensor_audit import audit_gyro
    camera=request['camera'];output=Path(request['output']);job=output/'learned_camera'
    source=Path(camera['source']).parents[3]
    rec=pycolmap.Reconstruction(camera['model'])
    # The tested adapter is for a single fixed-intrinsics phone camera.
    if len(rec.cameras)!=1:
        save_json(job/'recovery_report.json',{'status':'unsupported_calibration','accepted':False,
                  'note':'Recovery currently requires one fitted camera.'})
        return
    fitted_camera=next(iter(rec.cameras.values()))
    try:distortion=opencv_distortion(fitted_camera)
    except ValueError as error:
        save_json(job/'recovery_report.json',{'status':'unsupported_calibration','accepted':False,'note':str(error)})
        return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for learned-camera recovery')
    copied=job/'cached_matches';copied.mkdir()
    shutil.copy2(source/'database.db',copied/'database.db')
    (copied/'images').symlink_to((source/'images').resolve(),target_is_directory=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    np.random.seed(42)
    weights=json.loads((root/'runs/slam3r_checkpoints.json').read_text());models=[];identity={}
    started=time.monotonic()
    for name,cls in [('slam3r_i2p',Image2PointsModel),('slam3r_l2w',Local2WorldModel)]:
        config=json.loads(Path(weights[name]['config.json']).read_text())
        net=cls(**config);state=load_file(weights[name]['model.safetensors'])
        net.load_state_dict(state,strict=True)
        identity[name]={'path':weights[name]['model.safetensors'],'tensors_loaded_strictly':len(state),
                        'config_sha256':hashlib.sha256(Path(weights[name]['config.json']).read_bytes()).hexdigest()}
        print(name,'strictly loaded',len(state),'tensors',flush=True)
        del state;models.append(net.eval().cuda())
    names=sorted(p.name for p in (source/'images').glob('frame_*.jpg'))
    neural_images=source/'images'
    rectified=bool(np.any(np.abs(distortion)>1e-10))
    if rectified:
        neural_images=job/'rectified_images';neural_images.mkdir()
        k=fitted_camera.calibration_matrix()
        maps=cv2.initUndistortRectifyMap(k,distortion,None,k,
            (fitted_camera.width,fitted_camera.height),cv2.CV_32FC1)
        for name in names:
            rgb=cv2.imread(str(source/'images'/name))
            if rgb is None:raise FileNotFoundError(source/'images'/name)
            image=cv2.remap(rgb,*maps,interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(neural_images/name),image,[cv2.IMWRITE_JPEG_QUALITY,95])
    report={'status':'running','accepted':False,'weights':identity,'attempts':[]}
    report['neural_images_rectified']=rectified
    report['fitted_camera_model']=fitted_camera.model_name
    for clockwise in (0,90):
        attempt=job/f'rotation_{clockwise}';attempt.mkdir()
        images=neural_images
        if clockwise:
            images=attempt/'input_images';images.mkdir()
            for name in names:
                rgb=cv2.imread(str(neural_images/name))
                if rgb is None:raise FileNotFoundError(source/'images'/name)
                cv2.imwrite(str(images/name),cv2.rotate(rgb,cv2.ROTATE_90_CLOCKWISE),[cv2.IMWRITE_JPEG_QUALITY,95])
        dataset=Seq_Data(str(images),img_size=224,to_tensor=True)
        args=SimpleNamespace(device='cuda',win_r=3,num_scene_frame=8,initial_winsize=5,
             conf_thres_l2w=12.,conf_thres_i2p=1.5,num_points_save=500_000,keyframe_stride=2,
             buffer_size=60,buffer_strategy='reservoir',update_buffer_intv=1,norm_input=False,
             save_all_views=False,save_preds=True,save_for_eval=False)
        tick=time.monotonic()
        with torch.inference_mode():scene_recon_pipeline_offline(*models,dataset,args,str(attempt))
        row={'rotation_clockwise':clockwise,'inference_seconds':time.monotonic()-tick}
        try:
            points=np.load(attempt/'preds/registered_pcds.npy',mmap_mode='r')
            confidence=np.load(attempt/'preds/registered_confs.npy',mmap_mode='r')
            seeds,fit=fit_seeds(rec,names,points,confidence,clockwise)
            save_json(attempt/'seed_audit.json',fit)
            refined,tracks=verify_tracks(camera['model'],copied,seeds,attempt)
            model=attempt/'checked_model';model.mkdir();refined.write(model)
            gyro=audit_gyro(request['session'],model,request['fps'])
            rows=tracks['per_frame'];tracks['gyro']=gyro
            save_json(attempt/'tracks_audit.json',tracks)
            times=[int(Path(im.name).stem.split('_')[-1])/request['fps'] for im in refined.images.values() if im.has_pose]
            info={'start_seconds':min(times),'end_seconds':max(times),'registered_frames':len(times),
                  'full_run_selected_frames':len(names)}
            coverage=capture_coverage(info,request['duration'],request['minimum'])
            errors=[p.error for p in refined.points3D.values()]
            row.update(capture_coverage=coverage,new_measured_cameras=tracks['new_measured_cameras'],
                       gyro=gyro,median_reprojection_error_px=float(np.median(errors)) if errors else None)
            passed=(coverage['passed'] and (not gyro.get('usable') or gyro.get('passed')) and
                    refined.num_points3D()>=150 and errors and np.median(errors)<=2)
            if passed:
                container=output/'sparse_run';container.mkdir(exist_ok=True)
                folder=container/f'online_learned_{clockwise}';folder.mkdir()
                (folder/'images').symlink_to((source/'images').resolve(),target_is_directory=True)
                shutil.copy2(source/'database.db',folder/'database.db')
                target=folder/'blocks/0000/sparse/0';target.mkdir(parents=True);refined.write(target)
                save_json(folder/'current_map.json',{'binary_model':str(target.resolve())})
                save_json(container/'pipeline_manifest.json',json.loads((Path(request['source'])/'pipeline_manifest.json').read_text()))
                link_cached_candidates(request['source'],container)
                checked,_=select_camera_model(container,Path(request['session']),output,request['fps'])
                if capture_coverage(checked,request['duration'],request['minimum'])['passed']:
                    report.update(accepted=True,sparse_source=str(container.resolve()),
                                  selected_rotation_clockwise=clockwise,capture_coverage=coverage)
        except (RuntimeError,ValueError) as error:
            row['rejected_reason']=str(error)
        report['attempts'].append(row)
        save_json(job/'recovery_report.json',report)
        if report['accepted']:break
    report.update(status='accepted' if report['accepted'] else 'rejected',elapsed_seconds=time.monotonic()-started,
                  geometry_accuracy_validated=False,note='Camera coverage and measured-track consistency do not establish complete scene surfaces or metric accuracy.')
    save_json(job/'recovery_report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    worker(json.loads(parser.parse_args().request.read_text()))
