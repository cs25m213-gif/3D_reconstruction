"""Pose-conditioned pretrained depth and explicitly labelled surface completion.

Accepts any undistorted COLMAP dense workspace with images/, sparse/ and
stereo/depth_maps/. Uses the same fixed policy for every scene. Distances stay
in COLMAP units. No claim of complete or metrically accurate geometry is made.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from .hybrid_dense import camera_matrix, read_array, pose4, export_glb, make_volume, integrate
from .depth_cache_io import save_depth_cache


def save_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def remap_camera(array, source_k, target_k, shape, interpolation=cv2.INTER_NEAREST):
    """Map pinhole pixels using intrinsics, including resizing and crop offsets."""
    y, x = np.indices(shape, dtype=np.float32)
    x = (x-target_k[0, 2])/target_k[0, 0]*source_k[0, 0]+source_k[0, 2]
    y = (y-target_k[1, 2])/target_k[1, 1]*source_k[1, 1]+source_k[1, 2]
    return cv2.remap(array, x.astype(np.float32), y.astype(np.float32), interpolation)


def spatial_holdout(shape):
    y, x = np.indices(shape)
    return ((x//16 + 2*(y//16)) % 5) == 0


def fit_group_scale(depths, measured, sparse=False):
    """One scale for all cameras in a fragment; spatial blocks held out."""
    logs = []
    for d, m in zip(depths, measured):
        valid = np.isfinite(d) & (d > 0) & np.isfinite(m) & (m > 0) & ~spatial_holdout(d.shape)
        logs.append(np.log(m[valid]/d[valid])[::1 if sparse else 3])
    values = np.concatenate(logs)
    minimum = 30 if sparse else 100
    scale = float(np.exp(np.median(values))) if len(values) >= minimum else 1.
    checks = []
    for d, m in zip(depths, measured):
        valid = spatial_holdout(d.shape) & (m > 0) & (d > 0) & np.isfinite(d)
        err = np.abs(scale*d[valid]-m[valid])/m[valid]
        checks.append({'pixels': int(valid.sum()), 'training_pixels':len(values),
                       'scale_fit_valid':len(values)>=minimum,
                       'median': float(np.median(err)) if len(err) else None,
                       'p90': float(np.percentile(err, 90)) if len(err) else None})
    return scale, checks


def validation_anchors(ws, name, model, image, intrinsic, shape):
    """Separate stereo measurements from sparse SfM depth used for scale checks.

    Sparse feature depths remain validation anchors. They are never advertised
    as dense measured depth or integrated as a stereo-measured surface.
    """
    path=ws/'stereo/depth_maps'/(name+'.geometric.bin')
    camera=model.cameras[image.camera_id]
    if path.is_file():
        raw=read_array(path).astype(np.float32)
        raw[~np.isfinite(raw)|(raw<0)]=0
        measured=remap_camera(raw,camera_matrix(camera,raw.shape),intrinsic,shape)
        return measured,measured.copy(),'geometric_stereo'
    measured=np.zeros(shape,np.float32);anchors=measured.copy()
    transform=intrinsic@np.linalg.inv(camera.calibration_matrix());pose=pose4(image)
    for point in image.points2D:
        if not point.has_point3D() or point.point3D_id not in model.points3D:continue
        landmark=model.points3D[point.point3D_id]
        if not np.isfinite(landmark.error) or landmark.error>2 or landmark.track.length()<3:continue
        depth=float((pose[:3,:3]@landmark.xyz+pose[:3,3])[2])
        uv=transform@np.r_[point.xy,1.];x,y=np.rint(uv[:2]/uv[2]).astype(int)
        if depth>0 and 0<=x<shape[1] and 0<=y<shape[0]:
            # Multiple features landing in one prediction pixel count once.
            anchors[y,x]=depth if anchors[y,x]<=0 else min(anchors[y,x],depth)
    return measured,anchors,'triangulated_sfm_tracks'


def correct_local_scale(prior, measured):
    """Smooth measured residuals; spatial test blocks never enter the correction.

    A bounded residual field compensates for a model's spatial scale distortion.
    It is not a measurement in regions lacking measured anchors.
    """
    hold = spatial_holdout(prior.shape)
    valid = (prior > 0) & np.isfinite(prior) & (measured > 0) & np.isfinite(measured)
    train = valid & ~hold
    test = valid & hold
    if train.sum() < 400 or test.sum() < 100:
        return prior.copy(), {'used':False, 'reason':'insufficient spatial anchors'}
    residual = np.zeros(prior.shape, np.float32)
    residual[train] = np.clip(np.log(measured[train]/prior[train]), -np.log(2), np.log(2))
    sigma = max(4., .05*max(prior.shape))
    weight = cv2.GaussianBlur(train.astype(np.float32), (0,0), sigma)
    field = cv2.GaussianBlur(residual, (0,0), sigma)/np.maximum(weight, 1e-6)
    # Smoothly shrink correction in regions with no nearby measured evidence.
    field *= np.minimum(weight/.02,1.)
    corrected = (prior*np.exp(field)).astype(np.float32)
    before = np.abs(prior[test]-measured[test])/measured[test]
    after = np.abs(corrected[test]-measured[test])/measured[test]
    used = bool(np.median(after) < .8*np.median(before) and
                np.percentile(after,90) <= np.percentile(before,90))
    return (corrected if used else prior.copy()), {'used':used, 'test_pixels':int(test.sum()),
        'before_median':float(np.median(before)), 'after_median':float(np.median(after)),
        'before_p90':float(np.percentile(before,90)), 'after_p90':float(np.percentile(after,90)),
        'note':'Spatial holdout selects whether correction is useful; it is not independent ground truth.'}


def source_groups(names, by_name, points, size=4):
    """Use covisibility and parallax, retaining previous frames as context."""
    tracks = {n: {p.point3D_id for p in by_name[n].points2D if p.has_point3D()} for n in names}
    centers = {n: by_name[n].cam_from_world().inverse().translation for n in names}
    groups, assigned = [], set()
    for name in names:
        if name in assigned:
            continue
        def score(other):
            common = sorted(tracks[other] & tracks[name])
            if not common:
                return 0.
            sampled = common[::max(1,len(common)//48)]
            xyz = np.asarray([points[i].xyz for i in sampled])
            a, b = xyz-centers[name], xyz-centers[other]
            cosine = np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-9)
            angle = float(np.median(np.degrees(np.arccos(np.clip(cosine,-1,1)))))
            return len(common)*min(angle/3.,1.)*min(35./max(angle,1e-9),1.)
        ranked = sorted((n for n in names if n != name), key=score, reverse=True)
        # Pure rotations cannot set the input-pose scale; skip duplicate centres.
        others = [n for n in ranked if np.linalg.norm(centers[n]-centers[name]) > 1e-6]
        group = [name] + others[:size-1]
        if len(group) < 2:
            raise ValueError('Pose-conditioned depth needs at least two translated cameras')
        targets = [n for n in group if n not in assigned]
        assigned.update(targets)
        groups.append((group, targets))
    return groups


def infer(ws, out, names, model, by_name, args, only_names=None):
    import torch
    try:
        from depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        if exc.name != 'depth_anything_3':
            raise
        import sys
        source = Path(__file__).resolve().parents[2]/'tools/Depth-Anything-3/src'
        if not (source/'depth_anything_3').is_dir():
            raise
        sys.path.insert(0, str(source))
        from depth_anything_3.api import DepthAnything3
    torch.set_num_threads(args.threads)
    predictor = DepthAnything3.from_pretrained(args.model).to('cuda' if torch.cuda.is_available() else 'cpu')
    cache = out/'depth_cache'
    cache.mkdir(exist_ok=True)
    groups = getattr(args,'depth_groups',None) or source_groups(names, by_name, model.points3D, args.views)
    for index, (group, targets) in enumerate(groups):
        if only_names is not None:
            targets = [n for n in targets if n in only_names]
            if not targets:
                continue
        if all((cache/(n+'.npz')).exists() for n in targets):
            continue
        images = [str(ws/'images'/n) for n in group]
        poses = np.stack([pose4(by_name[n]) for n in group]).astype(np.float32)
        ks = np.stack([model.cameras[by_name[n].camera_id].calibration_matrix() for n in group]).astype(np.float32)
        started = time.monotonic()
        with torch.inference_mode():
            prediction = predictor.inference(images, extrinsics=poses, intrinsics=ks,
                align_to_input_ext_scale=True, process_res=args.resolution,
                process_res_method='upper_bound_resize', ref_view_strategy='first')
        depths = prediction.depth.astype(np.float32)
        measured = [];anchors=[];origins=[];object_measured=[]
        for i, n in enumerate(group):
            m,a,origin=validation_anchors(ws,n,model,by_name[n],prediction.intrinsics[i],depths[i].shape)
            object_measured.append(m.copy())
            if getattr(args, 'semantic_dir', None):
                from .semantic_scene import load_frame
                classes, _, _ = load_frame(args.semantic_dir, n, prediction.intrinsics[i], depths[i].shape)
                m[classes > 0] = 0
                a[classes > 0] = 0
            measured.append(m);anchors.append(a);origins.append(origin)
        scale, checks = fit_group_scale(depths, anchors,sparse=all(x=='triangulated_sfm_tracks' for x in origins))
        for i, n in enumerate(group):
            if n not in targets:
                continue
            depth = depths[i]*scale
            depth[~np.isfinite(depth) | (depth <= 0)] = 0
            checks[i]['anchor_source']=origins[i]
            if origins[i]=='triangulated_sfm_tracks':
                # Do not extrapolate a whole camera image from one or two
                # circularly fitted feature depths.
                check_valid=checks[i]['scale_fit_valid'] and checks[i]['pixels']>=10 and int((anchors[i]>0).sum())>=30
                if not check_valid:checks[i]['median']=checks[i]['p90']=None
            conf = prediction.conf[i] if prediction.conf is not None else np.ones_like(depth)
            temporary = cache/(n+'.tmp.npz')
            save_depth_cache(temporary, depth=depth, confidence=conf, measured=measured[i],object_measured=object_measured[i],
                anchors=anchors[i],anchor_source=origins[i],
                K=prediction.intrinsics[i], P=pose4(by_name[n]), rgb=prediction.processed_images[i],
                scale=scale, context=np.array(group), validation=json.dumps(checks[i]))
            temporary.replace(cache/(n+'.npz'))
        print(f'infer {index+1}/{len(groups)}: {len(targets)} new views; {time.monotonic()-started:.1f}s; scale {scale:.3g}', flush=True)
        del prediction
    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_student(ws, out, names, model, by_name, args):
    """Use the trained visual-scale student; return views requiring DA3 fallback."""
    import torch
    from .depth_student import DepthStudent

    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.student_checkpoint, map_location='cpu', weights_only=False)
    architecture = checkpoint.get('architecture', {'channels':48, 'planes':24})
    predictor = DepthStudent(**architecture).to(device).eval()
    predictor.load_state_dict(checkpoint['model'])
    cache = out/'depth_cache'
    cache.mkdir(exist_ok=True)
    groups = getattr(args,'depth_groups',None) or source_groups(names, by_name, model.points3D, args.views)
    fallback = set()
    height, width = 192, 256  # Resolution used by the saved checkpoint.
    for group, targets in groups:
        for name in targets:
            if (cache/(name+'.npz')).exists():
                continue
            context = [name] + [n for n in group if n != name][:2]
            rows = []
            for n in context:
                bgr = cv2.imread(str(ws/'images'/n), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(ws/'images'/n)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ih, iw = rgb.shape[:2]
                k = model.cameras[by_name[n].camera_id].calibration_matrix().astype(np.float32)
                k[0] *= width/iw
                k[1] *= height/ih
                rows.append((cv2.resize(rgb, (width,height), interpolation=cv2.INTER_AREA), k, pose4(by_name[n]).astype(np.float32)))
            while len(rows) < 3:
                rows.append(rows[-1])
            intrinsic = rows[0][1]
            measured, anchors, origin = validation_anchors(ws, name, model, by_name[name], intrinsic, (height,width))
            valid_anchor = anchors[np.isfinite(anchors)&(anchors>0)]
            if len(valid_anchor) < 30:
                fallback.add(name)
                print(f'student {name}: insufficient scale anchors; DA3 fallback', flush=True)
                continue
            scale = float(np.median(valid_anchor))
            if not np.isfinite(scale) or scale <= 0:
                fallback.add(name)
                continue
            rgb_tensor = torch.from_numpy(np.stack([r[0].transpose(2,0,1) for r in rows]).copy()).float()[None].to(device)/255.
            k_tensor = torch.from_numpy(np.stack([r[1] for r in rows]).copy())[None].to(device)
            poses = np.stack([r[2] for r in rows])
            poses[:,:3,3] /= scale
            p_tensor = torch.from_numpy(poses)[None].to(device)
            # BF16 measurably lowers this model's accuracy; GEOSTREAM_STUDENT_PRECISION=float32 avoids it.
            half = device.type=='cuda' and os.environ.get('GEOSTREAM_STUDENT_PRECISION','bf16')!='float32'
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=half):
                normalized_depth, confidence = predictor(rgb_tensor, k_tensor, p_tensor)
            depth = normalized_depth[0,0].float().cpu().numpy()*scale
            conf = confidence[0,0].float().cpu().numpy()
            if not np.isfinite(depth).all() or not np.isfinite(conf).all():
                fallback.add(name)
                print(f'student {name}: nonfinite output; DA3 fallback', flush=True)
                continue
            # Keep the same spatial holdout used by the DA3 backend.
            correction, checks = fit_group_scale([depth], [anchors], sparse=origin=='triangulated_sfm_tracks')
            check = checks[0]
            reliable = (check['scale_fit_valid'] and check['pixels'] >= 10 and
                        check['median'] is not None and check['median'] <= .2 and
                        check['p90'] is not None and check['p90'] <= .5)
            if not reliable:
                fallback.add(name)
                print(f'student {name}: geometric validation failed {check}; DA3 fallback', flush=True)
                continue
            depth = (depth*correction).astype(np.float32)
            if getattr(args, 'semantic_dir', None):
                from .semantic_scene import load_frame
                classes, _, _ = load_frame(args.semantic_dir, name, intrinsic, depth.shape)
                measured[classes > 0] = 0
                anchors[classes > 0] = 0
            check['anchor_source'] = origin
            temporary = cache/(name+'.tmp.npz')
            save_depth_cache(temporary, depth=depth, confidence=conf, measured=measured,
                object_measured=measured.copy(), anchors=anchors, anchor_source=origin,
                K=intrinsic, P=pose4(by_name[name]), rgb=rows[0][0],
                scale=correction, context=np.array(context), validation=json.dumps(check),
                depth_backend='student')
            temporary.replace(cache/(name+'.npz'))
            print(f'student {name}: passed geometric validation; median {check["median"]:.3f}', flush=True)
    del predictor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return fallback


def recurrent_updates(out,groups,checkpoint):
    """Update cached target depths; sparse held-out anchors gate each change."""
    import torch
    from .recurrent_depth import RecurrentDepthUpdater,warp_source_depth
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    predictor=RecurrentDepthUpdater().to(device).eval();predictor.load_state_dict(state['model'])
    cache=out/'depth_cache';rows=[]
    for group,targets in groups:
        for name in targets:
            path=cache/(name+'.npz')
            if not path.is_file():continue
            with np.load(path) as data:target={key:data[key].copy() for key in data.files}
            original=target['depth'].astype(np.float32)
            anchors=np.where(target['measured']>0,target['measured'],target['anchors']).astype(np.float32)
            scale_values=anchors[np.isfinite(anchors)&(anchors>0)]
            if len(scale_values)<30:continue
            scale=float(np.median(scale_values))
            previous=torch.from_numpy((original/scale)[None,None]).to(device)
            confidence=torch.from_numpy(target['confidence'].astype(np.float32)[None,None]).to(device)
            rgb=torch.from_numpy(target['rgb'].astype(np.float32).transpose(2,0,1)[None]).to(device)/255.
            intrinsic=target['K'];pose=target['P'];hidden=None;updates=0
            for source in group:
                if source==name or not (cache/(source+'.npz')).is_file():continue
                with np.load(cache/(source+'.npz')) as data:
                    src_depth=data['depth'].astype(np.float32)
                    warped=warp_source_depth(src_depth,data['K'],data['P'],intrinsic,pose,original.shape)
                    warped_conf=warp_source_depth(src_depth,data['K'],data['P'],intrinsic,pose,
                        original.shape,payload=data['confidence'].astype(np.float32))
                if (warped>0).mean()<.03:continue
                new=torch.from_numpy((warped/scale)[None,None]).to(device)
                new_conf=torch.from_numpy(warped_conf[None,None]).to(device)
                with torch.inference_mode():previous,confidence,hidden=predictor(previous,new,confidence,new_conf,rgb,hidden)
                updates+=1
            if not updates:continue
            candidate=previous[0,0].float().cpu().numpy()*scale
            candidate[~np.isfinite(candidate)]=0
            candidate[target['measured']>0]=target['measured'][target['measured']>0]
            hold=spatial_holdout(original.shape)&(anchors>0)&np.isfinite(anchors)
            if hold.sum()<10:
                rows.append({'target':name,'updates':updates,'accepted':False,'reason':'insufficient held-out anchors'})
                continue
            before=np.abs(original[hold]-anchors[hold])/anchors[hold]
            after=np.abs(candidate[hold]-anchors[hold])/anchors[hold]
            accepted=(np.median(after)<np.median(before) and
                np.percentile(after,90)<=1.05*np.percentile(before,90))
            rows.append({'target':name,'updates':updates,'accepted':bool(accepted),
                'before_median':float(np.median(before)),'after_median':float(np.median(after))})
            if accepted:
                target['depth_before_recurrent']=target['depth'].copy()
                target['depth']=candidate.astype(np.float32)
                target['recurrent_updates']=np.array(updates)
                temporary=cache/(name+'.recurrent.tmp.npz')
                np.savez_compressed(temporary,**target);temporary.replace(path)
    report={'attempted':len(rows),'accepted':sum(row['accepted'] for row in rows),
        'neural_updates':sum(row['updates'] for row in rows),
        'internal_anchor_gate_only':True,'independent_accuracy_validated':False,'frames':rows}
    save_json(out/'recurrent_report.json',report)
    del predictor
    if device.type=='cuda':torch.cuda.empty_cache()
    return report


def evidence_masks(frame, others, tolerance=.05):
    """Measured-source contradiction and prior consistency with occlusion handling."""
    d, k, p = frame['prior'], frame['K'], frame['P']
    y, x = np.indices(d.shape)
    cam = np.stack(((x-k[0, 2])/k[0, 0]*d, (y-k[1, 2])/k[1, 1]*d, d), -1).reshape(-1, 3)
    world = (cam-p[:3, 3])@p[:3, :3]
    center = -p[:3, :3].T@p[:3, 3]
    agree = np.zeros(d.size, np.uint8)
    measured_agree = np.zeros(d.size, np.uint8)
    conflict = np.zeros(d.size, np.uint8)
    for other in others:
        q, ko = other['P'], other['K']
        oc = -q[:3, :3].T@q[:3, 3]
        if np.linalg.norm(center-oc) < .025*np.median(d[d > 0]):
            continue
        pc = world@q[:3, :3].T+q[:3, 3]
        u = np.rint(pc[:, 0]/np.maximum(pc[:, 2], 1e-9)*ko[0, 0]+ko[0, 2]).astype(np.int64)
        v = np.rint(pc[:, 1]/np.maximum(pc[:, 2], 1e-9)*ko[1, 1]+ko[1, 2]).astype(np.int64)
        h, w = other['prior'].shape
        ok = (d.ravel() > 0) & (pc[:, 2] > 0) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
        ix = np.flatnonzero(ok)
        measured = other['measured'][v[ix], u[ix]]
        prior = other['prior'][v[ix], u[ix]]
        reference = np.where(measured > 0, measured, prior)
        relative = (pc[ix, 2]-reference)/np.maximum(reference, 1e-9)
        matched = (reference > 0) & (np.abs(relative) < tolerance)
        agree[ix] += matched
        measured_agree[ix] += matched & (measured > 0)
        # Behind a measured surface is occluded, not a contradiction.
        conflict[ix] += (measured > 0) & (relative < -tolerance)
    return ((agree >= 2) | (measured_agree >= 1)).reshape(d.shape), (conflict > 0).reshape(d.shape)


def modest_cleanup(mesh):
    """Remove tiny isolated fragments; never scale cutoff with scene size."""
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if len(labels):
        mesh.remove_triangles_by_mask(counts[labels] < 50)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def restrict_depth(frame, max_depth=None, near_percentile=None):
    """Restrict camera-axis depth for all evidence tiers, retaining raw anchors.

    Percentile is a per-view relative cutoff, not a metric distance estimate.
    """
    limits = [frame['depth_limit']]
    observed = frame['measured']
    observed = observed[np.isfinite(observed) & (observed > 0)]
    if max_depth is not None:
        if not np.isfinite(max_depth) or max_depth <= 0:
            raise ValueError('max_depth must be finite and positive (COLMAP units)')
        limits.append(max_depth)
    if near_percentile is not None:
        if not np.isfinite(near_percentile) or not 0 < near_percentile <= 100:
            raise ValueError('near_percentile must be in (0, 100]')
        if not len(observed):
            raise ValueError('Relative near filtering needs measured depth anchors')
        limits.append(float(np.percentile(observed, near_percentile)))
    limit = min(limits)
    valid = np.isfinite(frame['depth']) & (frame['depth'] > 0)
    removed = valid & (frame['depth'] > limit)
    frame['depth'] = np.where(valid & ~removed, frame['depth'], 0).astype(np.float32)
    frame['tier'] = np.where(valid & ~removed, frame['tier'], 0).astype(np.uint8)
    frame['depth_limit'] = limit
    return {'limit_visual_units': float(limit), 'removed_pixels': int(removed.sum()),
            'removed_fraction': float(removed.sum()/max(1, valid.sum()))}


def fuse_scale_groups(frames, voxel_fraction=.01, measured_repeat=2):
    """Bound voxel resolution per depth range, preserving a common world frame.

    Frame-depth ranges differ by at most a factor four within each submap.
    This avoids distant-depth clipping and enormous allocation at the nearest
    surface's resolution. Different-resolution submap seams remain possible.
    """
    import gc
    import open3d as o3d
    frames=[f for f in frames if np.any(f['depth']>0)]
    if not frames:raise ValueError('No validated depth frames to fuse')
    medians = np.array([np.median(f.get('anchors',f['measured'])[f.get('anchors',f['measured'])>0]) for f in frames])
    level_ids = np.floor(np.log(medians/medians.min())/np.log(4.)).astype(int)
    result = {n:o3d.geometry.TriangleMesh() for n in ('supported','completed')}
    levels=[]
    for progress, level in enumerate(np.unique(level_ids),1):
        selected = [f for f,lev in zip(frames,level_ids) if lev == level]
        scale = float(np.median(medians[level_ids==level]))
        trunc = max(f['depth_limit'] for f in selected)
        for name, tiers in (('supported',[2]),('completed',[2,3])):
            volume = make_volume(voxel_fraction*scale)
            integrate(volume,selected,[1],trunc,repeat=measured_repeat)
            integrate(volume,selected,tiers,trunc)
            piece = modest_cleanup(volume.extract_triangle_mesh())
            result[name] += piece
            del volume,piece
            gc.collect()
        levels.append({'frames':len(selected),'voxel':voxel_fraction*scale,'depth_limit':trunc})
        print(f'fusion scale group {progress}/{len(np.unique(level_ids))}: {len(selected)} frames, voxel {voxel_fraction*scale:.5g}',flush=True)
    return result,levels


def ray_audit(meshes, frames, out, workspace=None, max_views=16, semantic_directory=None):
    import open3d as o3d
    from html import escape
    if workspace is not None:
        import pycolmap
        model = pycolmap.Reconstruction(workspace/'sparse')
        by_name = {f['name']:f for f in frames}
        # Evaluate unselected registered cameras too; do not restrict coverage
        # reporting to the references that supplied depth to fusion.
        for im in model.images.values():
            if im.name in by_name or not (workspace/'images'/im.name).is_file():
                continue
            camera = model.cameras[im.camera_id]
            scale = 504/max(camera.height,camera.width)
            shape = (round(camera.height*scale),round(camera.width*scale))
            rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(workspace/'images'/im.name)),
                              (shape[1],shape[0])),cv2.COLOR_BGR2RGB)
            by_name[im.name] = {'name':im.name,'K':camera_matrix(camera,shape),'P':pose4(im),
                'prior':np.zeros(shape,np.float32),'measured':np.zeros(shape,np.float32),'rgb':rgb}
        frames = [by_name[n] for n in sorted(by_name)]
    if semantic_directory:
        from .semantic_scene import load_frame
        for f in frames:
            classes, _, _ = load_frame(semantic_directory, f['name'], f['K'], f['prior'].shape)
            f['semantic_excluded'] = classes > 0
            f['semantic_sky'] = classes == 1
            f['measured'] = np.where(classes > 0, 0, f['measured'])
    scenes = {}
    for name, mesh in meshes.items():
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        scenes[name] = scene
    rows, cards = [], []
    for i in np.unique(np.linspace(0, len(frames)-1, min(max_views, len(frames)), dtype=int)):
        f = frames[i]
        h, w = f['prior'].shape
        y, x = np.indices((h, w))
        k, p = f['K'], f['P']
        directions = np.stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones((h, w))), -1)@p[:3, :3]
        origins = np.broadcast_to(-p[:3, :3].T@p[:3, 3], directions.shape)
        rays = o3d.core.Tensor(np.concatenate((origins, directions), -1).astype(np.float32))
        panels = [cv2.cvtColor(f['rgb'], cv2.COLOR_RGB2BGR)]
        row = {'frame': f['name'], 'meshes': {}}
        for name, scene in scenes.items():
            hits = scene.cast_rays(rays)
            z = hits['t_hit'].numpy()
            valid = np.isfinite(z) & (z > 0)
            m = f['measured']
            overlap = valid & (m > 0)
            errors = np.abs(z[overlap]-m[overlap])/m[overlap]
            evaluated = ~f.get('semantic_excluded', np.zeros(valid.shape, bool))
            static_coverage = float(valid[evaluated].mean()) if evaluated.any() else None
            sky = f.get('semantic_sky', np.zeros(valid.shape, bool))
            row['meshes'][name] = {'ray_hit_fraction': static_coverage,
                'all_pixel_ray_hit_fraction': float(valid.mean()),
                'evaluated_pixels': int(evaluated.sum()),
                'semantic_excluded_fraction': float((~evaluated).mean()),
                'sky_ray_hit_fraction': float(valid[sky].mean()) if sky.any() else None,
                'measured_pixels': int(overlap.sum()),
                'measured_depth_median_error': float(np.median(errors)) if len(errors) else None,
                'measured_depth_p90_error': float(np.percentile(errors, 90)) if len(errors) else None,
                'measured_depth_under_5pct': float((errors < .05).mean()) if len(errors) else None}
            mesh = meshes[name]
            ids = hits['primitive_ids'].numpy()[valid].astype(np.int64)
            uv = hits['primitive_uvs'].numpy()[valid]
            weight = np.column_stack((1-uv.sum(1), uv))
            colors = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)[ids]]
            canvas = np.zeros((h, w, 3), np.uint8)
            canvas[valid] = np.clip((colors*weight[..., None]).sum(1)*255, 0, 255).astype(np.uint8)[..., ::-1]
            label=f'{name}: {static_coverage:.0%}' if static_coverage is not None else f'{name}: no static pixels'
            if semantic_directory:label+=' static'
            cv2.putText(canvas, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 0), 1)
            panels.append(canvas)
        filename = f'compare_{i:04d}.jpg'
        cv2.imwrite(str(out/filename), np.hstack(panels))
        cards.append(f'<p>{escape(f["name"])}</p><img src="{filename}" style="max-width:100%">')
        rows.append(row)
    (out/'coverage.html').write_text('<!doctype html><meta charset="utf-8"><title>Surface coverage</title>'
        '<body style="background:#111;color:white;font:16px sans-serif"><h1>Rendered surface coverage</h1>'
        '<p>Source, baseline when available, supported mesh, and completed mesh. '
        'Completed includes learned surfaces. Ray hits are not proof of geometric accuracy.</p>'+
        ('<p>Coverage evaluates static pixels only, excluding predicted sky, people and animals.</p>' if semantic_directory else '')+''.join(cards))
    return {'registered_views_considered':len(frames), 'sampled_views': rows,
        'coverage_domain': 'static pixels only; sky/person/animal excluded' if semantic_directory else 'all image pixels',
        'mean_ray_hit': {name: float(np.mean([r['meshes'][name]['ray_hit_fraction'] for r in rows
            if r['meshes'][name]['ray_hit_fraction'] is not None])) if any(r['meshes'][name]['ray_hit_fraction'] is not None for r in rows) else 0. for name in scenes}}


def geometry_agreement(audit):
    values=[r['meshes']['completed']['measured_depth_under_5pct']
            for r in audit['sampled_views'] if r['meshes']['completed']['measured_pixels'] >= 100]
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def fuse(ws, out, names, args):
    import open3d as o3d
    frames = []
    observations = []
    for name in names:
        with np.load(out/'depth_cache'/(name+'.npz')) as data:
            f = {key: data[key] for key in ('depth', 'confidence', 'measured', 'K', 'P', 'rgb')}
            f['anchors']=data['anchors'] if 'anchors' in data else f['measured'].copy()
            if 'object_measured' in data:f['object_measured']=data['object_measured']
            f['anchor_source']=str(data['anchor_source']) if 'anchor_source' in data else 'geometric_stereo'
            f['validation'] = json.loads(str(data['validation']))
            f['context'] = data['context'].tolist()
        f['name'], f['prior'] = name, f.pop('depth')
        if getattr(args, 'semantic_dir', None):
            from .semantic_scene import filter_frame
            observations.extend(filter_frame(f, args.semantic_dir, getattr(args, 'video_fps', None)))
        # Sparse SfM tracks are valid local scale anchors even when PatchMatch
        # produced no dense map for this camera. Keep them out of the measured
        # TSDF tier: they correct the prior only after the spatial holdout gate.
        f['prior'], f['local_correction'] = correct_local_scale(f['prior'], f['anchors'])
        if f['local_correction']['used']:
            f['validation'] = {'pixels':f['local_correction']['test_pixels'],
                'median':f['local_correction']['after_median'], 'p90':f['local_correction']['after_p90']}
        frames.append(f)
    by_name = {f['name']: f for f in frames}
    measured_samples = [f['measured'][f['measured'] > 0][::11] for f in frames]
    measured_samples = [sample for sample in measured_samples if len(sample)]
    anchor_samples = [f['anchors'][f['anchors'] > 0][::11] for f in frames]
    anchor_samples = [sample for sample in anchor_samples if len(sample)]
    # A sparse-track-only run can establish a visual-unit TSDF scale, but its
    # geometry has no independently measured depth and must fail that quality gate.
    scale_samples = measured_samples or anchor_samples
    if not scale_samples:
        raise ValueError('No valid stereo or sparse-track anchors for scale or TSDF settings')
    median = float(np.median(np.concatenate(scale_samples)))
    if not np.isfinite(median) or median <= 0:
        raise ValueError('No valid stereo or sparse-track anchors for scale or TSDF settings')
    rows = []
    depth_limits = []
    for i, f in enumerate(frames):
        neighbours = list(dict.fromkeys(f['context']+[n for n in names[max(0,i-3):i+4]]))
        support, conflict = evidence_masks(f, [by_name[n] for n in neighbours if n != f['name'] and n in by_name])
        d, m, conf = f['prior'], f['measured'], f['confidence']
        observed = f['anchors'][f['anchors'] > 0]
        depth_limit = 3*float(np.percentile(observed,95)) if len(observed) else 3*median
        depth_limits.append(depth_limit)
        f['depth_limit'] = depth_limit
        confidence_ok = np.isfinite(conf) & (conf >= np.percentile(conf[np.isfinite(conf)], 5))
        # Large jumps at object boundaries must not be connected by completion.
        from .hybrid_dense import edge_invalid
        candidate = (m <= 0) & (d > 0) & (d < depth_limit) & confidence_ok & ~edge_invalid(d, .08) & ~conflict
        validated = f['validation']['median']
        if validated is None or validated > .2:
            candidate[:] = False
        tier = np.zeros(d.shape, np.uint8)
        tier[candidate] = 3
        tier[candidate & support] = 2
        tier[m > 0] = 1
        f['tier'] = tier
        f['depth'] = np.where(m > 0, m, np.where(candidate, d, 0)).astype(np.float32)
        near_filter = None
        if getattr(args, 'max_depth', None) is not None or getattr(args, 'near_percentile', None) is not None:
            near_filter = restrict_depth(f, args.max_depth, args.near_percentile)
        rows.append({'frame': f['name'], 'measured': float((tier == 1).mean()),
                     'supported': float((tier == 2).mean()), 'inferred': float((tier == 3).mean()),
                     'contradicted': float(conflict.mean()), 'validation': f['validation'],
                     'local_correction': f['local_correction'], 'depth_limit':f['depth_limit'],
                     'anchor_source':f['anchor_source'],'validation_anchor_pixels':int((f['anchors']>0).sum()),
                     'near_filter': near_filter})
    visibility = None
    if getattr(args, 'global_visibility', False):
        from .global_visibility import guard_frames
        visibility = guard_frames(frames, max_references=getattr(args, 'visibility_references', 24))
        save_json(out/'global_visibility_report.json', visibility)
        print(f"Global visibility: rejected {visibility['mean_rejected_fraction']:.1%} of candidate depth pixels", flush=True)
    # Coverage must describe the retained pixels after the global gate.
    for row, frame in zip(rows, frames):
        for key, tier in (('measured', 1), ('supported', 2), ('inferred', 3)):
            row[key] = float((frame['tier'] == tier).mean())
    measured_repeat=getattr(args,'measured_repeat',None)
    if measured_repeat is None:
        if getattr(args,'include_unmeasured',False):
            observed=sum(int((f['tier']==1).sum()) for f in frames)
            learned=sum(int((f['tier']>=2).sum()) for f in frames)
            # Adding many learned views must not silently increase their
            # influence relative to unchanged stereo evidence. This is a
            # bounded priority heuristic, not calibrated sensor covariance.
            measured_repeat=min(32,max(2,int(np.ceil(4*learned/max(1,observed)))))
        else:
            measured_repeat=2
    fused, levels = fuse_scale_groups(frames, voxel_fraction=getattr(args, 'voxel_fraction', .01),measured_repeat=measured_repeat)
    meshes = {}
    baseline = ws/'scene_hybrid_mesh.ply'
    if not baseline.is_file():
        baseline = ws/'scene_mesh.ply'
    if baseline.is_file():
        meshes['baseline'] = o3d.io.read_triangle_mesh(str(baseline))
    for name, mesh in fused.items():
        if not len(mesh.triangles):
            raise RuntimeError(f'{name}: no mesh surface extracted')
        o3d.io.write_triangle_mesh(str(out/f'scene_{name}.ply'), mesh)
        export_glb(mesh, out/f'scene_{name}.glb')
        meshes[name] = mesh
    if getattr(args, 'semantic_dir', None):
        from .semantic_scene import write_observations
        write_observations(out, observations)
    audit = ray_audit(meshes, frames, out, ws, max_views=getattr(args, 'audit_views', 16),
                      semantic_directory=getattr(args, 'semantic_dir', None))
    rejected = [r['frame'] for r in rows if r['validation']['median'] is None or r['validation']['median'] > .2]
    issues = []
    if len(rejected) > .25*len(rows):
        issues.append('Learned and measured depth disagree in over 25% of views. Inspect calibration, poses, source depth and model predictions; completion was gated for those views.')
    if audit['mean_ray_hit']['completed'] < .9:
        issues.append('Completed mesh covers less than 90% of sampled image rays.')
    if sum(r['inferred'] for r in rows) > 0:
        issues.append('Completed mesh includes surfaces without independent measured depth confirmation.')
    agreement=geometry_agreement(audit)
    inconsistent=agreement is None or agreement < .8
    if inconsistent:
        issues.append('Rendered mesh fails measured-depth consistency: fewer than 80% of overlapping pixels agree within 5% depth, or insufficient checks exist. Do not accept high ray coverage as reconstruction success.')
    report = {'status': 'complete', 'model': args.model, 'frames': len(frames),
        'near_filter_policy': {'max_depth_visual_units': getattr(args, 'max_depth', None),
                              'per_view_percentile': getattr(args, 'near_percentile', None)},
        'coverage_mean': {k: float(np.mean([r[k] for r in rows])) for k in ('measured','supported','inferred')},
        'geometry_accuracy_validated': False, 'units': 'COLMAP visual units',
        'tsdf_scale_anchor_source': 'stereo_depth' if measured_samples else 'sparse_tracks_only',
        'quality_status': ('inconsistent_geometry' if inconsistent else
                           'inconsistent_depth' if len(rejected) > .25*len(rows) else
                           'high_rendered_coverage_unvalidated' if audit['mean_ray_hit']['completed'] >= .9 else
                           'incomplete'), 'per_frame': rows, 'surface_audit': audit,
        'rejected_prior_frames': rejected, 'issues': issues,
        'fusion_scale_groups':levels,
        'global_visibility':visibility,
        'fusion_policy': {'voxel_fraction':getattr(args, 'voxel_fraction', .01),
                          'measured_repeat':measured_repeat,
                          'audit_views':getattr(args, 'audit_views', 16),
                          'visibility_references':getattr(args, 'visibility_references', 24)},
        'mean_measured_depth_agreement_under_5pct':agreement,
        'semantic_filter': {'enabled': bool(getattr(args, 'semantic_dir', None)),
            'directory': str(args.semantic_dir) if getattr(args, 'semantic_dir', None) else None,
            'object_observations': len(observations),
            'positioned_observations': sum(r.get('position') is not None for r in observations)},
        'note': 'Inferred surfaces are learned estimates. Source and target COLMAP depths are correlated. '
                'Coverage describes retained depth pixels and rendered camera views, not whole-scene completeness.'}
    save_json(out/'multiview_report.json', report)
    import shutil
    shutil.copy2(out/'scene_completed.ply', out/'scene_mesh.ply')
    save_json(out/'dense_metrics.json', {'source_model': str(ws/'sparse'),
        'quality_warning': ('QUALITY CHECK FAILED: rendered geometry disagrees with measured depth. '
                            if inconsistent else '')+
                           'Completed mesh includes learned inferred surfaces. Geometry accuracy is unvalidated.'})
    from .mesh_viewer import write_mesh_viewer
    tool_root = Path(os.environ.get('GEOSTREAM_TOOLS', str(Path(__file__).resolve().parents[2]/'tools')))
    assets = args.three_assets or (tool_root/'three_0.169.0')
    if (ws/'GLTFLoader_local.js').exists():
        assets = ws
    if not (assets/'GLTFLoader_local.js').exists() and (ws/'GLTFLoader.js').exists():
        # Keep a local asset copy with the filename expected by the standard viewer.
        assets = out/'viewer_assets'
        assets.mkdir(exist_ok=True)
        for n in ('three.module.js','PLYLoader.js','OrbitControls.js','BufferGeometryUtils.js'):
            shutil.copy2(ws/n, assets/n)
        shutil.copy2(ws/'GLTFLoader.js', assets/'GLTFLoader_local.js')
    write_mesh_viewer(out, assets)
    page_path = out/'mesh_viewer.html'
    page = page_path.read_text().replace('</header>',
        '<a href="coverage.html">Rendered comparison</a> '
        '<a href="supported_viewer.html">Supported mesh</a> '
        '<a href="multiview_report.json">Depth evidence report</a></header>')
    page_path.write_text(page)
    supporting = page.replace('scene_mesh.ply', 'scene_supported.ply').replace(
        'Completed mesh includes learned inferred surfaces. Geometry accuracy is unvalidated.',
        'Measured and cross-view-supported depth. Geometry accuracy is unvalidated.')
    for key, values in (('mesh_faces', 'triangles'), ('mesh_vertices', 'vertices')):
        supporting = supporting.replace(f'"{key}": {len(getattr(meshes["completed"], values))}',
                                        f'"{key}": {len(getattr(meshes["supported"], values))}')
    supporting = supporting.replace('<a href="supported_viewer.html">Supported mesh</a>',
                                    '<a href="mesh_viewer.html">Completed mesh</a>')
    (out/'supported_viewer.html').write_text(supporting)
    print(json.dumps({'coverage_mean': report['coverage_mean'], 'rendered': audit['mean_ray_hit']}, indent=2), flush=True)
    return report


def audit_existing(ws, out, names, semantic_directory=None, max_views=16):
    import open3d as o3d
    frames=[]
    for name in names:
        with np.load(out/'depth_cache'/(name+'.npz')) as data:
            f={key:data[key] for key in ('K','P','rgb','measured')}
            f.update(name=name,prior=data['depth'])
            frames.append(f)
    baseline=ws/'scene_hybrid_mesh.ply'
    if not baseline.exists(): baseline=ws/'scene_mesh.ply'
    meshes={n:o3d.io.read_triangle_mesh(str(out/f'scene_{n}.ply')) for n in ('supported','completed')}
    if baseline.exists(): meshes={'baseline':o3d.io.read_triangle_mesh(str(baseline)),**meshes}
    report=json.loads((out/'multiview_report.json').read_text())
    report['surface_audit']=ray_audit(meshes,frames,out,ws,max_views=max_views,semantic_directory=semantic_directory)
    agreement=geometry_agreement(report['surface_audit'])
    report['mean_measured_depth_agreement_under_5pct']=agreement
    if agreement is None or agreement<.8:
        report['quality_status']='inconsistent_geometry'
        warning='QUALITY CHECK FAILED: rendered geometry disagrees with measured depth.'
        report['issues']=list(dict.fromkeys(report.get('issues',[])+[warning]))
        metrics_path=out/'dense_metrics.json'
        metrics=json.loads(metrics_path.read_text())
        metrics['quality_warning']=warning+' '+metrics.get('quality_warning','')
        save_json(metrics_path,metrics)
        for filename in ('mesh_viewer.html','supported_viewer.html'):
            page=out/filename
            if page.exists() and warning not in page.read_text():
                page.write_text(page.read_text().replace('</header>',f'<strong>{warning}</strong></header>'))
    elif report['quality_status']!='inconsistent_depth':
        report['quality_status']=('high_rendered_coverage_unvalidated'
            if report['surface_audit']['mean_ray_hit']['completed']>=.9 else 'incomplete')
    save_json(out/'multiview_report.json',report)
    print(json.dumps(report['surface_audit']['mean_ray_hit'],indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('all','infer','fuse','audit'), default='all')
    p.add_argument('--model', default='depth-anything/DA3-BASE')
    p.add_argument('--depth-backend', choices=('da3','student'), default='da3')
    p.add_argument('--student-checkpoint', type=Path,
                   default=Path(__file__).resolve().parents[2]/'runs/depth_student_20261028/best.pt')
    p.add_argument('--overlap-bins', action='store_true',
                   help='Experimental SIFT-overlap bins with verified track/parallax source pairs')
    p.add_argument('--recurrent-checkpoint',type=Path,
                   help='Experimental ConvGRU depth updater; each update must pass the held-out anchor gate')
    p.add_argument('--resolution', type=int, default=504)
    p.add_argument('--semantic-dir', type=Path, help='Verified semantic_scene masks in undistorted camera pixels')
    p.add_argument('--video-fps', type=float, help='Recorded FPS for time-labelled object observations')
    p.add_argument('--include-unmeasured',action='store_true',
                   help='Predict additional registered views; sparse SfM tracks validate their scale and remain separate from stereo-measured depth')
    p.add_argument('--views', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=0, help='Evenly spaced smoke-test references; 0 uses all')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--max-depth', type=float,
                   help='Maximum camera-axis depth in COLMAP visual units, NOT metres')
    p.add_argument('--near-percentile', type=float,
                   help='Relative per-view cutoff from measured depth, e.g. 70 retains nearer measured pixels')
    p.add_argument('--global-visibility', action='store_true',
                   help='Experimental whole-sequence free-space filter; can reduce coverage with inconsistent poses')
    p.add_argument('--voxel-fraction', type=float, default=.01)
    p.add_argument('--measured-repeat',type=int,choices=range(2,33),
                   help='Stereo observation integration weight; automatically prioritize measurements when extra learned views are added, bounded to 2..32')
    p.add_argument('--visibility-references', type=int, default=24)
    p.add_argument('--audit-views', type=int, default=16)
    p.add_argument('--three-assets', type=Path)
    args = p.parse_args()
    if args.depth_backend == 'student':
        args.student_checkpoint = args.student_checkpoint.resolve()
        if not args.student_checkpoint.is_file():
            p.error(f'Student checkpoint missing: {args.student_checkpoint}')
    if args.video_fps is not None and (not np.isfinite(args.video_fps) or args.video_fps <= 0):
        p.error('Video FPS must be finite and positive')
    if args.max_depth is not None and (not np.isfinite(args.max_depth) or args.max_depth <= 0):
        p.error('--max-depth must be finite and positive')
    if args.near_percentile is not None and (not np.isfinite(args.near_percentile) or not 0 < args.near_percentile <= 100):
        p.error('--near-percentile must be in (0, 100]')
    if not np.isfinite(args.voxel_fraction) or args.voxel_fraction <= 0 or args.visibility_references < 2 or args.audit_views < 2:
        p.error('Positive voxel fraction and at least two visibility/audit views required')
    import pycolmap
    ws, out = args.workspace.resolve(), args.output.resolve()
    if ws == out:
        raise ValueError('Choose a separate output folder to retain the comparison baseline')
    out.mkdir(parents=True, exist_ok=True)
    model = pycolmap.Reconstruction(ws/'sparse')
    by_name = {image.name: image for image in model.images.values()}
    names = sorted(n for n in by_name if (ws/'stereo/depth_maps'/(n+'.geometric.bin')).exists())
    if args.include_unmeasured:
        names=sorted(set(names)|{n for n,im in by_name.items() if im.has_pose and im.num_points3D>=30})
    if args.max_frames:
        names = [names[i] for i in np.unique(np.linspace(0,len(names)-1,min(args.max_frames,len(names)),dtype=int))]
    if len(names) < 2 or args.views < 2:
        raise ValueError('At least two images and two context views required')
    for n in names:
        if model.cameras[by_name[n].camera_id].model_name not in ('PINHOLE','SIMPLE_PINHOLE'):
            raise ValueError('Use an undistorted COLMAP dense workspace with pinhole cameras')
    if args.overlap_bins:
        from .plan_overlap_depth import workspace_groups
        args.depth_groups,bin_report=workspace_groups(ws,names,by_name,model.points3D,
            source_views=min(2,args.views-1))
        save_json(out/'overlap_bins.json',bin_report)
    files = list((ws/'sparse').glob('*.bin'))+[ws/'images'/n for n in names]+[
        ws/'stereo/depth_maps'/(n+'.geometric.bin') for n in names if (ws/'stereo/depth_maps'/(n+'.geometric.bin')).is_file()]
    semantic_identity = None
    if args.semantic_dir:
        args.semantic_dir = args.semantic_dir.resolve()
        semantic_manifest = json.loads((args.semantic_dir/'manifest.json').read_text())
        if json.loads((args.semantic_dir/'semantic_report.json').read_text())['status'] != 'complete':
            raise ValueError('Semantic predictions are not complete')
        for name in names:
            if hashlib.sha256((ws/'images'/name).read_bytes()).hexdigest() != semantic_manifest['images'].get(name):
                raise ValueError('Semantic cache RGB differs: '+name)
            camera=model.cameras[by_name[name].camera_id]
            cached_camera=semantic_manifest['cameras'].get(str(camera.camera_id),{})
            if (cached_camera.get('width') != camera.width or cached_camera.get('height') != camera.height
                or not np.allclose(np.asarray(cached_camera.get('K',np.zeros((3,3)))),camera.calibration_matrix(),rtol=0,atol=1e-7)):
                raise ValueError('Semantic cache camera differs: '+name)
        semantic_files = [args.semantic_dir/'manifest.json']+[args.semantic_dir/(n+'.npz') for n in names]
        semantic_identity = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in semantic_files}
    manifest = {'workspace': str(ws), 'model':args.model, 'resolution':args.resolution, 'views':args.views,
        'names':names,'include_unmeasured':args.include_unmeasured, 'inference_code':hashlib.sha256(''.join(inspect.getsource(f) for f in
            (infer, fit_group_scale, spatial_holdout, remap_camera, source_groups, validation_anchors)).encode()).hexdigest(),
        'files':[(str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns) for f in files]}
    if args.depth_backend == 'student':
        manifest['depth_backend'] = 'student'
        manifest['student_checkpoint'] = (str(args.student_checkpoint),
            hashlib.sha256(args.student_checkpoint.read_bytes()).hexdigest())
        manifest['student_inference_code'] = hashlib.sha256(inspect.getsource(infer_student).encode()).hexdigest()
    if args.recurrent_checkpoint:
        args.recurrent_checkpoint=args.recurrent_checkpoint.resolve()
        if not args.recurrent_checkpoint.is_file():p.error('Recurrent checkpoint missing')
        manifest['recurrent_checkpoint']=(str(args.recurrent_checkpoint),
            hashlib.sha256(args.recurrent_checkpoint.read_bytes()).hexdigest())
        manifest['recurrent_inference_code']=hashlib.sha256(inspect.getsource(recurrent_updates).encode()).hexdigest()
    if args.overlap_bins:
        manifest['overlap_bins'] = hashlib.sha256((out/'overlap_bins.json').read_bytes()).hexdigest()
    if semantic_identity is not None:
        manifest['semantic_masks'] = semantic_identity
    manifest = json.loads(json.dumps(manifest))
    path = out/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Inputs or policy changed; choose a new output directory')
    save_json(path, manifest)
    try:
        save_json(out/'run_report.json', {'status':'running','stage':args.stage})
        if args.stage in ('all','infer'):
            if args.depth_backend == 'student':
                fallback = infer_student(ws, out, names, model, by_name, args)
                if fallback:
                    infer(ws, out, names, model, by_name, args, only_names=fallback)
            else:
                infer(ws, out, names, model, by_name, args)
            if args.recurrent_checkpoint:
                groups=getattr(args,'depth_groups',None) or source_groups(names,by_name,model.points3D,args.views)
                recurrent_updates(out,groups,args.recurrent_checkpoint)
        if args.stage in ('all','fuse'):
            fuse(ws, out, names, args)
        if args.stage == 'audit':
            audit_existing(ws,out,names,args.semantic_dir,args.audit_views)
        final={'status':'complete','stage':args.stage,'geometry_accuracy_validated':False}
        if args.stage != 'infer':
            quality=json.loads((out/'multiview_report.json').read_text())['quality_status']
            final.update(quality_status=quality,
                         internal_quality_passed=quality=='high_rendered_coverage_unvalidated')
        save_json(out/'run_report.json', final)
    except Exception as exc:
        save_json(out/'run_report.json', {'status':'failed','stage':args.stage,'error':str(exc)})
        raise


if __name__ == '__main__':
    main()
