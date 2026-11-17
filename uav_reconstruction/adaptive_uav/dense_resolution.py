"""Resolution policies and calibrated RGB preparation from original video frames."""
from pathlib import Path
import hashlib
import json
import re

import cv2
import numpy as np

DETAIL_POLICIES = {
    'fast': {'dense_size': 640, 'depth_resolution': 336, 'voxel_fraction': .02, 'stereo_iterations': 1},
    'balanced': {'dense_size': 1280, 'depth_resolution': 504, 'voxel_fraction': .01, 'stereo_iterations': 2},
    'high': {'dense_size': 1920, 'depth_resolution': 672, 'voxel_fraction': .005, 'stereo_iterations': 3},
}


def prepare_dense_inputs(model_path, images, video, output, max_size):
    """Restore real source pixels when the pose cache has smaller RGB frames.

    Camera intrinsics and 2D observations scale together; world geometry and
    extrinsics stay in their original coordinates. Never enlarge cached pixels.
    """
    import pycolmap
    from .combined_reconstruct import save_json

    model_path, images, video, output = map(Path, (model_path, images, video, output))
    if max_size < 224:
        raise ValueError('Dense source resolution must be at least 224 pixels')
    rec = pycolmap.Reconstruction(model_path)
    registered = [im for im in rec.images.values() if im.has_pose]
    if not registered:
        raise ValueError('No registered images to prepare')
    cap = cv2.VideoCapture(str(video))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(vw, vh) <= 0:
        cap.release()
        raise ValueError('Invalid original video dimensions')
    limit = min(max_size, max(vw, vh))
    needs_original = any(max(rec.cameras[im.camera_id].width, rec.cameras[im.camera_id].height) < limit
                         for im in registered)
    if not needs_original:
        cap.release()
        return model_path, images, {'source': 'cached RGB', 'requested_long_side': max_size,
                                    'original_video_shape': [vh, vw], 'restored_original_pixels': False}
    if output.exists():
        cap.release()
        raise ValueError('Dense source preparation already exists; use a fresh output')
    image_dir = output / 'images'; image_dir.mkdir(parents=True)
    scaled_model = output / 'sparse'; scaled_model.mkdir()
    scales = {}
    for camera in rec.cameras.values():
        w, h = camera.width, camera.height
        if abs(w / h - vw / vh) > .01:
            cap.release()
            raise ValueError('Original video aspect ratio does not match the pose camera')
        factor = limit / max(w, h)
        camera.rescale(round(w * factor), round(h * factor))
        scales[camera.camera_id] = np.array([camera.width / w, camera.height / h])
    frames = {}
    for im in registered:
        match = re.fullmatch(r'frame_(\d+)\.(jpg|png)', im.name)
        if match is None:
            cap.release()
            raise ValueError('Original frame index is required to restore RGB: ' + im.name)
        frames[int(match[1])] = im
        for point in im.points2D:
            point.xy = point.xy * scales[im.camera_id]
    # Decode once, including intervening video frames, rather than repeatedly
    # seeking into long GOPs for every image.
    first, last = min(frames), max(frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for index in range(first, last + 1):
        if not cap.grab():
            cap.release()
            raise ValueError(f'Could not decode original frame {index}')
        if index not in frames:
            continue
        ok, rgb = cap.retrieve()
        if not ok:
            cap.release()
            raise ValueError(f'Could not retrieve original frame {index}')
        im = frames[index]; camera = rec.cameras[im.camera_id]
        rgb = cv2.resize(rgb, (camera.width, camera.height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(image_dir / im.name), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError('Failed to write original RGB frame ' + im.name)
    cap.release()
    rec.write(scaled_model)
    report = {'source': str(video.resolve()), 'requested_long_side': max_size,
              'effective_long_side': limit, 'original_video_shape': [vh, vw],
              'restored_original_pixels': True, 'frames': len(frames),
              'camera_and_observation_scaling': {str(k): v.tolist() for k, v in scales.items()},
              'note': 'RGB decoded from the original video; larger frames are not upsampled cached images.'}
    save_json(output / 'source_resolution.json', report)
    return scaled_model, image_dir, report


def matching_camera_resolution(old, current, names):
    """Do not promote lower-resolution stereo maps into a sharper run's cache."""
    before = {im.name: im for im in old.images.values() if im.has_pose}
    after = {im.name: im for im in current.images.values() if im.has_pose}
    return all((old.cameras[before[n].camera_id].width, old.cameras[before[n].camera_id].height) ==
               (current.cameras[after[n].camera_id].width, current.cameras[after[n].camera_id].height)
               for n in names)


def validate_cached_workspace(cached, current, video_identity, detail, recovery_views):
    """Reuse stereo only for the same video, poses, pixels and depth settings."""
    import pycolmap
    from .hybrid_dense import pose4, read_array
    cached,current=Path(cached).resolve(),Path(current).resolve()
    manifest=json.loads((cached.parent/'pipeline_manifest.json').read_text())
    if manifest.get('video')!=video_identity:
        raise ValueError('Cached stereo workspace belongs to a different video identity')
    old,new=pycolmap.Reconstruction(cached/'sparse'),pycolmap.Reconstruction(current/'sparse')
    before={im.name:im for im in old.images.values() if im.has_pose}
    after={im.name:im for im in new.images.values() if im.has_pose}
    if before.keys()!=after.keys():raise ValueError('Cached stereo camera set differs')
    if not matching_camera_resolution(old,new,before):raise ValueError('Cached stereo resolution differs')
    for name,im in before.items():
        target=after[name]
        if not np.allclose(pose4(im),pose4(target),rtol=0,atol=1e-7):
            raise ValueError('Cached stereo pose differs: '+name)
        a,b=old.cameras[im.camera_id],new.cameras[target.camera_id]
        if not np.allclose(a.calibration_matrix(),b.calibration_matrix(),rtol=0,atol=1e-6):
            raise ValueError('Cached stereo calibration differs: '+name)
        if hashlib.sha256((cached/'images'/name).read_bytes()).digest()!=hashlib.sha256((current/'images'/name).read_bytes()).digest():
            raise ValueError('Cached stereo RGB differs: '+name)
    settings=json.loads((cached/'depth_settings.json').read_text())
    plan=json.loads((cached/'dense_plan.json').read_text())
    expected=(detail['dense_size'],detail['stereo_iterations'],3.)
    actual=(settings.get('max_image_size'),settings.get('iterations'),settings.get('filter_min_triangulation_angle'))
    if actual!=expected or plan.get('max_reference_views')!=recovery_views or plan.get('max_source_views')!=4:
        raise ValueError('Cached stereo settings or reference budget differ')
    if settings.get('configuration')!=(cached/'stereo/patch-match.cfg').read_text():
        raise ValueError('Cached stereo source configuration differs')
    references=settings['references']
    if len(references)<8:raise ValueError('Insufficient cached stereo references')
    for name in references:
        if name not in before:raise ValueError('Unknown cached stereo camera: '+name)
        depth=read_array(cached/'stereo/depth_maps'/(name+'.geometric.bin'))
        camera=old.cameras[before[name].camera_id]
        if depth.shape!=(camera.height,camera.width):raise ValueError('Cached depth dimensions differ: '+name)
    return {'workspace':str(cached),'references':len(references),'video_identity_checked':True,
            'camera_poses_checked':True,'intrinsics_checked':True,'rgb_hashes_checked':True,
            'depth_settings_checked':True,'depth_shapes_checked':True}
