"""Estimate normals from locally planar, already validated depth measurements."""
import json
from functools import lru_cache
from pathlib import Path
import shutil

import cv2
import numpy as np


NORMAL_POLICY = {'version': 1, 'window': 25, 'min_samples': 24,
                 'max_relative_rms': .005, 'max_center_error': .01,
                 'max_angle_uncertainty_deg': 5.}


def read_array(path):
    with Path(path).open('rb') as stream:
        header = b''
        while header.count(b'&') < 3:
            byte = stream.read(1)
            if not byte or len(header) > 100:
                raise ValueError(f'Invalid COLMAP array header: {path}')
            header += byte
        width, height, channels = map(int, header[:-1].split(b'&'))
        values = np.frombuffer(stream.read(), dtype='<f4')
    return values.reshape((width, height, channels), order='F').transpose(1, 0, 2).squeeze()


def write_array(path, array):
    array = np.asarray(array, dtype='<f4')
    if array.ndim == 2:
        array = array[:, :, None]
    height, width, channels = array.shape
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(f'{width}&{height}&{channels}&'.encode('ascii'))
        stream.write(array.transpose(1, 0, 2).ravel(order='F').tobytes())
    temporary.replace(path)


def fit_depth_normals(depth, original, calibration, policy=NORMAL_POLICY):
    """Replace normals only where inverse-depth plane fits pass fixed checks.

    Depth coordinates are never filled or modified. Plane residuals and center
    agreement reject patches spanning depth boundaries. Inverse depth of a plane
    is affine in image coordinates, making its fitted coefficients a camera normal.
    """
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    q = np.zeros_like(depth)
    np.divide(1., depth, out=q, where=valid)
    y, x = np.indices(depth.shape, dtype=np.float64)
    weight = valid.astype(np.float64)
    window = policy['window']

    def total(a):
        return cv2.boxFilter(a, -1, (window, window), normalize=False,
                             borderType=cv2.BORDER_CONSTANT)

    n = total(weight)
    denominator = np.maximum(n, 1.)
    mx, my, mq = (total(a) / denominator for a in (weight*x, weight*y, q))
    vx = np.maximum(total(weight*x*x)/denominator - mx*mx, 0.)
    vy = np.maximum(total(weight*y*y)/denominator - my*my, 0.)
    vxy = total(weight*x*y)/denominator - mx*my
    vxq = total(x*q)/denominator - mx*mq
    vyq = total(y*q)/denominator - my*mq
    vq = np.maximum(total(q*q)/denominator - mq*mq, 0.)
    determinant = vx*vy - vxy*vxy
    safe = np.maximum(determinant, 1e-12)
    a = (vxq*vy - vyq*vxy)/safe
    b = (vyq*vx - vxq*vxy)/safe
    c = mq - a*mx - b*my
    residual = np.sqrt(np.maximum(vq-a*vxq-b*vyq, 0.))
    fx, fy, cx, cy = calibration[0, 0], calibration[1, 1], calibration[0, 2], calibration[1, 2]
    normal = np.stack([a*fx, b*fy, c+a*cx+b*cy], axis=-1)
    length = np.linalg.norm(normal, axis=-1)
    uncertainty = np.degrees(residual * np.sqrt((fx*fx*vy + fy*fy*vx) /
                                               np.maximum(denominator*safe, 1e-12)) /
                             np.maximum(length, 1e-12))
    predicted = a*x + b*y + c
    trusted = (valid & (n >= policy['min_samples']) & (determinant > 1e-3) &
               (residual <= policy['max_relative_rms']*mq) &
               (np.abs(predicted-q) <= policy['max_center_error']*q) &
               (uncertainty <= policy['max_angle_uncertainty_deg']) & (length > 0))
    result = np.asarray(original, dtype=np.float32).copy()
    result[trusted] = (-normal[trusted] / length[trusted, None]).astype(np.float32)
    return result, {'valid_depth_pixels': int(valid.sum()),
                    'reestimated_normal_pixels': int(trusted.sum()),
                    'fraction_of_valid_depth': float(trusted.sum()/max(valid.sum(), 1))}


def stabilize_workspace(workspace):
    import pycolmap
    workspace = Path(workspace)
    report_path = workspace / 'normal_estimation.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous['policy'] != NORMAL_POLICY:
            raise ValueError('Normal policy changed; use a fresh workspace')
        return previous
    if (workspace / 'scene_dense.ply').exists():
        raise ValueError('Normal estimation must precede fusion; use a fresh workspace')
    model = pycolmap.Reconstruction(workspace / 'sparse')
    images = {image.name: image for image in model.images.values()}
    names = (workspace / 'stereo/fusion.cfg').read_text().splitlines()
    raw = workspace / 'stereo/raw_normal_maps'
    raw.mkdir(exist_ok=True)
    rows = []
    for name in names:
        if not name.strip() or name.startswith('#'):
            continue
        normal_path = workspace / 'stereo/normal_maps' / (name + '.geometric.bin')
        original_path = raw / normal_path.name
        if not original_path.exists():
            normal_path.replace(original_path)
        original = read_array(original_path)
        depth = read_array(workspace / 'stereo/depth_maps' / (name + '.geometric.bin'))
        camera = model.cameras[images[name].camera_id]
        calibration = camera.calibration_matrix().copy()
        calibration[0] *= depth.shape[1]/camera.width
        calibration[1] *= depth.shape[0]/camera.height
        fitted, row = fit_depth_normals(depth, original, calibration)
        write_array(normal_path, fitted)
        rows.append({'frame': name, **row})
    report = {'policy': NORMAL_POLICY, 'frames': rows,
              'depth_values_changed': False, 'raw_normals_preserved': True,
              'valid_depth_pixels': sum(r['valid_depth_pixels'] for r in rows),
              'reestimated_normal_pixels': sum(r['reestimated_normal_pixels'] for r in rows)}
    report['cross_view_check'] = compare_normal_agreement(workspace)
    report['accepted'] = report['cross_view_check']['candidate_agreement'] >= report['cross_view_check']['original_agreement']
    if not report['accepted']:
        for row in rows:
            name = row['frame'] + '.geometric.bin'
            shutil.copy2(raw / name, workspace / 'stereo/normal_maps' / name)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def compare_normal_agreement(workspace, samples=12):
    """Compare normals at identical depths with identical source cameras."""
    import pycolmap
    workspace = Path(workspace)
    model = pycolmap.Reconstruction(workspace / 'sparse')
    images = {image.name: image for image in model.images.values()}
    plan = json.loads((workspace / 'dense_plan.json').read_text())['plan']
    selected = [plan[int(i)] for i in np.linspace(0, len(plan)-1, min(samples, len(plan)), dtype=int)]

    @lru_cache(maxsize=32)
    def array(name, folder):
        return read_array(workspace / 'stereo' / folder / (name + '.geometric.bin'))

    def camera(name):
        image = images[name]
        pose = image.cam_from_world().matrix()
        calibration = model.cameras[image.camera_id].calibration_matrix().copy()
        shape = array(name, 'depth_maps').shape
        c = model.cameras[image.camera_id]
        calibration[0] *= shape[1]/c.width
        calibration[1] *= shape[0]/c.height
        return pose[:, :3], pose[:, 3], calibration, np.linalg.inv(calibration)

    totals = np.zeros(3, dtype=int)
    for row in selected:
        name = row['reference']
        depth = array(name, 'depth_maps')
        rotation, translation, calibration, inverse = camera(name)
        y, x = np.mgrid[4:depth.shape[0]:16, 4:depth.shape[1]:16]
        y, x = y.ravel(), x.ravel()
        valid = depth[y, x] > 0
        y, x = y[valid], x[valid]
        z = depth[y, x]
        world = ((np.column_stack([x, y, np.ones(len(z))]) @ inverse.T)*z[:, None]-translation) @ rotation
        raw = array(name, 'raw_normal_maps')[y, x] @ rotation
        candidate = array(name, 'normal_maps')[y, x] @ rotation
        counts = np.zeros((len(z), 2), dtype=int)
        for source in row['sources']:
            other = source['name']
            sd = array(other, 'depth_maps')
            sr, st, sk, _ = camera(other)
            transformed = world @ sr.T + st
            projection = transformed @ sk.T
            uv_float = projection[:, :2]/np.maximum(projection[:, 2:], 1e-9)
            uv = np.rint(uv_float).astype(int)
            inside = ((transformed[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < sd.shape[1]) &
                      (uv[:, 1] >= 0) & (uv[:, 1] < sd.shape[0]))
            index = np.flatnonzero(inside)
            measured = sd[uv[index, 1], uv[index, 0]]
            consistent = ((measured > 0) & (np.abs(transformed[index, 2]-measured) <= .01*measured) &
                          (np.linalg.norm(uv_float[index]-uv[index], axis=1) <= 2.))
            index = index[consistent]
            for column, folder, ref_normals in ((0, 'raw_normal_maps', raw), (1, 'normal_maps', candidate)):
                source_normals = array(other, folder)[uv[index, 1], uv[index, 0]] @ sr
                dot = np.sum(source_normals * ref_normals[index], axis=1)
                counts[index[dot >= np.cos(np.deg2rad(10.))], column] += 1
        totals += [len(z), int(np.sum(counts[:, 0] >= 2)), int(np.sum(counts[:, 1] >= 2))]
    return {'sampled_valid_depth_pixels': int(totals[0]),
            'original_agreement': float(totals[1]/max(totals[0], 1)),
            'candidate_agreement': float(totals[2]/max(totals[0], 1)),
            'normal_tolerance_deg': 10, 'depth_relative_tolerance': .01,
            'method': 'two agreeing planned source views; identical positions before and after normal fitting'}
