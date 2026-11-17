"""Remove SfM doming by refining lens distortion in bundle adjustment.

With near-parallel nadir views, a frozen pinhole camera bends the whole trajectory: on UAVScenes
the stored models carry a +3.5% depth bias and up to 11.8 degrees of bend across 30 seconds, while
refining distortion alone brought the bias to about +0.1% on the clips tested. Focal length stays
fixed by default, because refining it with distortion is poorly constrained on this geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .utility_contracts import load_contract, write_json, write_status
from .validate_reference_geometry import PUBLISHED, area_of


def as_opencv(camera) -> list:
    """Same camera, written in the model whose distortion can be refined."""
    name, p = camera.model_name, list(map(float, camera.params))
    if name == 'PINHOLE':
        return p + [0., 0., 0., 0.]
    if name == 'SIMPLE_PINHOLE':
        return [p[0], p[0], p[1], p[2], 0., 0., 0., 0.]
    if name == 'SIMPLE_RADIAL':
        return [p[0], p[0], p[1], p[2], p[3], 0., 0., 0.]
    if name == 'RADIAL':
        return [p[0], p[0], p[1], p[2], p[3], p[4], 0., 0.]
    if name == 'OPENCV':
        return p
    raise ValueError(f'Cannot refine distortion for camera model {name}')


def refine(reconstruction, threads: int = 4, focal_scale: float = 1., refine_focal: bool = False) -> dict:
    import pycolmap
    report = {}
    for identifier, camera in reconstruction.cameras.items():
        params = as_opencv(camera)
        params[0] *= focal_scale
        params[1] *= focal_scale
        camera.model = pycolmap.CameraModelId.OPENCV
        camera.params = params
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = refine_focal
    options.refine_principal_point = False
    options.refine_extra_params = True
    solver = getattr(options, 'solver_options', None) or getattr(getattr(options, 'ceres', None), 'solver_options', None)
    if solver is None:
        raise RuntimeError('This pycolmap exposes no solver options, so the thread limit cannot be applied')
    solver.max_num_iterations, solver.num_threads = 60, threads
    summary = pycolmap.bundle_adjustment(reconstruction, options)
    for identifier, camera in reconstruction.cameras.items():
        report[str(identifier)] = [float(v) for v in camera.params]
    return {'cameras': report, 'focal_scale': focal_scale, 'refine_focal': refine_focal, 'solver': str(type(summary).__name__)}


def refine_model(model: Path, output: Path, threads: int = 4) -> tuple:
    """Keep the distortion model, so the existing undistortion step corrects the dense images."""
    import pycolmap
    reconstruction = pycolmap.Reconstruction(model)
    report = refine(reconstruction, threads)
    output.mkdir(parents=True, exist_ok=True)
    reconstruction.write_binary(output)
    return output, report


def world_from_camera(image) -> np.ndarray:
    pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    matrix = np.eye(4)
    matrix[:3, :4] = pose.matrix()
    return np.linalg.inv(matrix)


def anchors_for(reconstruction, image, width: int, height: int, grid: tuple, c_from_w: np.ndarray) -> np.ndarray:
    rows, columns = grid
    anchors = np.zeros(grid, np.float32)
    for point in image.points2D:
        if point.has_point3D():
            z = float((c_from_w @ np.r_[reconstruction.points3D[point.point3D_id].xyz, 1])[2])
            x, y = int(point.xy[0] * columns / width), int(point.xy[1] * rows / height)
            if z > 0 and 0 <= x < columns and 0 <= y < rows and (anchors[y, x] == 0 or z < anchors[y, x]):
                anchors[y, x] = z
    return anchors


def refine_clip(root: Path, contract: dict, name: str, output: Path, threads: int, factory_focal: bool) -> dict:
    """Write a pose, intrinsics and anchor overlay; the original caches stay untouched."""
    import pycolmap
    folder = root / contract['data']['cache_root'] / name
    infos = json.loads((folder / 'selected_infos.json').read_text())
    models = sorted((folder / 'reconstruction/sparse_run/online/blocks').glob('*/text_0'))
    if not models:
        return {'clip': name, 'status': 'no_sparse_model'}
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(models[0]))
    camera = reconstruction.cameras[next(iter(reconstruction.cameras))]
    width, height = camera.width, camera.height
    scene = infos[0]['OriginalSceneName']
    map_focal = float(infos[0]['P3x3'][0][0])
    scale = PUBLISHED[area_of(scene)]['K'][0] / map_focal if factory_focal else 1.
    report = refine(reconstruction, threads, scale)
    fitted = reconstruction.cameras[next(iter(reconstruction.cameras))]
    images = {int(i.name.split('_')[1].split('.')[0]): i for i in reconstruction.images.values() if i.has_pose}
    index = sorted(images)
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    centres, ratios = [], []
    for frame in index:
        world = world_from_camera(images[frame])
        centres.append((world[:3, 3], np.asarray(infos[frame]['T4x4'], np.float64)[:3, 3]))
    visual, metric = map(np.asarray, zip(*centres))
    gap = max(1, len(visual) // 4)
    alpha = float(np.median(np.linalg.norm(metric[gap:] - metric[:-gap], axis=1) / np.maximum(np.linalg.norm(visual[gap:] - visual[:-gap], axis=1), 1e-9)))
    cache = folder / 'reconstruction/multiview/depth_cache'
    for frame in index:
        file = cache / f'frame_{frame:06d}.jpg.npz'
        if not file.exists():
            continue
        with np.load(file) as cached:
            k, depth, old = cached['K'].astype(np.float64), cached['depth'], cached['anchors']
        grid = depth.shape
        world = world_from_camera(images[frame])
        c_from_w = np.linalg.inv(world)
        anchors = anchors_for(reconstruction, images[frame], width, height, grid, c_from_w)
        common = (anchors > 0) & (depth > 0)
        teacher_scale = float(np.median(anchors[common] / depth[common])) if common.sum() >= 30 else None
        k = k.copy()
        k[0, 0] *= scale
        k[1, 1] *= scale
        if teacher_scale is not None:
            ratios.append(teacher_scale)
        np.savez_compressed(target / f'frame_{frame:06d}.npz', P=c_from_w, K=k, anchors=anchors,
                            teacher_scale=np.float32(teacher_scale if teacher_scale is not None else np.nan))
    fx, fy, cx, cy, k1, k2, p1, p2 = map(float, fitted.params)
    summary = {'clip': name, 'status': 'refined', 'scene': scene, 'views': len(index), 'alpha_m_per_visual': alpha,
               'focal_scale': scale, 'distortion': {'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2},
               'median_teacher_scale': float(np.median(ratios)) if ratios else None}
    write_json(target / 'report.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--clips', nargs='*')
    parser.add_argument('--keep-map-focal', action='store_true', help='Keep the map focal instead of the factory calibration')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root = args.root.resolve()
    names = args.clips or sorted(p.name for p in (root / contract['data']['cache_root']).iterdir()
                                 if (p / 'selected_infos.json').exists() and 'GNSS' not in p.name)
    if args.dry_run:
        print(json.dumps({'clips': len(names)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        done = args.output / name / 'report.json'
        rows.append(json.loads(done.read_text()) if done.exists() else refine_clip(root, contract, name, args.output, args.threads, not args.keep_map_focal))
        print(json.dumps(rows[-1]), flush=True)
    write_json(args.output / 'report.json', {'clips': rows, 'focal': 'map' if args.keep_map_focal else 'factory'})
    write_status(args.output, 'stage1_dedome', 'complete', 'PASS' if all(r['status'] == 'refined' for r in rows) else 'PARTIAL', 'stage1_targets')


if __name__ == '__main__':
    main()
