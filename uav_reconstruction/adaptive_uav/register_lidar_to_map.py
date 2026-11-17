"""Stage 1: calibrate LiDAR against image pixels through the photogrammetric map.

RGB edges cannot register sparse LiDAR over vegetation. The Terra mesh shares its cameras with the
images, so a rendered mesh gives depth in image coordinates, and depth-to-depth comparison keeps its
power on trees and buildings. The map bridges the sensors; it is not independent truth.

The sensor clock offset is one physical quantity, so it is estimated once for the whole rig: the
median of per-flight fits on calibration clips only. Fitting it per area lets it trade against a
fixed pixel shift, because on a straight run both slide returns along the motion. With time frozen,
each area picks an extrinsic and a small bounded image-plane residual. Validation clips then score
that frozen result, refit time independently as a check, and face wrong-pairing controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .open3d_cpu import cpu_bindings
from .utility_contracts import THREADS, load_contract, sha256_file, write_json, write_status
from .validate_reference_geometry import Clip, PUBLISHED, clips, rotation_candidates

RENDER_STEP = 4
SCOPE = 'Agreement with the Terra map as a bridge from LiDAR to image pixels.'
POINTS_PER_FRAME = 6000
TIME_COARSE, TIME_FINE = np.arange(-.3, .3001, .03), np.arange(-.024, .0241, .006)
SHIFT_COARSE, SHIFT_FINE = np.arange(-30, 31, 6.), np.arange(-6, 7, 1.5)
RADIAL = np.arange(.98, 1.0201, .0025)
ROLL_DEG = np.arange(-.5, .51, .1)


def make_scene(path: Path):
    o3d = cpu_bindings()
    mesh = o3d.io.read_triangle_mesh(str(path))
    scene = o3d.t.geometry.RaycastingScene(nthreads=THREADS)
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(mesh.triangles, np.uint32)))
    return scene, o3d


def render_rays(scene, o3d, k: np.ndarray, pose: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Camera-Z depth along given native pixel coordinates; holes are zero."""
    rays = np.stack(((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones_like(xs, np.float64)), -1)
    directions = rays @ pose[:3, :3].T
    table = np.concatenate((np.broadcast_to(pose[:3, 3], directions.shape), directions), -1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(table), nthreads=THREADS)['t_hit'].numpy()  # In direction lengths, so this is Z.
    return np.where(np.isfinite(hit), hit, 0).astype(np.float32)


def render(scene, o3d, k: np.ndarray, pose: np.ndarray, width: int, height: int) -> np.ndarray:
    ys, xs = np.mgrid[0:height:RENDER_STEP, 0:width:RENDER_STEP]
    return render_rays(scene, o3d, k, pose, xs.astype(np.float64), ys.astype(np.float64))


def sample(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return cv2.remap(depth, (uv[:, 0] / RENDER_STEP).astype(np.float32)[None], (uv[:, 1] / RENDER_STEP).astype(np.float32)[None],
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]


def warp(uv: np.ndarray, centre: np.ndarray, dx: float, dy: float, radial: float, roll_deg: float) -> np.ndarray:
    """Image-plane correction applied to projected LiDAR pixels."""
    angle = np.radians(roll_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (uv - centre) @ rotation.T * radial + centre + [dx, dy]


IDENTITY = (0., 0., 1., 0.)


def spread(frames: list, theta: tuple = IDENTITY) -> float:
    """Median per-frame MAD of log(LiDAR/map). Small only if returns land on their own surfaces."""
    values = []
    for frame in frames:
        if frame['uv'] is None:
            continue
        d = sample(frame['depth'], warp(frame['uv'], frame['centre'], *theta))
        good = d > 1
        if good.sum() < 50:
            continue
        log_ratio = np.log(frame['z'][good] / d[good])
        values.append(np.median(np.abs(log_ratio - np.median(log_ratio))))
    return float(np.median(values)) if values else float('inf')


def depth_ratio(frames: list, theta: tuple = IDENTITY) -> float | None:
    values = []
    for frame in frames:
        if frame['uv'] is None:
            continue
        d = sample(frame['depth'], warp(frame['uv'], frame['centre'], *theta))
        good = d > 1
        if good.sum() >= 50:
            values.append(np.median(d[good] / frame['z'][good]))
    return float(np.median(values)) if values else None


def project(frames: list, rotation=None, translation=None, offset: float = 0.) -> None:
    for frame in frames:
        clip, i = frame['clip'], frame['image_index']
        try:
            camera = clip.in_camera(frame['scan_index'], frame['xyz'], rotation, translation, offset)
        except ValueError:
            frame['uv'] = None
            continue
        camera = camera[camera[:, 2] > 5]
        k = clip.k(i)
        frame['uv'] = (camera[:, :2] / camera[:, 2:]) * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
        frame['z'] = camera[:, 2]


def prepare(clip: Clip, scene, o3d, count: int, rng, wrong: bool = False) -> list:
    frames, usable = [], clip.sample(count)
    for position, i in enumerate(usable):
        scan = usable[(position + len(usable) // 2) % len(usable)] if wrong else i
        xyz = clip.scan(scan)
        if len(xyz) > POINTS_PER_FRAME:
            xyz = xyz[rng.choice(len(xyz), POINTS_PER_FRAME, replace=False)]
        k, info = clip.k(i), clip.infos[i]
        frames.append({'clip': clip, 'image_index': i, 'scan_index': scan, 'same': not wrong, 'xyz': xyz, 'uv': None, 'z': None,
                       'centre': np.array([k[0, 2], k[1, 2]]),
                       'depth': render(scene, o3d, k, clip.pose(i), int(info['Width']), int(info['Height']))})
    return frames


def fit_time(frames: list, rotation=None, translation=None) -> dict:
    curve = {}
    for offset in TIME_COARSE:
        project(frames, rotation, translation, float(offset))
        curve[float(offset)] = spread(frames)
    centre = min(curve, key=curve.get)
    for offset in centre + TIME_FINE:
        project(frames, rotation, translation, float(offset))
        curve[float(offset)] = spread(frames)
    best = min(curve, key=curve.get)
    return {'time_offset_s': float(best), 'spread': curve[best], 'spread_at_zero': curve[min(curve, key=abs)],
            'curve': {f'{k:.3f}': v for k, v in sorted(curve.items())}}


def fit_plane(frames: list) -> dict:
    """Bounded image-plane residual. A large value here means something is still unmodelled."""
    theta = list(IDENTITY)
    for offsets in (SHIFT_COARSE, SHIFT_FINE, SHIFT_FINE / 3):
        centre = list(theta[:2])
        grid = {(a, b): spread(frames, (centre[0] + a, centre[1] + b, theta[2], theta[3])) for a in offsets for b in offsets}
        (a, b), _ = min(grid.items(), key=lambda item: item[1])
        theta[0], theta[1] = centre[0] + a, centre[1] + b
        theta[2] = float(min(RADIAL, key=lambda r: spread(frames, (theta[0], theta[1], r, theta[3]))))
        theta[3] = float(min(ROLL_DEG, key=lambda r: spread(frames, (theta[0], theta[1], theta[2], r))))
    return {'dx_px': theta[0], 'dy_px': theta[1], 'radial_scale': theta[2], 'roll_deg': theta[3], 'spread': spread(frames, tuple(theta)),
            'map_over_lidar_depth': depth_ratio(frames, tuple(theta))}


def theta_of(row: dict) -> tuple:
    return tuple(row[k] for k in ('dx_px', 'dy_px', 'radial_scale', 'roll_deg'))


def load_area(area: str, rows: list, terra: Path, count: int, wrong_pairs: bool = True) -> dict:
    scene, o3d = make_scene(terra / area / 'Mesh.ply')
    rng = np.random.default_rng(0)
    data = {'calibration': [], 'validation': [], 'wrong': [], 'flights': {}}
    for row in rows:
        clip = Clip(row)
        frames = prepare(clip, scene, o3d, count, rng)
        data[row['role']].extend(frames)
        if row['role'] == 'validation' and wrong_pairs:
            data['wrong'] += prepare(clip, scene, o3d, max(count // 2, 4), rng, wrong=True)
        data['flights'].setdefault(row['scene'], {'calibration': [], 'validation': []})[row['role']].extend(frames)
    return data


def flight_offsets(area: str, rows: list, terra: Path, count: int) -> dict:
    """Time offset per flight, from calibration clips only."""
    data = load_area(area, [r for r in rows if r['role'] == 'calibration'], terra, count, wrong_pairs=False)
    return {name: fit_time(group['calibration'])['time_offset_s'] for name, group in data['flights'].items() if group['calibration']}


def calibrate_area(area: str, rows: list, terra: Path, count: int, rules: dict, offset: float) -> dict:
    data = load_area(area, rows, terra, count)
    calibration, validation, wrong = data['calibration'], data['validation'], data['wrong']
    candidates = {}
    for name, (rotation, translation) in rotation_candidates(area).items():
        project(calibration, rotation, translation, offset)
        candidates[name] = fit_plane(calibration)
    chosen = min(candidates, key=lambda n: candidates[n]['spread'])
    rotation, translation = rotation_candidates(area)[chosen]
    plane, theta = candidates[chosen], theta_of(candidates[chosen])
    # Everything below is frozen before validation clips are scored.
    project(validation)
    uncorrected = spread(validation)
    project(validation, rotation, translation, offset)
    corrected = spread(validation, theta)
    residual = fit_plane(validation)
    independent = fit_time(validation, rotation, translation)['time_offset_s']
    project(wrong, rotation, translation, offset)
    control = spread(wrong, theta)
    per_flight = {}
    for name, group in data['flights'].items():
        frames = group['calibration'] + group['validation']
        project(frames, rotation, translation, offset)
        speeds = [np.linalg.norm(f['clip'].pose(f['image_index'] + 1)[:3, 3] - f['clip'].pose(f['image_index'])[:3, 3]) * 2 for f in frames
                  if f['image_index'] + 1 < len(f['clip'].infos)]
        per_flight[name] = {'speed_mps': float(np.median(speeds)) if speeds else None, 'spread_with_calibration': spread(frames, theta),
                            'map_over_lidar_depth': depth_ratio(frames, theta),
                            'validation_time_offset_s': fit_time(group['validation'], rotation, translation)['time_offset_s'] if group['validation'] else None}
    magnitude = float(np.hypot(residual['dx_px'] - theta[0], residual['dy_px'] - theta[1]))
    power = control >= rules['min_wrong_pairing_ratio'] * corrected
    outcome = 'no_power' if not power else 'aligned' if magnitude <= rules['max_residual_px'] and corrected <= uncorrected else 'offset'
    return {'rotation_source': chosen, 'rotation': np.asarray(rotation).tolist(), 'translation': np.asarray(translation).tolist(),
            'time_offset_s': offset, 'plane': plane, 'candidates': candidates,
            'validation': {'spread_uncorrected': uncorrected, 'spread_corrected': corrected, 'spread_wrong_pairing': control, 'residual_px': magnitude,
                           'residual_fit': residual, 'independent_time_offset_s': independent, 'outcome': outcome},
            'flights': per_flight, 'published_factory_focal': PUBLISHED[area]['K'][0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--terra', type=Path)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--only', default='', help='Comma-separated areas, for a partial run')
    parser.add_argument('--time-only', action='store_true', help='Write per-flight offsets and stop, so areas can run in parallel')
    parser.add_argument('--time-offset', type=float, help='Use this rig time offset instead of estimating it')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root, rules = args.root.resolve(), contract['geometry']['map_registration']
    terra = args.terra or root / contract['data']['terra']
    rows = clips(root, contract)
    areas = sorted({r['area'] for r in rows if not args.only or r['area'] in args.only.split(',')})
    if args.dry_run:
        print(json.dumps({'areas': areas, 'clips': len(rows), 'threads': THREADS}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.time_only:
        for area in areas:
            write_json(args.output / f'time_offsets_{area}.json', flight_offsets(area, [r for r in rows if r['area'] == area], terra, args.frames))
        return
    output = args.output / 'calibration.json'
    report = json.loads(output.read_text()) if output.exists() else {'schema_version': 2, 'areas': {}, 'mesh_sha256': {}, 'scope': SCOPE}
    offset = args.time_offset if args.time_offset is not None else report.get('rig_time_offset_s')
    flights = {}
    if offset is None:
        for area in areas:
            flights.update(flight_offsets(area, [r for r in rows if r['area'] == area], terra, args.frames))
        offset = float(np.median(list(flights.values())))
        print('rig time offset', offset, json.dumps(flights), flush=True)
    report.update(rig_time_offset_s=offset, calibration_flight_offsets_s=flights or report.get('calibration_flight_offsets_s'))
    write_status(args.output, 'stage1_calibration', 'running', None, None)
    for area in areas:
        if area in report['areas']:
            continue
        report['mesh_sha256'][area] = sha256_file(terra / area / 'Mesh.ply')
        report['areas'][area] = calibrate_area(area, [r for r in rows if r['area'] == area], terra, args.frames, rules, offset)
        write_json(output, report)
        result = report['areas'][area]
        print(area, json.dumps({k: result[k] for k in ('rotation_source', 'time_offset_s', 'validation')}, default=float), flush=True)
    outcomes = {area: row['validation']['outcome'] for area, row in report['areas'].items()}
    write_status(args.output, 'stage1_calibration', 'complete', json.dumps(outcomes), 'stage1_targets')


if __name__ == '__main__':
    main()
