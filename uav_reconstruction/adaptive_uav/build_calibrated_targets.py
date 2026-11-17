"""Stage 1: build corrected supervision from the calibration, beside the provisional sidecars.

Each cached view gets, in decreasing trust: reliable LiDAR (static support, no conflicting returns,
no vehicle or water labels), the same from neighbouring scans for density, and dense map depth where
the mesh agrees with this frame's LiDAR. Nothing here overwrites the old targets. The new geometry
gate is written last: a flight only becomes QUALIFIED if its area passed map registration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .register_lidar_to_map import make_scene, render_rays, warp
from .utility_contracts import load_contract, sha256_file, write_json, write_status
from .validate_reference_geometry import CACHE_GRID, Clip, clips

SUPPORT_RADIUS_M, SUPPORT_HEIGHT_M, OCCLUSION_MARGIN = 1.5, .75, .02
REASONS = {0: 'no_return', 1: 'reliable', 2: 'no_neighbour_support', 3: 'inconsistent_with_neighbour_scans', 4: 'conflicting_returns', 5: 'excluded_label'}
MAP_MAX_BIAS, MAP_MAX_SPREAD, MAP_MIN_PIXELS = .02, .02, 200


def static_support(here: np.ndarray, others: np.ndarray) -> np.ndarray:
    """1 static, 2 nothing nearby to compare with, 3 no neighbour at this height."""
    if not len(others):
        return np.full(len(here), 2, np.uint8)
    tree = cKDTree(others[:, :2])
    _, neighbours = tree.query(here[:, :2], k=6, distance_upper_bound=SUPPORT_RADIUS_M, workers=1)
    found = neighbours < len(others)
    heights = np.where(found, others[np.minimum(neighbours, len(others) - 1), 2], np.inf)
    close = np.abs(heights - here[:, 2:3]).min(1) <= SUPPORT_HEIGHT_M
    return np.where(~found.any(1), 2, np.where(close, 1, 3)).astype(np.uint8)


def grid_cells(uv: np.ndarray, width: int, height: int) -> tuple:
    rows, columns = CACHE_GRID
    y = (np.floor(uv[:, 1]).astype(np.int64) * rows // height).clip(0, rows - 1)
    x = (np.floor(uv[:, 0]).astype(np.int64) * columns // width).clip(0, columns - 1)
    return y, x


def nearest(y, x, z, flag) -> tuple:
    """Z-buffer on the cache grid, keeping the nearest return's flag."""
    order = np.argsort(-z)
    depth, reason, far = np.zeros(CACHE_GRID, np.float32), np.zeros(CACHE_GRID, np.uint8), np.zeros(CACHE_GRID, np.float32)
    np.maximum.at(far, (y, x), z)
    depth[y[order], x[order]] = z[order]
    reason[y[order], x[order]] = flag[order]
    reason[(depth > 0) & (far > depth * (1 + OCCLUSION_MARGIN))] = 4
    return depth, reason


def label_mask(labels: Path | None, scene: str, image_name: str, exclude: list) -> np.ndarray | None:
    """True where semantic labels mark something unsafe for geometry."""
    if labels is None:
        return None
    stem = Path(image_name).stem
    folder = labels / f'interval1_{scene}' / 'interval1_CAM_label_id'
    path = next((folder / f'{stem}{suffix}' for suffix in ('.png', '.jpg') if (folder / f'{stem}{suffix}').exists()), None)
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        return None
    resized = cv2.resize(image, CACHE_GRID[::-1], interpolation=cv2.INTER_NEAREST)
    return np.isin(resized, exclude)


def map_grid(scene_handle, k: np.ndarray, pose: np.ndarray, width: int, height: int) -> np.ndarray:
    scene, o3d = scene_handle
    rows, columns = CACHE_GRID
    ys, xs = np.mgrid[0:rows, 0:columns]
    return render_rays(scene, o3d, k, pose, (xs + .5) * width / columns, (ys + .5) * height / rows)


def build_clip(row: dict, calibration: dict, handle, output: Path, labels: Path | None, exclude: list, neighbours: int) -> dict:
    clip = Clip(row)
    area = calibration['areas'].get(row['area'])
    if area is None:
        return {'clip': row['name'], 'status': 'not_calibrated'}
    rotation, translation = np.asarray(area['rotation']), np.asarray(area['translation'])
    offset, plane = area['time_offset_s'], area['plane']
    theta = (plane['dx_px'], plane['dy_px'], plane['radial_scale'], plane['roll_deg'])
    scans, skipped = {}, 0
    for i in clip.frames:
        try:
            pose = clip.lidar_pose(i, offset)
        except ValueError:
            skipped += 1
            continue
        xyz = clip.scan(i)
        camera = xyz @ rotation.T + translation
        scans[i] = {'world': camera @ pose[:3, :3].T + pose[:3, 3], 'at_image': clip.in_camera(i, xyz, rotation, translation, offset)}
    for i, scan in scans.items():
        others = [scans[j]['world'] for j in (i - 1, i + 1) if j in scans]
        scan['support'] = static_support(scan['world'], np.concatenate(others) if others else np.empty((0, 3)))
    cache = Path(row['folder']) / 'reconstruction/multiview/depth_cache'
    folder = output / row['name']
    folder.mkdir(parents=True, exist_ok=True)
    totals, agreements, trusted_maps = {name: 0 for name in REASONS.values()}, [], 0
    for i, scan in scans.items():
        if not (cache / f'frame_{i:06d}.jpg.npz').exists():
            continue
        info, k = clip.infos[i], clip.k(i)
        width, height = int(info['Width']), int(info['Height'])
        centre = np.array([k[0, 2], k[1, 2]])
        points = scan['at_image']
        front = points[:, 2] > .5
        uv = warp((points[front, :2] / points[front, 2:]) * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]], centre, *theta)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        y, x = grid_cells(uv[inside], width, height)
        depth, reason = nearest(y, x, points[front, 2][inside].astype(np.float32), scan['support'][front][inside])
        excluded = label_mask(labels, row['scene'], info['OriginalImageName'], exclude)
        if excluded is not None:
            reason[excluded & (depth > 0)] = 5
        reliable = reason == 1
        extra = np.zeros(CACHE_GRID, np.float32)
        inverse = np.linalg.inv(clip.pose(i))
        for j in range(i - neighbours, i + neighbours + 1):
            if j == i or j not in scans:
                continue
            good = scans[j]['support'] == 1
            camera = scans[j]['world'][good] @ inverse[:3, :3].T + inverse[:3, 3]
            camera = camera[camera[:, 2] > .5]
            nuv = warp((camera[:, :2] / camera[:, 2:]) * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]], centre, *theta)
            keep = (nuv[:, 0] >= 0) & (nuv[:, 0] < width) & (nuv[:, 1] >= 0) & (nuv[:, 1] < height)
            ny, nx = grid_cells(nuv[keep], width, height)
            order = np.argsort(-camera[keep, 2])
            layer = np.zeros(CACHE_GRID, np.float32)
            layer[ny[order], nx[order]] = camera[keep, 2][order]
            extra = np.where((extra == 0) | ((layer > 0) & (layer < extra)), layer, extra)
        extra[(depth > 0) | (excluded if excluded is not None else False)] = 0
        map_depth = np.zeros(CACHE_GRID, np.float32)
        map_ok = False
        if handle is not None:
            map_depth = map_grid(handle, k, clip.pose(i), width, height)
            shared = reliable & (map_depth > 1)
            if shared.sum() >= MAP_MIN_PIXELS:
                log_ratio = np.log(map_depth[shared] / depth[shared])
                bias, scatter = float(np.median(log_ratio)), float(np.median(np.abs(log_ratio - np.median(log_ratio))))
                agreements.append((bias, scatter))
                map_ok = abs(bias) <= MAP_MAX_BIAS and scatter <= MAP_MAX_SPREAD
            if excluded is not None:
                map_depth[excluded] = 0
        trusted_maps += map_ok
        temporary = folder / f'frame_{i:06d}.partial.npz'
        np.savez_compressed(temporary, measured_depth_z_m=np.where(reliable, depth, 0).astype(np.float32), measured_valid=reliable,
                            neighbour_depth_z_m=extra, raw_depth_z_m=depth, reason=reason,
                            map_depth_z_m=map_depth if map_ok else np.zeros(CACHE_GRID, np.float32), map_trusted=np.bool_(map_ok),
                            K_native=k, metric_reference_pose=clip.pose(i), native_shape=np.array([height, width]))
        temporary.replace(folder / f'frame_{i:06d}.npz')
        for code, name in REASONS.items():
            if code:
                totals[name] += int((reason == code).sum())
    bias = np.array(agreements) if agreements else np.zeros((0, 2))
    return {'clip': row['name'], 'scene': row['scene'], 'status': 'built', 'frames': len(scans), 'gap_skipped_scans': skipped, 'counts': totals,
            'map_trusted_frames': trusted_maps, 'map_median_log_bias': float(np.median(bias[:, 0])) if len(bias) else None,
            'map_median_log_spread': float(np.median(bias[:, 1])) if len(bias) else None}


def write_scales(root: Path, contract: dict, output: Path, overlay: Path | None) -> None:
    """Metric scale per clip, from de-domed camera centres when available."""
    old = json.loads((root / contract['data']['provisional_scale_calibration']).read_text())
    rows = {}
    for name, row in old.items():
        report = overlay / name / 'report.json' if overlay else None
        if report is not None and report.exists():
            rows[name] = {**row, 'scale_m_per_visual_unit': json.loads(report.read_text())['alpha_m_per_visual'], 'source': 'de-domed camera centres'}
        else:
            rows[name] = {**row, 'source': 'original camera centres'}
    write_json(output / 'scale_calibration.json', rows)


def write_gate(root: Path, contract: dict, calibration: dict, reports: list, output: Path) -> dict:
    old = json.loads((root / contract['provisional_geometry_gate']).read_text())
    flights = {}
    for scene, row in old['flights'].items():
        area = next(a for a in calibration['areas'] if scene.startswith(a)) if any(scene.startswith(a) for a in calibration['areas']) else None
        outcome = calibration['areas'][area]['validation']['outcome'] if area else 'not_measured'
        blocked = [r for r in row['reasons'] if 'calibration constants' not in r and 'registration' not in r and 'overlays' not in r and 'parallax' not in r]
        status = 'FAILED' if blocked or outcome == 'offset' else 'QUALIFIED' if outcome == 'aligned' else 'UNRESOLVED'
        flights[scene] = {'status': status, 'registration_outcome': outcome, 'blocking_reasons': blocked,
                          'previous_status': row['status'], 'inspection': 'pending human review',
                          'bridge': 'photogrammetric map; agreement with it is not independent truth'}
    gate = {'schema_version': 2, 'stage': 'stage1_targets', 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'flights': flights,
            'rules': contract['geometry']['map_registration'], 'target_reports': reports,
            'calibration_sha256': sha256_file(root / contract['geometry_calibration']),
            'status_counts': {s: sum(f['status'] == s for f in flights.values()) for s in ('VERIFIED', 'QUALIFIED', 'FAILED', 'UNRESOLVED')}}
    write_json(root / contract['geometry_gate'], gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--neighbours', type=int, default=2)
    parser.add_argument('--clips', nargs='*')
    parser.add_argument('--no-map', action='store_true', help='Skip mesh rendering')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root, data = args.root.resolve(), contract['data']
    calibration = json.loads((root / contract['geometry_calibration']).read_text())
    rows = [r for r in clips(root, contract) if not args.clips or r['name'] in args.clips]
    if args.dry_run:
        print(json.dumps({'clips': len(rows), 'areas': sorted(calibration['areas'])}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    labels = root / data['labels'] if data.get('labels') and (root / data['labels']).exists() else None
    exclude = contract['geometry']['exclude_label_ids']
    overlay = root / data['overlay'] if data.get('overlay') and (root / data['overlay']).exists() else None
    write_scales(root, contract, args.output, overlay)
    gate_file, reports, handles = root / contract['geometry_gate'], {}, {}
    if gate_file.exists():
        earlier, current = json.loads(gate_file.read_text()), sha256_file(root / contract['geometry_calibration'])
        if earlier.get('calibration_sha256', current) == current:  # Finished clips stay finished until their calibration changes.
            reports = {r['clip']: r for r in earlier.get('target_reports', [])}
    for row in rows:
        if row['name'] in reports:
            continue
        if not args.no_map and row['area'] not in handles:
            handles = {row['area']: make_scene(root / data['terra'] / row['area'] / 'Mesh.ply')}  # One mesh in memory at a time.
        reports[row['name']] = build_clip(row, calibration, handles.get(row['area']), args.output, labels, exclude, args.neighbours)
        print(json.dumps(reports[row['name']]), flush=True)
        write_gate(root, contract, calibration, list(reports.values()), args.output)
    gate = write_gate(root, contract, calibration, list(reports.values()), args.output)
    write_status(args.output, 'stage1_targets', 'complete', json.dumps(gate['status_counts']), 'stage2_data_controls')
    print(json.dumps(gate['status_counts']))


if __name__ == '__main__':
    main()
