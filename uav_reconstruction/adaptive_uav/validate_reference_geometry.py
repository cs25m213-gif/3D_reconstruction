"""Stage 1: measure whether projected LiDAR can be trusted as depth supervision.

Each check emits numbers, not a verdict by inspection. A flight is only ever
QUALIFIED automatically: VERIFIED also needs a person to review the overlays.
Terra maps and raw bags are not used here, so per-point timing stays a limitation.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .audit_lidar_alignment import RAW_CAMERA
from .build_cached_lidar_targets import downsample_sparse_min
from .lidar_depth_supervision import extrinsics_for_scene, project_returns
from .utility_contracts import load_contract, sha256_file, write_json, write_status

# Published camera/LiDAR calibration, github.com/sijieaaa/UAVScenes
# calibration_results.py at 909250d14a4460b2a62ea25f08f12ce6367427ac.
PUBLISHED = {
    'AMtown': {'R': [[.00298088, -.999728, -.0231416], [-.00504636, .0231263, -.99972], [.999983, .00309683, -.00497605]],
               't': [.0025563, .0567484, -.0512149], 'K': [1453.72, 1453.28, 1172.18, 1041.78],
               'dist': [-.121, .1113, .0016, .00013, -.062353]},
    'AMvalley': {'R': [[.00298068, -.999735, -.0231428], [-.00504595, .023132, -.99974], [.999985, .00309701, -.00497598]],
                 't': [-.0025563, .0567484, -.0512149], 'K': [1453.88, 1452.85, 1182.53, 1045.82],
                 'dist': [-.052, .1168, .0015, .00013, -.068564]},
    'HKairport': {'R': [[.00322743, -.999736, -.022768], [-.0560389, .0227496, -.999725], [.999979, .00335414, -.00552898]],
                  't': [.00242461, .0765454, -.0313375], 'K': [1451.28, 1451.29, 1177.5, 1043.5],
                  'dist': [-.0572, .1209, .00124, -.00018, -.06327]},
    'HKisland': {'R': [[.00352762, -.999765, -.0213775], [-.0111803, .0213369, -.99971], [.999931, .00376561, -.0111025]],
                 't': [-.0025563, .0470454, -.0513375], 'K': [1444.43, 1444.34, 1177.8, 1043.6],
                 'dist': [-.053, .121, .00127, .00043, -.06495]},
}
PUBLISHED_HK_GNSS = {'R': [[.00363212, -.999819, -.0213618], [-.0111679, .0214512, -.999591], [.999879, .00375613, -.0111134]],
                     't': [-.0021928, .0470312, -.0513126], 'K': [1444.43, 1444.34, 1179.5, 1044.9],
                     'dist': [-.056, .118, .00122, .00064, -.0627]}
AREAS = tuple(PUBLISHED)
GRID, CACHE_GRID = (192, 256), (420, 504)
SHIFT_RANGE, SHIFT_STEP = 30, 2
SWEEP_SCALES = np.linspace(.90, 1.10, 41, dtype=np.float32)
SWEEP_MIN_NCC, SWEEP_MIN_PROMINENCE = .7, .15
SWEEP_BASELINE_M = (8., 30.)
RANGE_BINS_M = ((0., 80.), (80., 110.), (110., 250.))
OCCLUSION_MARGIN = .02
SCAN_PERIOD_S = .1
FOCAL_SCALES = np.linspace(.97, 1.03, 13)
BORESIGHT_LIMIT_DEG, BORESIGHT_CELL_M = 1., 2.
SUPPORT_RADIUS_M, SUPPORT_HEIGHT_M, MIN_SEGMENT_YAW_DEG = 1.5, .75, 45.
REASONS = {0: 'no_return', 1: 'reliable', 2: 'no_neighbour_support', 3: 'inconsistent_with_neighbour_scans', 4: 'conflicting_returns'}


def area_of(scene: str) -> str:
    return next(area for area in AREAS if scene.startswith(area))


def clips(root: Path, contract: dict) -> list[dict]:
    data, geometry, rows = contract['data'], contract['geometry'], []
    scale_file = root / data['scale_calibration']
    # Calibration runs before the new targets exist, so it starts from the provisional scales.
    scales = json.loads((scale_file if scale_file.exists() else root / data['provisional_scale_calibration']).read_text())
    for folder in sorted((root / data['cache_root']).iterdir()):
        info_file = folder / 'selected_infos.json'
        lidar = root / data['lidar_root'] / folder.name
        if not info_file.exists() or not (lidar / 'manifest.json').exists():
            continue
        scene = json.loads(info_file.read_text())[0]['OriginalSceneName']
        if 'GNSS' in scene:
            continue
        role = 'calibration' if folder.name.endswith(geometry['calibration_batch_suffix']) else 'validation'
        rows.append({'name': folder.name, 'scene': scene, 'area': area_of(scene), 'role': role,
                     'folder': str(folder), 'lidar': str(lidar), 'sidecars': str(root / data['sidecars'] / folder.name),
                     'all_infos': str(root / data['lidar_root'] / 'all_infos' / f'{scene}.json'),
                     'alpha': scales.get(folder.name, {}).get('scale_m_per_visual_unit')})
    return rows


def interpolate_pose(times: np.ndarray, infos: list, timestamp: float) -> np.ndarray:
    """pose_at_time with a cached time axis, because calibration asks many thousands of times."""
    position = np.searchsorted(times, timestamp)
    if position == 0 or position == len(times):
        raise ValueError('LiDAR timestamp outside reference camera trajectory')
    lo, hi = position - 1, position
    if times[hi] - times[lo] > 1.:
        raise ValueError('Reference pose interpolation spans >1 second')
    left, right = np.asarray(infos[lo]['T4x4'], dtype=np.float64), np.asarray(infos[hi]['T4x4'], dtype=np.float64)
    alpha = (timestamp - times[lo]) / (times[hi] - times[lo])
    rotation = Slerp([times[lo], times[hi]], Rotation.from_matrix(np.stack((left[:3, :3], right[:3, :3]))))([timestamp])[0]
    result = np.eye(4)
    result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = (1 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    return result


class Clip:
    def __init__(self, row: dict):
        self.row = row
        self.folder, self.lidar = Path(row['folder']), Path(row['lidar'])
        self.infos = json.loads((self.folder / 'selected_infos.json').read_text())
        self.all_infos = json.loads(Path(row['all_infos']).read_text())
        self.all_times = np.array([float(Path(i['OriginalImageName']).stem) for i in self.all_infos])
        self.rows = {r['frame']: r for r in json.loads((self.lidar / 'manifest.json').read_text())}
        self.frames = sorted(self.rows)
        self.rotation, self.translation = extrinsics_for_scene(row['scene'])

    def gray(self, index: int) -> np.ndarray | None:
        return cv2.imread(str(self.folder / 'source/frames' / f'frame_{index:06d}.jpg'), cv2.IMREAD_GRAYSCALE)

    def scan(self, index: int) -> np.ndarray:
        return np.loadtxt(self.lidar / f'frame_{index:06d}.txt', dtype=np.float64).reshape(-1, 3)

    def pose(self, index: int) -> np.ndarray:
        return np.asarray(self.infos[index]['T4x4'], dtype=np.float64)

    def lidar_pose(self, index: int, offset: float = 0.) -> np.ndarray:
        return interpolate_pose(self.all_times, self.all_infos, self.rows[index]['lidar_time'] + offset)

    def in_camera(self, index: int, xyz: np.ndarray, rotation=None, translation=None, offset: float = 0.) -> np.ndarray:
        """Returns in the camera at image time; matches project_returns' convention."""
        rotation = self.rotation if rotation is None else rotation
        translation = self.translation if translation is None else translation
        camera = xyz @ np.asarray(rotation).T + np.asarray(translation)
        relative = np.linalg.inv(self.pose(index)) @ self.lidar_pose(index, offset)
        return camera @ relative[:3, :3].T + relative[:3, 3]

    def k(self, index: int) -> np.ndarray:
        return np.asarray(self.infos[index]['P3x3'], dtype=np.float64)

    def sample(self, count: int) -> list[int]:
        usable = [f for f in self.frames if 2 <= f < len(self.infos) - 2]
        return [usable[i] for i in np.unique(np.linspace(0, len(usable) - 1, count).astype(int))]


def rotation_error(matrix) -> float:
    r = np.asarray(matrix, dtype=np.float64)
    return float(np.abs(r @ r.T - np.eye(3)).max())


def check_constants(tolerance: float, orthonormal: float) -> dict:
    rows = {}
    for area, published in PUBLISHED.items():
        r, t = extrinsics_for_scene(area + '01')
        raw_k, raw_dist = RAW_CAMERA[area]
        raw_k = np.asarray(raw_k)
        repo_k = [raw_k[0, 0], raw_k[1, 1], raw_k[0, 2], raw_k[1, 2]]
        same = {'rotation': bool(np.abs(r - np.asarray(published['R'])).max() <= tolerance),
                'translation': bool(np.abs(t - np.asarray(published['t'])).max() <= tolerance),
                'raw_intrinsics': bool(np.abs(np.asarray(repo_k) - np.asarray(published['K'])).max() <= tolerance),
                'raw_distortion': bool(np.abs(np.asarray(raw_dist) - np.asarray(published['dist'])).max() <= tolerance)}
        gnss = bool(np.abs(r - np.asarray(PUBLISHED_HK_GNSS['R'])).max() <= tolerance)
        rows[area] = {'repo_matches_published': same, 'repo_rotation_is_hk_gnss': gnss,
                      'repo_rotation_orthonormal_error': rotation_error(r),
                      'published_rotation_orthonormal_error': rotation_error(published['R']),
                      'published_rotation_valid': rotation_error(published['R']) <= orthonormal,
                      'status': 'PASS' if all(same.values()) else 'FAIL'}
    return rows


def rotation_candidates(area: str) -> dict:
    """Named calibrations to compare by measurement, never by assumption."""
    r, t = extrinsics_for_scene(area + '01')
    rows = {'repo': (r, t), 'published': (np.asarray(PUBLISHED[area]['R']), np.asarray(PUBLISHED[area]['t']))}
    if area == 'HKairport':
        fixed = np.array(PUBLISHED[area]['R'])
        fixed[1, 0] = -.00560389  # One-decimal-place reading of the non-orthonormal entry.
        rows['published_decimal_fixed'] = (fixed, np.asarray(PUBLISHED[area]['t']))
    return rows


def check_poses(row: dict) -> dict:
    infos = json.loads(Path(row['all_infos']).read_text())
    t = np.array([float(Path(i['OriginalImageName']).stem) for i in infos])
    poses = np.array([i['T4x4'] for i in infos], dtype=np.float64)
    dt = np.diff(t)
    speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / np.maximum(dt, 1e-6)
    errors = [rotation_error(p[:3, :3]) for p in poses[::10]]
    return {'poses': len(poses), 'max_rotation_orthonormal_error': float(max(errors)),
            'min_determinant': float(np.linalg.det(poses[:, :3, :3]).min()),
            'time_monotonic': bool((dt > 0).all()), 'median_gap_s': float(np.median(dt)), 'max_gap_s': float(dt.max()),
            'speed_median_mps': float(np.median(speed)), 'speed_p99_mps': float(np.percentile(speed, 99)),
            'median_camera_z_dot_world_z': float(np.median(poses[:, 2, 2]))}


def direction_test(row: dict, pairs: int = 6) -> dict:
    """Adjacent scans should agree in the world only under the right pose direction."""
    clip = Clip(row)
    forward, inverse = [], []
    for i in clip.sample(pairs + 1)[:pairs]:
        if i + 1 not in clip.rows:
            continue
        a, b = [(clip.scan(j) @ clip.rotation.T + clip.translation)[::4] for j in (i, i + 1)]
        pa, pb = clip.lidar_pose(i), clip.lidar_pose(i + 1)
        ia, ib = np.linalg.inv(pa), np.linalg.inv(pb)
        forward.append(np.median(cKDTree(b @ pb[:3, :3].T + pb[:3, 3]).query(a @ pa[:3, :3].T + pa[:3, 3], workers=1)[0]))
        inverse.append(np.median(cKDTree(b @ ib[:3, :3].T + ib[:3, 3]).query(a @ ia[:3, :3].T + ia[:3, 3], workers=1)[0]))
    return {'clip': row['name'], 'camera_to_world_median_nn_m': float(np.median(forward)),
            'world_to_camera_median_nn_m': float(np.median(inverse))}


def check_pairing(row: dict, samples: int = 6) -> dict:
    """Is cache frame N the same picture as LiDAR frame N? Compares ±3 frames too."""
    clip = Clip(row)
    cache = clip.folder / 'reconstruction/multiview/depth_cache'
    files = sorted(cache.glob('frame_*.jpg.npz'))
    selected = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, samples).astype(int))]
    best, margins, ks = [], [], []
    for file in selected:
        index = int(file.name.split('_')[1].split('.')[0])
        with np.load(file) as cached:
            rgb, k = cached['rgb'], cached['K']
        small = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        small = (small - small.mean()) / (small.std() + 1e-6)
        scores = {}
        for offset in (-3, -2, -1, 0, 1, 2, 3):
            if not 0 <= index + offset < len(clip.infos):
                continue
            image = clip.gray(index + offset)
            if image is None:
                continue
            native = cv2.resize(image, rgb.shape[1::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
            scores[offset] = float((small * ((native - native.mean()) / (native.std() + 1e-6))).mean())
        best.append(max(scores, key=scores.get))
        margins.append(scores[0] - max(v for o, v in scores.items() if o != 0))
        ref = clip.k(index)
        ks.append(abs(float(k[0, 0]) / (ref[0, 0] * rgb.shape[1] / clip.infos[index]['Width']) - 1))
    return {'clip': row['name'], 'samples': len(best), 'best_offsets': best, 'min_margin': float(min(margins)),
            'median_margin': float(np.median(margins)), 'max_focal_relative_difference': float(max(ks))}


def pixel_chain(native=(2048, 2448)) -> dict:
    """Follow native pixels through the real target path and compare with the RGB path."""
    h, w = native
    errors = []
    for u, v in ((100, 90), (612, 512), (1224, 1024), (1836, 1536), (2347, 1957), (733, 1381)):
        depth = np.zeros((h, w), np.float32)
        depth[v, u] = 10.
        cached = downsample_sparse_min(depth, CACHE_GRID)
        final = downsample_sparse_min(cached, GRID)
        gy, gx = np.argwhere(final > 0)[0]
        image = np.zeros((h, w), np.float32)
        image[max(v - 8, 0):v + 9, max(u - 8, 0):u + 9] = 1.
        small = cv2.resize(cv2.resize(image, CACHE_GRID[::-1], interpolation=cv2.INTER_AREA), GRID[::-1],
                           interpolation=cv2.INTER_AREA)
        ys, xs = np.mgrid[:small.shape[0], :small.shape[1]]
        cx, cy = float((xs * small).sum() / small.sum()), float((ys * small).sum() / small.sum())
        errors.append([float(gx - cx), float(gy - cy)])
    errors = np.asarray(errors)
    scale = GRID[1] / w
    return {'target_minus_rgb_grid_px_mean': errors.mean(0).tolist(), 'target_minus_rgb_grid_px_max_abs': np.abs(errors).max(0).tolist(),
            'target_minus_rgb_native_px_mean': (errors.mean(0) / scale).tolist(),
            'k_half_pixel_grid_px': float((scale - 1) / 2),
            'k_note': 'K is scaled by s, but pixel centres map as (u+.5)s-.5, so K and RGB differ by this constant.'}


def check_projection(row: dict, samples: int = 6) -> dict:
    clip = Clip(row)
    rows = []
    for i in clip.sample(samples):
        depth, stats = project_returns(clip.scan(i), clip.infos[i], camera_at_lidar=clip.lidar_pose(i))
        rows.append({**stats, 'collisions': stats['in_image_returns'] - stats['valid_pixels']})
    total = lambda key: sum(r[key] for r in rows)
    return {'clip': row['name'], 'input_returns': total('input_returns'), 'positive_z_fraction': total('positive_z_returns') / max(total('input_returns'), 1),
            'in_image_fraction': total('in_image_returns') / max(total('input_returns'), 1),
            'collision_fraction': total('collisions') / max(total('in_image_returns'), 1),
            'median_range_minus_z_m': float(np.median([r['mean_range_minus_z_m'] for r in rows])),
            'median_depth_m': float(np.median([r['depth_median_m'] for r in rows]))}


def check_timing(row: dict) -> dict:
    clip = Clip(row)
    offsets = np.abs([r['sync_offset_ms'] for r in clip.rows.values()])
    centres = np.array([clip.pose(i)[:3, 3] for i in clip.frames])
    times = np.array([clip.rows[i]['image_time'] for i in clip.frames])
    speed = np.linalg.norm(np.diff(centres, axis=0), axis=1) / np.maximum(np.diff(times), 1e-6)
    return {'clip': row['name'], 'frames': len(clip.frames), 'median_abs_sync_ms': float(np.median(offsets)),
            'max_abs_sync_ms': float(offsets.max()), 'speed_p95_mps': float(np.percentile(speed, 95)),
            'scan_gap_s_max': float(np.diff(sorted(r['lidar_time'] for r in clip.rows.values())).max())}


def boundaries(camera: np.ndarray, k: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Midpoints of consecutive returns across a depth jump; same rule as score_pairs."""
    h, w = shape
    uv = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k, np.zeros(5))[0].reshape(-1, 2)
    midpoint = .5 * (uv[:-1] + uv[1:])
    jump = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    valid = ((camera[:-1, 2] > 1) & (camera[1:, 2] > 1) & (jump > 2) & (jump < 30) &
             (np.abs(np.diff(camera[:, 2])) > 2) & np.isfinite(midpoint).all(1))
    xy = midpoint[valid]
    edge = SHIFT_RANGE + 2
    return xy[(xy[:, 0] > edge) & (xy[:, 0] < w - edge) & (xy[:, 1] > edge) & (xy[:, 1] < h - edge)].astype(int)


def shift_map(gradient: np.ndarray, xy: np.ndarray) -> np.ndarray:
    steps = np.arange(-SHIFT_RANGE, SHIFT_RANGE + 1, SHIFT_STEP)
    result = np.empty((len(steps), len(steps)), np.float64)
    for row, dy in enumerate(steps):
        for column, dx in enumerate(steps):
            result[row, column] = gradient[xy[:, 1] + dy, xy[:, 0] + dx].mean()
    return (result - result.mean()) / (result.std() + 1e-9)


def registration_clip(row: dict, frames: int = 30) -> dict | None:
    clip = Clip(row)
    rng = np.random.default_rng(sum(map(ord, row['name'])))
    real, wrong, shuffled, perturbed, counts = [], [], [], [], []
    focal = np.zeros(len(FOCAL_SCALES))
    tilt = cv2.Rodrigues(np.array([0., np.radians(.5), 0.]))[0]
    selected = clip.sample(frames)
    for i in selected:
        image = clip.gray(i)
        other = clip.gray(selected[(selected.index(i) + len(selected) // 2) % len(selected)])
        if image is None or other is None:
            continue
        gradient, other_gradient = [
            cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)), (0, 0), 2.)
            for a in (image, other)]
        try:
            camera = clip.in_camera(i, clip.scan(i))
        except ValueError:
            continue
        k = clip.k(i)
        xy = boundaries(camera, k, image.shape)
        if len(xy) < 50:
            continue
        real.append(shift_map(gradient, xy))
        for f, scale in enumerate(FOCAL_SCALES):
            zoomed = k.copy()
            zoomed[[0, 1], [0, 1]] *= scale
            at = boundaries(camera, zoomed, image.shape)
            focal[f] += gradient[at[:, 1], at[:, 0]].mean() / gradient.mean() if len(at) else 0.
        wrong.append(shift_map(other_gradient, xy))
        random_xy = rng.integers([32, 32], [image.shape[1] - 32, image.shape[0] - 32], size=xy.shape)
        shuffled.append(shift_map(gradient, random_xy))
        tilted = boundaries(camera @ tilt.T, k, image.shape)
        if len(tilted) >= 50:
            perturbed.append(shift_map(gradient, tilted))
        counts.append(len(xy))
    if not real:
        return None
    return {'clip': row['name'], 'scene': row['scene'], 'area': row['area'], 'role': row['role'],
            'real': np.stack(real), 'wrong': np.stack(wrong), 'shuffled': np.stack(shuffled),
            'perturbed': np.stack(perturbed) if perturbed else None, 'boundary_points': int(sum(counts)),
            'focal': focal / len(real)}


def peak(maps: np.ndarray) -> dict:
    mean = maps.mean(0)
    y, x = np.unravel_index(np.argmax(mean), mean.shape)
    steps = np.arange(-SHIFT_RANGE, SHIFT_RANGE + 1, SHIFT_STEP)
    ys, xs = np.ogrid[:mean.shape[0], :mean.shape[1]]
    away = (ys - y) ** 2 + (xs - x) ** 2 > 16
    centre = float(np.median(mean[away]))
    spread = 1.4826 * float(np.median(np.abs(mean[away] - centre))) + 1e-9
    return {'dx': int(steps[x]), 'dy': int(steps[y]), 'z': float((mean[y, x] - centre) / spread),
            'frames': int(len(maps))}


def peak_interval(maps: np.ndarray, draws: int = 200) -> dict:
    rng = np.random.default_rng(0)
    rows = [peak(maps[rng.integers(0, len(maps), len(maps))]) for _ in range(draws)]
    dx, dy = np.array([r['dx'] for r in rows]), np.array([r['dy'] for r in rows])
    return {'dx_95': np.percentile(dx, [2.5, 97.5]).tolist(), 'dy_95': np.percentile(dy, [2.5, 97.5]).tolist()}


def registration_summary(rows: list[dict], geometry: dict) -> dict:
    summary = {}
    for scope in sorted({r['scene'] for r in rows} | {r['area'] for r in rows}):
        subset = [r for r in rows if scope in (r['scene'], r['area'])]
        result = {}
        for role in ('calibration', 'validation'):
            chosen = [r for r in subset if r['role'] == role]
            if not chosen:
                continue
            real = np.concatenate([r['real'] for r in chosen])
            result[role] = {**peak(real), **peak_interval(real), 'boundary_points': sum(r['boundary_points'] for r in chosen),
                            'wrong_pairing': peak(np.concatenate([r['wrong'] for r in chosen])),
                            'shuffled_positions': peak(np.concatenate([r['shuffled'] for r in chosen]))}
            tilted = [r['perturbed'] for r in chosen if r['perturbed'] is not None]
            if tilted:
                result[role]['rotation_0p5deg'] = peak(np.concatenate(tilted))
            curve = np.mean([r['focal'] for r in chosen], axis=0)
            result[role]['focal_scale_curve'] = {f'{s:.3f}': float(v) for s, v in zip(FOCAL_SCALES, curve)}
            result[role]['best_focal_scale'] = float(FOCAL_SCALES[int(np.argmax(curve))])
            power = result[role]['z'] >= geometry['registration_peak_min_z']
            clean = max(result[role]['wrong_pairing']['z'], result[role]['shuffled_positions']['z']) < result[role]['z']
            close = np.hypot(result[role]['dx'], result[role]['dy']) <= geometry['registration_tolerance_native_px']
            result[role]['outcome'] = ('no_power' if not (power and clean) else 'aligned' if close else 'offset')
        summary[scope] = result
    return summary


def sweep_frame(clip: Clip, index: int, sources: list[int], points: np.ndarray, rng, count: int = 500):
    ref = clip.gray(index)
    grays = [clip.gray(s) for s in sources]
    if ref is None or any(g is None for g in grays):
        return None
    ref = cv2.GaussianBlur(ref, (0, 0), 1.).astype(np.float32)
    grays = [cv2.GaussianBlur(g, (0, 0), 1.).astype(np.float32) for g in grays]
    h, w = ref.shape
    k = clip.k(index)
    uvw = (k @ points.T).T
    uv = uvw[:, :2] / uvw[:, 2:]
    z = points[:, 2]
    keep = (z > 5) & (uv[:, 0] > 30) & (uv[:, 0] < w - 30) & (uv[:, 1] > 30) & (uv[:, 1] < h - 30)
    uv, z = uv[keep], z[keep]
    if len(uv) < 50:
        return None
    pick = rng.choice(len(uv), min(count, len(uv)), replace=False)
    uv, z = uv[pick].astype(np.float32), z[pick].astype(np.float32)
    offsets = np.arange(-10, 11, 2, dtype=np.float32)
    dy, dx = np.meshgrid(offsets, offsets, indexing='ij')
    dx, dy = dx.ravel(), dy.ravel()
    px, py = uv[:, 0, None] + dx, uv[:, 1, None] + dy
    patches = cv2.remap(ref, px, py, cv2.INTER_LINEAR)
    patches = patches - patches.mean(1, keepdims=True)
    energy = np.sqrt((patches ** 2).sum(1))
    textured = energy / np.sqrt(patches.shape[1]) > 4.
    if textured.sum() < 20:
        return None
    uv, z, px, py, patches, energy = uv[textured], z[textured], px[textured], py[textured], patches[textured], energy[textured]
    fx, fy, cx, cy = map(float, (k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
    rays = np.stack(((px - cx) / fx, (py - cy) / fy, np.ones_like(px)), -1) * z[:, None, None]
    scores = np.zeros((len(SWEEP_SCALES), len(uv)), np.float32)
    for source, gray in zip(sources, grays):
        relative = np.linalg.inv(clip.pose(source)) @ clip.pose(index)
        rotation, translation = relative[:3, :3].astype(np.float32), relative[:3, 3].astype(np.float32)
        for s, scale in enumerate(SWEEP_SCALES):
            camera = rays * scale @ rotation.T + translation
            mapx = fx * camera[..., 0] / camera[..., 2] + cx
            mapy = fy * camera[..., 1] / camera[..., 2] + cy
            warped = cv2.remap(gray, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped = warped - warped.mean(1, keepdims=True)
            scores[s] += (warped * patches).sum(1) / (np.sqrt((warped ** 2).sum(1)) * energy + 1e-6)
    scores /= len(sources)
    best = scores.argmax(0)
    rows = np.arange(scores.shape[1])
    ncc = scores[best, rows]
    prominence = ncc - np.median(scores, axis=0)
    interior = (best > 0) & (best < len(SWEEP_SCALES) - 1)
    accepted = interior & (ncc >= SWEEP_MIN_NCC) & (prominence >= SWEEP_MIN_PROMINENCE)
    radius = np.hypot(uv[:, 0] - cx, uv[:, 1] - cy) / np.hypot(w / 2, h / 2)
    return {'scale': SWEEP_SCALES[best[accepted]], 'z': z[accepted], 'radius': radius[accepted],
            'ncc': ncc[accepted], 'tested': int(len(uv))}


def pick_sources(clip: Clip, index: int) -> list[int]:
    centre = clip.pose(index)[:3, 3]
    sources = []
    for direction in (-1, 1):
        for j in range(index + direction, len(clip.infos) if direction > 0 else -1, direction):
            if SWEEP_BASELINE_M[0] <= np.linalg.norm(clip.pose(j)[:3, 3] - centre) <= SWEEP_BASELINE_M[1]:
                sources.append(j)
                break
    return sources


def depth_sweep_clip(row: dict, frames: int = 12) -> dict | None:
    clip = Clip(row)
    rng = np.random.default_rng(sum(map(ord, row['name'])))
    real, control, used = [], [], 0
    far = max(1, len(clip.infos) // 2)
    for i in clip.sample(frames):
        sources = pick_sources(clip, i)
        if not sources:
            continue
        try:
            points = clip.in_camera(i, clip.scan(i))
        except ValueError:
            continue
        result = sweep_frame(clip, i, sources, points, rng)
        if result is None:
            continue
        used += 1
        real.append({**result, 'frame': i})
        # Wrong pairing: this frame's points against the view at the far end of the clip.
        wrong = sweep_frame(clip, (i + far) % len(clip.infos), sources, points, rng)
        if wrong is not None:
            control.append(wrong)
    if not real:
        return None
    return {'clip': row['name'], 'scene': row['scene'], 'area': row['area'], 'role': row['role'], 'frames': used,
            'real': real, 'control_tested': sum(c['tested'] for c in control), 'control_accepted': sum(len(c['scale']) for c in control)}


def bias_stats(frames: list[dict]) -> dict | None:
    medians = [100 * (float(np.median(f['scale'])) - 1) for f in frames if len(f['scale']) >= 5]
    if len(medians) < 3:
        return None
    rng = np.random.default_rng(0)
    draws = [np.median(rng.choice(medians, len(medians))) for _ in range(1000)]
    return {'parallax_minus_lidar_percent': float(np.median(medians)),
            'ci95_percent': np.percentile(draws, [2.5, 97.5]).tolist(), 'frames': len(medians),
            'points': int(sum(len(f['scale']) for f in frames))}


def depth_sweep_summary(rows: list[dict]) -> dict:
    summary = {}
    for scope in sorted({r['scene'] for r in rows} | {r['area'] for r in rows}):
        subset = [r for r in rows if scope in (r['scene'], r['area'])]
        frames = [f for r in subset for f in r['real']]
        tested = sum(f['tested'] for f in frames)
        accepted = sum(len(f['scale']) for f in frames)
        result = {'all': bias_stats(frames), 'tested': tested, 'accepted': accepted,
                  'wrong_pairing_tested': sum(r['control_tested'] for r in subset),
                  'wrong_pairing_accepted': sum(r['control_accepted'] for r in subset),
                  'by_range_m': {}, 'by_radius': {}, 'by_role': {}}
        for role in ('calibration', 'validation'):
            result['by_role'][role] = bias_stats([f for r in subset if r['role'] == role for f in r['real']])
        for low, high in RANGE_BINS_M:
            part = [{'scale': f['scale'][(f['z'] >= low) & (f['z'] < high)]} for f in frames]
            result['by_range_m'][f'{low:.0f}-{high:.0f}'] = bias_stats(part)
        for label, low, high in (('inner', 0., .4), ('middle', .4, .7), ('outer', .7, 2.)):
            part = [{'scale': f['scale'][(f['radius'] >= low) & (f['radius'] < high)]} for f in frames]
            result['by_radius'][label] = bias_stats(part)
        summary[scope] = result
    return summary


def segments(row: dict, parts: int = 2) -> list[dict]:
    """Split a clip so a turning flight is not described by one heading."""
    clip = Clip(row)
    usable, poses = [], []
    for i in clip.frames:
        try:
            poses.append(clip.lidar_pose(i))
            usable.append(i)
        except ValueError:  # Real pose gap; the target builder rejects these scans too.
            continue
    rows = []
    for chunk in np.array_split(np.arange(len(usable)), parts):
        if len(chunk) < 8:
            continue
        chosen = [poses[j] for j in chunk]
        axis = np.mean([p[:2, 0] for p in chosen], axis=0)
        rows.append({'name': f"{row['name']}#{chunk[0]}", 'yaw': float(np.degrees(np.arctan2(axis[1], axis[0]))),
                     'straightness': float(np.linalg.norm(axis)), 'poses': chosen,
                     'camera': [clip.scan(usable[j])[::4] @ clip.rotation.T for j in chunk],
                     'translation': clip.translation})
    return rows


def segment_cells(segment: dict, tilt: np.ndarray) -> np.ndarray:
    """(cell key, mean height) for one segment after a camera-frame rotation error."""
    parts = []
    for camera, pose in zip(segment['camera'], segment['poses']):
        parts.append((camera @ tilt.T + segment['translation']) @ pose[:3, :3].T + pose[:3, 3])
    points = np.concatenate(parts)
    keys = np.floor(points[:, :2] / BORESIGHT_CELL_M).astype(np.int64)
    packed = (keys[:, 0] + 100000) * 200001 + keys[:, 1] + 100000
    unique, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    heights = np.bincount(inverse, points[:, 2]) / counts
    keep = counts >= 4
    return np.c_[unique[keep], heights[keep]]


def pair_spread(first: np.ndarray, second: np.ndarray) -> float | None:
    _, a, b = np.intersect1d(first[:, 0], second[:, 0], return_indices=True)
    if len(a) < 300:
        return None
    difference = first[a, 1] - second[b, 1]
    return float(1.4826 * np.median(np.abs(difference - np.median(difference))))


def overlapping_pairs(parts: list[dict]) -> list[tuple[int, int]]:
    cells = [segment_cells(p, np.eye(3)) for p in parts]
    pairs = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            turn = abs((parts[i]['yaw'] - parts[j]['yaw'] + 180) % 360 - 180)
            if turn >= MIN_SEGMENT_YAW_DEG and pair_spread(cells[i], cells[j]) is not None:
                pairs.append((i, j))
    return pairs


def spread_at(parts, pairs, tilt: np.ndarray) -> list[float]:
    used = sorted({i for pair in pairs for i in pair})
    cells = {i: segment_cells(parts[i], tilt) for i in used}
    return [pair_spread(cells[i], cells[j]) or np.nan for i, j in pairs]


def search_boresight(parts, pairs, base: np.ndarray) -> dict:
    """Smallest disagreement over roll/pitch; base is the assumed error already present."""
    best = (np.inf, 0., 0.)
    for step, span, centre in ((.25, BORESIGHT_LIMIT_DEG, (0., 0.)), (.05, .25, None)):
        centre = centre if centre is not None else best[1:]
        for a in np.arange(centre[0] - span, centre[0] + span + 1e-6, step):
            for b in np.arange(centre[1] - span, centre[1] + span + 1e-6, step):
                tilt = cv2.Rodrigues(np.radians([a, b, 0.]))[0] @ base
                value = float(np.nanmean(spread_at(parts, pairs, tilt)))
                if value < best[0]:
                    best = (value, float(a), float(b))
    return {'spread_m': best[0], 'correction_x_deg': best[1], 'correction_y_deg': best[2]}


def boresight_area(item) -> dict:
    area, rows = item
    parts = [segment for row in rows for segment in segments(row)]
    pairs = overlapping_pairs(parts)
    if not pairs:
        return {'area': area, 'status': 'no_observable_pairs', 'pairs': 0}
    identity = np.eye(3)
    injected = cv2.Rodrigues(np.radians([.3, -.2, 0.]))[0]
    found, control = search_boresight(parts, pairs, identity), search_boresight(parts, pairs, injected)
    recovered = [control['correction_x_deg'] - found['correction_x_deg'], control['correction_y_deg'] - found['correction_y_deg']]
    baseline = float(np.nanmean(spread_at(parts, pairs, identity)))
    candidates = {}
    repo_r, _ = rotation_candidates(area)['repo']
    for name, (rotation, _) in rotation_candidates(area).items():
        # Spread with the candidate substituted for the repo rotation.
        change = rotation @ repo_r.T
        candidates[name] = {'spread_m': float(np.nanmean(spread_at(parts, pairs, change))),
                            'difference_from_repo_deg': float(np.degrees(np.linalg.norm(cv2.Rodrigues(change)[0])))}
    per_pair = []
    for pair in pairs[:6]:
        per_pair.append({'segments': [parts[pair[0]]['name'], parts[pair[1]]['name']],
                         'turn_deg': float(abs((parts[pair[0]]['yaw'] - parts[pair[1]]['yaw'] + 180) % 360 - 180)),
                         **search_boresight(parts, [pair], identity)})
    return {'area': area, 'status': 'measured', 'pairs': len(pairs), 'baseline_spread_m': baseline, 'best': found,
            'candidates': candidates, 'per_pair': per_pair,
            'control_injected_deg': [.3, -.2], 'control_recovered_deg': recovered}


def decomposition_clip(row: dict) -> dict | None:
    """Signed metric error of teacher and anchors, and where the scale could come from."""
    if not row['alpha']:
        return None
    teacher, anchors, depth_m, stored_scale, centres = [], [], [], [], []
    cache = Path(row['folder']) / 'reconstruction/multiview/depth_cache'
    infos = json.loads((Path(row['folder']) / 'selected_infos.json').read_text())
    for target in sorted(Path(row['sidecars']).glob('frame_*.npz')):
        file = cache / (target.stem + '.jpg.npz')
        if not file.exists():
            continue
        with np.load(target) as sidecar:
            measured = sidecar['measured_depth_z_m']
        with np.load(file) as cached:
            d, a, pose = cached['depth'], cached['anchors'], cached['P'].astype(np.float64)
            stored_scale.append(float(cached['scale']))
        valid = measured > 0
        depth_m.append(measured[valid])
        teacher.append(d[valid] * row['alpha'] - measured[valid])
        mask = valid & np.isfinite(a) & (a > 0)
        anchors.append(np.c_[a[mask], measured[mask]])
        index = int(target.stem.split('_')[1])
        if np.isfinite(pose).all():
            centres.append((-pose[:3, :3].T @ pose[:3, 3], np.asarray(infos[index]['T4x4'])[:3, 3]))
    if not depth_m:
        return None
    depth_m, teacher, anchors = map(np.concatenate, (depth_m, teacher, anchors))
    visual, metric = map(np.asarray, zip(*centres))
    gap = max(1, len(visual) // 4)
    ratios = np.linalg.norm(metric[gap:] - metric[:-gap], axis=1) / np.maximum(np.linalg.norm(visual[gap:] - visual[:-gap], axis=1), 1e-9)
    lidar_alpha = float(np.median(anchors[:, 1] / anchors[:, 0]))
    return {'clip': row['name'], 'scene': row['scene'], 'pixels': len(depth_m), 'median_lidar_m': float(np.median(depth_m)),
            'teacher_minus_lidar_median_m': float(np.median(teacher)),
            'teacher_minus_lidar_percent': float(100 * np.median(teacher / depth_m)),
            'anchor_pixels': len(anchors),
            'anchor_minus_lidar_percent': float(100 * (row['alpha'] / lidar_alpha - 1)),
            'alpha_umeyama': row['alpha'], 'alpha_pairwise_baselines': float(np.median(ratios)),
            'alpha_from_lidar_anchors': lidar_alpha, 'median_stored_cache_scale': float(np.median(stored_scale))}


def reliability_clip(item) -> dict | None:
    """Per-return static support, written beside (never over) the provisional sidecars."""
    row, output = item
    clip = Clip(row)
    batch = Path(output) / row['name']
    batch.mkdir(parents=True, exist_ok=True)
    totals, by_range, agree = {name: 0 for name in REASONS.values()}, {}, []
    worlds = {}

    def world(index):
        if index not in worlds:
            camera = clip.scan(index) @ clip.rotation.T + clip.translation
            pose = clip.lidar_pose(index)
            worlds[index] = camera @ pose[:3, :3].T + pose[:3, 3]
        return worlds[index]

    for target in sorted(Path(row['sidecars']).glob('frame_*.npz')):
        index = int(target.stem.split('_')[1])
        try:
            points = clip.in_camera(index, clip.scan(index))
            here = world(index)
            nearby = [world(j) for j in (index - 1, index + 1) if j in clip.rows]
        except ValueError:
            continue
        k = clip.k(index)
        front = (points[:, 2] > .5) & (points[:, 2] < 250)
        uv = (points[:, :2] / np.maximum(points[:, 2:], 1e-6)) * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
        w, h = int(clip.infos[index]['Width']), int(clip.infos[index]['Height'])
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        support = np.zeros(len(points), np.uint8)
        if nearby:
            others = np.concatenate(nearby)
            tree = cKDTree(others[:, :2])
            distance, neighbours = tree.query(here[:, :2], k=6, distance_upper_bound=SUPPORT_RADIUS_M, workers=1)
            found = neighbours < len(others)
            heights = np.where(found, others[np.minimum(neighbours, len(others) - 1), 2], np.inf)
            support = np.where(~found.any(1), 2, np.where(np.abs(heights - here[:, 2:3]).min(1) <= SUPPORT_HEIGHT_M, 1, 3)).astype(np.uint8)
        else:
            support[:] = 2
        ch, cw = CACHE_GRID
        y = (np.floor(uv[inside, 1]).astype(np.int64) * ch // h).clip(0, ch - 1)
        x = (np.floor(uv[inside, 0]).astype(np.int64) * cw // w).clip(0, cw - 1)
        z, flag = points[inside, 2].astype(np.float32), support[inside]
        order = np.argsort(-z)
        depth = np.zeros(CACHE_GRID, np.float32)
        reason = np.zeros(CACHE_GRID, np.uint8)
        far = np.zeros(CACHE_GRID, np.float32)
        np.maximum.at(far, (y, x), z)
        depth[y[order], x[order]] = z[order]  # Nearest is written last, as in the z-buffer.
        reason[y[order], x[order]] = flag[order]
        valid = depth > 0
        reason[valid & (far > depth * (1 + OCCLUSION_MARGIN))] = 4
        with np.load(target) as sidecar:
            original = sidecar['measured_depth_z_m']
        both = valid & (original > 0)
        agree.append(float(np.mean(np.abs(depth[both] - original[both]) < .05)) if both.any() else 0.)
        np.savez_compressed(batch / target.name, measured_depth_z_m=depth, raw_valid=valid, reliable=reason == 1, reason=reason)
        for code, name in REASONS.items():
            if code:
                totals[name] += int((reason == code).sum())
        for low, high in RANGE_BINS_M:
            key, band = f'{low:.0f}-{high:.0f}', (depth >= low) & (depth < high) & valid
            pair = by_range.setdefault(key, [0, 0])
            pair[0], pair[1] = pair[0] + int(band.sum()), pair[1] + int((band & (reason == 1)).sum())
        for j in list(worlds):
            if j < index - 1:
                del worlds[j]
    return {'clip': row['name'], 'scene': row['scene'], 'frames': len(agree), 'counts': totals, 'by_range_m': by_range,
            'sidecar_reproduction_rate': float(np.mean(agree)) if agree else None}


def overlay_clip(item) -> list[str]:
    """Native RGB with projected returns beside a deliberately shifted control."""
    row, output = item
    clip = Clip(row)
    written = []
    for i in clip.sample(9)[2::3]:
        image = cv2.imread(str(clip.folder / 'source/frames' / f'frame_{i:06d}.jpg'))
        try:
            camera = clip.in_camera(i, clip.scan(i))
        except ValueError:
            continue
        if image is None:
            continue
        k, h = clip.k(i), image.shape[0]
        xy = boundaries(camera, k, image.shape[:2])
        if len(xy) < 30:
            continue
        centre = np.median(xy, axis=0).astype(int)
        cx, cy = int(np.clip(centre[0], 400, image.shape[1] - 401)), int(np.clip(centre[1], 300, h - 301))
        uv = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k, np.zeros(5))[0].reshape(-1, 2)
        low, high = np.percentile(camera[:, 2], [2, 98])
        hue = (255 * np.clip((camera[:, 2] - low) / max(high - low, 1), 0, 1)).astype(np.uint8)
        colours = cv2.applyColorMap(hue[:, None], cv2.COLORMAP_TURBO)[:, 0]
        panels = []
        for shift in (0, 25):
            canvas = image.copy()
            for (u, v), colour in zip(uv.astype(int), colours):
                if 0 <= u + shift < canvas.shape[1] and 0 <= v < h:
                    cv2.circle(canvas, (int(u + shift), int(v)), 2, tuple(int(c) for c in colour), -1, cv2.LINE_AA)
            crop = canvas[cy - 300:cy + 300, cx - 400:cx + 400].copy()
            label = f"{row['scene']} frame {i}: {'as projected' if not shift else 'control, shifted +25 px'}"
            cv2.putText(crop, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(crop)
        path = Path(output) / f"{row['name']}_frame_{i:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(path.name)
    return written


def run_pool(function, items, workers: int):
    with Pool(workers) as pool:
        return [row for row in pool.imap(function, items) if row is not None]


def flight_gate(scene: str, results: dict, geometry: dict, intrinsic_f: dict) -> dict:
    """Precedence: FAILED, then UNRESOLVED, then QUALIFIED. VERIFIED needs a human review."""
    area = area_of(scene)
    reasons, of_flight = [], lambda rows: [r for r in rows if r['clip'].startswith(scene)]
    constants = results['constants'][area]
    if constants['status'] != 'PASS':
        reasons.append('calibration constants differ from the published values for this area')
    pairing = of_flight(results['pairing'])
    if any(set(p['best_offsets']) != {0} or p['min_margin'] <= 0 for p in pairing):
        reasons.append('cached RGB does not match its LiDAR frame index')
    pose = results['poses'][scene]
    if pose['max_rotation_orthonormal_error'] > geometry['rotation_orthonormal_tolerance'] or not pose['time_monotonic']:
        reasons.append('invalid reference poses')
    direction = of_flight(results['direction'])
    if not all(d['camera_to_world_median_nn_m'] < d['world_to_camera_median_nn_m'] for d in direction):
        reasons.append('pose direction test did not favour camera-to-world')
    registration = results['registration'].get(scene, {})
    scoped = {role: row.get('outcome') for role, row in registration.items()}
    area_scoped = {role: row.get('outcome') for role, row in results['registration'].get(area, {}).items()}
    outcomes = scoped if set(scoped.values()) - {'no_power'} else area_scoped
    failed = bool(reasons)
    if outcomes and all(value == 'offset' for value in outcomes.values()):
        failed = True
        reasons.append('significant registration offset on calibration and validation clips')
    depth = results['depth_sweep'].get(scene, {})
    bias = depth.get('all') or {}
    tolerance = geometry['depth_bias_tolerance_percent']
    interval = bias.get('ci95_percent')
    depth_outcome = ('insufficient' if not interval else 'outside_tolerance' if min(map(abs, interval)) > tolerance and interval[0] * interval[1] > 0 else 'within_tolerance')
    classes = {}
    for label, row in depth.get('by_range_m', {}).items():
        value = row and row['parallax_minus_lidar_percent']
        classes[label] = {'parallax_minus_lidar_percent': value, 'ci95_percent': row and row['ci95_percent'],
                          'status': 'UNRESOLVED' if not row else 'QUALIFIED'}
    timing = of_flight(results['timing'])
    median_z = float(np.median([p['median_depth_m'] for p in of_flight(results['projection'])]))
    speed = max(t['speed_p95_mps'] for t in timing)
    deskew_px = speed * SCAN_PERIOD_S * intrinsic_f[area] / median_z
    if failed:
        status = 'FAILED'
    elif not outcomes or 'no_power' in outcomes.values():
        status = 'UNRESOLVED'
        reasons.append('RGB registration could not be measured with enough significance')
    else:
        status = 'QUALIFIED'
        reasons.append('Automated checks only; overlays await human review')
    if depth_outcome != 'within_tolerance':
        reasons.append(f'parallax/LiDAR depth disagreement {depth_outcome}')
    return {'status': status, 'reasons': reasons, 'registration_outcomes': outcomes, 'depth_outcome': depth_outcome,
            'parallax_minus_lidar_percent': bias.get('parallax_minus_lidar_percent'), 'ci95_percent': interval,
            'depth_classes': classes, 'scan_motion_bound_native_px': float(deskew_px),
            'per_point_timing': 'unavailable in extracted scans; raw bags needed, so deskew is not claimed',
            'inspection': 'pending human review'}


def build_gate(results: dict, contract: dict, flights: list[str]) -> dict:
    geometry = contract['geometry']
    intrinsic = {area: PUBLISHED[area]['K'][0] for area in AREAS}
    rows = {scene: flight_gate(scene, results, geometry, intrinsic) for scene in flights}
    counts = {status: sum(r['status'] == status for r in rows.values()) for status in ('VERIFIED', 'QUALIFIED', 'FAILED', 'UNRESOLVED')}
    return {'schema_version': 1, 'stage': 'stage1_geometry', 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'contract_id': contract['contract_id'], 'tolerances': {k: v for k, v in geometry.items() if k != 'tolerance_rationale'},
            'tolerance_rationale': geometry['tolerance_rationale'], 'status_counts': counts, 'flights': rows,
            'trusted_for_supervision': [], 'qualified_for_diagnostics': sorted(s for s, r in rows.items() if r['status'] == 'QUALIFIED'),
            'measurement_scope': 'Non-GNSS UAVScenes clips in the 30-batch cache; LiDAR xyz without per-point time. Terra maps and raw bags not used.',
            'source_sha256': results['source_sha256']}


def render_report(gate: dict, results: dict) -> str:
    lines = ['# Stage 1 calibration and geometry report', '', gate['measurement_scope'], '',
             f"Status counts: {gate['status_counts']}. Trusted for supervision: {gate['trusted_for_supervision'] or 'none yet'}.", '',
             '| Flight | Status | Registration | Parallax minus LiDAR % (95% CI) | Scan motion bound px |', '|---|---|---|---:|---:|']
    for scene, row in gate['flights'].items():
        value = row['parallax_minus_lidar_percent']
        text = 'n/a' if value is None else f"{value:.2f} ({row['ci95_percent'][0]:.2f} to {row['ci95_percent'][1]:.2f})"
        lines.append(f"| {scene} | {row['status']} | {row['registration_outcomes']} | {text} | {row['scan_motion_bound_native_px']:.1f} |")
    lines += ['', '## Calibration constants', '']
    for area, row in results['constants'].items():
        lines.append(f"- {area}: {row['status']}; repo rotation is HK_GNSS: {row['repo_rotation_is_hk_gnss']}; published rotation orthonormal error {row['published_rotation_orthonormal_error']:.5f}")
    lines += ['', '## Boresight', '']
    for row in results['boresight']:
        lines.append(f"- {row['area']}: {json.dumps({k: v for k, v in row.items() if k not in ('per_pair', 'area')})}")
    lines += ['', '## Signed error decomposition', '', '| Clip | Teacher - LiDAR % | Anchor - LiDAR % | Umeyama alpha | Pairwise alpha | LiDAR-implied alpha |', '|---|---:|---:|---:|---:|---:|']
    for row in results['decomposition']:
        lines.append(f"| {row['clip']} | {row['teacher_minus_lidar_percent']:.2f} | {row['anchor_minus_lidar_percent']:.2f} | {row['alpha_umeyama']:.4f} | {row['alpha_pairwise_baselines']:.4f} | {row['alpha_from_lidar_anchors']:.4f} |")
    lines += ['', '## Reasons by flight', '']
    for scene, row in gate['flights'].items():
        lines.append(f"- {scene}: " + '; '.join(row['reasons']))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--only', default='', help='Comma-separated areas, for a partial diagnostic run')
    parser.add_argument('--checks', default='structure,decomposition,boresight,registration,sweep,masks,overlays')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs; a dry run can never mark a gate')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root = args.root.resolve()
    all_clips = clips(root, contract)
    chosen = all_clips if not args.only else [c for c in all_clips if c['area'] in args.only.split(',')]
    if args.dry_run:
        print(json.dumps({'clips': len(all_clips), 'flights': len({c['scene'] for c in all_clips})}, indent=2))
        return
    cv2.setNumThreads(1)
    results, checks, geometry = {}, args.checks.split(','), contract['geometry']
    args.output.mkdir(parents=True, exist_ok=True)
    write_status(args.output, 'stage1_geometry', 'running', None, None)
    flights = list({c['scene']: c for c in chosen}.values())
    if 'structure' in checks:
        results['constants'] = check_constants(geometry['published_constant_tolerance'], geometry['rotation_orthonormal_tolerance'])
        results['pixel_chain'] = pixel_chain()
        results['poses'] = dict(zip([f['scene'] for f in flights], run_pool(check_poses, flights, args.workers)))
        results['direction'] = run_pool(direction_test, chosen, args.workers)
        results['pairing'] = run_pool(check_pairing, chosen, args.workers)
        results['projection'] = run_pool(check_projection, chosen, args.workers)
        results['timing'] = run_pool(check_timing, chosen, args.workers)
        print('structure done', flush=True)
    if 'decomposition' in checks:
        results['decomposition'] = run_pool(decomposition_clip, chosen, args.workers)
        print('decomposition done', flush=True)
    if 'boresight' in checks:
        items = [(area, [c for c in chosen if c['area'] == area]) for area in sorted({c['area'] for c in chosen})]
        results['boresight'] = run_pool(boresight_area, items, args.workers)
        print('boresight done', flush=True)
    if 'registration' in checks:
        rows = run_pool(registration_clip, chosen, args.workers)
        results['registration'] = registration_summary(rows, geometry)
        print('registration done', flush=True)
    if 'sweep' in checks:
        rows = run_pool(depth_sweep_clip, chosen, args.workers)
        results['depth_sweep'] = depth_sweep_summary(rows)
        print('sweep done', flush=True)
    if 'masks' in checks:
        items = [(c, str(args.output / 'measurements')) for c in chosen]
        results['reliability'] = run_pool(reliability_clip, items, args.workers)
        print('masks done', flush=True)
    if 'overlays' in checks:
        items = [(c, str(args.output / 'overlays')) for c in chosen if c['role'] == 'validation']
        results['overlays'] = sorted(name for names in run_pool(overlay_clip, items, args.workers) for name in names)
        print('overlays done', flush=True)
    results['source_sha256'] = sha256_file(Path(__file__))
    write_json(args.output / 'measurements_raw.json', results)
    if set(checks) >= {'structure', 'decomposition', 'boresight', 'registration', 'sweep'} and not args.only:
        gate = build_gate(results, contract, [f['scene'] for f in flights])
        write_json(args.output / 'geometry_gate.json', gate)
        (args.output / 'calibration_report.md').write_text(render_report(gate, results))
        write_status(args.output, 'stage1_geometry', 'complete', json.dumps(gate['status_counts']), 'stage2_data_controls',
                     'Process complete; quality gate is per flight in geometry_gate.json.')
        print(json.dumps(gate['status_counts']), flush=True)


if __name__ == '__main__':
    main()
