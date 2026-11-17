"""Where a fused reconstruction fails, which stage failed there, and which frames built it.

Steps 1-3 of selector credit assignment, analysis only (no training, no counterfactual swaps):

1. Fuse the selected frames twice: once with fixed reference poses, once with the de-domed SfM
   poses aligned to the reference frame. Score both against the fixed LiDAR reference population
   per plan-view cell (precision, recall, F) instead of one global number.
2. Label each failed cell with the earliest stage that explains it:
     no_candidate_view  no frame of the clip sees the cell; data ceiling, not the selector's fault
     not_selected       some candidate frame sees it, no selected frame does
     pose               fixed-pose fusion is good there, SfM-pose fusion is not
     depth              selected views' depth disagrees with their own LiDAR inside the cell
     depth_unverified   selected views see the cell but have no LiDAR pixels there to check depth
     fusion             poses and depths are fine, the fused surface is still wrong
3. Record provenance: how many fused points each frame put in each cell, and group frames into
   temporal bins around each selected keyframe, so failures can be traced to bins and frames.

Visibility is a frustum test without occlusion; adequate for nadir UAV views, not for facades.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.spatial import cKDTree

from .counterfactual_replay import Clip, calibration_scales, eligible, fuse, role_of
from .reconstruction_quality import ReferencePopulation, cell_keys, surface_metrics
from .student_inference import load_student
from .utility_contracts import THREADS, load_contract, sha256_file, write_json

LABELS = ('no_candidate_view', 'not_selected', 'pose', 'depth', 'depth_unverified', 'fusion')
DEPTH_TRUNC = 250.


def decode(keys: np.ndarray) -> np.ndarray:
    return np.stack((keys // 2 ** 31 - 2 ** 30, keys % 2 ** 31 - 2 ** 30), axis=1)


def cell_index(points: np.ndarray, cells: np.ndarray, size: float) -> np.ndarray:
    """Index into the sorted ROI cell keys; -1 outside the reference footprint."""
    if not len(points):
        return np.empty(0, np.int64)
    keys = cell_keys(points, size)
    found = np.clip(np.searchsorted(cells, keys), 0, len(cells) - 1)
    return np.where(cells[found] == keys, found, -1)


def back_project(depth: np.ndarray, k: np.ndarray, world_from_camera: np.ndarray, stride: int = 2) -> np.ndarray:
    """Same pixel convention as Open3D integration, so provenance matches what was fused."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]
    ok = (z > 0) & np.isfinite(z) & (z < DEPTH_TRUNC)
    xs, ys, z = xs[ok], ys[ok], z[ok].astype(np.float64)
    camera = np.c_[(xs - k[0, 2]) / k[0, 0] * z, (ys - k[1, 2]) / k[1, 1] * z, z]
    return camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]


def per_cell_surface(predicted: np.ndarray, reference: ReferencePopulation, ref_cell: np.ndarray) -> dict:
    """Per-cell versions of surface_metrics: recall over reference points, precision over predicted."""
    n = len(reference.cells)
    tol = reference.tolerance_m
    n_ref = np.bincount(ref_cell, minlength=n).astype(np.float64)
    out = {'n_ref': n_ref, 'recall': np.zeros(n), 'precision': np.full(n, np.nan), 'n_pred': np.zeros(n),
           'accuracy_m': np.full(n, np.nan), 'completeness_m': np.full(n, np.nan)}
    pred_cell = cell_index(predicted, reference.cells, reference.cell)
    surface, pred_cell = predicted[pred_cell >= 0], pred_cell[pred_cell >= 0]
    if not len(surface):
        return out
    to_pred = cKDTree(surface).query(reference.points, workers=THREADS)[0]
    to_ref = reference.tree.query(surface, workers=THREADS)[0]
    out['recall'] = np.bincount(ref_cell, weights=(to_pred <= tol).astype(float), minlength=n) / np.maximum(n_ref, 1)
    out['n_pred'] = np.bincount(pred_cell, minlength=n).astype(np.float64)
    hits = np.bincount(pred_cell, weights=(to_ref <= tol).astype(float), minlength=n)
    has = out['n_pred'] > 0
    out['precision'][has] = hits[has] / out['n_pred'][has]
    for values, cells, key in ((to_ref, pred_cell, 'accuracy_m'), (to_pred, ref_cell, 'completeness_m')):
        order = np.argsort(cells, kind='stable')
        split = np.split(values[order], np.flatnonzero(np.diff(cells[order])) + 1)
        for c, chunk in zip(np.unique(cells), split):
            out[key][c] = float(np.median(chunk))
    return out


def f_score(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(precision, nan=0.)
    return np.where(p + recall > 0, 2 * p * recall / np.maximum(p + recall, 1e-12), 0.)


def visibility(view, reference: ReferencePopulation, ref_cell: np.ndarray, n_ref: np.ndarray) -> np.ndarray:
    """Fraction of each cell's reference points inside this frame's frustum (reference pose)."""
    pose, k = view.reference_pose, view.K
    h, w = view.rgb.shape[:2]
    camera = (reference.points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = k[0, 0] * camera[:, 0] / z + k[0, 2]
        v = k[1, 1] * camera[:, 1] / z + k[1, 2]
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.bincount(ref_cell, weights=inside.astype(float), minlength=len(n_ref)) / np.maximum(n_ref, 1)


def lidar_depth_error(view, sidecar: Path, reference: ReferencePopulation) -> tuple:
    """Point error of this frame's fused depth against its own LiDAR, per LiDAR pixel, by cell."""
    if not sidecar.exists():
        return np.empty(0, np.int64), np.empty(0)
    with np.load(sidecar) as measured:
        lidar = measured['measured_depth_z_m'].copy()
        for mask in ('reliable', 'measured_valid'):
            if mask in measured:
                lidar = lidar * measured[mask]
    ys, xs = np.nonzero(lidar > 0)
    if not len(ys):
        return np.empty(0, np.int64), np.empty(0)
    k, pose = view.K, view.reference_pose
    z = lidar[ys, xs].astype(np.float64)
    rays = np.c_[(xs + .5 - k[0, 2]) / k[0, 0], (ys + .5 - k[1, 2]) / k[1, 1], np.ones(len(z))]
    world = (rays * z[:, None]) @ pose[:3, :3].T + pose[:3, 3]
    depth = view.fusion_depth
    H, W = lidar.shape
    h, w = depth.shape
    pred = depth[np.minimum((ys * h / H).astype(int), h - 1), np.minimum((xs * w / W).astype(int), w - 1)].astype(np.float64)
    valid = (pred > 0) & np.isfinite(pred)
    # Missing prediction counts as the full LiDAR range: no surface is worse than any surface.
    error = np.where(valid, np.abs(pred - z), z) * np.linalg.norm(rays, axis=1)
    cells = cell_index(world, reference.cells, reference.cell)
    keep = cells >= 0
    return cells[keep], error[keep]


def temporal_bins(n_views: int, selected: list) -> np.ndarray:
    """Each selected keyframe owns the candidates nearer to it than to its neighbours."""
    edges = [(a + b) / 2 for a, b in zip(selected, selected[1:])]
    return np.searchsorted(np.asarray(edges), np.arange(n_views), side='right')


def plot(path: Path, cells: np.ndarray, size: float, label: np.ndarray, f_sfm: np.ndarray, f_fixed: np.ndarray, views, selected, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    xy = decode(cells)
    lo = xy.min(0)
    shape = tuple(xy.max(0) - lo + 1)
    colours = ['#2e7d32', '#9e9e9e', '#d32f2f', '#7b1fa2', '#f57c00', '#fbc02d', '#1976d2']
    names = ('good',) + LABELS
    grid_label = np.full(shape, np.nan)
    grid_label[xy[:, 0] - lo[0], xy[:, 1] - lo[1]] = label
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    extent = (lo[0] * size, (lo[0] + shape[0]) * size, lo[1] * size, (lo[1] + shape[1]) * size)
    for ax, values, name in ((axes[0], f_fixed, 'F, fixed reference poses'), (axes[1], f_sfm, 'F, SfM poses')):
        grid = np.full(shape, np.nan)
        grid[xy[:, 0] - lo[0], xy[:, 1] - lo[1]] = values
        image = ax.imshow(grid.T, origin='lower', extent=extent, vmin=0, vmax=1, cmap='viridis')
        ax.set_title(name)
        fig.colorbar(image, ax=ax, shrink=.8)
    axes[2].imshow(grid_label.T, origin='lower', extent=extent, cmap=ListedColormap(colours), vmin=-.5, vmax=len(names) - .5,
                   interpolation='nearest')
    axes[2].set_title('Failure stage (SfM-pose run)')
    axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colours], labels=list(names), loc='upper left',
                   bbox_to_anchor=(1.01, 1), fontsize=8)
    centres = np.array([v.reference_pose[:2, 3] for v in views])
    for ax in axes:
        ax.plot(centres[:, 0], centres[:, 1], '-', color='white', lw=.8, alpha=.7)
        ax.scatter(centres[selected, 0], centres[selected, 1], s=14, c='black', edgecolors='white', linewidths=.5, zorder=3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    return value


def analyse(args) -> dict:
    started = time.monotonic()
    contract = load_contract(args.config, ('utility.quality_protocol',))
    root, settings, data = args.root.resolve(), contract['utility'], contract['data']
    gate_path = root / contract['geometry_gate']
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    cache_root = root / data['cache_root']
    suffix = contract['geometry']['calibration_batch_suffix']
    if args.clip.endswith(suffix):
        raise SystemExit('Calibration clips fit the scale, so they cannot also be scored')
    area = args.clip.rstrip('0123456789_').rstrip('0123456789')
    same_area = sorted(p.name for p in cache_root.iterdir() if p.name.startswith(area) and (p / 'selected_infos.json').exists())
    scales = calibration_scales(root, contract, same_area)

    clip = Clip(root, contract, args.clip)
    status = eligible(gate, clip.scene, args.diagnostic)
    if status is None:
        raise SystemExit(f'Reference for {clip.scene} is not VERIFIED/QUALIFIED; use --diagnostic for unpromotable output')
    if clip.scene not in scales or not len(clip.points):
        raise SystemExit('No calibration scale or LiDAR reference for this clip')
    clip.apply_scale(scales[clip.scene])
    selected = clip.choose_keyframes(args.keyframe_ratio or settings['keyframe_baseline_ratio'])
    backend = 'cached_teacher'
    if args.student:
        clip.use_student(load_student(args.student, args.device), args.device)
        # use_student overwrites the calibrated depth with anchor-fitted depth; reapply the same
        # calibration ratio so student and teacher runs share one metric scale.
        for view in clip.views:
            view.fusion_depth = view.fusion_depth * clip.depth_scale
            view.depth_m = view.depth_m * clip.depth_scale
        backend = 'student_past_only_keyframe_sources'
    views = clip.views
    reference = ReferencePopulation(clip.points, settings['tolerance_m'], status, settings['quality_protocol'])
    ref_cell = cell_index(reference.points, reference.cells, reference.cell)
    n_cells, voxel = len(reference.cells), settings['voxel_m']

    # 1. Two fusions of the same selected frames: only the poses differ.
    runs = {}
    for name, pose_of in (('fixed', lambda v: v.reference_pose), ('sfm', lambda v: v.aligned_pose)):
        points = fuse([(views[p], pose_of(views[p])) for p in selected], voxel)
        runs[name] = {'global': surface_metrics(points, reference), 'cells': per_cell_surface(points, reference, ref_cell)}
    n_ref = runs['fixed']['cells']['n_ref']
    f_fixed = f_score(runs['fixed']['cells']['precision'], runs['fixed']['cells']['recall'])
    f_sfm = f_score(runs['sfm']['cells']['precision'], runs['sfm']['cells']['recall'])

    # 3. Provenance and evidence for every frame, selected or not.
    visible = np.stack([visibility(v, reference, ref_cell, n_ref) for v in views]) >= args.visible_fraction
    contribution = np.zeros((len(views), n_cells))
    depth_err_sum, depth_err_n, depth_err_med = np.zeros((len(views), n_cells)), np.zeros((len(views), n_cells)), np.full((len(views), n_cells), np.nan)
    sidecars = root / data['sidecars'] / clip.name
    for i, view in enumerate(views):
        cells = cell_index(back_project(view.fusion_depth, view.fusion_k, view.reference_pose), reference.cells, reference.cell)
        contribution[i] = np.bincount(cells[cells >= 0], minlength=n_cells)
        cells, error = lidar_depth_error(view, sidecars / f'frame_{view.index:06d}.npz', reference)
        if len(cells):
            depth_err_sum[i] = np.bincount(cells, weights=error, minlength=n_cells)
            depth_err_n[i] = np.bincount(cells, minlength=n_cells)
            order = np.argsort(cells, kind='stable')
            for c, chunk in zip(np.unique(cells), np.split(error[order], np.flatnonzero(np.diff(cells[order])) + 1)):
                depth_err_med[i, c] = float(np.median(chunk))

    # 2. Stage labels, earliest explaining stage first.
    sel = np.zeros(len(views), bool)
    sel[selected] = True
    measured = n_ref >= args.min_ref_points
    failed = measured & (f_sfm < args.fail_f)
    seen_any, seen_sel = visible.any(0), visible[sel].any(0)
    sel_lidar_px = depth_err_n[sel].sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN cells: no selected LiDAR there
        sel_depth_med = np.nanmedian(depth_err_med[sel], axis=0)
    # Ordered: a cell takes the first rule that explains it. Codes index ('good', *LABELS).
    rules = ((1, ~seen_any), (2, ~seen_sel), (3, f_fixed >= args.good_f),
             (5, sel_lidar_px < args.min_lidar_pixels), (4, np.nan_to_num(sel_depth_med) > args.depth_tolerance_m),
             (6, np.ones(n_cells, bool)))
    label = np.zeros(n_cells, int)
    rest = failed.copy()
    for code, condition in rules:
        hit = rest & condition
        label[hit] = code
        rest &= ~hit
    label = np.where(measured, label, -1)

    bins = temporal_bins(len(views), selected)
    share = contribution * sel[:, None]
    share = share / np.maximum(share.sum(0, keepdims=True), 1)

    cell_rows = []
    xy = decode(reference.cells)
    for c in range(n_cells):
        builders = np.flatnonzero(share[:, c] > 0)
        cell_rows.append({'cell': c, 'x_m': (xy[c, 0] + .5) * reference.cell, 'y_m': (xy[c, 1] + .5) * reference.cell,
                          'n_ref': int(n_ref[c]), 'label': 'unmeasured' if label[c] < 0 else ('good', *LABELS)[label[c]],
                          'f_fixed': clean(f_fixed[c]), 'f_sfm': clean(f_sfm[c]),
                          'recall_fixed': clean(runs['fixed']['cells']['recall'][c]), 'recall_sfm': clean(runs['sfm']['cells']['recall'][c]),
                          'precision_fixed': clean(runs['fixed']['cells']['precision'][c]), 'precision_sfm': clean(runs['sfm']['cells']['precision'][c]),
                          'accuracy_sfm_m': clean(runs['sfm']['cells']['accuracy_m'][c]), 'completeness_sfm_m': clean(runs['sfm']['cells']['completeness_m'][c]),
                          'candidates_seeing': int(visible[:, c].sum()), 'selected_seeing': int(visible[sel, c].sum()),
                          'selected_depth_err_median_m': clean(sel_depth_med[c]), 'selected_lidar_pixels': int(sel_lidar_px[c]),
                          'builders': ' '.join(f'{views[i].index}:{share[i, c]:.2f}' for i in builders[np.argsort(-share[builders, c])])})

    frame_rows = []
    for i, view in enumerate(views):
        own = depth_err_n[i].sum()
        row = {'position': i, 'frame_index': view.index, 'bin': int(bins[i]), 'selected': bool(sel[i]),
               'cells_visible': int(visible[i].sum()), 'cells_built': int((contribution[i] > 0).sum()),
               'depth_err_mean_m': clean(depth_err_sum[i].sum() / own) if own else None, 'lidar_pixels': int(own),
               'failed_cells_visible': int((visible[i] & failed).sum())}
        for code, name in enumerate(LABELS, start=1):
            # Selected frames: share of failed cells they built. Candidates: failed cells they could see.
            row[f'blame_{name}'] = clean(share[i, label == code].sum()) if sel[i] else int((visible[i] & (label == code)).sum())
        frame_rows.append(row)

    bin_rows = []
    for b, p in enumerate(selected):
        members = np.flatnonzero(bins == b)
        alternatives = [m for m in members if m != p]
        missed = label == 2  # not_selected: a candidate saw it, no selected frame did
        reach = {m: int((visible[m] & missed).sum()) for m in alternatives}
        best = max(reach, key=reach.get) if reach else None
        built = share[p]
        bin_rows.append({'bin': b, 'selected_position': p, 'selected_frame': views[p].index,
                         'candidate_frames': ' '.join(str(views[m].index) for m in alternatives),
                         'cells_built': int((built > 0).sum()), 'failed_share_built': clean(built[failed].sum()),
                         **{f'share_{name}': clean(built[label == code].sum()) for code, name in enumerate(LABELS, start=1)},
                         'selected_depth_err_m': frame_rows[p]['depth_err_mean_m'],
                         'missed_cells_reachable_in_bin': int((visible[alternatives] & missed).any(0).sum()) if alternatives else 0,
                         'best_alternative_frame': views[best].index if best is not None else None,
                         'best_alternative_reach': reach.get(best, 0) if best is not None else 0})

    counts = {name: int((label == code).sum()) for code, name in enumerate(LABELS, start=1)}
    ref_weighted = {name: clean(n_ref[label == code].sum() / max(n_ref[measured].sum(), 1)) for code, name in enumerate(LABELS, start=1)}
    summary = {'clip': clip.name, 'flight': clip.scene, 'split': role_of(clip.scene, contract), 'reference_status': status,
               'reference_identity': reference.identity(), 'depth_backend': backend,
               'student_checkpoint': str(args.student) if args.student else None,
               'student_sha256': sha256_file(args.student) if args.student else None,
               'config_sha256': sha256_file(args.config), 'calibration_scale': clip.depth_scale,
               'frames': len(views), 'selected_positions': selected, 'selected_frames': [views[p].index for p in selected],
               'cell_size_m': reference.cell, 'tolerance_m': reference.tolerance_m, 'voxel_m': voxel,
               'thresholds': {'fail_f': args.fail_f, 'good_f': args.good_f, 'depth_tolerance_m': args.depth_tolerance_m,
                              'min_ref_points': args.min_ref_points, 'min_lidar_pixels': args.min_lidar_pixels,
                              'visible_fraction': args.visible_fraction},
               'global': {name: run['global'] for name, run in runs.items()},
               'cells': {'total': n_cells, 'measured': int(measured.sum()), 'failed': int(failed.sum()), 'by_label': counts,
                         'reference_point_fraction_by_label': ref_weighted},
               'limitations': ['Analysis only: no counterfactual swaps, no selector update.',
                               'Selection is the rule-based baseline-ratio keyframe policy, not the neural selector.',
                               'Visibility is a frustum test without occlusion.',
                               'SfM poses are the de-domed full-clip trajectory aligned to the reference, not a fresh SfM run on the selected frames.',
                               'Plan-view cells: vertical structure inside a cell is merged.'],
               'seconds': round(time.monotonic() - started, 1)}
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'cells.csv', cell_rows)
    write_csv(out / 'frames.csv', frame_rows)
    write_csv(out / 'bins.csv', bin_rows)
    np.savez_compressed(out / 'arrays.npz', cells=reference.cells, label=label, f_fixed=f_fixed, f_sfm=f_sfm, n_ref=n_ref,
                        visible=visible, contribution=contribution, depth_err_median=depth_err_med, bins=bins, selected=np.asarray(selected),
                        frame_index=np.asarray([v.index for v in views]))
    write_json(out / 'summary.json', summary)
    plot(out / 'failure_map.png', reference.cells, reference.cell, np.where(label < 0, np.nan, label), f_sfm, f_fixed, views, selected,
         f'{clip.name}: {counts}  (cell {reference.cell:.0f} m)')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--clip', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--student', type=Path, help='Past-only student depth; cached teacher depth otherwise')
    p.add_argument('--device', default='cpu')
    p.add_argument('--keyframe-ratio', type=float, help='Baseline/depth spacing for the selection; contract value by default')
    p.add_argument('--fail-f', type=float, default=.5, help='Cell F-score below this (SfM-pose run) is a failure')
    p.add_argument('--good-f', type=float, default=.5, help='Fixed-pose F at or above this blames poses for an SfM-run failure')
    p.add_argument('--depth-tolerance-m', type=float, default=1., help='Median selected-view LiDAR point error above this blames depth')
    p.add_argument('--min-ref-points', type=int, default=20)
    p.add_argument('--min-lidar-pixels', type=int, default=20)
    p.add_argument('--visible-fraction', type=float, default=.5, help='Share of a cell inside a frustum to count as seen')
    p.add_argument('--diagnostic', action='store_true', help='Allow unresolved references; output is not promotable')
    args = p.parse_args()
    summary = analyse(args)
    print(json.dumps({k: summary[k] for k in ('clip', 'selected_frames', 'global', 'cells', 'seconds')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
