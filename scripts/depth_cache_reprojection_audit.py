"""Index and audit reuse of high-coverage measured depth across a saved scene.

Run with: .venv/bin/python depth_cache_reprojection_audit.py --workspace <dense_workspace>
"""
import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'uav_reconstruction'))
from adaptive_uav.hybrid_dense import camera_matrix, read_array


def project_depth(source, target):
    depth, K = source['depth'], source['K']
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[ys, xs]
    camera = np.stack(((xs - K[0, 2]) / K[0, 0] * z,
                       (ys - K[1, 2]) / K[1, 1] * z, z), axis=1)
    rotation, translation = source['pose'][:3, :3], source['pose'][:3, 3]
    world = (camera - translation) @ rotation
    rotation, translation = target['pose'][:3, :3], target['pose'][:3, 3]
    seen = world @ rotation.T + translation
    Ko = target['K']
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.rint(seen[:, 0] / seen[:, 2] * Ko[0, 0] + Ko[0, 2]).astype(int)
        v = np.rint(seen[:, 1] / seen[:, 2] * Ko[1, 1] + Ko[1, 2]).astype(int)
    h, w = target['depth'].shape
    good = (np.isfinite(seen).all(axis=1) & (seen[:, 2] > 0) &
            (u >= 0) & (u < w) & (v >= 0) & (v < h))
    output = np.full(h * w, np.inf, np.float32)
    np.minimum.at(output, v[good] * w + u[good], seen[good, 2].astype(np.float32))
    output[~np.isfinite(output)] = 0
    return output.reshape(h, w)


def evaluate(target, source_names, frames, min_views=2):
    maps = [project_depth(frames[name], target) for name in source_names]
    if not maps:
        return {'sources': 0, 'two_view_coverage': 0., 'new_on_target_missing': 0.,
                'heldout_measured_pixels': 0, 'heldout_median_error': None,
                'heldout_p90_error': None, 'heldout_under_5pct': None}
    stack = np.stack(maps)
    values = np.where(stack > 0, stack, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        median = np.nanmedian(values, axis=0)
    agreeing = (stack > 0) & (np.abs(stack - median) <= .035 * median)
    support = agreeing.sum(axis=0) >= min_views
    measured = target['depth'] > 0
    held = support & measured
    error = np.abs(median[held] - target['depth'][held]) / target['depth'][held]
    return {'sources': len(source_names),
            'min_views': min_views,
            'two_view_coverage': float(support.mean()),
            'new_on_target_missing': float((support & ~measured).mean()),
            'heldout_measured_pixels': int(held.sum()),
            'heldout_median_error': float(np.median(error)) if len(error) else None,
            'heldout_p90_error': float(np.percentile(error, 90)) if len(error) else None,
            'heldout_under_5pct': float((error < .05).mean()) if len(error) else None}


def main():
    import pycolmap
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path,
                        default=ROOT / 'runs/session_20261011_202810_corrected/dense_hybrid_trial')
    ws = parser.parse_args().workspace.resolve()
    report = json.loads((ws / 'hybrid_dense_report.json').read_text())
    plan = json.loads((ws / 'dense_plan.json').read_text())['plan']
    model = pycolmap.Reconstruction(ws / 'sparse')
    by_name = {image.name: image for image in model.images.values()}
    measured_fraction = {row['frame']: row['measured'] for row in report['per_frame']}
    names = [row['reference'] for row in plan]
    cache_candidates = sorted(names, key=lambda name: measured_fraction[name], reverse=True)
    cutoff = float(np.median(list(measured_fraction.values())))
    high = [name for name in cache_candidates if measured_fraction[name] >= cutoff]
    index = {'selection': 'measured-depth fraction >= scene median, sorted by coverage',
             'coverage_cutoff': cutoff, 'note': 'Index of existing full-resolution source images and COLMAP measured depth maps; source frames are not duplicated.',
             'frames': [
                 {'frame': name, 'measured_fraction': measured_fraction[name],
                  'image': str(ws / 'images' / name),
                  'depth': str(ws / 'stereo/depth_maps' / (name + '.geometric.bin'))}
                 for name in high]}
    (ws / 'high_coverage_frame_cache.json').write_text(json.dumps(index, indent=2) + '\n')
    frames, tracks = {}, {}
    for name in names:
        image = by_name[name]
        depth = read_array(ws / 'stereo/depth_maps' / (name + '.geometric.bin'))[::4, ::4]
        frames[name] = {'depth': depth, 'K': camera_matrix(model.cameras[image.camera_id], depth.shape),
                        'pose': image.cam_from_world().matrix()}
        tracks[name] = {p.point3D_id for p in image.points2D if p.has_point3D()}
    rows = []
    for entry in plan[::8]:
        name = entry['reference']
        candidates = [other for other in high if other != name]
        ranked = sorted(candidates, key=lambda other: (
            len(tracks[name] & tracks[other]) * measured_fraction[other]), reverse=True)
        chosen = [other for other in ranked if len(tracks[name] & tracks[other]) >= 100][:12]
        planned = [source['name'] for source in entry['sources'] if source['name'] in frames]
        union = list(dict.fromkeys(planned + chosen))
        rows.append({'frame': name, 'measured_fraction': measured_fraction[name],
                     'high_coverage_source_frames': chosen,
                     'planned_source_frames': planned,
                     'high_coverage_cache': evaluate(frames[name], chosen, frames),
                     'planned_sources': evaluate(frames[name], planned, frames),
                     'planned_plus_cache': evaluate(frames[name], union, frames),
                     'planned_plus_cache_one_view': evaluate(frames[name], union, frames, 1),
                     'planned_plus_cache_three_views': evaluate(frames[name], union, frames, 3)})
    summary = {'sampled_targets': len(rows), 'high_coverage_cached_frames': len(high),
               'method': 'quarter-resolution measured-depth reprojection; compare one, two, and three source map agreement within 3.5%'
                         '; target measured pixels are held out for error checks',
               'rows': rows}
    for key in ('high_coverage_cache', 'planned_sources', 'planned_plus_cache',
                'planned_plus_cache_one_view', 'planned_plus_cache_three_views'):
        summary[key + '_mean_new_fraction'] = float(np.mean([r[key]['new_on_target_missing'] for r in rows]))
        heldout = [
            r[key]['heldout_median_error'] for r in rows
            if r[key]['heldout_median_error'] is not None]
        summary[key + '_median_heldout_error'] = float(np.median(heldout)) if heldout else None
    (ws / 'depth_cache_reprojection_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
