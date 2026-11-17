"""Measure where cached hybrid depth pixels are lost before TSDF fusion."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'uav_reconstruction'))
from adaptive_uav.hybrid_dense import (camera_matrix, cross_view, edge_invalid,
                                        frame_id, read_array, reference_names)


def audit(workspace: Path, stride: int = 4):
    import pycolmap

    report = json.loads((workspace / 'hybrid_dense_report.json').read_text())
    model = pycolmap.Reconstruction(workspace / 'sparse')
    plan = {row['reference']: [source['name'] for source in row['sources']]
            for row in json.loads((workspace / 'dense_plan.json').read_text())['plan']}
    by_name = {image.name: image for image in model.images.values()}
    names = sorted((n for n in reference_names(workspace) if n in by_name), key=frame_id)
    measured, aligned, matrices = {}, {}, {}
    for name in names:
        m = read_array(workspace / 'stereo/depth_maps' / (name + '.geometric.bin')).astype(np.float32)
        m[~np.isfinite(m)] = 0
        info = report['alignment'][name]
        if info['ok']:
            raw = np.load(workspace / 'prior_depth' / (name + '.npy'))
            a = (float(info['scale']) * cv2.resize(raw, (m.shape[1], m.shape[0]))
                 + float(info['shift'])).astype(np.float32)
            far = 3 * float(np.percentile(m[m > 0], 95))
            a[~np.isfinite(a) | (a <= 0) | (a > far)] = 0
        else:
            a = np.zeros_like(m)
        measured[name], aligned[name] = m, a
        matrices[name] = camera_matrix(model.cameras[by_name[name].camera_id], m.shape)
    combined = {n: np.where(measured[n] > 0, measured[n], aligned[n]) for n in names}
    high_cache = [row['frame'] for row in json.loads(
        (workspace / 'high_coverage_frame_cache.json').read_text())['frames']
        if row['frame'] in measured]
    tracks = {name: {p.point3D_id for p in by_name[name].points2D if p.has_point3D()}
              for name in names}
    coverage = {name: float((measured[name] > 0).mean()) for name in names}
    one = SimpleNamespace(tol=.035, min_agree=1)
    two = SimpleNamespace(tol=.035, min_agree=2)
    rows = []
    for name in names[::stride]:
        m, a = measured[name], aligned[name]
        blank = m <= 0
        candidate = blank & (a > 0)
        edged = candidate & ~edge_invalid(a)
        near = sorted((n for n in names if n != name),
                      key=lambda n: abs(frame_id(n) - frame_id(name)))[:6]
        neighbours = [by_name[n] for n in near]
        planned = [by_name[n] for n in plan.get(name, []) if n in measured and n != name]
        joint_names = list(dict.fromkeys(near + [image.name for image in planned]))
        joint = [by_name[n] for n in joint_names]
        ranked_cache = sorted((n for n in high_cache if n != name),
                              key=lambda n: len(tracks[name] & tracks[n]) * coverage[n],
                              reverse=True)
        cached_names = [n for n in ranked_cache
                        if len(tracks[name] & tracks[n]) >= 100][:12]
        with_cache = [by_name[n] for n in dict.fromkeys(joint_names + cached_names)]
        kwargs = (edged, a, by_name[name], matrices[name], neighbours)
        supported_one = cross_view(*kwargs, combined, model, one)
        supported_two = cross_view(*kwargs, combined, model, two)
        measured_one = cross_view(*kwargs, measured, model, one)
        measured_two = cross_view(*kwargs, measured, model, two)
        supported_plan = cross_view(edged, a, by_name[name], matrices[name],
                                    planned, combined, model, two)
        supported_joint = cross_view(edged, a, by_name[name], matrices[name],
                                     joint, combined, model, two)
        measured_plan = cross_view(edged, a, by_name[name], matrices[name],
                                   planned, measured, model, two)
        supported_cache = cross_view(edged, a, by_name[name], matrices[name],
                                     with_cache, combined, model, two)
        measured_cache = cross_view(edged, a, by_name[name], matrices[name],
                                    with_cache, measured, model, two)
        total = m.size
        rows.append({'frame': name, 'alignment_ok': report['alignment'][name]['ok'],
                     'measured': float((m > 0).sum() / total),
                     'blank_with_aligned_prior': float(candidate.sum() / total),
                     'after_prior_edge_gate': float(edged.sum() / total),
                     'one_view_prior_or_measured': float(supported_one.sum() / total),
                     'two_views_prior_or_measured': float(supported_two.sum() / total),
                     'one_view_measured_only': float(measured_one.sum() / total),
                     'two_views_measured_only': float(measured_two.sum() / total),
                     'two_views_planned_sources': float(supported_plan.sum() / total),
                     'two_views_temporal_or_planned': float(supported_joint.sum() / total),
                     'two_views_temporal_planned_cache': float(supported_cache.sum() / total),
                     'two_views_temporal_planned_cache_measured_only': float(measured_cache.sum() / total),
                     'cache_sources': len(cached_names),
                     'cache_new_vs_temporal_planned': float((supported_cache & ~supported_joint).sum() / total),
                     'two_views_planned_measured_only': float(measured_plan.sum() / total),
                     'planned_new_vs_temporal': float((supported_plan & ~supported_two).sum() / total),
                     'joint_new_vs_temporal': float((supported_joint & ~supported_two).sum() / total)})
    keys = ('measured', 'blank_with_aligned_prior', 'after_prior_edge_gate',
            'one_view_prior_or_measured', 'two_views_prior_or_measured',
            'one_view_measured_only', 'two_views_measured_only')
    keys += ('two_views_planned_sources', 'two_views_temporal_or_planned',
             'two_views_planned_measured_only', 'planned_new_vs_temporal',
             'joint_new_vs_temporal', 'two_views_temporal_planned_cache',
             'two_views_temporal_planned_cache_measured_only',
             'cache_sources', 'cache_new_vs_temporal_planned')
    result = {'method': f'every {stride}th planned reference frame; exact hybrid gates;'
                        ' aligned prior reconstructed from saved scale and shift',
              'sampled_frames': len(rows),
              'mean_fraction': {key: float(np.mean([r[key] for r in rows])) for key in keys},
              'per_frame': rows}
    (workspace / 'hybrid_failure_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--stride', type=int, default=4)
    args = parser.parse_args()
    result = audit(args.workspace, args.stride)
    print(json.dumps({key: value for key, value in result.items() if key != 'per_frame'}, indent=2))
