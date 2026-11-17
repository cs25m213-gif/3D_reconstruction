"""Step 4 of selector credit assignment: per-bin counterfactual swaps, analysis only.

For a fixed-budget selection, each bin's selected frame is replaced in turn by every other candidate
in that bin, one bin at a time, with all other selected frames kept. The student depth is recomputed
for every frame whose past-only sources change (the swapped frame and the next two selected frames),
then the selection is fused with fixed reference poses and scored per cell against the LiDAR
reference. Pose recovery is held fixed on purpose so only selection and depth can change.

Outputs per bin: the best alternative, its regret (local to the cells the bin can see, and global),
and the rank of the frame that was actually chosen. labels.jsonl holds one ranking target per bin.
The combined swap applies every bin's best alternative together to check whether gains add up.
With --attribution, failed cells from failure_attribution are cross-tabulated against whether any
single swap fixes them, separating selection-caused depth failures from student-caused ones.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .counterfactual_replay import Clip, calibration_scales, eligible, fuse, role_of
from .failure_attribution import LABELS, cell_index, f_score, per_cell_surface, temporal_bins, visibility, write_csv
from .reconstruction_quality import ReferencePopulation, surface_metrics
from .student_inference import anchor_scale, load_student, predict
from .utility_contracts import load_contract, sha256_file, write_json


class DepthCache:
    """Past-only student depth keyed by (frame, sources), so baseline and swaps share computation."""

    def __init__(self, clip, model, device):
        self.clip, self.model, self.device, self.store, self.calls = clip, model, device, {}, 0

    def view(self, position: int, sources: tuple):
        key = (position, sources)
        if key not in self.store:
            views = self.clip.views
            view = views[position]
            depth, _ = predict(self.model, [(v.rgb, v.feature_k, v.feature_pose) for v in [view] + [views[s] for s in sources]],
                               anchor_scale(view.anchors_m), self.device, anchors=view.anchors_m)
            depth = depth.astype(np.float32) * self.clip.depth_scale  # same calibration as failure_attribution
            k = view.K.copy()
            k[0] *= depth.shape[1] / view.rgb.shape[1]
            k[1] *= depth.shape[0] / view.rgb.shape[0]
            self.store[key] = replace(view, fusion_depth=depth, fusion_k=k,
                                      fusion_rgb=cv2.resize(view.rgb, depth.shape[::-1], interpolation=cv2.INTER_AREA))
            self.calls += 1
        return self.store[key]


def realise(selection: list, cache: DepthCache) -> list:
    """Same source rule as Clip.use_student: the two latest earlier selected frames, newest first."""
    selection = sorted(selection)
    return [cache.view(p, tuple(selection[:i][-2:][::-1])) for i, p in enumerate(selection)]


def evaluate(selection: list, cache: DepthCache, reference, ref_cell, voxel: float) -> dict:
    points = fuse([(v, v.reference_pose) for v in realise(selection, cache)], voxel)
    cells = per_cell_surface(points, reference, ref_cell)
    return {'global': surface_metrics(points, reference), 'f': f_score(cells['precision'], cells['recall']), 'recall': cells['recall']}


def local_f(f: np.ndarray, weight: np.ndarray, region: np.ndarray) -> float:
    w = weight * region
    return float((f * w).sum() / w.sum()) if w.sum() else float('nan')


def plot(path: Path, bins: list, base_f, combined_f, cells, size, views, selection, combined_selection, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .failure_attribution import decode
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    labels = [f"b{r['bin']}\n{r['selected_frame']}→{r['best_alternative_frame']}" if r['best_alternative_frame'] is not None
              else f"b{r['bin']}\n{r['selected_frame']}" for r in bins]
    regret = [r['regret_local_f'] or 0. for r in bins]
    axes[0].bar(range(len(bins)), regret, color=['#d32f2f' if v > 0 else '#9e9e9e' for v in regret])
    axes[0].set_xticks(range(len(bins)), labels, fontsize=7)
    axes[0].set_ylabel('local F gain of best swap (regret)')
    axes[0].set_title('Per-bin regret: selected → best alternative')
    xy = decode(cells)
    lo = xy.min(0)
    shape = tuple(xy.max(0) - lo + 1)
    grid = np.full(shape, np.nan)
    grid[xy[:, 0] - lo[0], xy[:, 1] - lo[1]] = combined_f - base_f
    extent = (lo[0] * size, (lo[0] + shape[0]) * size, lo[1] * size, (lo[1] + shape[1]) * size)
    image = axes[1].imshow(grid.T, origin='lower', extent=extent, cmap='RdBu', vmin=-1, vmax=1)
    fig.colorbar(image, ax=axes[1], shrink=.8, label='ΔF per cell (combined best swaps − chosen)')
    centres = np.array([v.reference_pose[:2, 3] for v in views])
    axes[1].plot(centres[:, 0], centres[:, 1], '-', color='black', lw=.8, alpha=.6)
    axes[1].scatter(centres[selection, 0], centres[selection, 1], s=22, c='black', label='chosen', zorder=3)
    axes[1].scatter(centres[combined_selection, 0], centres[combined_selection, 1], s=40, facecolors='none', edgecolors='#d32f2f',
                    linewidths=1.4, label='best swaps', zorder=4)
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('y (m)')
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(args) -> dict:
    started = time.monotonic()
    contract = load_contract(args.config, ('utility.quality_protocol',))
    root, settings, data = args.root.resolve(), contract['utility'], contract['data']
    gate_path = root / contract['geometry_gate']
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    cache_root = root / data['cache_root']
    if args.clip.endswith(contract['geometry']['calibration_batch_suffix']):
        raise SystemExit('Calibration clips fit the scale, so they cannot also be scored')
    area = args.clip.rstrip('0123456789_')
    scales = calibration_scales(root, contract, sorted(p.name for p in cache_root.iterdir()
                                                       if p.name.startswith(area) and (p / 'selected_infos.json').exists()))
    clip = Clip(root, contract, args.clip)
    status = eligible(gate, clip.scene, args.diagnostic)
    if status is None:
        raise SystemExit(f'Reference for {clip.scene} is not VERIFIED/QUALIFIED; use --diagnostic for unpromotable output')
    if clip.scene not in scales or not len(clip.points):
        raise SystemExit('No calibration scale or LiDAR reference for this clip')
    clip.apply_scale(scales[clip.scene])
    selection = clip.choose_keyframes(args.keyframe_ratio or settings['keyframe_baseline_ratio'])
    views, voxel = clip.views, settings['voxel_m']
    reference = ReferencePopulation(clip.points, settings['tolerance_m'], status, settings['quality_protocol'])
    ref_cell = cell_index(reference.points, reference.cells, reference.cell)
    n_ref = np.bincount(ref_cell, minlength=len(reference.cells)).astype(float)
    measured = n_ref >= args.min_ref_points
    weight = n_ref * measured
    visible = np.stack([visibility(v, reference, ref_cell, n_ref) for v in views]) >= args.visible_fraction
    cache = DepthCache(clip, load_student(args.student, args.device), args.device)

    base = evaluate(selection, cache, reference, ref_cell, voxel)
    base_failed = measured & (base['f'] < args.fail_f)
    bins = temporal_bins(len(views), selection)
    swap_rows, bin_rows, label_rows = [], [], []
    fixed_by_any = np.zeros(len(reference.cells), bool)
    best_choice = list(selection)
    for b, chosen in enumerate(selection):
        members = [int(m) for m in np.flatnonzero(bins == b)]
        region = visible[members].any(0) & measured
        base_local = local_f(base['f'], weight, region)
        options = [{'frame': views[chosen].index, 'position': chosen, 'local_f': base_local,
                    'global_f': base['global']['f_score'], 'recall': base['global']['recall'], 'chosen': True}]
        for m in members:
            if m == chosen:
                continue
            trial = sorted(selection[:b] + [m] + selection[b + 1:])
            result = evaluate(trial, cache, reference, ref_cell, voxel)
            fixed = base_failed & (result['f'] >= args.fail_f)
            broken = measured & ~base_failed & (result['f'] < args.fail_f)
            fixed_by_any |= fixed
            option = {'frame': views[m].index, 'position': m, 'local_f': local_f(result['f'], weight, region),
                      'global_f': result['global']['f_score'], 'recall': result['global']['recall'], 'chosen': False}
            options.append(option)
            swap_rows.append({'bin': b, 'selected_frame': views[chosen].index, 'candidate_frame': views[m].index,
                              'global_f': round(option['global_f'], 4), 'delta_global_f': round(option['global_f'] - base['global']['f_score'], 4),
                              'delta_recall': round(option['recall'] - base['global']['recall'], 4),
                              'local_f': round(option['local_f'], 4), 'delta_local_f': round(option['local_f'] - base_local, 4),
                              'failed_cells_fixed': int(fixed.sum()), 'good_cells_broken': int(broken.sum())})
        ranked = sorted(options, key=lambda o: -o['local_f'])
        best = ranked[0]
        if not best['chosen'] and best['local_f'] - base_local > args.tie_margin:
            best_choice[b] = best['position']
        bin_rows.append({'bin': b, 'selected_frame': views[chosen].index, 'alternatives': len(options) - 1,
                         'region_cells': int(region.sum()), 'local_f_chosen': round(base_local, 4),
                         'best_alternative_frame': None if best['chosen'] else best['frame'],
                         'regret_local_f': round(max(best['local_f'] - base_local, 0.), 4),
                         'regret_global_f': round(max(o['global_f'] for o in options) - base['global']['f_score'], 4),
                         'chosen_rank': 1 + next(i for i, o in enumerate(ranked) if o['chosen']), 'options': len(options)})
        if len(options) > 1:
            label_rows.append({'clip': clip.name, 'flight': clip.scene, 'keyframe_ratio': args.keyframe_ratio or settings['keyframe_baseline_ratio'],
                               'bin': b, 'context_frames': [views[p].index for p in selection if p != chosen],
                               'options': [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()} for o in options],
                               'target': 'maximise local_f; ties within tie_margin are not preferences', 'tie_margin': args.tie_margin})

    combined = evaluate(best_choice, cache, reference, ref_cell, voxel) if best_choice != selection else base
    gains = [r['regret_global_f'] for r in bin_rows]
    summary = {'clip': clip.name, 'flight': clip.scene, 'split': role_of(clip.scene, contract), 'reference_status': status,
               'student_checkpoint': str(args.student), 'student_sha256': sha256_file(args.student), 'config_sha256': sha256_file(args.config),
               'keyframe_ratio': args.keyframe_ratio or settings['keyframe_baseline_ratio'], 'poses': 'fixed reference poses for every branch',
               'selected_frames': [views[p].index for p in selection], 'combined_best_frames': [views[p].index for p in sorted(best_choice)],
               'base': base['global'], 'combined': combined['global'],
               'sum_of_single_bin_global_gains': round(float(sum(gains)), 4),
               'combined_global_gain': round(combined['global']['f_score'] - base['global']['f_score'], 4),
               'bins': len(selection), 'bins_with_alternatives': sum(r['alternatives'] > 0 for r in bin_rows),
               'bins_where_chosen_was_best': sum(r['chosen_rank'] == 1 for r in bin_rows if r['alternatives']),
               'failed_cells': int(base_failed.sum()), 'failed_cells_fixable_by_one_swap': int((base_failed & fixed_by_any).sum()),
               'depth_calls': cache.calls, 'thresholds': {'fail_f': args.fail_f, 'tie_margin': args.tie_margin,
                                                          'visible_fraction': args.visible_fraction, 'min_ref_points': args.min_ref_points},
               'limitations': ['One bin swapped at a time; interactions only checked by the combined run.',
                               'Fixed reference poses: SfM registration of the swapped frame is not tested.',
                               'Selection baseline is the rule-based keyframe policy, not the neural selector.']}
    if args.attribution:
        arrays = np.load(args.attribution / 'arrays.npz')
        if [views[p].index for p in selection] != arrays['frame_index'][arrays['selected']].tolist():
            raise SystemExit('Attribution run used a different selection')
        label = arrays['label']
        summary['attribution_crosstab'] = {name: {'cells': int((label == code).sum()),
                                                  'fixable_by_one_swap': int(((label == code) & fixed_by_any).sum())}
                                           for code, name in enumerate(LABELS, start=1)}
    summary['seconds'] = round(time.monotonic() - started, 1)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'swaps.csv', swap_rows)
    write_csv(out / 'bins.csv', bin_rows)
    with (out / 'labels.jsonl').open('w') as stream:
        for row in label_rows:
            stream.write(json.dumps(row) + '\n')
    np.savez_compressed(out / 'arrays.npz', cells=reference.cells, base_f=base['f'], combined_f=combined['f'], fixed_by_any=fixed_by_any,
                        base_failed=base_failed, selection=np.asarray(selection), combined=np.asarray(sorted(best_choice)))
    write_json(out / 'summary.json', summary)
    plot(out / 'swaps.png', bin_rows, base['f'], combined['f'], reference.cells, reference.cell, views, selection, sorted(best_choice),
         f"{clip.name} ratio {summary['keyframe_ratio']}: F {base['global']['f_score']:.3f} → {combined['global']['f_score']:.3f} with best swaps")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--clip', required=True)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--keyframe-ratio', type=float)
    p.add_argument('--attribution', type=Path, help='failure_attribution output for the same clip and ratio')
    p.add_argument('--fail-f', type=float, default=.5)
    p.add_argument('--tie-margin', type=float, default=.005, help='Local F gain needed to count a swap as better')
    p.add_argument('--visible-fraction', type=float, default=.5)
    p.add_argument('--min-ref-points', type=int, default=20)
    p.add_argument('--diagnostic', action='store_true')
    args = p.parse_args()
    s = run(args)
    print(json.dumps({k: s[k] for k in ('clip', 'keyframe_ratio', 'selected_frames', 'combined_best_frames', 'sum_of_single_bin_global_gains',
                                        'combined_global_gain', 'bins_with_alternatives', 'bins_where_chosen_was_best', 'failed_cells',
                                        'failed_cells_fixable_by_one_swap', 'depth_calls', 'seconds')
                      } | {'base_f': round(s['base']['f_score'], 4), 'combined_f': round(s['combined']['f_score'], 4),
                           'crosstab': s.get('attribution_crosstab')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
