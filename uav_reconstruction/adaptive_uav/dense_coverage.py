"""Select depth cameras by spatial support at the depth filter's parallax angle."""
import argparse
import json
from pathlib import Path

import numpy as np


def recovery_workspace(source, output, max_views=32, source_views=4):
    """Expand reference coverage with the same 3-degree parallax criterion.

    Pose/image files and optional photometric initializations are reused. All
    geometric maps are recalculated because the source-view sets have changed.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new dense recovery workspace to preserve existing maps')
    output.mkdir(parents=True)
    for name in ('images', 'sparse'):
        (output / name).symlink_to((source / name).resolve(), target_is_directory=True)
    for name in ('depth_maps', 'normal_maps', 'consistency_graphs'):
        target = output / 'stereo' / name
        target.mkdir(parents=True)
        if name != 'consistency_graphs':
            for file in (source / 'stereo' / name).glob('*.photometric.bin'):
                (target / file.name).symlink_to(file.resolve())
    report = plan_coverage(output, max_views, source_views, min_angle=3.)
    (output / 'refinement_provenance.json').write_text(json.dumps({
        'source_workspace': str(source), 'camera_poses_reused': True,
        'parallax_threshold_relaxed': False, 'photometric_initialization_reused': True,
        'geometric_depth_recomputed_for_all_references': True,
        'reason': 'Allocate temporal anchors across all stereo-supported cameras instead of only a small check subset.'
    }, indent=2) + '\n')
    return report


def spatial_weights(xy, width, height, grid_size=8):
    cells = np.clip((np.asarray(xy) / [width, height] * grid_size).astype(int),
                    0, grid_size - 1)
    keys = cells[:, 1] * grid_size + cells[:, 0]
    counts = np.bincount(keys, minlength=grid_size ** 2)
    weights = 1. / counts[keys]
    return weights / weights.sum()


def choose_sources(masks, weights, limit=6, baseline=()):
    """Reward two usable source observations in every occupied image cell."""
    counts = np.zeros(len(weights), dtype=int)
    chosen = []
    for _ in range(limit):
        marginal = np.where(counts == 0, .5, np.where(counts == 1, 1., .02))
        scores = {name: float(np.sum(weights * mask * marginal))
                  for name, mask in masks.items() if name not in chosen}
        if not scores:
            break
        name = max(scores, key=scores.get)
        chosen.append(name)
        counts += masks[name]
    fallback = [name for name in baseline if name in masks][:limit]
    old_counts = sum((masks[name] for name in fallback), np.zeros(len(weights), int))
    if np.sum(weights * (old_counts >= 2)) > np.sum(weights * (counts >= 2)):
        chosen, counts = fallback, old_counts
    return chosen, float(np.sum(weights * (counts >= 2)))


def two_source_core(graph, selected):
    """Remove references whose sources cannot themselves obtain geometric depth."""
    selected = set(selected)
    while True:
        valid = {name for name in selected if len(graph[name] & selected) >= 2}
        if valid == selected:
            return valid
        selected = valid


def supported_reference_subset(graph, budget, preferred=()):
    """Try connected neighbourhoods when temporal anchors miss a stereo group.

    This bounded graph search changes camera selection, never the pair gates.
    A graph edge still requires real shared tracks at the requested angle.
    """
    core = two_source_core(graph, graph)
    if len(core) <= budget:
        return core
    preferred = set(preferred)
    best = set()
    for root in sorted(core):
        selected, frontier = {root}, [root]
        while frontier and len(selected) < budget:
            current = frontier.pop(0)
            neighbours = sorted((graph[current] & core) - selected,
                                key=lambda n: (-len(graph[n] & selected),
                                               n not in preferred, -len(graph[n] & core), n))
            for name in neighbours:
                if len(selected) >= budget:
                    break
                selected.add(name)
                frontier.append(name)
        selected = two_source_core(graph, selected)
        if (len(selected), len(selected & preferred)) > (len(best), len(best & preferred)):
            best = selected
    return best


def plan_coverage(workspace, max_views=96, source_views=6, seed_plan=None,
                  min_angle=3.):
    import pycolmap
    workspace = Path(workspace)
    reconstruction = pycolmap.Reconstruction(workspace / 'sparse')
    images = {image.name: image for image in reconstruction.images.values()
              if image.has_pose and image.num_points3D >= 30}
    names = sorted(images)
    if len(names) < 5 or max_views < 5 or source_views < 2:
        raise ValueError('Dense reconstruction needs at least five cameras and two sources')
    if seed_plan:
        selected = {row['reference'] for row in seed_plan['plan']}
        if len(selected) > max_views or not selected.issubset(images):
            raise ValueError('Seed references do not fit the camera budget/model')
    else:
        # Reserve a quarter of the budget for useful supporting cameras instead
        # of spending every slot on temporal subsampling alone.
        anchors = min(len(names), max(5, 3 * max_views // 4))
        selected = {max((names[int(i)] for i in group),
                        key=lambda name: images[name].num_points3D)
                    for group in np.array_split(np.arange(len(names)), anchors)}
    seeds = sorted(selected)
    centers = {name: np.asarray(image.cam_from_world().inverse().translation)
               for name, image in images.items()}
    tracks = {name: {point.point3D_id for point in image.points2D if point.has_point3D()}
              for name, image in images.items()}
    cache = {}

    def support(name):
        if name in cache:
            return cache[name]
        image = images[name]
        camera = reconstruction.cameras[image.camera_id]
        points = [point for point in image.points2D if point.has_point3D()]
        ids = [point.point3D_id for point in points]
        xyz = np.array([reconstruction.points3D[pid].xyz for pid in ids])
        weights = spatial_weights([point.xy for point in points], camera.width, camera.height)
        rays = xyz - centers[name]
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        masks, details = {}, {}
        for other in names:
            if other == name:
                continue
            common = np.array([pid in tracks[other] for pid in ids])
            if common.sum() < 30:
                continue
            other_rays = xyz - centers[other]
            other_rays /= np.maximum(np.linalg.norm(other_rays, axis=1, keepdims=True), 1e-12)
            angles = np.degrees(np.arccos(np.clip((rays * other_rays).sum(1), -1, 1)))
            usable = common & (angles >= min_angle) & (angles <= 45)
            if usable.sum() < 30:
                continue
            masks[other] = usable
            details[other] = {'name': other, 'common_points': int(common.sum()),
                              'usable_points': int(usable.sum()),
                              'median_angle_deg': float(np.median(angles[usable]))}
        cache[name] = weights, masks, details
        return cache[name]

    counts = {}
    for name in seeds:
        weights, masks, _ = support(name)
        counts[name] = sum((mask for other, mask in masks.items() if other in selected),
                           np.zeros(len(weights), int))
    while len(selected) < min(max_views, len(names)):
        scores = {name: 0. for name in names if name not in selected}
        for ref in seeds:
            weights, masks, _ = support(ref)
            marginal = np.where(counts[ref] == 0, .5,
                                np.where(counts[ref] == 1, 1., .02))
            for name in scores:
                if name in masks:
                    scores[name] += float(np.sum(weights * masks[name] * marginal))
        pick = max(scores, key=scores.get)
        if scores[pick] <= 0:
            break
        selected.add(pick)
        for ref in seeds:
            masks = support(ref)[1]
            if pick in masks:
                counts[ref] += masks[pick]
    # Every selected source must itself support geometric depth estimation.
    initial_selection = set(selected)
    graph = {name: set(support(name)[1]) for name in selected}
    selected = two_source_core(graph, selected)
    selection_recovered = False
    if len(selected) < 5:
        graph = {name: set(support(name)[1]) for name in names}
        selected = supported_reference_subset(graph, max_views, seeds)
        full_core = two_source_core(graph, graph)
        diagnostic = {
            'registered_candidates': len(names), 'max_reference_views': max_views,
            'filter_min_triangulation_angle': min_angle,
            'minimum_usable_shared_tracks': 30,
            'temporal_selection_before_pruning': len(initial_selection),
            'all_camera_two_source_core': len(full_core),
            'recovered_reference_views': len(selected),
            'source_counts': {name: len(graph[name]) for name in names},
            'parallax_threshold_relaxed': False,
            'reason': ('camera_budget_or_selection' if len(full_core) >= 5
                       else 'insufficient_supported_camera_graph'),
        }
        (workspace / 'dense_planning_diagnostics.json').write_text(
            json.dumps(diagnostic, indent=2) + '\n')
        if len(selected) < 5:
            raise RuntimeError(
                'Insufficient spatial overlap and parallax for dense reconstruction: '
                f'{len(full_core)}/{len(names)} cameras have two supported sources '
                f'in the full graph; {len(selected)} fit the {max_views}-view budget. '
                'See dense_planning_diagnostics.json')
        selection_recovered = True
    old_rows = {row['reference']: row for row in (seed_plan or {}).get('plan', [])}
    rows = []
    for name in sorted(selected):
        weights, masks, details = support(name)
        masks = {other: mask for other, mask in masks.items() if other in selected}
        old_names = [s['name'] for s in old_rows.get(name, {}).get('sources', [])]
        chosen, fraction = choose_sources(masks, weights, source_views, old_names)
        row = {'reference': name, 'sources': [details[other] for other in chosen],
               'spatial_sparse_support_two_sources': fraction}
        if old_names:
            old_count = sum((masks[s] for s in old_names if s in masks),
                            np.zeros(len(weights), int))
            row['previous_spatial_sparse_support_two_sources'] = float(
                np.sum(weights * (old_count >= 2)))
        rows.append(row)
    report = {'registered_cameras': reconstruction.num_reg_images(),
              'reference_views': len(rows), 'max_reference_views': max_views,
              'max_source_views': source_views, 'filter_min_triangulation_angle': min_angle,
              'planner_version': 3, 'selection_recovered': selection_recovered,
              'added_references': sorted(selected - set(seeds)),
              'rejected_references': sorted(set(seeds) - selected), 'plan': rows,
              'selection': 'temporal anchors plus support cameras; two source observations per image cell at the depth filter angle',
              'support_metric_note': 'Sparse-feature support is a planning diagnostic, not surface coverage or accuracy.'}
    (workspace / 'dense_plan.json').write_text(json.dumps(report, indent=2) + '\n')
    (workspace / 'stereo').mkdir(exist_ok=True)
    (workspace / 'stereo/patch-match.cfg').write_text(''.join(
        row['reference'] + '\n' + ', '.join(s['name'] for s in row['sources']) + '\n' for row in rows))
    (workspace / 'stereo/fusion.cfg').write_text(''.join(row['reference'] + '\n' for row in rows))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra-views', type=int, default=32)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Use a new output directory to preserve previous results')
    if args.extra_views < 0:
        raise ValueError('Extra view count must be nonnegative')
    output.mkdir(parents=True)
    for name in ('images', 'sparse'):
        (output / name).symlink_to((source / name).resolve(), target_is_directory=True)
    for folder in ('depth_maps', 'normal_maps', 'consistency_graphs'):
        destination = output / 'stereo' / folder
        destination.mkdir(parents=True)
        if folder != 'consistency_graphs':
            for path in (source / 'stereo' / folder).glob('*.photometric.bin'):
                (destination / path.name).symlink_to(path.resolve())
    seed = json.loads((source / 'dense_plan.json').read_text())
    report = plan_coverage(output, len(seed['plan']) + args.extra_views,
                           seed['max_source_views'], seed)
    provenance = {'source_workspace': str(source), 'camera_poses_reused': True,
                  'photometric_initialization_reused': True,
                  'geometric_depth_recomputed_for_all_references': True,
                  'reason': 'source sets now cover spatial regions at the filter parallax threshold'}
    (output / 'refinement_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    pairs = [r for r in report['plan'] if 'previous_spatial_sparse_support_two_sources' in r]
    print(json.dumps({'references': report['reference_views'], 'added': len(report['added_references']),
                      'sparse_support_before': float(np.mean([r['previous_spatial_sparse_support_two_sources'] for r in pairs])),
                      'sparse_support_after': float(np.mean([r['spatial_sparse_support_two_sources'] for r in pairs]))}, indent=2))


if __name__ == '__main__':
    main()
