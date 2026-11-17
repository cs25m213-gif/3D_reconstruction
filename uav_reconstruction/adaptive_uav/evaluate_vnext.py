"""Stage 9: paired policy comparison at an exactly matched keyframe budget.

Every policy gets the same frames, depth, poses and reference population, so the only difference is
which frames it chose. Promotion needs a measured gain over both the rule and coverage-gain policies,
across flights, on a trusted reference. Complete-video results come from evaluate_mesh.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .counterfactual_replay import Clip, calibration_scales, fuse, role_of
from .embedding_cache import Descriptors
from .graph_selector import load_utility_checkpoint
from .reconstruction_quality import DEFAULT_WEIGHTS, ReferencePopulation, cluster_interval, quality, surface_metrics
from .selector_policy import UtilityPolicy, coverage_gain, random_policy, rule, uniform
from .student_inference import load_student
from .utility_contracts import load_contract, sha256_file, write_json, write_status

MIN_FLIGHTS = 3


def score(chosen: list, reference, voxel: float) -> dict:
    from .remote_geometry import geometry_metrics
    metrics = geometry_metrics(chosen, reference, voxel,
                               lambda: surface_metrics(fuse([(v, v.reference_pose) for v in chosen], voxel), reference))
    return {**metrics, 'quality': quality(metrics, None, DEFAULT_WEIGHTS), 'keyframes': len(chosen)}


def promotion(rows: list, policies: list, margin: float) -> dict:
    clips = sorted({r['clip'] for r in rows})
    paired, lookup = {}, {(r['clip'], r['policy']): r for r in rows}
    for policy in policies:
        for rival in ('rule', 'coverage_gain'):
            if policy == rival:
                continue
            gains: dict = {}
            for clip in clips:
                gains.setdefault(lookup[(clip, policy)]['scene'], []).append(lookup[(clip, policy)]['quality'] - lookup[(clip, rival)]['quality'])
            paired[f'{policy}_vs_{rival}'] = cluster_interval(gains)
    matched = all(len({lookup[(c, p)]['keyframes'] for p in policies}) == 1 for c in clips)
    trusted = all(r['reference_status'] in ('VERIFIED', 'QUALIFIED') for r in rows)
    wins = [paired.get(f'utility_vs_{rival}') for rival in ('rule', 'coverage_gain')]
    gain = all(w and w['ci95'] and w['mean'] > margin and w['ci95'][0] > 0 and w['clusters'] >= MIN_FLIGHTS for w in wins)
    return {'paired': paired, 'budget_matched': matched, 'trusted_reference': trusted, 'measured_gain': gain,
            'development_candidate': bool(gain and trusted and matched), 'promote':False,
            'full_video_validated':False, 'causal_online_evidence':False,
            'blocking_reasons':['Fixed-reference-pose clip evaluation is not an online full-video validation gate'],
            'minimum_flights': MIN_FLIGHTS, 'margin': margin, 'scope': 'Clip-level selection with fixed reference poses, not a full-video result.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--utility-checkpoint', type=Path)
    parser.add_argument('--student-checkpoint', type=Path, help='Past-only student depth instead of cached teacher depth')
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--clips', nargs='*')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config, ('utility.quality_protocol', 'utility.scalarization'))
    root, settings, utility = args.root.resolve(), contract['evaluation'], contract['utility']
    every = sorted(p.name for p in (root / contract['data']['cache_root']).iterdir() if (p / 'selected_infos.json').exists() and 'GNSS' not in p.name)
    names = args.clips or [n for n in every if role_of(n, contract) == settings['split']]
    if args.dry_run:
        print(json.dumps({**settings, 'clips': names}))
        return
    gate_file = root / contract['geometry_gate']
    gate = json.loads(gate_file.read_text()) if gate_file.exists() else {}
    scales = calibration_scales(root, contract, every)
    policies = {'rule': rule, 'uniform': uniform, 'random': random_policy, 'coverage_gain': coverage_gain}
    if args.utility_checkpoint:
        model, data = load_utility_checkpoint(args.utility_checkpoint, args.device)
        policies['utility'] = UtilityPolicy(model, data, args.device)
    student = load_student(args.student_checkpoint, args.device) if args.student_checkpoint else None
    descriptors, rows = Descriptors(args.embeddings), []
    for name in names:
        if name.endswith(contract['geometry']['calibration_batch_suffix']):
            continue
        clip = Clip(root, contract, name)
        if clip.scene not in scales or not len(clip.points):
            continue
        clip.apply_scale(scales[clip.scene])
        clip.choose_keyframes(utility['keyframe_baseline_ratio'])
        if student is not None:
            clip.use_student(student, args.device)
        status = ((gate.get('flights') or {}).get(clip.scene) or {}).get('status', 'UNRESOLVED')
        reference = ReferencePopulation(clip.points, utility['tolerance_m'], status, utility['quality_protocol'])
        selections = [(policy, select(clip, settings['keyframes_per_clip'], descriptors)) for policy, select in policies.items()]
        from .remote_geometry import geometry_metrics_batch
        metrics = geometry_metrics_batch([chosen for _, chosen in selections], reference, utility['voxel_m'],
                    lambda: [surface_metrics(fuse([(v, v.reference_pose) for v in chosen], utility['voxel_m']), reference)
                             for _, chosen in selections])
        for (policy, chosen), measured in zip(selections, metrics):
            rows.append({'clip': name, 'scene': clip.scene, 'policy': policy, 'reference_status': status,
                         **measured, 'quality': quality(measured, None, DEFAULT_WEIGHTS), 'keyframes': len(chosen)})
            print(json.dumps({k: rows[-1][k] for k in ('clip', 'policy', 'keyframes', 'f_score', 'quality')}), flush=True)
    if not rows:
        raise SystemExit('No evaluable clips')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'scene_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision = promotion(rows, list(policies), settings['promotion_margin_f'])
    decision.update(utility_checkpoint_sha256=sha256_file(args.utility_checkpoint) if args.utility_checkpoint else None,
                    diagnostic_labels=bool(data.get('diagnostic_labels',True)) if args.utility_checkpoint else True)
    write_json(args.output / 'paired_metrics.json', decision['paired'])
    from .remote_geometry import routing_provenance
    decision['geometry_execution'] = routing_provenance()
    write_json(args.output / 'promotion.json', decision)
    write_status(args.output, 'stage9_evaluation', 'complete', 'PROMOTE' if decision['promote'] else 'NOT_PROMOTED', None)
    print(json.dumps({k: v for k, v in decision.items() if k != 'paired'}, indent=2))


if __name__ == '__main__':
    main()
