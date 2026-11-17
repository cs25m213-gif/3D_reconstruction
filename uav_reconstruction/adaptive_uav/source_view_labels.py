"""Stage 4: label which earlier views give the best depth for a frame.

For a frame, try pairs of verified past keyframes as student sources and score each pair against
reliable measurements of that frame. Two views that each look useful can be redundant together, so
pairs are what get labelled and ranked.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np

from .counterfactual_replay import Clip, eligible, role_of
from .embedding_cache import Descriptors
from .selector_features import MapState, as_pose
from .student_inference import anchor_scale, load_student, predict
from .sequential_replay import StudentDepthProvider
from .utility_contracts import Budget, load_contract, write_json, write_status


def view_target(root: Path, contract: dict, clip: Clip, view) -> np.ndarray | None:
    file = root / contract['data']['sidecars'] / clip.name / f'frame_{view.index:06d}.npz'
    if not file.exists():
        return None
    with np.load(file) as data:
        depth = data['measured_depth_z_m'].copy()
        if 'reliable' in data:
            depth = depth * data['reliable']
        if 'map_depth_z_m' in data:
            dense = data['map_depth_z_m']
            depth = np.where((depth == 0) & (dense > 0), dense, depth)
    return depth


def absrel(prediction: np.ndarray, target: np.ndarray) -> float:
    resized = cv2.resize(prediction, target.shape[::-1], interpolation=cv2.INTER_NEAREST)
    good = (target > 0) & (resized > 0)
    return float((np.abs(resized[good] - target[good]) / target[good]).mean()) if good.sum() >= 100 else float('nan')


def label_clip(root: Path, contract: dict, clip: Clip, model, descriptors, output: Path, stream, device: str, limit: int) -> int:
    settings = contract['utility']
    candidates, written = settings['source_candidates'], 0
    state = MapState()
    provider, history = StudentDepthProvider(model,device), []
    views = clip.views
    for number, position in enumerate(clip.keyframes):
        view = views[position]
        target = view_target(root, contract, clip, view)
        pose = view.feature_pose
        if number >= 3 and target is not None and written < limit:
            budget = Budget(**settings['budgets']['loose']).state()
            row = state.features(view, as_pose(view, pose), budget, descriptors.get(clip.name, view.index, view.rgb))
            by_id = {v.frame_id: v for v in views}
            scale = anchor_scale(view.anchors_m)
            usable = []
            for slot, frame_id in enumerate(row['token_ids']):
                ratio = np.linalg.norm(by_id[frame_id].feature_pose[:3, 3] - pose[:3, 3]) / scale
                if settings['source_baseline'][0] <= ratio <= settings['source_baseline'][1]:
                    usable.append(slot)
            pairs = []
            for a, b in combinations(usable[:candidates], 2):
                sources = [(by_id[row['token_ids'][s]].rgb, by_id[row['token_ids'][s]].feature_k, by_id[row['token_ids'][s]].feature_pose) for s in (a, b)]
                error = absrel(predict(model, [(view.rgb, view.feature_k, pose)] + sources, scale, device,
                                       anchors=view.anchors_m)[0], target)
                if np.isfinite(error):
                    pairs.append([a, b, error])
            if len(pairs) >= 2:
                file = output / 'source_features' / f'{clip.name}_{view.index}.npz'
                np.savez_compressed(file, **{k: v for k, v in row.items() if k != 'token_ids'})
                stream.write(json.dumps({'clip': clip.name, 'flight': clip.scene, 'split': role_of(clip.scene, contract), 'reference': view.index,
                                         'pairs': pairs, 'features': str(file)}) + '\n')
                stream.flush()
                written += 1
        # Future policy features must observe the same student backend, not cached DA3 depth.
        prepared, _, _ = provider(view,pose,history)
        state.commit(prepared, as_pose(prepared, pose), descriptors.get(clip.name, view.index, view.rgb))
        state.observe(prepared.frame_id, prepared.depth_m, prepared.K, pose, prepared.confidence)
        history.append((prepared,pose.copy()))
        history=history[-64:]
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--student-checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--clips', nargs='*')
    parser.add_argument('--roles', nargs='*', default=['train', 'validation'])
    parser.add_argument('--limit', type=int, default=20, help='Labelled frames per clip')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root, output = args.root.resolve(), args.output.resolve()
    gate_file = root / contract['geometry_gate']
    gate = json.loads(gate_file.read_text()) if gate_file.exists() else None
    names = args.clips or sorted(p.name for p in (root / contract['data']['cache_root']).iterdir() if (p / 'selected_infos.json').exists()
                                 and 'GNSS' not in p.name and role_of(p.name, contract) in args.roles and not p.name.endswith(contract['geometry']['calibration_batch_suffix']))
    if args.dry_run:
        print(json.dumps({'clips': names}))
        return
    (output / 'source_features').mkdir(parents=True, exist_ok=True)
    model, descriptors, counts = load_student(args.student_checkpoint, args.device), Descriptors(args.embeddings), {}
    with (output / 'source_labels.jsonl').open('w') as stream:
        for name in names:
            clip = Clip(root, contract, name)
            if eligible(gate, clip.scene, args.diagnostic) is None:
                continue
            clip.choose_keyframes(contract['utility']['keyframe_baseline_ratio'])
            counts[name] = label_clip(root, contract, clip, model, descriptors, output, stream, args.device, args.limit)
            print(name, counts[name], flush=True)
    write_json(output / 'source_labels_report.json', {'clips': counts, 'student_checkpoint': str(args.student_checkpoint),
               'descriptor': descriptors.name, 'normalization':'sparse_anchor_median_then_anchor_fit',
               'causal_online_evidence':False,'qualification':'Poses and RGB anchors still originate from full-clip SfM'})


if __name__ == '__main__':
    main()
