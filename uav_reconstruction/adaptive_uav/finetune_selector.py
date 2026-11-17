"""Fine-tune the selector on real reconstruction reward, on training flights only.

This is the proposal's closed loop. The policy samples a frame set, the set is reconstructed and
scored, and the reward is its gain over the rule on that clip. Nothing differentiates through
fusion: REINFORCE with a per-clip baseline pushes the policy toward sets that truly did better.
It starts from a supervised checkpoint and writes a new one, which must still pass evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from .counterfactual_replay import Clip, calibration_scales, eligible, fuse, role_of
from .embedding_cache import Descriptors
from .graph_selector import load_utility_checkpoint
from .reconstruction_quality import DEFAULT_WEIGHTS, ReferencePopulation, quality, surface_metrics
from .remote_geometry import geometry_metrics, routing_provenance
from .selector_policy import UtilityPolicy, rule
from .student_inference import load_student
from .utility_contracts import load_contract, write_json, write_status


def reward(chosen: list, reference, voxel: float) -> float:
    metrics = geometry_metrics(chosen, reference, voxel,
                               lambda: surface_metrics(fuse([(v, v.reference_pose) for v in chosen], voxel), reference))
    return quality(metrics, None, DEFAULT_WEIGHTS)


def training_plateau(completed, best_at, patience):
    """A budget heuristic on training reward only; held-out evaluation is still required."""
    return patience > 0 and best_at > 0 and completed >= max(1000, patience) and completed-best_at >= patience


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--student-checkpoint', type=Path, help='Past-only student depth instead of cached teacher depth')
    parser.add_argument('--rollouts', type=int, default=200)
    parser.add_argument('--patience-rollouts', type=int, default=1000,
                        help='Stop after this many attempts without a better full-cycle training mean, after at least 1000 attempts; 0 disables')
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--entropy', type=float, default=.01)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--allow-diagnostic', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.rollouts < 1 or args.patience_rollouts < 0: parser.error('Need positive rollouts and nonnegative patience')
    if not args.dry_run and any((args.output/name).exists() for name in ('best.pt','rollouts.jsonl','report.json')):
        raise SystemExit('Existing fine-tuning artifacts: use a fresh output directory; this trainer does not resume')
    contract = load_contract(args.config, ('utility.quality_protocol', 'utility.scalarization'))
    root, settings = args.root.resolve(), contract['utility']
    every = sorted(p.name for p in (root / contract['data']['cache_root']).iterdir() if (p / 'selected_infos.json').exists() and 'GNSS' not in p.name)
    names = [n for n in every if role_of(n, contract) == 'train' and not n.endswith(contract['geometry']['calibration_batch_suffix'])]
    if args.dry_run:
        print(json.dumps({'clips': names, 'rollouts': args.rollouts,'patience_rollouts':args.patience_rollouts}))
        return
    device = torch.device(args.device)
    model, data = load_utility_checkpoint(args.checkpoint, device)
    if data['diagnostic_labels'] and not args.allow_diagnostic:
        raise SystemExit('This checkpoint came from diagnostic labels. Pass --allow-diagnostic for a non-promotable run.')
    random.seed(contract['selector']['seed']); torch.manual_seed(contract['selector']['seed'])
    gate_file = root / contract['geometry_gate']
    gate = json.loads(gate_file.read_text()) if gate_file.exists() else {}
    scales = calibration_scales(root, contract, every)
    count = contract['evaluation']['keyframes_per_clip']
    descriptors = Descriptors(args.embeddings)
    policy = UtilityPolicy(model, data, device)
    student = load_student(args.student_checkpoint, args.device) if args.student_checkpoint else None
    clips, baselines = [], {}
    for name in names:
        clip = Clip(root, contract, name)
        status = eligible(gate, clip.scene, args.allow_diagnostic)
        if status is None or clip.scene not in scales or not len(clip.points):
            continue  # A reward needs a reference that passed the gate, same as the labels.
        clip.apply_scale(scales[clip.scene])
        clip.choose_keyframes(settings['keyframe_baseline_ratio'])
        if student is not None:
            clip.use_student(student, args.device)
        reference = ReferencePopulation(clip.points, settings['tolerance_m'], status, settings['quality_protocol'])
        baselines[name] = reward(rule(clip, count), reference, settings['voxel_m'])
        clips.append((clip, reference))
    if not clips:
        raise SystemExit('No training clips with a reference')
    model.eval()
    model.gru.train()  # CUDA RNNs only backpropagate in training mode; this GRU has no dropout.
    # Only the decision path moves: early layers keep what supervised ranking learned.
    trainable = [p for name, p in model.named_parameters() if name.split('.')[0] in ('gru', 'utility')]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.split('.')[0] in ('gru', 'utility'))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=contract['selector']['weight_decay'])
    args.output.mkdir(parents=True, exist_ok=True)
    moving, recent, best = {name: 0. for name in baselines}, [], None
    optimizer_steps, skipped_updates, completed, best_at = 0, 0, 0, 0
    # Even a bounded run shorter than one clip cycle must leave a usable original candidate.
    torch.save({**data,'model':model.state_dict(),'finetuned_from':str(args.checkpoint),
                'rollouts':0,'finetune_optimizer_steps':0,'mean_gain_over_rule':None},args.output/'best.pt')
    with (args.output / 'rollouts.jsonl').open('a') as log:
        for step in range(args.rollouts):
            clip, reference = clips[step % len(clips)]
            chosen, log_probabilities, entropies = policy.run(clip, count, descriptors, args.temperature)
            gain = reward(chosen, reference, settings['voxel_m']) - baselines[clip.name]
            advantage = gain - moving[clip.name]
            moving[clip.name] = .9 * moving[clip.name] + .1 * gain
            recent = (recent + [gain])[-len(clips):]
            if log_probabilities:
                loss = -(advantage * torch.stack(log_probabilities).sum() + args.entropy * torch.stack(entropies).sum())
                if torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
                    if torch.isfinite(norm):
                        optimizer.step(); optimizer_steps += 1
                    else:
                        optimizer.zero_grad(set_to_none=True); skipped_updates += 1
                else:
                    skipped_updates += 1
            log.write(json.dumps({'step': step, 'clip': clip.name, 'gain_over_rule': gain, 'advantage': advantage}) + '\n')
            log.flush()
            completed = step + 1
            mean = float(np.mean(recent))
            if len(recent) == len(clips) and (best is None or mean > best):
                best = mean; best_at = completed
                temporary = args.output / 'best.tmp'
                torch.save({**data, 'model': model.state_dict(), 'finetuned_from': str(args.checkpoint), 'rollouts': completed,
                            'finetune_optimizer_steps':optimizer_steps, 'mean_gain_over_rule': best}, temporary)
                temporary.replace(args.output / 'best.pt')
            print(json.dumps({'step': step, 'gain': gain, 'mean_recent': mean}), flush=True)
            if training_plateau(completed,best_at,args.patience_rollouts):break
    write_json(args.output / 'report.json', {'rollouts': completed, 'planned_rollouts':args.rollouts,
               'selected_checkpoint_rollouts':best_at,'early_stopped':completed<args.rollouts,
               'stop_reason':'training_reward_plateau' if completed<args.rollouts else 'rollout_budget',
               'patience_rollouts':args.patience_rollouts,'optimizer_steps': optimizer_steps,
               'skipped_updates': skipped_updates, 'best_mean_gain_over_rule': best,
               'geometry_execution': routing_provenance(),
               'scope': 'Training-flight TSDF point-cloud reward with fixed reference poses, not fresh video-to-mesh reconstruction; only independent evaluation can establish gain.'})
    write_status(args.output, 'stage5_finetune', 'complete', 'FINETUNED', 'stage9_evaluation')


if __name__ == '__main__':
    main()
