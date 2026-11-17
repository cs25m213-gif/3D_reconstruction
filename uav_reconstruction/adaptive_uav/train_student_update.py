"""Update the existing depth student on corrected geometry, in a new folder.

There is no separate distillation stage: this continues from a saved student, trains against the
trust-ordered loss and validates on reliable LiDAR. Inference and training default to float32,
because BF16 measurably lowered v2's accuracy. --temporal then trains only the small temporal
branch around the frozen updated student. Every model is scored on the same pixels as its control.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .depth_student import DepthStudent
from .student_dataset import UpdateDataset, discover, unlabelled_clip, auxiliary_manifest
from .student_losses import DEFAULT_WEIGHTS, student_loss
from .temporal_depth_student import TemporalDepthStudent, warp_hidden, warp_prior
from .utility_contracts import load_contract, sha256_file, state_identity, write_json, write_status

MIN_ANCHOR_PIXELS = 30  # Same minimum as fit_group_scale for sparse anchors.


def frame_of(batch: dict, t: int) -> dict:
    return {key: value[:, t] for key, value in batch.items() if torch.is_tensor(value) and value.ndim >= 3}


def to_device(batch: dict, device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def relative(frame: dict, previous: dict) -> torch.Tensor:
    return frame['P'][:, 0] @ torch.linalg.inv(previous['P'][:, 0])


def forward_sequence(model, batch: dict, weights: dict, autocast, temporal: bool) -> tuple:
    """Run chronologically. With more than one frame, the last prediction constrains the next."""
    total, logged, previous, hidden = 0., collections.Counter(), None, None
    steps = batch['rgb'].shape[1]
    for t in range(steps):
        frame = frame_of(batch, t)
        prior, wide, visibility = None, (None, None, None), None
        if previous is not None:
            k = frame['K'][:, 0]
            warped, warped_confidence, warped_valid = warp_prior(previous['depth'], previous['confidence'], k, relative(frame, previous['frame']),
                                                                previous_k=previous['frame']['K'][:,0])
            prior, wide = (warped, warped_valid), (warped, warped_confidence, warped_valid)
        with autocast():
            if temporal:
                memory = None
                if hidden is not None:
                    memory = warp_hidden(hidden, warped, frame['K'][:, 0], torch.linalg.inv(relative(frame, previous['frame'])),
                                         previous_k=previous['frame']['K'][:,0])
                depth, confidence, visibility, hidden = model(frame['rgb'], frame['K'], frame['P'], *wide, memory)
            else:
                depth, confidence = model(frame['rgb'], frame['K'], frame['P'])
        depth, confidence = depth.float(), confidence.float()
        loss, parts = student_loss(frame, depth, confidence, weights, prior, None if visibility is None else visibility.float())
        total = total + loss
        logged.update({k: v / steps for k, v in parts.items()})
        previous = {'depth': depth.detach(), 'confidence': confidence.detach(), 'frame': frame}
    return total / steps, dict(logged)


@torch.no_grad()
def evaluate(model, loader, device, autocast) -> dict:
    """Reliable LiDAR only, so the score never depends on the targets the model trained with.

    Deployment rescales each student frame to its SfM anchors, and the teacher was already scaled
    that way, so that is the score that decides. The raw score stays in the log.
    """
    model.eval()
    scenes: dict = collections.defaultdict(lambda: collections.Counter())
    for batch in loader:
        batch = to_device(batch, device)
        frame = frame_of(batch, -1)
        with autocast():
            output = model(frame['rgb'], frame['K'], frame['P'])
        depth = output[0].float()
        valid, target = frame['measured_valid'], frame['measured']
        factor = batch['metric_per_normalized'].to(depth).float()[:, None, None, None]
        for b, scene in enumerate(batch['scene']):
            mask = valid[b]
            pixels = int(mask.sum())
            if not pixels:
                continue
            known = (frame['anchor'][b] > 0) & (depth[b] > 0)
            fitted = depth[b] * (frame['anchor'][b][known] / depth[b][known]).median() if int(known.sum()) >= MIN_ANCHOR_PIXELS else depth[b]
            for name, prediction in (('student', fitted), ('raw_student', depth[b]), ('teacher', frame['teacher'][b])):
                delta = prediction - target[b]
                row = {f'{name}_rel': float((delta.abs() / target[b].clamp_min(1e-4))[mask].sum()),
                       f'{name}_mae': float((delta.abs() * factor[b])[mask].sum()), f'{name}_bias': float((delta * factor[b])[mask].sum())}
                scenes[scene].update(row)
                scenes['all'].update(row)
            scenes[scene]['pixels'] += pixels
            scenes['all']['pixels'] += pixels
    result = {}
    for scene, row in scenes.items():
        n = max(row['pixels'], 1)
        result[scene] = {'pixels': row['pixels'], 'student_absrel': row['student_rel'] / n, 'student_mae_m': row['student_mae'] / n,
                         'student_bias_m': row['student_bias'] / n, 'raw_student_absrel': row['raw_student_rel'] / n, 'raw_student_bias_m': row['raw_student_bias'] / n,
                         'teacher_absrel': row['teacher_rel'] / n, 'teacher_mae_m': row['teacher_mae'] / n,
                         'teacher_bias_m': row['teacher_bias'] / n}
    return result


def save_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def data_identities(clips):
    """Track auxiliary caches as well as supervised inputs when checking resume."""
    identities = []
    for clip in clips:
        for index in clip.indices:
            paths = [clip.cache / f'frame_{index:06d}.jpg.npz']
            if clip.targets: paths.append(clip.targets / f'frame_{index:06d}.npz')
            if clip.overlay: paths.append(clip.overlay / f'frame_{index:06d}.npz')
            for path in paths:
                if path.is_file():
                    stat = path.stat()
                    identities.append((str(path), stat.st_size, stat.st_mtime_ns))
    return sorted(identities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--sequence', type=int, default=2, help='Frames per sample; two or more enables multiview consistency')
    parser.add_argument('--cached-sources', action='store_true', help='Use original contexts, which can include future frames')
    parser.add_argument('--precision', choices=('float32', 'bf16'), default='float32')
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--unlabelled', type=Path, nargs='*', default=[], help='depth_cache folders of your own sessions')
    parser.add_argument('--aux-manifest', type=Path, action='append', default=[], help='Completed auxiliary manifests; only training-role clips are admitted')
    parser.add_argument('--allow-unresolved', action='store_true', help='Train on flights without a passed gate; diagnostic only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int, help='Stop each epoch early; for smoke tests only')
    parser.add_argument('--weight', action='append', default=[], metavar='NAME=VALUE', help='Override a loss weight, e.g. teacher=0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root = args.root.resolve()
    if args.dry_run:
        print(json.dumps({'weights': DEFAULT_WEIGHTS, 'splits': contract['splits']}))
        return
    weights = {**DEFAULT_WEIGHTS, **contract.get('student', {}).get('weights', {})}
    for item in args.weight:
        name, value = item.split('=')
        if name not in weights:
            raise SystemExit(f'Unknown loss term {name}')
        weights[name] = float(value)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    roles = discover(root, contract, trusted_only=not args.allow_unresolved)
    if not roles['train'] or not roles['validation']:
        raise SystemExit(f"No usable clips: { {k: len(v) for k, v in roles.items()} }. Run calibration first, or pass --allow-unresolved for a diagnostic run.")
    causal = not args.cached_sources
    train = UpdateDataset(roles['train'], args.sequence, causal, augment=True)
    auxiliary = [unlabelled_clip(p) for p in args.unlabelled]
    for manifest in args.aux_manifest:
        auxiliary.extend(auxiliary_manifest(manifest))
    if auxiliary:
        train = ConcatDataset((train, UpdateDataset(auxiliary, args.sequence, causal, augment=True)))
    validation = UpdateDataset(roles['validation'], 1, causal)
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, shuffle=True, drop_last=True, **common)
    validation_loader = DataLoader(validation, shuffle=False, **common)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = args.precision == 'bf16' and device.type == 'cuda'
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)
    start = args.init_checkpoint or root / contract['depth_checkpoint']
    if args.output.resolve() == start.resolve().parent:
        raise SystemExit('--output must be a new directory, not the checkpoint being updated')
    state = torch.load(start, map_location='cpu', weights_only=False)
    student = DepthStudent(48, 24)
    student.load_state_dict(state['model'] if 'model' in state else state)
    model = (TemporalDepthStudent(student) if args.temporal else student).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / 'latest.pt'
    identities = data_identities([clip for clips in roles.values() for clip in clips] + auxiliary)
    source_dir=Path(__file__).parent
    run_identity=state_identity(config=sha256_file(args.config),start=sha256_file(start),weights=weights,
        data=identities,causal=causal,precision=args.precision,sequence=args.sequence,batch=args.batch_size,
        learning_rate=args.learning_rate,unlabelled=[str(p) for p in args.unlabelled],
        auxiliary_manifests={str(p):sha256_file(p) for p in args.aux_manifest},
        source={n:sha256_file(source_dir/n) for n in ('train_student_update.py','student_dataset.py','student_losses.py','student_inference.py',
                                                    'depth_student.py','temporal_depth_student.py','build_cached_lidar_targets.py','train_depth_student.py')})
    first, best, stale, steps, skipped = 1, float('inf'), 0, 0, 0
    if latest.exists() and not args.resume:
        raise SystemExit(f'{latest} exists; use --resume or a new output folder')
    if args.resume and latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get('run_identity')!=run_identity:
            raise SystemExit('Training inputs/code/config changed or old resume metadata is missing; use a new output directory')
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer'])
        first, best, stale, steps, skipped = saved['epoch'] + 1, saved['best'], saved['stale'], saved['optimizer_steps'], saved['skipped']
        torch.set_rng_state(saved['rng']['torch'].cpu()); np.random.set_state(saved['rng']['numpy']); random.setstate(saved['rng']['python'])
    if args.resume and (args.output / 'config.json').exists():
        baseline = json.loads((args.output / 'config.json').read_text())['baseline']  # Weights here are already updated.
    else:
        baseline = evaluate(student.to(device), validation_loader, device, autocast)
    write_json(args.output / 'config.json', {'args': {k: str(v) for k, v in vars(args).items()}, 'weights': weights, 'start_checkpoint': str(start),
                                             'trainable_parameters': sum(p.numel() for p in trainable), 'clips': {k: [c.name for c in v] for k, v in roles.items()}, 'auxiliary_clips': [c.name for c in auxiliary], 'map_only_clips': [c.name for c in roles['train'] if c.map_only],
                                             'examples': {'train': len(train), 'validation': len(validation)}, 'causal_sources': causal, 'baseline': baseline})
    write_status(args.output, 'stage5_student', 'running', None, None)
    started = time.monotonic()
    with (args.output / 'optimizer_steps.jsonl').open('a') as log:
        for epoch in range(first, args.epochs + 1):
            model.train()
            if args.temporal:
                model.student.eval()
            losses = []
            for index, batch in enumerate(train_loader, 1):
                if args.max_steps and index > args.max_steps:
                    break
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                total, parts = forward_sequence(model, batch, weights, autocast, args.temporal)
                if not torch.isfinite(total):
                    skipped += 1
                    continue
                total.backward()
                norm = nn.utils.clip_grad_norm_(trainable, 2.)
                if not torch.isfinite(norm):
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                optimizer.step()
                steps += 1
                losses.append(float(total.detach()))
                if index % 50 == 0:
                    log.write(json.dumps({'epoch': epoch, 'step': steps, 'loss': float(np.mean(losses[-50:])), 'gradient_norm': float(norm), **parts}) + '\n')
                    log.flush()
            metrics = evaluate(model, validation_loader, device, autocast)
            score = metrics['all']['student_absrel']
            payload = {'model': student.state_dict(), 'epoch': epoch, 'validation': metrics, 'optimizer_steps': steps, 'precision': args.precision,
                       'architecture': {'channels': 48, 'planes': 24}, 'temporal': model.state_dict() if args.temporal else None,
                       'architecture_id': TemporalDepthStudent.architecture_id if args.temporal else 'depth_student_48ch_24planes',
                       'start_checkpoint': str(start),'run_identity':run_identity,'evaluation_protocol':'sparse_anchor_fit_v1'}
            if score < best:
                best, stale = score, 0
                save_atomic(args.output / 'best.pt', payload)
            else:
                stale += 1
            save_atomic(latest, {'weights': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best': best, 'stale': stale,
                                 'optimizer_steps': steps, 'skipped': skipped, 'run_identity':run_identity,
                                 'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}})
            row = {'epoch': epoch, 'train_loss': float(np.mean(losses)) if losses else None, 'validation': metrics['all'], 'best_absrel': best, 'optimizer_steps': steps,
                   'skipped_updates': skipped, 'seconds': time.monotonic() - started}
            print(json.dumps(row), flush=True)
            if stale >= args.patience:
                break
    control = baseline['all']['student_absrel']
    report = {'best_validation_absrel': best, 'control_validation_absrel': control, 'teacher_validation_absrel': baseline['all']['teacher_absrel'],
              'improved_on_control': best < control, 'beat_teacher': best < baseline['all']['teacher_absrel'], 'optimizer_steps': steps, 'skipped_updates': skipped,
              'scope': 'Depth against reliable LiDAR on validation flights; not a 3D result.'}
    report.update(checkpoint_sha256=sha256_file(args.output/'best.pt') if (args.output/'best.pt').exists() else None,
                  evaluation_protocol='sparse_anchor_fit_v1', validation_pixels=baseline['all']['pixels'],
                  diagnostic_labels=args.allow_unresolved)
    write_json(args.output / 'report.json', report)
    write_status(args.output, 'stage5_student', 'complete', 'IMPROVED' if report['improved_on_control'] else 'NOT_IMPROVED', 'stage4_labels')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
