"""Retrain the existing depth student with measured LiDAR sidecars.

Teacher caches and the original checkpoint remain unchanged. Scene areas are
split before training: AMtown/HKairport train, AMvalley validate, HKisland test.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .build_cached_lidar_targets import downsample_sparse_min
from .depth_student import DepthStudent
from .train_depth_student import TeacherCacheDataset


ROLES = {'AMtown': 'train', 'HKairport': 'train',
         'AMvalley': 'validation', 'HKisland': 'test'}


class MeasuredTeacherDataset(TeacherCacheDataset):
    def __init__(self, paths, sidecars: Path, scales: dict, augment: bool = False):
        super().__init__(paths)
        self.sidecars = sidecars
        self.scales = scales
        self.augment = augment

    def __getitem__(self, index):
        item = super().__getitem__(index)
        file = self.paths[index]
        batch = file.parents[3].name
        frame = file.name.split('.')[0]
        target = self.sidecars / batch / (frame + '.npz')
        with np.load(target) as data:
            measured = downsample_sparse_min(data['measured_depth_z_m'], self.size)
        metric_per_visual = self.scales[batch]['scale_m_per_visual_unit']
        normalized = measured / (metric_per_visual * item['scale'])
        item['measured'] = torch.from_numpy(normalized[None].astype(np.float32))
        item['metric_per_normalized'] = metric_per_visual * item['scale']
        item['measured_valid'] = torch.from_numpy(((normalized > .25) & (normalized < 4.))[None])
        if self.augment:
            # One transform for all views retains their appearance correspondence.
            gain = .75 + .5 * torch.rand(())
            offset = -.08 + .16 * torch.rand(())
            gamma = .8 + .4 * torch.rand(())
            item['rgb'] = (item['rgb'] * gain + offset).clamp(0, 1).pow(gamma)
        return item


def discover(cache_root: Path, sidecars: Path, scales: dict) -> tuple[dict, dict]:
    paths = {role: [] for role in ('train', 'validation', 'test')}
    scenes = {role: set() for role in paths}
    for batch in sorted(cache_root.iterdir()):
        info_file = batch / 'selected_infos.json'
        if not info_file.exists() or batch.name not in scales or not scales[batch.name]['usable']:
            continue
        scene = scales[batch.name]['scene']
        if 'GNSS' in scene:
            continue
        role = next((value for prefix, value in ROLES.items() if scene.startswith(prefix)), None)
        if role is None:
            continue
        for target in sorted((sidecars / batch.name).glob('frame_*.npz')):
            file = batch / 'reconstruction/multiview/depth_cache' / (target.stem + '.jpg.npz')
            if file.exists():
                paths[role].append(str(file.resolve()))
                scenes[role].add(scene)
    return paths, {k: sorted(v) for k, v in scenes.items()}


def loss_fn(batch, prediction, confidence, measured_weight=4., teacher_weight=.5):
    measured = batch['measured']
    measured_valid = batch['measured_valid']
    teacher = batch['teacher']
    teacher_valid = batch['teacher_valid'] & ~measured_valid
    log_prediction = prediction.clamp_min(1e-4).log()
    measured_error = (log_prediction - measured.clamp_min(1e-4).log()).abs()
    teacher_error = (log_prediction - teacher.clamp_min(1e-4).log()).abs()
    measured_loss = measured_error[measured_valid].mean()
    teacher_loss = teacher_error[teacher_valid].mean() if teacher_valid.any() else prediction.sum() * 0
    anchors = batch['anchor']
    anchor_valid = (anchors > .25) & (anchors < 4.) & ~measured_valid
    anchor_loss = (log_prediction - anchors.clamp_min(1e-4).log()).abs()[anchor_valid].mean() if anchor_valid.any() else prediction.sum() * 0
    confidence_target = torch.exp(-5 * measured_error.detach()).clamp(0, 1)
    probability = confidence[measured_valid].clamp(1e-5, 1-1e-5)
    target = confidence_target[measured_valid]
    confidence_loss = -(target * probability.log() + (1-target) * (1-probability).log()).mean()
    total = measured_weight * measured_loss + teacher_weight * teacher_loss + .2 * anchor_loss + .05 * confidence_loss
    return total, {'measured': float(measured_loss.detach()), 'teacher': float(teacher_loss.detach()),
                   'anchor': float(anchor_loss.detach()), 'confidence': float(confidence_loss.detach())}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {'student_sum': 0., 'teacher_sum': 0., 'student_mae_m_sum': 0.,
              'teacher_mae_m_sum': 0., 'student_bias_m_sum': 0., 'teacher_bias_m_sum': 0.,
              'pixels': 0, 'frames': 0}
    for batch in loader:
        rgb, k, p = (batch[name].to(device) for name in ('rgb', 'K', 'P'))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            prediction, _ = model(rgb, k, p)
        prediction = prediction.float().cpu()
        valid = batch['measured_valid']
        target = batch['measured']
        teacher = batch['teacher']
        factor = batch['metric_per_normalized'].float()[:, None, None, None]
        count = int(valid.sum())
        if count == 0:
            continue
        student_delta = prediction - target
        teacher_delta = teacher - target
        totals['student_sum'] += float((student_delta.abs() / target.clamp_min(1e-4))[valid].sum())
        totals['teacher_sum'] += float((teacher_delta.abs() / target.clamp_min(1e-4))[valid].sum())
        totals['student_mae_m_sum'] += float((student_delta.abs() * factor)[valid].sum())
        totals['teacher_mae_m_sum'] += float((teacher_delta.abs() * factor)[valid].sum())
        totals['student_bias_m_sum'] += float((student_delta * factor)[valid].sum())
        totals['teacher_bias_m_sum'] += float((teacher_delta * factor)[valid].sum())
        totals['pixels'] += count
        totals['frames'] += len(batch['path'])
    n = max(1, totals['pixels'])
    return {'frames': totals['frames'], 'measured_pixels': totals['pixels'],
            'student_absrel': totals['student_sum'] / n, 'teacher_absrel': totals['teacher_sum'] / n,
            'student_mae_m': totals['student_mae_m_sum'] / n,
            'teacher_mae_m': totals['teacher_mae_m_sum'] / n,
            'student_signed_bias_m': totals['student_bias_m_sum'] / n,
            'teacher_signed_bias_m': totals['teacher_bias_m_sum'] / n}


def save(path: Path, value):
    temp = path.with_suffix('.tmp')
    if path.suffix == '.pt':
        torch.save(value, temp)
    else:
        temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, default=Path('runs/uavscenes_30_batch'))
    parser.add_argument('--sidecars', type=Path, default=Path('runs/uavscenes_cached_lidar_targets'))
    parser.add_argument('--original', type=Path, default=Path('runs/depth_student_20261028/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/depth_student_lidar'))
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--measured-weight', type=float, default=4.)
    parser.add_argument('--teacher-weight', type=float, default=.5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--channels', type=int, default=48)
    parser.add_argument('--planes', type=int, default=24)
    parser.add_argument('--skip-test', action='store_true', help='Leave final test sealed until model selection')
    parser.add_argument('--extra-direct-root', type=Path,
                        help='Supplement training with separated-scale UAVScenes LiDAR groups')
    parser.add_argument('--extra-direct-targets', type=Path,
                        default=Path('runs/uavscenes_multiflight_measured'))
    parser.add_argument('--direct-max-baseline', type=float, default=.4,
                        help='Maximum source baseline divided by median measured depth')
    parser.add_argument('--direct-sfm-anchors', type=Path,
                        help='Use RGB-triangulated sparse anchors and scale for direct groups')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init-checkpoint', type=Path,
                        help='Update this existing student instead of starting from random weights')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    scales = json.loads((args.sidecars / 'scale_calibration.json').read_text())
    paths, scenes = discover(args.cache_root, args.sidecars, scales)
    if min(map(len, paths.values())) == 0:
        raise RuntimeError(f'Empty measured train/validation/test split: { {k: len(v) for k, v in paths.items()} }')
    save(args.output / 'dataset_manifest.json', {'scenes': scenes,
         'frames': {k: len(v) for k, v in paths.items()}, 'sidecars': str(args.sidecars.resolve()),
         'metric_scale_source': 'camera-centre Sim3; no LiDAR depth fit',
         'augment': args.augment, 'measured_weight': args.measured_weight,
         'teacher_weight': args.teacher_weight, 'learning_rate': args.learning_rate,
         'architecture': {'channels': args.channels, 'planes': args.planes},
         'extra_direct_root': str(args.extra_direct_root) if args.extra_direct_root else None,
         'direct_max_baseline': args.direct_max_baseline,
         'direct_sfm_anchors': str(args.direct_sfm_anchors) if args.direct_sfm_anchors else None})
    datasets = {role: MeasuredTeacherDataset(entries, args.sidecars, scales,
                                            augment=args.augment and role == 'train')
                for role, entries in paths.items()}
    if args.extra_direct_root:
        from .direct_lidar_dataset import DirectLidarDataset
        additional = DirectLidarDataset(args.extra_direct_root, args.extra_direct_targets,
                                        augment=args.augment,
                                        max_normalized_baseline=args.direct_max_baseline,
                                        sparse_anchor_root=args.direct_sfm_anchors)
        datasets['train'] = ConcatDataset((datasets['train'], additional))
        save(args.output / 'direct_data.json', {'groups': len(additional),
             'scenes': ['AMtown01', 'HKairport01'],
             'scale': ('RGB-triangulated sparse anchors' if args.direct_sfm_anchors else
                       'median of disjoint LiDAR anchor pixels; other pixels supervise depth')})
    loaders = {role: DataLoader(datasets[role],
                               batch_size=args.batch_size, shuffle=role == 'train',
                               num_workers=args.workers, pin_memory=True,
                               persistent_workers=args.workers > 0)
               for role in paths}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthStudent(channels=args.channels, planes=args.planes).to(device)
    if args.init_checkpoint:
        if args.init_checkpoint.resolve().parent == args.output.resolve():
            raise SystemExit('--output must be a new directory, not the checkpoint being updated')
        start_state = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(start_state['model'] if 'model' in start_state else start_state)
    original = torch.load(args.original, map_location=device, weights_only=False)
    baseline_model = DepthStudent().to(device)
    baseline_model.load_state_dict(original['model'])
    baseline_roles = ('validation',) if args.skip_test else ('validation', 'test')
    baseline = {role: evaluate(baseline_model, loaders[role], device) for role in baseline_roles}
    save(args.output / 'baseline.json', baseline)
    del baseline_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    latest = args.output / 'latest.pt'
    first_epoch, best, stale, steps = 1, float('inf'), 0, 0
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        first_epoch = state['epoch'] + 1
        best = state['best']
        steps = state['optimizer_steps']
    elif latest.exists():
        raise ValueError('Existing training checkpoint; use --resume')
    started = time.monotonic()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        losses = []
        for index, batch in enumerate(loaders['train'], 1):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            if not batch['measured_valid'].any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                depth, confidence = model(batch['rgb'], batch['K'], batch['P'])
                total, parts = loss_fn(batch, depth.float(), confidence.float(),
                                       args.measured_weight, args.teacher_weight)
            if not torch.isfinite(total):
                print(f'nonfinite loss at epoch {epoch}, step {index}', flush=True)
                continue
            total.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(norm):
                optimizer.zero_grad(set_to_none=True)
                print(f'nonfinite gradient at epoch {epoch}, step {index}', flush=True)
                continue
            optimizer.step()
            steps += 1
            losses.append(float(total.detach()))
            if index % 50 == 0:
                print(json.dumps({'epoch': epoch, 'step': index, 'steps_total': steps,
                                  'loss_recent': float(np.mean(losses[-50:])), 'parts': parts}), flush=True)
        validation = evaluate(model, loaders['validation'], device)
        score = validation['student_absrel']
        if score < best:
            best, stale = score, 0
            save(args.output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch,
                                          'validation': validation, 'optimizer_steps': steps,
                                          'architecture': {'channels': args.channels, 'planes': args.planes}})
        else:
            stale += 1
        save(latest, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch, 'best': best, 'optimizer_steps': steps,
                      'architecture': {'channels': args.channels, 'planes': args.planes}})
        status = {'status': 'training', 'epoch': epoch, 'optimizer_steps': steps,
                  'elapsed_seconds': time.monotonic() - started,
                  'train_loss': float(np.mean(losses)) if losses else None,
                  'validation': validation, 'best_validation_absrel': best,
                  'parameters': sum(p.numel() for p in model.parameters())}
        save(args.output / 'status.json', status)
        print(json.dumps(status), flush=True)
        if stale >= args.patience:
            break
    state = torch.load(args.output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    test = None if args.skip_test else evaluate(model, loaders['test'], device)
    save(args.output / 'status.json', {'status': 'complete', 'epoch': epoch,
         'optimizer_steps': steps, 'elapsed_seconds': time.monotonic() - started,
         'best_validation_absrel': best, 'heldout_test': test,
         'baseline': baseline, 'deployed': False})
    print(json.dumps({'heldout_test': test, 'checkpoint': str(args.output / 'best.pt')}), flush=True)


if __name__ == '__main__':
    main()
