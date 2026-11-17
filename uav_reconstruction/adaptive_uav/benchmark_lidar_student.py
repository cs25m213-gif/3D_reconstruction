"""Matched measured-depth benchmark for DA3, original and LiDAR students."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .depth_student import DepthStudent
from .train_depth_student_lidar import MeasuredTeacherDataset, discover


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def metric_bucket():
    return {'pixels': 0, 'absrel_sum': 0., 'abs_m_sum': 0., 'sq_m_sum': 0., 'bias_m_sum': 0.}


def update(bucket, delta_m, target_m):
    bucket['pixels'] += int(target_m.size)
    bucket['absrel_sum'] += float(np.sum(np.abs(delta_m) / target_m))
    bucket['abs_m_sum'] += float(np.sum(np.abs(delta_m)))
    bucket['sq_m_sum'] += float(np.sum(delta_m ** 2))
    bucket['bias_m_sum'] += float(np.sum(delta_m))


def finish(bucket):
    count = bucket['pixels']
    if not count:
        return {'pixels': 0}
    return {'pixels': count, 'absrel': bucket['absrel_sum'] / count,
            'mae_m': bucket['abs_m_sum'] / count,
            'rmse_m': (bucket['sq_m_sum'] / count) ** .5,
            'signed_bias_m': bucket['bias_m_sum'] / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, default=Path('runs/uavscenes_30_batch'))
    parser.add_argument('--sidecars', type=Path, default=Path('runs/uavscenes_cached_lidar_targets'))
    parser.add_argument('--original', type=Path, default=Path('runs/depth_student_20261028/best.pt'))
    parser.add_argument('--lidar-student', type=Path, default=Path('runs/depth_student_lidar/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/depth_student_lidar/benchmark.json'))
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    args = parser.parse_args()
    before = sha256(args.lidar_student)
    scales = json.loads((args.sidecars / 'scale_calibration.json').read_text())
    paths, scenes = discover(args.cache_root, args.sidecars, scales)
    dataset = MeasuredTeacherDataset(paths[args.split], args.sidecars, scales)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    for name, checkpoint in (('original_student', args.original), ('lidar_student', args.lidar_student)):
        model = DepthStudent().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model'])
        model.eval()
        models[name] = model
    results = {}
    counts = {}
    for scene in scenes[args.split]:
        results[scene] = {name: metric_bucket() for name in ('da3', *models)}
        counts[scene] = 0
    results['all'] = {name: metric_bucket() for name in ('da3', *models)}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            predictions = {'da3': batch['teacher'].numpy()}
            rgb, k, p = (batch[key].to(device, non_blocking=True) for key in ('rgb', 'K', 'P'))
            for name, model in models.items():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    depth, _ = model(rgb, k, p)
                predictions[name] = depth.float().cpu().numpy()
            measured = batch['measured'].numpy()
            valid = batch['measured_valid'].numpy()
            factor = batch['metric_per_normalized'].numpy()
            for i, path in enumerate(batch['path']):
                scene = scales[Path(path).parents[3].name]['scene']
                counts[scene] += 1
                mask = valid[i, 0]
                target_m = measured[i, 0][mask] * factor[i]
                for name, prediction in predictions.items():
                    delta_m = (prediction[i, 0][mask] - measured[i, 0][mask]) * factor[i]
                    update(results[scene][name], delta_m, target_m)
                    update(results['all'][name], delta_m, target_m)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    after = sha256(args.lidar_student)
    if before != after:
        raise RuntimeError('LiDAR student checkpoint changed during benchmarking')
    report = {'split': args.split, 'scenes': scenes[args.split], 'frames': counts,
              'metrics': {scene: {name: finish(value) for name, value in methods.items()}
                          for scene, methods in results.items()},
              'elapsed_seconds_all_three': elapsed,
              'peak_gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None,
              'checkpoint_sha256': before,
              'parameter_count': {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
              'scale_source': 'reference camera-centre alignment, no LiDAR depth fit',
              'caveats': ['LiDAR sidecar projections have not passed dynamic-object and image-edge alignment gates.',
                          'Original student may have seen some test RGB/DA3 caches during earlier distillation.',
                          'This is a depth benchmark, not a held-out video-to-mesh validation.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
