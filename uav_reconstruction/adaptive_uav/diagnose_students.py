"""Matched five-model depth benchmark and real RGB/LiDAR visual gallery."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .depth_student import DepthStudent
from .train_depth_student_lidar import MeasuredTeacherDataset, discover


CHECKPOINTS = {
    'original': Path('runs/depth_student_20261028/best.pt'),
    'v1': Path('runs/depth_student_lidar/best.pt'),
    'v4': Path('runs/depth_student_lidar_v4_more_data/best.pt'),
    'v5': Path('runs/depth_student_lidar_v5_matched_baseline/best.pt'),
}
NAMES = ('DA3', 'original', 'v1', 'v4', 'v5')
RANGES = ((0, 40), (40, 80), (80, 120), (120, 200), (200, 1000))


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(prediction, target):
    difference = prediction - target
    absolute = np.abs(difference)
    return {'pixels': int(len(target)), 'absrel': float(np.mean(absolute / target)),
            'mae_m': float(absolute.mean()), 'rmse_m': float(np.sqrt(np.mean(difference**2))),
            'signed_bias_m': float(difference.mean()),
            'median_abs_m': float(np.median(absolute)),
            'p90_abs_m': float(np.percentile(absolute, 90)),
            'p95_abs_m': float(np.percentile(absolute, 95))}


def picture(rgb_path, predicted, measured, mask, factor, scene, frame, scores, output):
    original = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(cv2.resize(original, (512, 384), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    h, w = mask.shape
    measured_m = measured * factor
    valid_depth = np.where(mask, measured_m, np.nan)
    overlay = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).copy()
    depth_color = plt.get_cmap('viridis')(np.clip((measured_m - 20) / 140, 0, 1))[:, :, :3]
    overlay[mask] = (.55 * overlay[mask] + .45 * depth_color[mask] * 255).astype(np.uint8)
    edge = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 80, 160)
    edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_NEAREST)
    edge_overlay = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).copy()
    edge_overlay[edge > 0] = [255, 255, 0]
    edge_overlay[mask] = [255, 30, 30]
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), constrained_layout=True)
    panels = [('RGB (decoded)', rgb, None), ('Projected LiDAR overlay', overlay, None),
              ('Measured Z (m)', valid_depth, 'depth'), ('Measured mask', mask, 'mask')]
    for name in NAMES:
        panels.append((f'{name} Z (m)', predicted[name] * factor, 'depth'))
    for name in NAMES:
        panels.append((f'{name} |error| (m)', np.where(mask, np.abs(predicted[name]-measured)*factor, np.nan), 'error'))
    panels.extend([('RGB edges + LiDAR pixels', edge_overlay, None),
                   ('Error key', np.zeros((h, w)), 'blank')])
    for ax, (title, data, style) in zip(axes.flat, panels):
        if style == 'depth':
            ax.imshow(data, cmap='viridis', vmin=20, vmax=160)
        elif style == 'error':
            ax.imshow(data, cmap='magma', vmin=0, vmax=20)
        elif style == 'mask':
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
        elif style == 'blank':
            ax.text(.02, .9, 'Depth: 20–160 m\nError: 0–20 m\nBlack/white mask = invalid/valid',
                    va='top', transform=ax.transAxes)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.suptitle(f'{scene} frame {frame} | LiDAR pixels {mask.sum()} | DA3 {scores["DA3"]["absrel"]:.3f} | v1 {scores["v1"]["absrel"]:.3f} | v5 {scores["v5"]["absrel"]:.3f}', fontsize=13)
    fig.savefig(output, dpi=110)
    plt.close(fig)


def run(split, output, gallery=False):
    output.mkdir(parents=True, exist_ok=True)
    sidecars = Path('runs/uavscenes_cached_lidar_targets')
    scales = json.loads((sidecars / 'scale_calibration.json').read_text())
    paths, scene_split = discover(Path('runs/uavscenes_30_batch'), sidecars, scales)
    loader = DataLoader(MeasuredTeacherDataset(paths[split], sidecars, scales),
                        batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, metadata = {}, {}
    for name, path in CHECKPOINTS.items():
        state = torch.load(path, map_location='cpu', weights_only=False)
        architecture = state.get('architecture', {'channels': 48, 'planes': 24})
        model = DepthStudent(**architecture).to(device).eval()
        model.load_state_dict(state['model'])
        models[name] = model
        metadata[name] = {'path': str(path.resolve()), 'sha256': hash_file(path),
                          'best_epoch': state.get('epoch'), 'architecture': architecture,
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'optimizer_saved': 'optimizer' in state}
    per_frame, visual_data, accum = [], {}, {scene: {name: [] for name in NAMES} for scene in (*scene_split[split], 'all')}
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            images = batch['rgb'].to(device)
            intrinsics = batch['K'].to(device)
            poses = batch['P'].to(device)
            predictions = {'DA3': batch['teacher'].numpy()}
            for name, model in models.items():
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    depth, _ = model(images, intrinsics, poses)
                predictions[name] = depth.float().cpu().numpy()
            for i, path in enumerate(batch['path']):
                cache = Path(path)
                batch_name = cache.parents[3].name
                scene = scales[batch_name]['scene']
                frame = cache.name.split('.')[0]
                mask = batch['measured_valid'][i, 0].numpy()
                measured = batch['measured'][i, 0].numpy()
                factor = float(batch['metric_per_normalized'][i])
                target = measured[mask] * factor
                scores = {}
                for name in NAMES:
                    prediction = predictions[name][i, 0]
                    value = prediction[mask] * factor
                    scores[name] = metrics(value, target)
                    accum[scene][name].append((value, target))
                    accum['all'][name].append((value, target))
                row = {'scene': scene, 'batch': batch_name, 'frame': frame, 'cache': path,
                       'measured_pixels': int(mask.sum()), 'metric_per_normalized': factor,
                       'scores': scores}
                per_frame.append(row)
                if gallery:
                    visual_data[(scene, frame, batch_name)] = (
                        {name: predictions[name][i, 0].copy() for name in NAMES},
                        measured.copy(), mask.copy(), factor, scores, cache)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    summary = {}
    for scene, methods in accum.items():
        summary[scene] = {}
        for name, pairs in methods.items():
            prediction = np.concatenate([x for x, _ in pairs])
            target = np.concatenate([y for _, y in pairs])
            summary[scene][name] = metrics(prediction, target)
            summary[scene][name]['depth_ranges'] = {
                f'{lo}-{hi}m': metrics(prediction[(target >= lo) & (target < hi)], target[(target >= lo) & (target < hi)])
                for lo, hi in RANGES if ((target >= lo) & (target < hi)).any()}
    report = {'split': split, 'scenes': scene_split[split], 'frames': len(per_frame),
              'models': metadata, 'summary': summary, 'per_frame': per_frame,
              'seconds': time.monotonic() - started,
              'peak_allocated_gpu_gb': torch.cuda.max_memory_allocated()/1e9 if device.type == 'cuda' else None,
              'same_inputs_and_masks': True,
              'scale_source': 'camera-centre Sim3; no LiDAR depth fit',
              'warning': 'LiDAR image alignment and dynamics remain unresolved; cached DA3 inference time excluded.'}
    (output / f'five_model_{split}.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / f'per_frame_{split}.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scene', 'batch', 'frame', 'pixels', *[f'{name}_{metric}' for name in NAMES for metric in ('absrel', 'mae_m', 'p95_abs_m')]])
        for row in per_frame:
            writer.writerow([row['scene'], row['batch'], row['frame'], row['measured_pixels'],
                             *[row['scores'][name][metric] for name in NAMES for metric in ('absrel', 'mae_m', 'p95_abs_m')]])
    if gallery:
        gallery_dir = output / 'visual_comparisons'
        gallery_dir.mkdir(exist_ok=True)
        originals = gallery_dir / 'original_rgb'
        originals.mkdir(exist_ok=True)
        cards = []
        for scene in scene_split[split]:
            scene_rows = [row for row in per_frame if row['scene'] == scene]
            if split == 'validation':
                ranked = sorted(scene_rows, key=lambda row: row['scores']['v1']['absrel'])
                selected = [ranked[0], ranked[len(ranked)//2], ranked[-1],
                            max(scene_rows, key=lambda row: abs(row['scores']['v1']['absrel']-row['scores']['DA3']['absrel']))]
                selected = list({(row['batch'], row['frame']): row for row in selected}.values())
                for candidate in (ranked[len(ranked)//4], ranked[3*len(ranked)//4], *ranked):
                    if len(selected) >= 4:
                        break
                    if all((candidate['batch'], candidate['frame']) != (row['batch'], row['frame']) for row in selected):
                        selected.append(candidate)
            else:
                ordered = sorted(scene_rows, key=lambda row: (row['batch'], row['frame']))
                selected = [ordered[j] for j in (0, len(ordered)//3, 2*len(ordered)//3, len(ordered)-1)]
            for row in selected:
                key = (scene, row['frame'], row['batch'])
                predicted, measured, mask, factor, scores, cache = visual_data[key]
                frame_num = int(row['frame'].split('_')[-1])
                rgb_path = cache.parents[3] / 'source/frames' / f'frame_{frame_num:06d}.jpg'
                image_name = f'{split}_{scene}_{row["batch"]}_{row["frame"]}.png'
                picture(rgb_path, predicted, measured, mask, factor, scene, row['frame'], scores, gallery_dir / image_name)
                original_name = image_name.replace('.png', '.jpg')
                shutil.copy2(rgb_path, originals / original_name)
                cards.append(f'<figure><img src="visual_comparisons/{html.escape(image_name)}" width="900"><figcaption>{html.escape(scene)} {html.escape(row["batch"])} {html.escape(row["frame"])} — DA3 {scores["DA3"]["absrel"]:.3%}; v1 {scores["v1"]["absrel"]:.3%}; v5 {scores["v5"]["absrel"]:.3%} — <a href="visual_comparisons/original_rgb/{html.escape(original_name)}">original RGB</a></figcaption></figure>')
        (output / f'visual_gallery_{split}.html').write_text('<!doctype html><meta charset="utf-8"><title>Depth student diagnosis</title><style>body{font:16px sans-serif;background:#111;color:#ddd}figure{margin:2em auto;max-width:950px}img{max-width:100%}</style><h1>Real RGB/LiDAR depth comparisons — '+split+'</h1><p>Depth scale 20–160 m; error scale 0–20 m; masks show measured pixels only. RGB source frames decoded from UAVScenes.</p>'+''.join(cards))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output', type=Path, default=Path('runs/depth_student_diagnosis'))
    parser.add_argument('--gallery', action='store_true')
    args = parser.parse_args()
    report = run(args.split, args.output, args.gallery)
    print(json.dumps({'split': report['split'], 'frames': report['frames'], 'seconds': report['seconds'],
                      'summary': {name: report['summary']['all'][name]['absrel'] for name in NAMES}}, indent=2))
