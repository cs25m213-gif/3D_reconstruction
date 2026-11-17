"""Measure supervision masks and actual gradients for cache/direct batches."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .depth_student import DepthStudent
from .direct_lidar_dataset import DirectLidarDataset
from .train_depth_student_lidar import MeasuredTeacherDataset, discover, loss_fn


def run(output=Path('runs/depth_student_diagnosis/effective_loss_counts.json')):
    root = Path('runs/uavscenes_cached_lidar_targets')
    scales = json.loads((root / 'scale_calibration.json').read_text())
    paths, _ = discover(Path('runs/uavscenes_30_batch'), root, scales)
    cached = MeasuredTeacherDataset(paths['train'][:2], root, scales)
    direct = DirectLidarDataset(Path('runs/uavscenes_multiflight'), Path('runs/uavscenes_multiflight_measured'),
                                max_normalized_baseline=.16, augment=False)
    samples = {'cached_only': [cached[0], cached[1]],
               'direct_only': [direct[0], direct[1]],
               'mixed': [cached[0], direct[0]]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthStudent().to(device)
    state = torch.load('runs/depth_student_lidar/best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    results = {}
    for name, items in samples.items():
        batch = {}
        for key in items[0]:
            values = [row[key] for row in items]
            if torch.is_tensor(values[0]):
                batch[key] = torch.stack(values).to(device)
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values, device=device)
            else:
                batch[key] = values
        measured = batch['measured_valid']
        teacher = batch['teacher_valid'] & ~measured
        anchors = (batch['anchor'] > .25) & (batch['anchor'] < 4) & ~measured
        model.zero_grad(set_to_none=True)
        depth, confidence = model(batch['rgb'], batch['K'], batch['P'])
        total, parts = loss_fn(batch, depth.float(), confidence.float(), 2, 1)
        total.backward()
        gradients = {module: float(sum(p.grad.norm().item() for p in getattr(model, module).parameters() if p.grad is not None))
                     for module in ('encoder', 'volume', 'refine')}
        results[name] = {'measured_pixels': int(measured.sum()), 'teacher_pixels': int(teacher.sum()),
                         'anchor_pixels': int(anchors.sum()), 'loss': float(total.detach()),
                         'parts': parts, 'gradient_l2_sum_by_module': gradients}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n')
    return results


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
