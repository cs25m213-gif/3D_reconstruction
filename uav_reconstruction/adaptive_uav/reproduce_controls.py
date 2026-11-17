"""Stage 2: recompute the recorded DA3 and v2 validation numbers from frozen inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .depth_student import DepthStudent
from .train_depth_student_lidar import MeasuredTeacherDataset, discover, evaluate
from .utility_contracts import load_contract, write_json, write_status

RECORDED = {'teacher_absrel': .05351, 'student_absrel': .05960, 'teacher_mae_m': 4.450, 'student_mae_m': 4.604}
TOLERANCE = {'teacher_absrel': .0002, 'student_absrel': .0002, 'teacher_mae_m': .02, 'student_mae_m': .02}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, help='Partial smoke run; a partial run cannot pass the gate')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root, data = args.root.resolve(), contract['data']
    scales = json.loads((root / data['scale_calibration']).read_text())
    paths, _ = discover(root / data['cache_root'], root / data['sidecars'], scales)
    if args.dry_run:
        print(json.dumps({role: len(rows) for role, rows in paths.items()}))
        return
    files = paths['validation'][:args.limit] if args.limit else paths['validation']
    loader = torch.utils.data.DataLoader(MeasuredTeacherDataset(files, root / data['sidecars'], scales), batch_size=2, num_workers=args.workers)
    state = torch.load(root / contract['depth_checkpoint'], map_location='cpu', weights_only=False)
    model = DepthStudent(48, 24)
    model.load_state_dict(state['model'] if 'model' in state else state)
    device = torch.device(args.device)
    measured = evaluate(model.to(device), loader, device)
    differences = {k: measured[k] - RECORDED[k] for k in RECORDED}
    comparable = device.type == 'cuda'  # Recorded values used BF16 autocast, which the trainer enables on CUDA only.
    passed = comparable and not args.limit and all(abs(differences[k]) <= TOLERANCE[k] for k in RECORDED)
    report = {'scope': 'cached teacher and v2 on the recorded validation frames', 'measured': measured, 'recorded': RECORDED,
              'differences': differences, 'tolerance': TOLERANCE, 'partial': bool(args.limit), 'precision': 'bf16 autocast' if comparable else 'float32',
              'status': 'PASS' if passed else 'FAIL' if comparable and not args.limit else 'NOT_COMPARABLE'}
    write_json(args.output / 'baseline_metrics.json', report)
    write_status(args.output / 'stage2', 'stage2_controls', 'complete', report['status'], 'stage3_memory' if passed else None)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report['status'] == 'FAIL' else 0)


if __name__ == '__main__':
    main()
