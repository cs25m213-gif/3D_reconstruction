"""CPU forward-call microbenchmark for decisions from the cached cascade experiment.

Executes requested student predictions for real, replays unrequested cached outputs
only to satisfy the offline evaluation loader, and checks freshly predicted depths.
This is NOT a live-video or complete reconstruction runtime measurement.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import counterfactual_replay as cr
from . import reconstruction_selector as rs
from . import test_online_cascade as experiment
from .contribution_labels import load_clip
from .student_inference import load_student


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads((a.work / 'experiment.json').read_text())
    for k in ('work', 'base', 'labels', 'greedy', 'config', 'student'):
        cfg[k] = Path(cfg[k])
    rs.CFG.__dict__.update(cfg)
    rs.CFG.root = Path.cwd()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    validation = json.loads((a.work / 'validation_rows.json').read_text())
    tune = json.loads((a.work / 'tune.json').read_text())
    # Deterministic one-clip-per-flight profiling sample; no selection by performance.
    sample = [next(r for r in validation if r['flight'] == f)
              for f in sorted({r['flight'] for r in validation})]
    contract = rs.load_contract(rs.CFG.config, ('utility.quality_protocol',))
    student = load_student(rs.CFG.student, 'cpu')
    original = cr.predict
    results = []
    try:
        for row in sample:
            name = row['clip']
            chosen = next(r for r in row['rows'] if r['margin'] == tune['margin'])
            modes = {'all_frames': list(range(row['frames'])),
                     'cheap': row['rows'][0]['requested_depth_frames'],
                     'chosen_cascade': chosen['requested_depth_frames'],
                     'online_depth_aware': row['rows'][-1]['requested_depth_frames']}
            cache = rs.CFG.base / 'student_depth' / f'{rs._student_key()}_v2' / f'{name}.npz'
            with np.load(cache) as d:
                cached = [(d[f'd{i}'].copy(), d[f'c{i}'].copy()) for i in range(int(d['n']))]
            for mode, requested in modes.items():
                wanted = set(requested)
                state = {'index': 0, 'actual_forward_calls': 0, 'prediction_seconds': 0., 'max_abs_depth_difference': 0.}

                def count_forward(module, args, output):
                    state['actual_forward_calls'] += 1

                hook = student.register_forward_hook(count_forward)

                def counted(*args, **kwargs):
                    i = state['index']
                    state['index'] += 1
                    expected = cached[i]
                    if i not in wanted:
                        return expected
                    start = time.perf_counter()
                    result = original(*args, **kwargs)
                    state['prediction_seconds'] += time.perf_counter() - start
                    difference = float(np.max(np.abs(result[0] - expected[0])))
                    state['max_abs_depth_difference'] = max(state['max_abs_depth_difference'], difference)
                    if not np.allclose(result[0], expected[0], rtol=1e-4, atol=1e-3):
                        raise RuntimeError(f'{name} frame {i}: cached/fresh depth mismatch {difference}')
                    return result

                cr.predict = counted
                start = time.perf_counter()
                try:
                    load_clip(rs.CFG.root, contract, name, student, 'cpu')
                finally:
                    cr.predict = original
                    hook.remove()
                if state['index'] != row['frames'] or state['actual_forward_calls'] != len(wanted):
                    raise RuntimeError(f'Forward accounting mismatch: {state}')
                result = {'clip': name, 'flight': row['flight'], 'mode': mode,
                          'requested': len(wanted), 'total_frames': row['frames'],
                          'offline_loader_seconds': time.perf_counter() - start, **state}
                results.append(result)
                experiment.save(a.work / 'cpu_forward_profile.json', {
                    'status': 'running', 'scope': __doc__, 'rows': results})
                print(json.dumps(result), flush=True)
    finally:
        cr.predict = original
    experiment.save(a.work / 'cpu_forward_profile.json', {
        'status': 'complete', 'scope': __doc__, 'rows': results,
        'limitations': ['Fixed precomputed RGB source buffers and reference poses.',
                        'One clip per validation flight; cold/warm cache and mode-order effects.',
                        'Prediction time includes source RGB encodings but excludes DINO and tracking.',
                        'Unrequested depth outputs are replayed solely for the offline loader.']})


if __name__ == '__main__':
    main()
