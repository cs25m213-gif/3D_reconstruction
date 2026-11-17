"""Fit visual-to-metric scale from camera centres, never from held-out LiDAR."""
import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import weighted_umeyama


def calibrate(root: Path, output: Path) -> dict:
    rows = {}
    for batch in sorted(p for p in root.iterdir() if (p / 'selected_infos.json').exists()):
        infos = json.loads((batch / 'selected_infos.json').read_text())
        cache = batch / 'reconstruction/multiview/depth_cache'
        visual, metric, indices = [], [], []
        for file in sorted(cache.glob('frame_*.jpg.npz')):
            index = int(file.name.split('_')[1].split('.')[0])
            with np.load(file) as data:
                pose = data['P'].astype(np.float64)
            if not np.isfinite(pose).all():
                continue
            visual.append(-pose[:3, :3].T @ pose[:3, 3])
            metric.append(np.asarray(infos[index]['T4x4'])[:3, 3])
            indices.append(index)
        if len(visual) < 8:
            rows[batch.name] = {'usable': False, 'reason': 'fewer than eight posed caches'}
            continue
        visual, metric = np.asarray(visual), np.asarray(metric)
        scale, rotation, translation = weighted_umeyama(visual[::2], metric[::2])
        errors = np.linalg.norm(scale * (rotation @ visual[1::2].T).T + translation - metric[1::2], axis=1)
        rows[batch.name] = {'usable': bool(np.isfinite(scale) and 0 < scale < 1000 and np.median(errors) < 2.0),
                            'scene': infos[0]['OriginalSceneName'], 'posed_frames': len(visual),
                            'calibration_frames': len(visual[::2]), 'heldout_frames': len(errors),
                            'scale_m_per_visual_unit': float(scale),
                            'heldout_position_median_m': float(np.median(errors)),
                            'heldout_position_p90_m': float(np.percentile(errors, 90))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + '\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.cache_root, args.output)
    print(json.dumps({'usable': sum(r['usable'] for r in result.values()), 'batches': len(result),
                      'scales': {k: v.get('scale_m_per_visual_unit') for k, v in result.items() if v['usable']}}, indent=2))
