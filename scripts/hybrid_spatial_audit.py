"""Audit learned-depth alignment on spatially held-out measured depth regions."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'uav_reconstruction'))
from adaptive_uav.hybrid_dense import huber_fit, read_array, reference_names


def audit(workspace: Path):
    rows = []
    for name in reference_names(workspace):
        measured_path = workspace / 'stereo/depth_maps' / (name + '.geometric.bin')
        prior_path = workspace / 'prior_depth' / (name + '.npy')
        if not measured_path.is_file() or not prior_path.is_file():
            continue
        measured = read_array(measured_path)[::4, ::4]
        prior = cv2.resize(np.load(prior_path), (measured.shape[1], measured.shape[0]))
        h, w = measured.shape
        yy, xx = np.indices((h, w))
        # Spatial cells are withheld together. This tests extrapolation across
        # regions, rather than a random pixel split of neighboring observations.
        cells = (yy * 8 // h) * 8 + (xx * 8 // w)
        valid = np.isfinite(measured) & (measured > 0) & np.isfinite(prior) & (prior > 0)
        held = valid & ((cells * 17 + 3) % 4 == 0)
        train = valid & ~held
        if min(int(held.sum()), int(train.sum())) < 400:
            rows.append({'frame': name, 'status': 'insufficient_spatial_anchors'})
            continue
        d, m = prior[train].astype(float), measured[train].astype(float)
        rng = np.random.default_rng(0)
        if len(d) > 30000:
            indices = rng.choice(len(d), 30000, replace=False)
            d, m = d[indices], m[indices]
        scale, shift = huber_fit(d, m)
        error = np.abs(scale * prior[held] + shift - measured[held]) / measured[held]
        rows.append({'frame': name, 'status': 'ok' if scale > 0 else 'invalid_scale',
                     'anchors_train': int(train.sum()), 'anchors_heldout': int(held.sum()),
                     'median_relative_error': float(np.median(error)),
                     'p90_relative_error': float(np.percentile(error, 90))})
    good = [row for row in rows if row['status'] == 'ok']
    result = {'method': '8x8 spatial-cell holdout on measured pixels, 1/4 resolution',
              'scope': 'Tests prior alignment only where measured depth exists; cannot validate blank surfaces.',
              'frames': len(rows), 'frames_valid': len(good),
              'median_frame_median_relative_error': float(np.median([r['median_relative_error'] for r in good])) if good else None,
              'median_frame_p90_relative_error': float(np.median([r['p90_relative_error'] for r in good])) if good else None,
              'frames_p90_over_10_percent': sum(r['p90_relative_error'] > .1 for r in good),
              'per_frame': rows}
    (workspace / 'spatial_alignment_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    print(json.dumps({k: v for k, v in audit(parser.parse_args().workspace).items() if k != 'per_frame'}, indent=2))
