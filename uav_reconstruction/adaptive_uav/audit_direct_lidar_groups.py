"""Check geometric overlap of added UAVScenes training groups."""
import argparse
import json
from pathlib import Path

import numpy as np

from .direct_lidar_dataset import DirectLidarDataset


def audit(source: Path, targets: Path, output: Path, max_baseline: float = .4) -> dict:
    dataset = DirectLidarDataset(source, targets, augment=False,
                                 max_normalized_baseline=max_baseline)
    rows = []
    for scene, reference, sources, _ in dataset.groups:
        info = dataset.manifests[scene]
        with np.load(targets / scene / f'frame_{reference:06d}.npz') as data:
            depth = data['measured_depth_z_m']
        yy, xx = np.nonzero(depth > 0)
        if len(yy) > 1000:
            positions = np.linspace(0, len(yy)-1, 1000).astype(int)
            yy, xx = yy[positions], xx[positions]
        z = depth[yy, xx]
        k0 = np.asarray(info[reference]['info']['P3x3'])
        xyz = np.stack(((xx-k0[0, 2])*z/k0[0, 0], (yy-k0[1, 2])*z/k0[1, 1], z), 1)
        ref = np.asarray(info[reference]['info']['T4x4'])
        world = xyz @ ref[:3, :3].T + ref[:3, 3]
        for source_row in sources:
            src = np.asarray(source_row['info']['T4x4'])
            camera = (world - src[:3, 3]) @ src[:3, :3]
            k = np.asarray(source_row['info']['P3x3'])
            zz = camera[:, 2]
            u = k[0, 0]*camera[:, 0]/np.maximum(zz, 1e-6)+k[0, 2]
            v = k[1, 1]*camera[:, 1]/np.maximum(zz, 1e-6)+k[1, 2]
            width, height = source_row['info']['Width'], source_row['info']['Height']
            good = (zz > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            baseline = np.linalg.norm(src[:3, 3] - ref[:3, 3])
            rows.append({'scene': scene, 'reference': reference,
                         'source': source_row['info']['OriginalImageName'],
                         'baseline_m': float(baseline), 'in_source_fraction': float(good.mean()),
                         'sampled_points': len(z)})
    by_scene = {}
    for scene in sorted({r['scene'] for r in rows}):
        subset = [r for r in rows if r['scene'] == scene]
        overlap = np.array([r['in_source_fraction'] for r in subset])
        by_scene[scene] = {'groups': len(subset)//2, 'median_overlap': float(np.median(overlap)),
                           'p10_overlap': float(np.percentile(overlap, 10)),
                           'under_50_percent': int((overlap < .5).sum()),
                           'median_baseline_m': float(np.median([r['baseline_m'] for r in subset]))}
    result = {'groups': len(dataset), 'by_scene': by_scene, 'rows': rows,
              'note': 'Frustum overlap only; does not prove unobstructed shared surfaces or accurate RGB alignment.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('runs/uavscenes_multiflight'))
    parser.add_argument('--targets', type=Path, default=Path('runs/uavscenes_multiflight_measured'))
    parser.add_argument('--output', type=Path, default=Path('runs/uavscenes_multiflight_measured/group_audit.json'))
    parser.add_argument('--max-baseline', type=float, default=.4)
    args = parser.parse_args()
    result = audit(args.source, args.targets, args.output, args.max_baseline)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
