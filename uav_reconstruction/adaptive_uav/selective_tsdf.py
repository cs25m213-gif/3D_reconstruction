"""CPU local TSDF integration of uncertain regions from OpenMVS depth maps.

Exports independent refinement candidates. They are not merged into the
validated textured mesh automatically: seam/quality validation is required.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from .open3d_cpu import cpu_bindings


def uncertain_blocks(points, voxel_size, threshold=.65, max_blocks=3):
    buckets = {}
    for p in points:
        key = tuple(np.floor(np.asarray(p['xyz']) / voxel_size).astype(int))
        bucket = buckets.setdefault(key, [])
        bucket.append(p['uncertainty'])
    ranked = [(float(np.mean(values)), key, len(values)) for key, values in buckets.items()
              if len(values) >= 3 and np.mean(values) >= threshold]
    ranked.sort(reverse=True)
    return ranked[:max_blocks]


def refine(reconstruction, utilities, max_blocks=3, threshold=.65):
    o3d = cpu_bindings()
    start = time.monotonic()
    spec = importlib.util.spec_from_file_location('openmvs_utils', utilities)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    uncertain = json.loads((reconstruction / 'uncertainty_map.json').read_text())
    xyz = np.asarray([p['xyz'] for p in uncertain])
    # Quantiles limit sensitivity to isolated sparse outliers. Units remain SfM units.
    span = np.quantile(xyz, .95, axis=0) - np.quantile(xyz, .05, axis=0)
    block_size = max(float(np.linalg.norm(span)) / 8, 1e-6)
    blocks = uncertain_blocks(uncertain, block_size, threshold, max_blocks)
    output = reconstruction / 'selective_tsdf'
    output.mkdir(exist_ok=True)
    rows = []
    paths = sorted((reconstruction / 'dense').glob('*.dmap'))
    if not paths:
        raise RuntimeError('No OpenMVS depth maps available')
    for uncertainty, key, support in blocks:
        lower = np.asarray(key) * block_size - .1 * block_size
        upper = lower + 1.2 * block_size
        voxel_length = block_size / 64
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length, sdf_trunc=4 * voxel_length,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        integrations, kept_pixels = 0, 0
        for path in paths:
            data = utils.loadDMAP(str(path))
            depth = np.asarray(data['depth_map'])
            h, w = depth.shape
            yy, xx = np.indices((h, w))
            k = data['depth_K']
            camera = np.stack(((xx - k[0,2]) * depth / k[0,0],
                               (yy - k[1,2]) * depth / k[1,1], depth), axis=-1)
            world = camera @ data['R'] + data['C']
            mask = (depth > 0) & np.isfinite(depth) & np.all((world >= lower) & (world <= upper), axis=-1)
            if data['has_conf']:
                conf = data['confidence_map']
                valid_conf = conf[(depth > 0) & np.isfinite(conf)]
                if valid_conf.size:
                    # Confidence scale belongs to OpenMVS; use a per-view quantile.
                    mask &= conf >= np.quantile(valid_conf, .25)
            if mask.sum() < 32:
                continue
            masked = np.where(mask, depth, 0).astype(np.float32)
            color = o3d.geometry.Image(np.zeros((h,w,3), dtype=np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, o3d.geometry.Image(masked),
                depth_scale=1., depth_trunc=float(depth[mask].max() * 1.01), convert_rgb_to_intensity=False)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w,h,k[0,0],k[1,1],k[0,2],k[1,2])
            extrinsic = np.eye(4)
            extrinsic[:3,:3] = data['R']
            extrinsic[:3,3] = -data['R'] @ data['C']
            volume.integrate(rgbd, intrinsic, extrinsic)
            integrations += 1
            kept_pixels += int(mask.sum())
        mesh = volume.extract_triangle_mesh()
        bbox = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
        mesh = mesh.crop(bbox)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.compute_vertex_normals()
        name = f"block_{len(rows):03d}.ply"
        if len(mesh.triangles):
            o3d.io.write_triangle_mesh(str(output / name), mesh)
        rows.append({'key': list(map(int,key)), 'uncertainty': uncertainty, 'sparse_support': support,
                     'integrated_views': integrations, 'processed_pixels': kept_pixels,
                     'triangles': len(mesh.triangles), 'voxel_size_scene_units': voxel_length,
                     'file': name if len(mesh.triangles) else None})
    report = {'status': 'candidate_patches_only', 'blocks': rows, 'seconds': time.monotonic()-start,
              'threshold': threshold, 'global_mesh_modified': False}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reconstruction', type=Path)
    p.add_argument('--utilities', type=Path, default=Path('apps/openMVS-source/scripts/python/MvsUtils.py'))
    p.add_argument('--max-blocks', type=int, default=3)
    p.add_argument('--threshold', type=float, default=.65)
    args = p.parse_args()
    refine(args.reconstruction.resolve(), args.utilities.resolve(), args.max_blocks, args.threshold)


if __name__ == '__main__':
    main()
