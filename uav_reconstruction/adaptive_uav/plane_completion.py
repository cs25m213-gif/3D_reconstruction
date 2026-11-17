"""Infer bounded planar patches from geometric depth and independent view checks.

These patches are model-based inference, not additional stereo measurements.
The measured mesh is preserved and inferred faces are exported separately.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
import trimesh


def read_depth(path):
    with Path(path).open('rb') as stream:
        header = b''
        while header.count(b'&') < 3:
            byte = stream.read(1)
            if not byte:
                raise ValueError('Truncated COLMAP depth header')
            header += byte
        width, height, channels = map(int, header[:-1].split(b'&'))
        values = np.frombuffer(stream.read(), dtype=np.float32)
    return values.reshape((width, height, channels), order='F').transpose(1, 0, 2)[:, :, 0]


def fit_plane(points, tolerance, seed=0, min_inlier_fraction=.75):
    """Robust plane fit with spread and residual gates; reject curved surfaces."""
    points = np.asarray(points)
    if len(points) < 80:
        return None
    rng = np.random.default_rng(seed)
    sampled = points[rng.choice(len(points), min(1200, len(points)), replace=False)]
    best = np.zeros(len(sampled), dtype=bool)
    for _ in range(160):
        a, b, c = sampled[rng.choice(len(sampled), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        keep = np.abs((sampled - a) @ normal) < tolerance
        if keep.sum() > best.sum():
            best = keep
    if best.mean() < min_inlier_fraction:
        return None
    center = sampled[best].mean(0)
    _, singular, axes = np.linalg.svd(sampled[best] - center, full_matrices=False)
    if singular[1] < .05 * singular[0]:
        return None
    normal = axes[-1]
    offset = -normal @ center
    residual = np.abs(points @ normal + offset)
    inliers = residual < tolerance
    if inliers.mean() < min_inlier_fraction:
        return None
    return normal, offset, inliers


def ray_plane(rays, center, normal, offset):
    denominator = rays @ normal
    distance = np.full(len(rays), np.nan)
    np.divide(-(normal @ center + offset), denominator, out=distance,
              where=np.abs(denominator) > 1e-8)
    return center + distance[:, None] * rays, distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--three-assets', type=Path, required=True)
    p.add_argument('--max-views', type=int, default=48)
    args = p.parse_args()
    started = time.monotonic()
    work, output = args.workspace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = pycolmap.Reconstruction(work / 'sparse')
    images = {i.name: i for i in model.images.values()}
    plan = json.loads((work / 'dense_plan.json').read_text())['plan']
    chosen = np.unique(np.linspace(0, len(plan) - 1, min(args.max_views, len(plan)), dtype=int))
    cloud = trimesh.load(work / 'scene_dense.ply', process=False)
    cache = {}
    def load(name):
        if name not in cache:
            im = images[name]
            depth = read_depth(work / 'stereo/depth_maps' / (name + '.geometric.bin'))
            rgb = cv2.imread(str(work / 'images' / name))
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mean = cv2.boxFilter(gray, -1, (21, 21))
            std = np.sqrt(np.maximum(cv2.boxFilter(gray * gray, -1, (21, 21)) - mean * mean, 0))
            camera = model.cameras[im.camera_id]
            k = camera.calibration_matrix().copy()
            k[0] *= depth.shape[1] / camera.width
            k[1] *= depth.shape[0] / camera.height
            pose = im.cam_from_world().matrix()
            camera_points = cloud.vertices @ pose[:, :3].T + pose[:, 3]
            uv = camera_points @ k.T
            uv = np.divide(uv[:, :2], uv[:, 2:3], out=np.full((len(uv), 2), -1.), where=uv[:, 2:3] > 0)
            uv = np.rint(uv - .5).astype(int)
            inside = (camera_points[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
            fused_depth = np.full(depth.size, np.inf)
            np.minimum.at(fused_depth, uv[inside, 1] * depth.shape[1] + uv[inside, 0], camera_points[inside, 2])
            fused_depth = fused_depth.reshape(depth.shape)
            cache[name] = {'depth': depth, 'image': rgb,
                           'lab': cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32),
                           'flat': std < 4, 'fused_depth': fused_depth,
                           'K': k, 'R': pose[:, :3], 't': pose[:, 3]}
        return cache[name]
    vertices, faces, colors = [], [], []
    accepted, rejected = [], []
    previous_points = []
    count = 0
    for index in chosen:
        row = plan[int(index)]
        name = row['reference']; data = load(name)
        n, labels, regions, _ = cv2.connectedComponentsWithStats(data['flat'].astype(np.uint8), 8)
        candidates = sorted(range(1, n), key=lambda j: regions[j, cv2.CC_STAT_AREA], reverse=True)[:8]
        for label in candidates:
            area = int(regions[label, cv2.CC_STAT_AREA])
            if area < 3000:
                continue
            region = labels == label
            # Reliable stereo seeds often sit along the textured boundary of a
            # blank wall. Include that narrow ring, using only fused samples.
            seed_region = cv2.dilate(region.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
            yy, xx = np.nonzero(seed_region & np.isfinite(data['fused_depth']))
            if len(xx) < 80:
                rejected.append({'frame': name, 'reason': 'insufficient reliable depth seeds', 'area': area})
                continue
            z = data['fused_depth'][yy, xx]
            rays = np.column_stack([xx + .5, yy + .5, np.ones(len(xx))]) @ np.linalg.inv(data['K']).T
            points = (rays * z[:, None] - data['t']) @ data['R']
            tolerance = .005 * float(np.median(z))
            # A connected flat-color region may contain several wall planes.
            # Fit the dominant cluster and restrict inference to its seed hull.
            result = fit_plane(points, tolerance, min_inlier_fraction=.35)
            if result is None:
                rejected.append({'frame': name, 'reason': 'depth seeds not a stable plane', 'area': area})
                continue
            normal, offset, inliers = result
            # Seed support must extend across the region, not just a tiny corner.
            hull = cv2.convexHull(np.column_stack([xx[inliers], yy[inliers]]).astype(np.float32))
            if cv2.contourArea(hull) < .1 * area:
                rejected.append({'frame': name, 'reason': 'plane seeds cover too little of region', 'area': area})
                continue
            hull_mask = np.zeros(region.shape, dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, np.rint(hull[:, 0]).astype(np.int32), 1)
            step = 4
            gy, gx = np.mgrid[2:region.shape[0]:step, 2:region.shape[1]:step]
            eligible = (region[gy, gx] & (hull_mask[gy, gx] > 0)).ravel()
            rays = np.column_stack([gx.ravel() + .5, gy.ravel() + .5, np.ones(gx.size)]) @ np.linalg.inv(data['K']).T @ data['R']
            center = -data['R'].T @ data['t']
            xyz, distance = ray_plane(rays, center, normal, offset)
            valid = eligible & np.isfinite(distance) & (distance > 0)
            # Extrapolation is bounded relative to the observed seed depths.
            valid &= (distance > np.percentile(z[inliers], 2) * .5) & (distance < np.percentile(z[inliers], 98) * 2)
            indices = np.flatnonzero(valid)
            if len(indices) < 80:
                continue
            pixel_votes = np.zeros(gx.size, dtype=np.uint8)
            confirming = []
            ref_lab = data['lab'][gy.ravel()[indices], gx.ravel()[indices]]
            for source in row['sources']:
                other = load(source['name'])
                camxyz = xyz[indices] @ other['R'].T + other['t']
                uv = camxyz @ other['K'].T
                uv = np.divide(uv[:, :2], uv[:, 2:3], out=np.full((len(uv), 2), -1.), where=uv[:, 2:3] > 0)
                uv = np.rint(uv - .5).astype(int)
                inside = (camxyz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < other['depth'].shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < other['depth'].shape[0])
                j = np.flatnonzero(inside)
                if not len(j):
                    continue
                ox, oy = uv[j].T
                observed = other['depth'][oy, ox]
                measured = observed > 0
                compatible = np.abs(observed - camxyz[j, 2]) <= .02 * camxyz[j, 2]
                # Independent geometric depths must confirm the plane before
                # color agreement can extend it through textureless pixels.
                seed_checks = measured & other['flat'][oy, ox]
                if seed_checks.sum() < 30 or compatible[seed_checks].mean() < .8:
                    continue
                color_ok = np.linalg.norm(other['lab'][oy, ox] - ref_lab[j], axis=1) < 15
                agrees = color_ok & (~measured | compatible) & other['flat'][oy, ox]
                pixel_votes[indices[j[agrees]]] += 1
                confirming.append(source['name'])
            keep = valid & (pixel_votes >= 2)
            if len(confirming) < 2 or keep.sum() < 100:
                rejected.append({'frame': name, 'reason': 'plane lacks two independent confirming views', 'area': area})
                continue
            if previous_points:
                nearest = cKDTree(np.vstack(previous_points)).query(xyz[keep])[0]
                if np.mean(nearest < tolerance * 2) > .9:
                    continue
            grid_ids = np.arange(gx.size).reshape(gx.shape)
            quads = np.stack([grid_ids[:-1, :-1], grid_ids[:-1, 1:], grid_ids[1:, :-1], grid_ids[1:, 1:]], axis=-1).reshape(-1, 4)
            quads = quads[keep[quads].all(1)]
            if not len(quads):
                continue
            used = np.unique(quads)
            remap = np.full(gx.size, -1, dtype=int); remap[used] = np.arange(len(used)) + count
            patch_faces = np.vstack([quads[:, [0, 1, 2]], quads[:, [1, 3, 2]]])
            vertices.append(xyz[used]); faces.append(remap[patch_faces])
            colors.append(cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB)[gy.ravel()[used], gx.ravel()[used]])
            previous_points.append(xyz[used][::8]); count += len(used)
            accepted.append({'frame': name, 'region_pixels': area, 'seed_points': len(points),
                             'seed_inlier_fraction': float(inliers.mean()), 'confirming_views': confirming,
                             'vertices': len(used), 'faces': len(patch_faces),
                             'plane_normal': normal.tolist(), 'plane_offset': float(offset),
                             'seed_tolerance_scene_units': tolerance})
        print(f'[{name}] accepted planes: {len(accepted)}', flush=True)
    report = {'method': 'bounded planar inference from geometric seeds and two independently depth-confirming, color-compatible views',
              'processed_reference_views': len(chosen), 'accepted_planes': accepted,
              'rejected_regions': rejected, 'elapsed_seconds': time.monotonic() - started,
              'geometry_accuracy_validated': False, 'inferred_surface': True}
    (output / 'plane_completion.json').write_text(json.dumps(report, indent=2) + '\n')
    if not vertices:
        print('No plane passed the support gates; measured mesh unchanged', flush=True)
        return
    inferred = trimesh.Trimesh(vertices=np.vstack(vertices), faces=np.vstack(faces),
                               vertex_colors=np.vstack(colors), process=False)
    inferred.export(output / 'inferred_planes.ply')
    measured = trimesh.load(work / 'scene_mesh_colored.ply', process=False)
    combined = trimesh.util.concatenate([measured, inferred])
    combined.export(output / 'scene_mesh.ply')
    combined.export(output / 'scene_mesh_colored.ply')
    scene = trimesh.Scene()
    scene.add_geometry(measured, node_name='measured_stereo_surface')
    scene.add_geometry(inferred, node_name='inferred_planar_surface')
    scene.export(output / 'scene_colored.glb')
    metrics = json.loads((work / 'dense_metrics.json').read_text())
    metrics.update(source_model=str(work / 'sparse'), status='measured_plus_inferred_mesh',
                   measured_faces=len(measured.faces), inferred_faces=len(inferred.faces),
                   mesh_faces=len(combined.faces), mesh_vertices=len(combined.vertices),
                   quality_warning='Includes inferred planar patches confirmed by neighboring depth seeds. Added surfaces are not new stereo measurements; geometric accuracy is unverified.')
    (output / 'dense_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    from .mesh_viewer import write_mesh_viewer
    print(write_mesh_viewer(output, args.three_assets), flush=True)


if __name__ == '__main__':
    main()
