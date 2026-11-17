"""Cheap, separately labelled planar hypotheses for textureless walls and ceilings.

This module never changes COLMAP depth, the fused cloud, or the measured mesh.
Only planes anchored by stereo depth and checked in another view are exported.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .depth_normals import read_array


def smooth_neutral_regions(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (11, 11))
    variance = cv2.boxFilter(gray * gray, -1, (11, 11)) - mean * mean
    mask = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 95) &
            (variance < 12**2)).astype(np.uint8)
    # Strong color/structure edges partition adjoining walls and foreground.
    edges = cv2.dilate(cv2.Canny(image, 35, 80), np.ones((3, 3), np.uint8))
    mask[edges > 0] = 0
    return mask


def fit_plane(points, seed=0):
    """RANSAC on a bounded sample, followed by a held-out depth check."""
    points = np.asarray(points, np.float64)
    if len(points) < 400:
        return None
    rng = np.random.default_rng(seed)
    samples = points[rng.choice(len(points), min(3500, len(points)), replace=False)]
    order = rng.permutation(len(samples))
    training = samples[order[:int(.8 * len(samples))]]
    heldout = samples[order[int(.8 * len(samples)):]]
    tolerance = .018 * float(np.median(training[:, 2]))
    best = np.zeros(len(training), bool)
    for _ in range(160):
        a, b, c = training[rng.choice(len(training), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        inliers = np.abs((training-a) @ normal) < tolerance
        if inliers.sum() > best.sum():
            best = inliers
    if best.mean() < .70:
        return None
    center = training[best].mean(0)
    _, _, vh = np.linalg.svd(training[best]-center, full_matrices=False)
    normal = vh[-1]
    offset = -normal @ center
    if normal[2] < 0:
        normal, offset = -normal, -offset
    residual = np.abs(heldout @ normal + offset)
    if np.median(residual) > tolerance or np.mean(residual < 2*tolerance) < .8:
        return None
    return {'normal': normal, 'offset': float(offset),
            'inlier_fraction': float(best.mean()),
            'heldout_fraction': float(np.mean(residual < 2*tolerance)),
            'tolerance': tolerance, 'median_anchor_depth': float(np.median(training[:, 2]))}


def camera_matrix(camera, shape):
    matrix = camera.calibration_matrix().copy()
    matrix[0] *= shape[1] / camera.width
    matrix[1] *= shape[0] / camera.height
    return matrix


def plane_depth(plane, matrix, x, y):
    normal = plane['normal']
    ray_x = (x-matrix[0, 2]) / matrix[0, 0]
    ray_y = (y-matrix[1, 2]) / matrix[1, 1]
    denominator = normal[0]*ray_x + normal[1]*ray_y + normal[2]
    return np.divide(-plane['offset'], denominator,
                     out=np.full_like(ray_x, np.nan, dtype=np.float64),
                     where=np.abs(denominator) > 1e-4)


def cross_view_check(world_points, reference_name, model, names, workspace):
    """Require other measured depths to agree; absent depth gives no evidence."""
    image_by_name = {image.name: image for image in model.images.values()}
    reference_id = int(reference_name[6:12])
    nearby = sorted((name for name in names if name != reference_name),
                    key=lambda name: abs(int(name[6:12])-reference_id))[:4]
    checks, agrees, occludes = 0, 0, 0
    for name in nearby:
        image = image_by_name[name]
        depth_path = workspace / 'stereo/depth_maps' / (name+'.geometric.bin')
        if not depth_path.exists():
            continue
        depth = read_array(depth_path)
        matrix = camera_matrix(model.cameras[image.camera_id], depth.shape)
        pose = image.cam_from_world().matrix()
        points = world_points @ pose[:, :3].T + pose[:, 3]
        positive = points[:, 2] > 0
        uv = points @ matrix.T
        x = np.rint(uv[:, 0]/np.maximum(uv[:, 2], 1e-10)).astype(int)
        y = np.rint(uv[:, 1]/np.maximum(uv[:, 2], 1e-10)).astype(int)
        visible = positive & (x >= 0) & (y >= 0) & (x < depth.shape[1]) & (y < depth.shape[0])
        if not visible.any():
            continue
        indices = np.flatnonzero(visible)
        measured = depth[y[indices], x[indices]]
        observed = measured > 0
        indices, measured = indices[observed], measured[observed]
        if len(indices) == 0:
            continue
        error = (measured-points[indices, 2])/np.maximum(measured, 1e-6)
        checks += len(error)
        agrees += int((np.abs(error) < .035).sum())
        occludes += int((error < -.035).sum())
    return {'checked': checks, 'agreed': agrees, 'occluded': occludes,
            'agreement_fraction': agrees/max(checks, 1)}


def candidate(workspace, model, image, names, component, labels, image_rgb, depth, seed):
    region = labels == component
    anchor = region & (depth > 0) & np.isfinite(depth)
    y, x = np.where(anchor)
    if len(x) < 400:
        return None
    matrix = camera_matrix(model.cameras[image.camera_id], depth.shape)
    z = depth[y, x]
    points = np.column_stack(((x-matrix[0, 2])/matrix[0, 0]*z,
                              (y-matrix[1, 2])/matrix[1, 1]*z, z))
    plane = fit_plane(points, seed)
    if plane is None:
        return None
    # Do not extend through nearby measured geometry incompatible with the plane.
    predicted_anchor = plane_depth(plane, matrix, x, y)
    outlier = np.zeros(region.shape, np.uint8)
    outlier[y, x] = (np.abs(predicted_anchor-z) > .035*z).astype(np.uint8)
    excluded = cv2.dilate(outlier, np.ones((17, 17), np.uint8)) > 0
    distance = cv2.distanceTransform((~anchor).astype(np.uint8), cv2.DIST_L2, 5)
    missing = region & ~anchor & ~excluded & (distance <= 110)
    if missing.sum() < 1200:
        return None
    # Cross-view checks sample the proposed missing region, never the fitted anchors.
    yy, xx = np.where(missing)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(xx), min(3000, len(xx)), replace=False)
    xx, yy = xx[take], yy[take]
    zz = plane_depth(plane, matrix, xx, yy)
    plausible = np.isfinite(zz) & (zz > .5*plane['median_anchor_depth']) & (zz < 2*plane['median_anchor_depth'])
    if plausible.sum() < 500:
        return None
    camera_points = np.column_stack(((xx[plausible]-matrix[0, 2])/matrix[0, 0]*zz[plausible],
                                     (yy[plausible]-matrix[1, 2])/matrix[1, 1]*zz[plausible], zz[plausible]))
    pose = image.cam_from_world().inverse().matrix()
    world_points = camera_points @ pose[:, :3].T + pose[:, 3]
    check = cross_view_check(world_points, image.name, model, names, workspace)
    if check['checked'] < 40 or check['agreement_fraction'] < .75 or check['occluded']/check['checked'] > .15:
        return None
    # Global plane identity allows duplicate camera patches to be suppressed.
    world_normal = pose[:, :3] @ plane['normal']
    world_offset = plane['offset'] - world_normal @ pose[:, 3]
    if world_normal[np.argmax(np.abs(world_normal))] < 0:
        world_normal, world_offset = -world_normal, -world_offset
    report = {'frame': image.name, 'component_pixels': int(region.sum()),
              'measured_anchor_pixels': len(x), 'candidate_missing_pixels': int(missing.sum()),
              'inlier_fraction': plane['inlier_fraction'],
              'heldout_fraction': plane['heldout_fraction'],
              'cross_view': check, 'normal_world': world_normal.tolist(),
              'offset_world': float(world_offset),
              'maximum_extrapolation_pixels': 110,
              'classification': 'inferred_plane'}
    return {'report': report, 'missing': missing, 'plane': plane, 'matrix': matrix,
            'pose': pose, 'image': image_rgb}


def mesh_patch(proposal, step=12):
    import trimesh
    missing = proposal['missing']
    height, width = missing.shape
    xs = np.arange(step//2, width, step)
    ys = np.arange(step//2, height, step)
    xx, yy = np.meshgrid(xs, ys)
    # A cell is accepted only when its corners stay in the segmented plane.
    present = missing[yy, xx]
    depth = plane_depth(proposal['plane'], proposal['matrix'], xx, yy)
    median = proposal['plane']['median_anchor_depth']
    present &= np.isfinite(depth) & (depth > .5*median) & (depth < 2*median)
    cam = np.stack(((xx-proposal['matrix'][0, 2])/proposal['matrix'][0, 0]*depth,
                    (yy-proposal['matrix'][1, 2])/proposal['matrix'][1, 1]*depth,
                    depth), axis=-1)
    world = cam @ proposal['pose'][:, :3].T + proposal['pose'][:, 3]
    indices = np.arange(xx.size).reshape(xx.shape)
    a, b, c, d = indices[:-1, :-1], indices[:-1, 1:], indices[1:, 1:], indices[1:, :-1]
    good = present[:-1, :-1] & present[:-1, 1:] & present[1:, 1:] & present[1:, :-1]
    faces = np.concatenate((np.stack((a[good], b[good], c[good]), 1),
                            np.stack((a[good], c[good], d[good]), 1)))
    if len(faces) < 100:
        return None
    colors = cv2.cvtColor(proposal['image'], cv2.COLOR_BGR2RGBA)[yy, xx]
    mesh = trimesh.Trimesh(vertices=world.reshape(-1, 3), faces=faces,
                           vertex_colors=colors.reshape(-1, 4), process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def run(workspace, max_references=64):
    import pycolmap
    import trimesh
    workspace = Path(workspace)
    started = time.monotonic()
    model = pycolmap.Reconstruction(workspace / 'sparse')
    image_by_name = {image.name: image for image in model.images.values()}
    plan = json.loads((workspace / 'dense_plan.json').read_text())
    names = [row['reference'] for row in plan['plan']]
    selected = [names[int(i)] for i in np.linspace(0, len(names)-1,
                                                  min(len(names), max_references), dtype=int)]
    proposals = []
    for name in selected:
        depth = read_array(workspace / 'stereo/depth_maps' / (name+'.geometric.bin'))
        rgb = cv2.imread(str(workspace / 'images' / name))
        if rgb is None:
            continue
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(smooth_neutral_regions(rgb))
        components = sorted(range(1, len(stats)),
                            key=lambda index: stats[index, cv2.CC_STAT_AREA], reverse=True)
        for component in components[:4]:
            if stats[component, cv2.CC_STAT_AREA] < 8000:
                continue
            result = candidate(workspace, model, image_by_name[name], names,
                               component, labels, rgb, depth,
                               seed=int(name[6:12])+component)
            if result is not None:
                proposals.append(result)
    proposals.sort(key=lambda item: item['report']['candidate_missing_pixels'], reverse=True)
    chosen, meshes = [], []
    preview_dir = workspace / 'quality'
    preview_dir.mkdir(exist_ok=True)
    for stale in preview_dir.glob('inferred_plane_*.jpg'):
        stale.unlink()
    for proposal in proposals:
        row = proposal['report']
        normal, offset = np.array(row['normal_world']), row['offset_world']
        duplicate = any(abs(np.dot(normal, previous['normal_world'])) > .985 and
                        abs(offset-previous['offset_world']) < .03*max(abs(offset), 1)
                        for previous in chosen)
        if duplicate:
            continue
        mesh = mesh_patch(proposal)
        if mesh is None:
            continue
        chosen.append(row)
        preview = proposal['image'].copy()
        mask = proposal['missing']
        preview[mask] = (.55*preview[mask] + .45*np.array([255, 190, 30])).astype(np.uint8)
        preview_name = f'inferred_plane_{len(chosen):02d}.jpg'
        cv2.imwrite(str(preview_dir / preview_name),
                    np.hstack((proposal['image'], preview)))
        row['preview'] = 'quality/' + preview_name
        meshes.append(mesh)
        if len(meshes) >= 12:
            break
    report = {'status': 'proposals_available' if meshes else 'insufficient_evidence',
              'method': 'neutral low-texture segmentation + measured-depth RANSAC plane + held-out and neighboring-depth checks',
              'measured_mesh_modified': False, 'geometry_accuracy_validated': False,
              'reference_views_checked': len(selected),
              'candidate_count': len(proposals), 'accepted_plane_count': len(meshes),
              'accepted_planes': chosen,
              'elapsed_seconds': time.monotonic()-started,
              'note': 'Planar patches are inferred, not measured. No surface is created without measured anchors and cross-view support.'}
    (workspace / 'planar_completion_report.json').write_text(json.dumps(report, indent=2)+'\n')
    inferred_path = workspace / 'scene_inferred_planes.glb'
    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.export(inferred_path)
        report['inferred_faces'] = len(combined.faces)
        (workspace / 'planar_completion_report.json').write_text(json.dumps(report, indent=2)+'\n')
    else:
        inferred_path.unlink(missing_ok=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.workspace), indent=2))
