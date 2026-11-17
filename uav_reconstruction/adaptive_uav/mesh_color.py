"""Project captured RGB onto an existing mesh with mesh and stereo occlusion checks."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import trimesh

from .hybrid_dense import camera_matrix, pose4, read_array


def project_visible(scene, vertices, pose, intrinsic, shape):
    """Return vertex indices/pixels whose first ray intersection is this surface.

    Ray directions have camera-axis z=1, so t_hit and projected depth share units.
    This prevents colors from being projected through the mesh onto hidden walls.
    """
    camera_points = vertices @ pose[:3, :3].T + pose[:3, 3]
    depth = camera_points[:, 2]
    pixels = camera_points @ intrinsic.T
    pixels = pixels[:, :2] / np.maximum(depth[:, None], 1e-12)
    height, width = shape[:2]
    valid = (np.isfinite(pixels).all(axis=1) & (depth > 1e-6) &
             (pixels[:, 0] >= 1) & (pixels[:, 0] < width - 2) &
             (pixels[:, 1] >= 1) & (pixels[:, 1] < height - 2))
    indices = np.flatnonzero(valid)
    if not len(indices):
        return indices, pixels[indices], depth[indices]
    center = -pose[:3, :3].T @ pose[:3, 3]
    directions = (vertices[indices] - center) / depth[indices, None]
    rays = np.column_stack((np.broadcast_to(center, directions.shape), directions))
    hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
    # A relative tolerance is independent of the arbitrary COLMAP scale.
    visible = np.isfinite(hit) & (np.abs(hit - depth[indices]) <= .005 * depth[indices])
    indices = indices[visible]
    return indices, pixels[indices], depth[indices]


def color_mesh(mesh_path, workspace, output, max_views=96, preserve_colors=False, semantic_dir=None):
    started = time.monotonic()
    mesh_path, workspace, output = map(Path, (mesh_path, workspace, output))
    if max_views < 1:
        raise ValueError('max_views must be positive')
    output.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = np.asarray(mesh.vertices)
    if not len(vertices) or not len(mesh.faces):
        raise ValueError('Cannot color an empty mesh')
    normals = np.asarray(mesh.vertex_normals)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    reconstruction = pycolmap.Reconstruction(workspace / 'sparse')
    images = sorted((i for i in reconstruction.images.values() if i.has_pose), key=lambda i: i.name)
    if not images:
        raise ValueError('No posed RGB views')
    selected = np.unique(np.linspace(0, len(images) - 1, min(max_views, len(images)), dtype=int))
    # Blend at most three comparably good views, rather than all distant views.
    scores = np.zeros((len(vertices), 3), np.float32)
    colors = np.zeros((len(vertices), 3, 3), np.float32)
    stereo_supported = np.zeros(len(vertices), bool)
    views = []
    for index in selected:
        image = images[index]
        camera = reconstruction.cameras[image.camera_id]
        if camera.model_name not in ('PINHOLE', 'SIMPLE_PINHOLE'):
            raise ValueError('RGB coloring requires undistorted pinhole images')
        rgb = cv2.imread(str(workspace / 'images' / image.name))
        if rgb is None:
            raise FileNotFoundError(workspace / 'images' / image.name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        pose = pose4(image)
        intrinsic = camera_matrix(camera, rgb.shape)
        ids, uv, depth = project_visible(scene, vertices, pose, intrinsic, rgb.shape)
        visible_count = len(ids)
        evidence = np.zeros(len(ids), bool)
        depth_path = workspace / 'stereo' / 'depth_maps' / (image.name + '.geometric.bin')
        if depth_path.is_file() and len(ids):
            measured = read_array(depth_path)
            dx = np.clip(np.rint(uv[:, 0] * measured.shape[1] / rgb.shape[1]).astype(int), 0, measured.shape[1] - 1)
            dy = np.clip(np.rint(uv[:, 1] * measured.shape[0] / rgb.shape[0]).astype(int), 0, measured.shape[0] - 1)
            observed = measured[dy, dx]
            known = np.isfinite(observed) & (observed > 0)
            evidence = known & (np.abs(depth - observed) <= .05 * observed)
            # Unmeasured pixels still need mesh visibility. Known stereo surfaces
            # reject projecting their appearance onto a different learned surface.
            keep = ~known | evidence
            ids, uv, depth, evidence = ids[keep], uv[keep], depth[keep], evidence[keep]
        if len(ids):
            if semantic_dir:
                from .semantic_scene import load_frame
                classes, _, _ = load_frame(semantic_dir, image.name, intrinsic, rgb.shape[:2])
                keep = classes[uv[:, 1].astype(int), uv[:, 0].astype(int)] == 0
                ids, uv, depth, evidence = ids[keep], uv[keep], depth[keep], evidence[keep]
        if len(ids):
            x, y = uv[:, 0].astype(int), uv[:, 1].astype(int)
            fx, fy = (uv[:, 0] - x)[:, None], (uv[:, 1] - y)[:, None]
            sampled = (rgb[y, x] * (1 - fx) * (1 - fy) + rgb[y, x + 1] * fx * (1 - fy) +
                       rgb[y + 1, x] * (1 - fx) * fy + rgb[y + 1, x + 1] * fx * fy)
            center = -pose[:3, :3].T @ pose[:3, 3]
            to_camera = center - vertices[ids]
            distances = np.linalg.norm(to_camera, axis=1)
            # Absolute cosine handles inconsistently oriented learned triangles.
            cosine = np.abs(np.sum(normals[ids] * to_camera, axis=1) / distances)
            margin = np.minimum.reduce((uv[:, 0], uv[:, 1], rgb.shape[1] - 1 - uv[:, 0], rgb.shape[0] - 1 - uv[:, 1]))
            weight = ((.1 + .9 * cosine ** 2) * np.minimum(1., margin / 20) /
                      np.maximum(distances ** 2, 1e-12)).astype(np.float32)
            slot = np.argmin(scores[ids], axis=1)
            better = weight > scores[ids, slot]
            scores[ids[better], slot[better]] = weight[better]
            # Blend in linear light; exported RGB bytes are sRGB.
            srgb = sampled[better].astype(np.float32) / 255
            colors[ids[better], slot[better]] = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
            stereo_supported[ids[evidence]] = True
        views.append({'frame': image.name, 'mesh_visible_vertices': visible_count,
                      'accepted_rgb_vertices': len(ids), 'stereo_agreeing_vertices': int(evidence.sum())})
        print(f'RGB {len(views)}/{len(selected)}: {image.name}, {len(ids)} visible vertices', flush=True)
    weights = np.where(scores >= .5 * scores.max(axis=1, keepdims=True), scores, 0)
    totals = weights.sum(axis=1)
    colored = totals > 0
    linear = (colors[colored] * weights[colored, :, None]).sum(axis=1) / totals[colored, None]
    srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
    vertex_colors = np.full((len(vertices), 4), 160, np.uint8)
    vertex_colors[:, 3] = 255
    inherited = preserve_colors and mesh.visual.kind == 'vertex'
    if inherited:
        vertex_colors = np.asarray(mesh.visual.vertex_colors, np.uint8).copy()
    vertex_colors[colored, :3] = np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
    ply, glb = output / 'scene_mesh_colored.ply', output / 'scene_colored.glb'
    mesh.export(ply)
    # PLY stores sRGB bytes; glTF COLOR_0 is linear. An unlit material preserves
    # captured appearance instead of adding highlights to already lit RGB.
    glb_colors = vertex_colors.copy()
    srgb_all = vertex_colors[:, :3].astype(np.float32) / 255
    linear_all = np.where(srgb_all <= .04045, srgb_all / 12.92, ((srgb_all + .055) / 1.055) ** 2.4)
    glb_colors[:, :3] = np.rint(linear_all * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=glb_colors)

    def unlit(tree):
        tree.setdefault('extensionsUsed', []).append('KHR_materials_unlit')
        tree['materials'] = [{'doubleSided': True, 'pbrMetallicRoughness': {
            'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0, 'roughnessFactor': 1},
            'extensions': {'KHR_materials_unlit': {}}}]
        for item in tree['meshes']:
            for primitive in item['primitives']:
                primitive['material'] = 0

    mesh.export(glb, tree_postprocessor=unlit)
    report = {'source_mesh': str(mesh_path.resolve()), 'rgb_workspace': str(workspace.resolve()),
              'method': 'visibility-checked RGB projection; up to three nearby frontal views',
              'geometry_changed': False, 'vertices': len(vertices), 'faces': len(mesh.faces),
              'views_used': len(views), 'colored_vertices': int(colored.sum()),
              'colored_vertex_fraction': float(colored.mean()),
              'vertices_with_stereo_agreeing_rgb': int(stereo_supported.sum()),
              'uncolored_vertices': int((~colored).sum()),
              'unobserved_color': 'existing mesh RGB retained' if inherited else 'neutral gray; no invented RGB',
              'preserved_existing_colors': bool(inherited),
              'color_encoding': 'sRGB PLY; linear RGB glTF with unlit double-sided material',
              'ply': str(ply.resolve()), 'glb': str(glb.resolve()),
              'elapsed_seconds': time.monotonic() - started, 'per_view': views}
    (output / 'color_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-views', type=int, default=96)
    parser.add_argument('--preserve-colors',action='store_true',help='Keep existing mesh RGB where no new projection passes the visibility checks')
    parser.add_argument('--semantic-dir', type=Path)
    args = parser.parse_args()
    report = color_mesh(args.mesh, args.workspace, args.output, args.max_views,preserve_colors=args.preserve_colors,semantic_dir=args.semantic_dir)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_view'}), flush=True)


if __name__ == '__main__':
    main()
