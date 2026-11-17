"""Trim mesh faces that extrapolate beyond measured dense geometry."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def support_mask(vertices, faces, cloud, tolerance):
    tree = cKDTree(np.asarray(cloud))
    keep = np.zeros(len(faces), dtype=bool)
    for start in range(0, len(faces), 50000):
        triangle = np.asarray(vertices)[np.asarray(faces)[start:start + 50000]]
        samples = np.stack([triangle.mean(1), (triangle[:, 0] + triangle[:, 1]) / 2,
                            (triangle[:, 1] + triangle[:, 2]) / 2,
                            (triangle[:, 2] + triangle[:, 0]) / 2], axis=1)
        distance = tree.query(samples.reshape(-1, 3))[0].reshape(-1, 4)
        keep[start:start + len(triangle)] = (distance <= tolerance).all(1)
    return keep


def trim_mesh(workspace):
    workspace = Path(workspace)
    raw = workspace / 'scene_mesh_raw.ply'
    mesh_path = workspace / 'scene_mesh.ply'
    if not raw.exists():
        mesh_path.replace(raw)
    mesh = trimesh.load(raw, process=False)
    cloud = trimesh.load(workspace / 'scene_dense.ply', process=False)
    tree = cKDTree(cloud.vertices)
    sample = cloud.vertices[::max(1, len(cloud.vertices) // 10000)]
    spacing = float(np.median(tree.query(sample, k=2)[0][:, 1]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise RuntimeError('Dense cloud has invalid sample spacing')
    tolerance = 8 * spacing
    keep = support_mask(mesh.vertices, mesh.faces, cloud.vertices, tolerance)
    report = {'raw_faces': len(mesh.faces), 'supported_faces': int(keep.sum()),
              'tolerance_scene_units': tolerance, 'median_dense_spacing': spacing,
              'retained_surface_area_fraction': float(mesh.area_faces[keep].sum() / max(mesh.area, 1e-12)),
              'passed': bool(keep.sum() >= 1000),
              'method': 'centroid and three edge midpoints must lie near measured dense points',
              'geometric_accuracy_validated': False}
    (workspace / 'mesh_support.json').write_text(json.dumps(report, indent=2) + '\n')
    if keep.sum() < 1000:
        raise RuntimeError('Too few mesh faces supported by dense measurements')
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    mesh.export(mesh_path)
    return report


def color_supported_mesh(workspace):
    """Preserve geometry and assign RGB from measured fused samples."""
    workspace = Path(workspace)
    mesh = trimesh.load(workspace / 'scene_mesh.ply', process=False)
    cloud = trimesh.load(workspace / 'scene_dense.ply', process=False)
    nearest = cKDTree(cloud.vertices).query(mesh.vertices)[1]
    mesh.visual.vertex_colors = np.asarray(cloud.visual.vertex_colors)[nearest]
    mesh.export(workspace / 'scene_mesh_colored.ply')
    mesh.export(workspace / 'scene_colored.glb')
    return {'method': 'RGB from nearest fused measurement; mesh positions and faces preserved',
            'colored_vertices': len(mesh.vertices), 'colored_faces': len(mesh.faces)}
