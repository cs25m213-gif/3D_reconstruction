"""Automatic coverage and missing-evidence report for a completed dense run."""
import json
from pathlib import Path

import cv2
import numpy as np

from .depth_normals import read_array


def audit_dense(workspace):
    import pycolmap
    import trimesh
    workspace = Path(workspace)
    plan = json.loads((workspace / 'dense_plan.json').read_text())
    model = pycolmap.Reconstruction(workspace / 'sparse')
    mesh = trimesh.load(workspace / 'scene_mesh.ply', process=False)
    image_by_name = {image.name: image for image in model.images.values()}
    depth_rows = []
    textureless, pixels = 0, 0
    for entry in plan['plan']:
        name = entry['reference']
        depth = read_array(workspace / 'stereo/depth_maps' / (name + '.geometric.bin'))
        gray = cv2.imread(str(workspace / 'images' / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'Missing source image {name}')
        gray = cv2.resize(gray, (depth.shape[1], depth.shape[0])).astype(np.float32)
        mean = cv2.boxFilter(gray, -1, (11, 11))
        std = np.sqrt(np.maximum(cv2.boxFilter(gray*gray, -1, (11, 11))-mean*mean, 0))
        textureless += int((std < 2).sum())
        pixels += depth.size
        depth_rows.append({'frame': name, 'valid_depth_fraction': float(np.mean(depth > 0))})
    images = sorted(image_by_name.values(), key=lambda image: image.name)
    samples = [images[int(i)] for i in np.linspace(0, len(images)-1, min(9, len(images)), dtype=int)]
    views = []
    audit_dir = workspace / 'quality'
    audit_dir.mkdir(exist_ok=True)
    for index, image in enumerate(samples):
        camera = model.cameras[image.camera_id]
        width = 360
        height = round(width*camera.height/camera.width)
        k = camera.calibration_matrix().copy()
        k[0] *= width/camera.width
        k[1] *= height/camera.height
        pose = image.cam_from_world().matrix()
        points = np.asarray(mesh.vertices) @ pose[:, :3].T + pose[:, 3]
        projection = points @ k.T
        uv = projection[:, :2]/np.maximum(projection[:, 2:], 1e-9)
        faces = np.asarray(mesh.faces)
        front = (points[faces, 2] > 1e-5).all(1)
        triangles = uv[faces]
        inside = ((triangles[:, :, 0].max(1) >= 0) & (triangles[:, :, 0].min(1) < width) &
                  (triangles[:, :, 1].max(1) >= 0) & (triangles[:, :, 1].min(1) < height))
        mask = np.zeros((height*2, width*2), np.uint8)
        # Supersampling reduces the coverage inflation from rasterizing tiny faces.
        for triangle in triangles[front & inside]:
            coordinates = np.rint(np.clip(triangle, -1e6, 1e6)*32).astype(np.int32)
            cv2.fillConvexPoly(mask, coordinates, 255, shift=4)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA) >= 128
        source = cv2.resize(cv2.imread(str(workspace / 'images' / image.name)), (width, height))
        overlay = source.copy()
        overlay[mask] = (.6*overlay[mask] + .4*np.array([60, 220, 60])).astype(np.uint8)
        overlay[~mask] = (.65*overlay[~mask]).astype(np.uint8)
        cv2.putText(overlay, f'est. visible surface {mask.mean():.0%}', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.imwrite(str(audit_dir / f'coverage_{index:02d}.jpg'), np.hstack([source, overlay]))
        views.append({'frame': image.name, 'estimated_visible_coverage': float(mask.mean()),
                      'comparison': f'quality/coverage_{index:02d}.jpg'})
    coverage = np.array([row['estimated_visible_coverage'] for row in views])
    median_depth = float(np.median([row['valid_depth_fraction'] for row in depth_rows]))
    support = [row['spatial_sparse_support_two_sources'] for row in plan['plan']
               if 'spatial_sparse_support_two_sources' in row]
    issues = []
    if textureless/max(pixels, 1) > .25:
        issues.append('Large low-texture areas lack reliable visual depth; use textured overlap or measured depth for these surfaces.')
    if support and np.mean(support) < .5:
        issues.append('Many regions lack two source observations with useful parallax; more lateral movement and overlapping views are needed.')
    if median_depth < .5:
        issues.append('Less than half of image pixels have consistent geometric depth; mesh completion cannot establish their actual shape.')
    broad = bool(coverage.mean() >= .7 and np.percentile(coverage, 10) >= .5)
    report = {'status': 'broad_visible_coverage' if broad else 'incomplete',
              'geometry_accuracy_validated': False, 'view_count': len(views),
              'mean_estimated_visible_coverage': float(coverage.mean()),
              'median_geometric_depth_fraction': median_depth,
              'low_texture_pixel_fraction': float(textureless/max(pixels, 1)),
              'mean_spatial_sparse_support': float(np.mean(support)) if support else None,
              'method': 'supersampled CPU triangle silhouette at registered camera poses; image occupancy is not whole-scene completeness or geometric accuracy',
              'issues': issues, 'sampled_views': views, 'depth_views': depth_rows}
    planar_path = workspace / 'planar_completion_report.json'
    planar = json.loads(planar_path.read_text()) if planar_path.exists() else None
    if planar is not None:
        report['separate_inferred_planes'] = {
            'status': planar['status'], 'accepted_plane_count': planar['accepted_plane_count'],
            'measured_mesh_modified': False}
    (workspace / 'quality_report.json').write_text(json.dumps(report, indent=2) + '\n')
    metrics_path = workspace / 'dense_metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    inferred_count = int(planar.get('accepted_plane_count', 0)) if planar else 0
    inferred_file = 'scene_inferred_planes.glb' if inferred_count and (workspace / 'scene_inferred_planes.glb').is_file() else None
    hybrid = {
        'schema_version': 1,
        'status': 'measured_plus_inferred' if inferred_file else 'measured_only',
        'measured_geometry': {
            'dense_points': metrics.get('dense_points'),
            'mesh_vertices': int(len(mesh.vertices)),
            'mesh_faces': int(len(mesh.faces)),
            'mesh_file': 'scene_colored.glb',
            'coverage_status': report['status'],
            'mean_estimated_visible_coverage': report['mean_estimated_visible_coverage'],
        },
        'inferred_geometry': {
            'method': planar.get('method') if planar else None,
            'plane_count': inferred_count,
            'mesh_faces': int(planar.get('inferred_faces', 0)) if planar else 0,
            'mesh_file': inferred_file,
            'visible_by_default': False,
            'measured_mesh_modified': False,
            'reference_views_checked': planar.get('reference_views_checked') if planar else None,
            'plane_evidence': [
                {'frame': row['frame'],
                 'measured_anchor_pixels': row['measured_anchor_pixels'],
                 'heldout_fraction': row['heldout_fraction'],
                 'cross_view': row['cross_view'],
                 'preview': row.get('preview')}
                for row in planar.get('accepted_planes', [])
            ] if planar else [],
        },
        'assessment': {
            'geometry_accuracy_validated': False,
            'inferred_planes_included_in_coverage_score': False,
            'quality_report': 'quality_report.json',
            'limits': report['issues'],
            'note': 'The inferred layer adds plausible planar surfaces but does not establish whole-room completeness or metric accuracy.',
        },
    }
    (workspace / 'hybrid_report.json').write_text(json.dumps(hybrid, indent=2) + '\n')
    cards = ''.join(f'<figure><figcaption>{row["frame"]}</figcaption><img src="{Path(row["comparison"]).name}" width="720"></figure>' for row in views)
    text = '<p>' + '</p><p>'.join(issues) + '</p>'
    inferred_cards = ''
    if planar and planar['accepted_plane_count']:
        inferred_cards = ('<h2>Optional estimated planes</h2><p>Blue overlays show planar '
                          'estimates anchored to measured depth and checked in another view. '
                          'They remain separate from the measured mesh and do not change the coverage score. '
                          '<a href="../mesh_viewer.html">Use the viewer toggle</a>.</p>')
        inferred_cards += ''.join(
            f'<figure><figcaption>{row["frame"]}: inferred region</figcaption>'
            f'<img src="{Path(row["preview"]).name}" width="720"></figure>'
            for row in planar['accepted_planes'])
    (audit_dir / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Coverage report</title>'
        '<style>body{background:#111b24;color:#e5edf4;font:16px system-ui;max-width:900px;margin:24px auto}img{max-width:100%}a{color:#8bdcff}</style>'
        '<a href="../mesh_viewer.html">Back to colored mesh</a> · '
        '<a href="../hybrid_report.json">Hybrid report JSON</a><h1>Coverage report</h1>'
        '<p>Original image on the left; estimated visible mesh is green on the right. '
        'These views measure image occupancy, not whole-scene completeness or accuracy.</p>'+text+inferred_cards+cards)
    return report
