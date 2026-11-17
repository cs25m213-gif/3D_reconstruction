"""Choose bounded dense work from shared landmarks and triangulation angles."""
import json
from pathlib import Path

import numpy as np


def pair_quality(center_a, center_b, landmarks):
    """Return median triangulation angle; rotation alone gives zero parallax."""
    if len(landmarks) == 0:
        return 0.0
    a = np.asarray(landmarks) - center_a
    b = np.asarray(landmarks) - center_b
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float(np.median(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))))


def plan_dense(workspace, max_views=96, source_views=6, min_support=200):
    import pycolmap
    workspace = Path(workspace)
    reconstruction = pycolmap.Reconstruction(workspace / 'sparse')
    images = sorted((i for i in reconstruction.images.values()
                     if i.has_pose and i.num_points3D >= min_support), key=lambda i: i.name)
    if len(images) < 5:
        raise RuntimeError('Too few supported camera poses for dense reconstruction')
    if len(images) > max_views:
        # Temporal bins preserve full recording coverage; support breaks ties.
        images = [max((images[int(i)] for i in group), key=lambda im: im.num_points3D)
                  for group in np.array_split(np.arange(len(images)), max_views)]
    centers = {i.name: np.asarray(i.cam_from_world().inverse().translation) for i in images}
    tracks = {i.name: {p.point3D_id for p in i.points2D if p.has_point3D()} for i in images}
    neighbors = {i.name: [] for i in images}
    for j, a in enumerate(images):
        for b in images[j + 1:]:
            common = sorted(tracks[a.name] & tracks[b.name])
            if len(common) < 30:
                continue
            sampled = common[::max(1, len(common) // 64)][:64]
            xyz = np.array([reconstruction.points3D[p].xyz for p in sampled])
            angle = pair_quality(centers[a.name], centers[b.name], xyz)
            if not .8 <= angle <= 45:
                continue
            score = len(common) * min(angle / 3., 1.)
            for ref, src in ((a.name, b.name), (b.name, a.name)):
                neighbors[ref].append({'name': src, 'common_points': len(common),
                                       'median_angle_deg': angle, 'score': score})
    active = set(neighbors)
    while True:
        valid = {name for name in active if sum(n['name'] in active for n in neighbors[name]) >= 2}
        if valid == active:
            break
        active = valid
    if len(active) < 5:
        raise RuntimeError('Camera map lacks connected dense pairs with useful parallax')
    plan = []
    for name in sorted(active):
        sources = sorted((n for n in neighbors[name] if n['name'] in active),
                         key=lambda n: n['score'], reverse=True)[:source_views]
        plan.append({'reference': name, 'sources': sources})
    report = {'registered_cameras': reconstruction.num_reg_images(), 'reference_views': len(plan),
              'max_reference_views': max_views, 'max_source_views': source_views,
              'rejected_references': sorted(set(neighbors) - active), 'plan': plan,
              'selection': 'temporal coverage, shared sparse landmarks, 0.8 to 45 degree median parallax'}
    (workspace / 'dense_plan.json').write_text(json.dumps(report, indent=2) + '\n')
    (workspace / 'stereo/patch-match.cfg').write_text(''.join(
        row['reference'] + '\n' + ', '.join(n['name'] for n in row['sources']) + '\n' for row in plan))
    (workspace / 'stereo/fusion.cfg').write_text(''.join(row['reference'] + '\n' for row in plan))
    return report
