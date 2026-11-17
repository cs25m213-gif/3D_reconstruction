"""Hybrid dense reconstruction: measured depth + aligned learned prior + global planes.

Drop into uav_reconstruction/adaptive_uav/hybrid_dense.py. Standalone: it has no
relative imports, so it also runs as `python hybrid_dense.py --workspace dense`.

Why: PatchMatch cannot measure textureless walls/ceilings/doors, so per-frame
coverage stalls at 11-60%. This module keeps PatchMatch as the "measured" layer
and adds two clearly labelled layers:

  tier 1  measured   COLMAP PatchMatch geometric depth (unchanged)
  tier 2  supported  monocular metric depth prior, scale/shift-aligned to measured
                     depth + sparse points, kept only if neighbouring views agree
  tier 3  inferred   global world-space planes, extended only near pixels where
                     the plane already agrees with measured/supported depth

Fusion is CPU TSDF (Open3D), so the 4 GB GPU is only used for the prior and
PatchMatch, never both at once (stage "all" runs the prior in its own process).

Pipeline order (run by --stage all):
  1. prior : Depth Anything V2 metric-indoor, fp16, batch 1  -> prior_depth/*.npy
  2. align : robust per-frame scale/shift, held-out check, edge and cross-view gates
  3. fuse  : TSDF (measured weighted x2) -> scene_hybrid_mesh.ply/.glb,
             scene_hybrid_provenance.ply
  4. planes: sequential RANSAC on the fused cloud, anchored extension per frame
             -> scene_hybrid_planes.glb
  5. report: hybrid_dense_report.json + quality/hybrid_*.jpg (measured/supported/inferred)

Units: the COLMAP model has arbitrary scale. All thresholds are therefore
relative to the median measured depth, never absolute metres.

----------------------------------------------------------------------------
INTEGRATION (already wired in this tree)
----------------------------------------------------------------------------
reconstruct.py passes --hybrid to fast_session. The dense_cuda stage stops after
PatchMatch depth maps; this module then produces the final colored TSDF mesh.
A failure here marks the saved-video run failed. Outputs are:
  hybrid_dense_report.json   (dense_quality.py owns hybrid_report.json)
  scene_hybrid_mesh.glb/.ply, scene_hybrid_provenance.ply, scene_hybrid_planes.glb
Manual run:  python -m adaptive_uav.hybrid_dense --workspace <run>/dense --stage all

Capture (largest remaining lever): sweep sideways instead of rotating in place,
add one high pass (ceiling) and one low pass (floor), lock exposure, close the loop.
`capture_diagnostics` in the report tells you when parallax was too low.

Dependencies: torch, transformers, open3d, scipy, trimesh, pycolmap, opencv, numpy.
"""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

PRIOR_MODEL = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf'  # verify id; -Small- if VRAM is tight


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def frame_id(name):
    return int(name[6:12])


def read_array(path):
    """Read a COLMAP .bin depth/normal map."""
    with open(path, 'rb') as stream:
        width, height, channels = np.genfromtxt(stream, delimiter='&', max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        stream.seek(0)
        delimiters = 0
        while delimiters < 3:
            if stream.read(1) == b'&':
                delimiters += 1
        array = np.fromfile(stream, np.float32)
    array = array.reshape((width, height, channels), order='F')
    return np.transpose(array, (1, 0, 2)).squeeze()


def camera_matrix(camera, shape):
    matrix = camera.calibration_matrix().copy()
    matrix[0] *= shape[1] / camera.width
    matrix[1] *= shape[0] / camera.height
    return matrix


def reference_names(ws):
    plan = ws / 'dense_plan.json'
    if plan.exists():
        return [row['reference'] for row in json.loads(plan.read_text())['plan']]
    lines = (ws / 'stereo/patch-match.cfg').read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')][::2]


def pose4(image):
    matrix = np.eye(4)
    matrix[:3, :4] = image.cam_from_world().matrix()[:3, :4]
    return matrix


# --------------------------------------------------------------------------
# stage 1: learned depth prior (GPU, own process)
# --------------------------------------------------------------------------
def run_prior(ws, size=518, model_id=PRIOR_MODEL):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    out = ws / 'prior_depth'
    out.mkdir(exist_ok=True)
    names = reference_names(ws)
    manifest_path = out / 'prior_manifest.json'
    manifest = {'model_id': model_id, 'long_side_px': size}
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Cached prior uses another model or size; choose a new workspace')
    elif list(out.glob('*.npy')):
        raise RuntimeError('Cached prior has no model manifest; use --stage fuse for this legacy run')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    todo = [n for n in names if not (out / (n + '.npy')).exists()]
    if not todo:
        return
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id, torch_dtype=torch.float16).cuda().eval()
    for name in todo:
        image = cv2.imread(str(ws / 'images' / name))
        if image is None:
            raise FileNotFoundError(ws / 'images' / name)
        h, w = image.shape[:2]
        scale = size / max(h, w)
        nh, nw = max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14)
        small = cv2.cvtColor(cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)
        inputs = processor(images=small, do_resize=False, return_tensors='pt')
        with torch.inference_mode():
            depth = model(pixel_values=inputs['pixel_values'].half().cuda()).predicted_depth
        temporary = out / (name + '.tmp.npy')
        np.save(temporary, depth[0].float().cpu().numpy())
        temporary.replace(out / (name + '.npy'))
        del inputs, depth
        torch.cuda.empty_cache()
    missing = [name for name in names if not (out / (name + '.npy')).is_file()]
    if missing:
        raise RuntimeError(f'Learned depth missing for {len(missing)} reference frames')


# --------------------------------------------------------------------------
# stage 2: alignment and gates (CPU)
# --------------------------------------------------------------------------
def huber_fit(d, m, iterations=8):
    """Robust fit m ~ s*d + t."""
    A = np.c_[d, np.ones_like(d)]
    weights = np.ones_like(d)
    solution = np.array([1., 0.])
    for _ in range(iterations):
        root = np.sqrt(weights)
        solution = np.linalg.lstsq(A * root[:, None], m * root, rcond=None)[0]
        residual = np.abs(A @ solution - m)
        delta = 1.345 * max(1.4826 * np.median(residual), 1e-9)
        weights = np.where(residual <= delta, 1., delta / np.maximum(residual, 1e-12))
    return solution


def sparse_anchors(points, image, K, shape):
    """Use only sparse points observed in this image; avoid occluded global points."""
    visible = [points[p.point3D_id].xyz for p in image.points2D
               if p.has_point3D() and p.point3D_id in points]
    if not visible:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float64)
    points = np.asarray(visible, np.float64)
    pose = image.cam_from_world().matrix()
    cam = points @ pose[:, :3].T + pose[:, 3]
    uv = cam @ K.T
    x = np.rint(uv[:, 0] / np.maximum(uv[:, 2], 1e-9)).astype(int)
    y = np.rint(uv[:, 1] / np.maximum(uv[:, 2], 1e-9)).astype(int)
    ok = (cam[:, 2] > 1e-6) & (x >= 0) & (y >= 0) & (x < shape[1]) & (y < shape[0])
    return x[ok], y[ok], cam[ok, 2]


def align_frame(prior_small, measured, points, image, K, args, rng):
    h, w = measured.shape
    prior = cv2.resize(prior_small, (w, h), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(measured > 0)
    m = measured[ys, xs].astype(np.float64)
    if len(m) < 4 * args.min_anchors:
        sx, sy, sz = sparse_anchors(points, image, K, (h, w))
        xs, ys, m = np.r_[xs, sx], np.r_[ys, sy], np.r_[m, sz]
    if len(m) > 30000:
        pick = rng.choice(len(m), 30000, replace=False)
        xs, ys, m = xs[pick], ys[pick], m[pick]
    d = prior[ys, xs].astype(np.float64)
    good = np.isfinite(d) & (d > 0) & np.isfinite(m) & (m > 0)
    xs, ys, m, d = xs[good], ys[good], m[good], d[good]
    if len(m) < args.min_anchors:
        return {'ok': False, 'reason': 'too few anchors', 'anchors': int(len(m))}
    order = rng.permutation(len(m))
    cut = int(.8 * len(m))
    train, held = order[:cut], order[cut:]
    s, t = huber_fit(d[train], m[train])
    error = float(np.median(np.abs(s * d[held] + t - m[held]) / np.maximum(m[held], 1e-9)))
    if s <= 0 or error > args.max_align_error:
        return {'ok': False, 'reason': 'held-out alignment error', 'anchors': int(len(m)),
                'heldout_median_error': error}
    aligned = (s * prior + t).astype(np.float32)
    far = 3 * float(np.percentile(m, 95))
    aligned[~np.isfinite(aligned) | (aligned <= 0) | (aligned > far)] = 0
    return {'ok': True, 'depth': aligned, 'scale': float(s), 'shift': float(t),
            'heldout_median_error': error, 'anchors': int(len(m))}


def edge_invalid(depth, relative=0.04):
    """Prior blurs depth edges; remove flying pixels there."""
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8
    bad = (np.hypot(gx, gy) / np.maximum(depth, 1e-6) > relative).astype(np.uint8)
    return cv2.dilate(bad, np.ones((5, 5), np.uint8)) > 0


def cross_view(candidate, depth, image, K, neighbours, combined, model, args):
    """Keep prior pixels geometrically consistent with separated camera views.

    Other prior maps are still learned estimates, so this is consistency evidence,
    not a measured-depth validation.
    """
    ys, xs = np.nonzero(candidate)
    out = np.zeros_like(candidate)
    if len(xs) == 0:
        return out
    z = depth[ys, xs].astype(np.float64)
    cam = np.stack(((xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z), 1)
    inverse = image.cam_from_world().inverse().matrix()
    world = cam @ inverse[:, :3].T + inverse[:, 3]
    center = inverse[:, 3]
    agree = np.zeros(len(z), int)
    seen = np.zeros(len(z), int)
    for other in neighbours:
        other_center = other.cam_from_world().inverse().matrix()[:, 3]
        if np.linalg.norm(other_center-center) / max(float(np.median(z)), 1e-9) < .025:
            continue
        other_depth = combined.get(other.name)
        if other_depth is None:
            continue
        Ko = camera_matrix(model.cameras[other.camera_id], other_depth.shape)
        pose = other.cam_from_world().matrix()
        pc = world @ pose[:, :3].T + pose[:, 3]
        uv = pc @ Ko.T
        u = np.rint(uv[:, 0] / np.maximum(uv[:, 2], 1e-9)).astype(int)
        v = np.rint(uv[:, 1] / np.maximum(uv[:, 2], 1e-9)).astype(int)
        ok = (pc[:, 2] > 1e-6) & (u >= 0) & (v >= 0) & \
             (u < other_depth.shape[1]) & (v < other_depth.shape[0])
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        measured = other_depth[v[idx], u[idx]]
        has = measured > 0
        idx, measured = idx[has], measured[has]
        error = (measured - pc[idx, 2]) / np.maximum(measured, 1e-9)
        visible = error > -args.tol            # closer in the other view = occluded, ignore
        idx, error = idx[visible], error[visible]
        seen[idx] += 1
        agree[idx] += (np.abs(error) < args.tol)
    good = (agree >= args.min_agree) & (agree >= .66 * seen)
    out[ys[good], xs[good]] = True
    return out


# --------------------------------------------------------------------------
# stage 3: TSDF fusion (CPU)
# --------------------------------------------------------------------------
def make_volume(voxel):
    import open3d as o3d
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


def integrate(volume, frames, tiers, depth_trunc, repeat=1):
    import open3d as o3d
    for f in frames:
        depth = np.where(np.isin(f['tier'], tiers), f['depth'], 0).astype(np.float32)
        if not depth.any():
            continue
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(f['rgb'])),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        h, w = depth.shape
        K = f['K']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        for _ in range(repeat):
            volume.integrate(rgbd, intrinsic, f['P'])


def clean_mesh(mesh, diagnostics=None):
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    threshold = max(500, .01 * len(labels))
    discarded = counts[labels] < threshold if len(labels) else np.zeros(0, bool)
    if diagnostics is not None:
        diagnostics.update({'raw_faces': int(len(labels)),
                            'connected_components': int(len(counts)),
                            'largest_component_faces': int(counts.max()) if len(counts) else 0,
                            'minimum_component_faces': int(threshold),
                            'discarded_faces': int(discarded.sum()),
                            'discarded_components': int((counts < threshold).sum()),
                            'component_size_bins': {
                                f'{low}-{high - 1}': {
                                    'components': int(((counts >= low) & (counts < high)).sum()),
                                    'faces': int(counts[(counts >= low) & (counts < high)].sum())}
                                for low, high in ((1, 50), (50, 500), (500, 2000),
                                                  (2000, 5000), (5000, 10000),
                                                  (10000, 1000000000))}})
    if len(labels):
        mesh.remove_triangles_by_mask(discarded)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def export_glb(mesh, path):
    import trimesh
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    colors = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    rgba = np.c_[colors, np.full(len(colors), 255, np.uint8)] if len(colors) == len(vertices) else None
    trimesh.Trimesh(vertices, faces, vertex_colors=rgba, process=False).export(path)


# --------------------------------------------------------------------------
# stage 4: global planes, anchored extension
# --------------------------------------------------------------------------
def global_planes(pcd, threshold, max_planes, min_fraction):
    planes, rest, total = [], pcd, len(pcd.points)
    for _ in range(max_planes):
        if len(rest.points) < max(500, min_fraction * total):
            break
        plane, inliers = rest.segment_plane(distance_threshold=threshold, ransac_n=3,
                                            num_iterations=1500)
        if len(inliers) < min_fraction * total:
            break
        normal = np.array(plane[:3], np.float64)
        norm = np.linalg.norm(normal)
        normal, offset = normal / norm, plane[3] / norm
        points = np.asarray(rest.points)[inliers]
        u = np.cross(normal, [1., 0, 0] if abs(normal[0]) < .9 else [0, 1., 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        uv = np.c_[points @ u, points @ v]
        planes.append({'normal': normal, 'offset': float(offset), 'u': u, 'v': v,
                       'lo': uv.min(0), 'hi': uv.max(0), 'inliers': int(len(inliers))})
        rest = rest.select_by_index(inliers, invert=True)
    return planes


def infer_frame(f, planes, args, median_depth):
    depth, tier, K, P = f['depth'], f['tier'], f['K'], f['P']
    h, w = depth.shape
    R, t = P[:3, :3], P[:3, 3]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    rx, ry = (xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1]
    empty = depth <= 0
    local_min = cv2.erode(np.where(depth > 0, depth, 1e9).astype(np.float32),
                          np.ones((31, 31), np.uint8))
    lab = cv2.cvtColor(f['rgb'], cv2.COLOR_RGB2LAB).astype(np.float32)
    image_edges = cv2.dilate(cv2.Canny(f['rgb'], 45, 100), np.ones((3, 3), np.uint8)) > 0
    best = np.full((h, w), np.inf, np.float32)
    best_z = np.zeros((h, w), np.float32)
    margin = args.plane_margin_fraction * median_depth
    for plane in planes:
        normal_c = R @ plane['normal']
        offset_c = plane['offset'] - normal_c @ t
        den = normal_c[0] * rx + normal_c[1] * ry + normal_c[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            z = -offset_c / den
        valid = np.isfinite(z) & (np.abs(den) > .05) & (z > 0) & (z < 3 * median_depth)
        agree = valid & (depth > 0) & (np.abs(depth - z) < args.tol * depth)
        if agree.sum() < 400:
            continue
        # Extend the observed plane's appearance; plain walls and yellow doors
        # can both qualify, while dark foreground objects stay excluded.
        anchor_lab = lab[agree][::max(1, int(agree.sum()) // 5000)]
        median_lab = np.median(anchor_lab, axis=0)
        color_distance = np.linalg.norm(lab-median_lab, axis=2)
        anchor_spread = float(np.median(np.linalg.norm(anchor_lab-median_lab, axis=1)))
        color_limit = float(np.clip(2.5*anchor_spread, 18, 40))
        distance = cv2.distanceTransform((~agree).astype(np.uint8), cv2.DIST_L2, 5)
        in_front = (local_min < 1e8) & (z < .95 * local_min)   # plane cannot sit in front of seen geometry
        cand = (empty & valid & (distance <= args.extrapolation_px) &
                (distance < best) & ~in_front & ~image_edges &
                (color_distance <= color_limit))
        yy, xx = np.nonzero(cand)
        if len(yy) == 0:
            continue
        zc = z[yy, xx]
        cam = np.stack((rx[yy, xx] * zc, ry[yy, xx] * zc, zc), 1)
        world = (cam - t) @ R
        uv = np.stack((world @ plane['u'], world @ plane['v']), 1)
        inside = np.all((uv >= plane['lo'] - margin) & (uv <= plane['hi'] + margin), axis=1)
        yy, xx = yy[inside], xx[inside]
        best[yy, xx] = distance[yy, xx]
        best_z[yy, xx] = z[yy, xx]
    new = np.isfinite(best) & (best_z > 0)
    depth[new] = best_z[new]
    tier[new] = 3
    return int(new.sum())


# --------------------------------------------------------------------------
# diagnostics, previews, report
# --------------------------------------------------------------------------
def capture_diagnostics(names, by_name, median_depth, step=5):
    ordered = sorted(names, key=frame_id)
    centers = np.array([by_name[n].cam_from_world().inverse().matrix()[:3, 3] for n in ordered])
    if len(centers) <= step:
        return {'status': 'too few frames'}
    ratio = np.linalg.norm(centers[step:] - centers[:-step], axis=1) / median_depth
    low = float((ratio < .03).mean())
    return {'median_baseline_over_depth': float(np.median(ratio)),
            'low_parallax_fraction': low,
            'warning': ('Rotation-dominated capture: sweep sideways and add ceiling/floor passes.'
                        if low > .4 else None)}


def write_previews(ws, frames, args):
    out = ws / 'quality'
    out.mkdir(exist_ok=True)
    for old in out.glob('hybrid_*.jpg'):
        old.unlink()
    colors = {1: (60, 200, 80), 2: (60, 130, 240), 3: (255, 190, 30)}  # RGB
    for k, i in enumerate(np.linspace(0, len(frames) - 1, min(args.previews, len(frames)), dtype=int)):
        f = frames[i]
        overlay = f['rgb'].copy()
        for tier, color in colors.items():
            mask = f['tier'] == tier
            overlay[mask] = (.5 * overlay[mask] + .5 * np.array(color)).astype(np.uint8)
        pair = cv2.cvtColor(np.hstack((f['rgb'], overlay)), cv2.COLOR_RGB2BGR)
        text = 'M %.0f%%  S %.0f%%  I %.0f%%' % tuple(100 * (f['tier'] == t).mean() for t in (1, 2, 3))
        cv2.putText(pair, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        cv2.imwrite(str(out / f'hybrid_{k:02d}_{f["name"]}'), pair)


def write_quality_page(ws, report):
    rows = report['coverage_mean']
    images = sorted((ws / 'quality').glob('hybrid_*.jpg'))
    cards = ''.join(f'<figure><img src="{html.escape(path.name)}" loading="lazy">'
                    f'<figcaption>{html.escape(path.name)}</figcaption></figure>' for path in images)
    page = ('''<!doctype html><meta charset="utf-8"><title>Hybrid coverage</title>
<style>body{margin:2rem auto;max-width:1100px;background:#101820;color:#ecf4fb;font:16px system-ui}
a{color:#8bdcff}img{width:100%;height:auto}figure{margin:1.5rem 0;padding:1rem;background:#1d2b36}
small{color:#b9cbda}</style><h1>Hybrid coverage audit</h1>'''
            f'<p>Measured {rows["measured"]:.1%} · prior-consistent {rows["supported"]:.1%} '
            f'· inferred planes {rows["inferred"]:.1%}</p>'
            '<p>These are mean image-pixel fractions. Geometry accuracy is unvalidated. '
            'Blue prior pixels can agree with other learned priors without independent measurement.</p>'
            '<p><a href="../hybrid_dense_report.json">Full JSON report</a> · '
            '<a href="../mesh_viewer.html">Colored mesh viewer</a></p>' + cards)
    (ws / 'quality/index.html').write_text(page)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def run_hybrid(ws, args):
    import open3d as o3d
    import pycolmap
    from scipy.spatial import cKDTree
    started = time.monotonic()
    o3d.utility.random.seed(0)
    rng = np.random.default_rng(0)
    model = pycolmap.Reconstruction(ws / 'sparse')
    by_name = {image.name: image for image in model.images.values()}
    points = model.points3D
    names = sorted((n for n in reference_names(ws) if n in by_name), key=frame_id)

    measured, aligned, info, meta = {}, {}, {}, {}
    samples = []
    for name in names:
        depth_path = ws / 'stereo/depth_maps' / (name + '.geometric.bin')
        prior_path = ws / 'prior_depth' / (name + '.npy')
        image_path = ws / 'images' / name
        if not (depth_path.exists() and prior_path.exists() and image_path.exists()):
            continue
        m = read_array(depth_path).astype(np.float32)
        m[~np.isfinite(m)] = 0
        K = camera_matrix(model.cameras[by_name[name].camera_id], m.shape)
        result = align_frame(np.load(prior_path).astype(np.float32), m, points, by_name[name], K, args, rng)
        info[name] = {k: v for k, v in result.items() if k != 'depth'}
        measured[name] = m
        aligned[name] = result['depth'] if result['ok'] else np.zeros_like(m)
        bgr = cv2.resize(cv2.imread(str(image_path)), (m.shape[1], m.shape[0]))
        meta[name] = {'K': K, 'P': pose4(by_name[name]), 'rgb': cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)}
        samples.append(m[m > 0][::97])
    present = list(measured)
    if not present or not len(np.concatenate(samples)):
        raise RuntimeError('No measured depth; run PatchMatch first')
    median_depth = float(np.median(np.concatenate(samples)))

    combined = {n: np.where(measured[n] > 0, measured[n], aligned[n]) for n in present}
    frames = []
    for name in present:
        m, a = measured[name], aligned[name]
        candidate = (a > 0) & (m <= 0) & ~edge_invalid(a)
        near = sorted((n for n in present if n != name),
                      key=lambda n: abs(frame_id(n) - frame_id(name)))[:args.neighbours]
        supported = cross_view(candidate, a, by_name[name], meta[name]['K'],
                               [by_name[n] for n in near], combined, model, args)
        tier = np.zeros(m.shape, np.uint8)
        tier[m > 0] = 1
        tier[supported] = 2
        depth = np.where(tier == 1, m, np.where(tier == 2, a, 0)).astype(np.float32)
        frames.append({'name': name, 'depth': depth, 'tier': tier, **meta[name]})
    del combined, aligned

    voxel = args.voxel_fraction * median_depth
    trunc = args.depth_trunc_factor * median_depth
    full, meas_only = make_volume(voxel), make_volume(voxel)
    integrate(full, frames, [1], trunc, repeat=args.measured_weight)
    integrate(full, frames, [2], trunc)
    integrate(meas_only, frames, [1], trunc)
    mesh_cleanup = {}
    mesh = clean_mesh(full.extract_triangle_mesh(), mesh_cleanup)
    if len(mesh.triangles) == 0:
        raise RuntimeError('TSDF produced an empty mesh')
    o3d.io.write_triangle_mesh(str(ws / 'scene_hybrid_mesh.ply'), mesh)
    export_glb(mesh, ws / 'scene_hybrid_mesh.glb')
    measured_points = np.asarray(meas_only.extract_point_cloud().points)
    vertices = np.asarray(mesh.vertices)
    if len(measured_points):
        distance, _ = cKDTree(measured_points).query(vertices, distance_upper_bound=2 * voxel)
        is_measured = np.isfinite(distance)
    else:
        is_measured = np.zeros(len(vertices), bool)
    provenance = o3d.geometry.TriangleMesh(mesh)
    provenance.vertex_colors = o3d.utility.Vector3dVector(
        np.where(is_measured[:, None], [.24, .78, .31], [.24, .51, .94]))
    o3d.io.write_triangle_mesh(str(ws / 'scene_hybrid_provenance.ply'), provenance)

    # Global planes must originate from measured geometry. Learned prior points
    # can extend a plane, but cannot establish one by mutual self-consistency.
    cloud = meas_only.extract_point_cloud().voxel_down_sample(2 * voxel)
    planes = global_planes(cloud, 1.5 * voxel, args.max_planes, args.plane_min_fraction)
    inferred_pixels = sum(infer_frame(f, planes, args, median_depth) for f in frames)
    inferred_path = ws / 'scene_hybrid_planes.glb'
    inferred_faces = 0
    if inferred_pixels:
        inferred_volume = make_volume(voxel)
        integrate(inferred_volume, frames, [3], trunc)
        inferred_mesh = inferred_volume.extract_triangle_mesh()
        inferred_faces = len(inferred_mesh.triangles)
        if inferred_faces:
            inferred_mesh.paint_uniform_color([1., .75, .12])
            export_glb(inferred_mesh, inferred_path)
    if not inferred_faces:
        inferred_path.unlink(missing_ok=True)

    per_frame = []
    for f in frames:
        total = f['tier'].size
        per_frame.append({'frame': f['name'],
                          'measured': float((f['tier'] == 1).sum() / total),
                          'supported': float((f['tier'] == 2).sum() / total),
                          'inferred': float((f['tier'] == 3).sum() / total)})
    mean = {k: float(np.mean([r[k] for r in per_frame])) for k in ('measured', 'supported', 'inferred')}
    measured_plus_supported = mean['measured'] + mean['supported']
    with_inferred = measured_plus_supported + mean['inferred']
    write_previews(ws, frames, args)
    report = {'status': 'complete', 'median_measured_depth': median_depth, 'voxel': voxel,
              'frames': len(frames), 'alignment': info,
              'frames_alignment_failed': [n for n, r in info.items() if not r['ok']],
              'coverage_mean': mean, 'coverage_measured_plus_supported': measured_plus_supported,
              'coverage_with_inferred': with_inferred, 'coverage_target': args.coverage_target,
              'planes': [{'inliers': p['inliers'], 'normal': p['normal'].tolist(),
                          'offset': p['offset']} for p in planes],
              'inferred_faces': inferred_faces, 'per_frame': per_frame,
              'mesh_cleanup': mesh_cleanup,
              'capture_diagnostics': capture_diagnostics(present, by_name, median_depth),
              'quality_status': ('broad_coverage' if measured_plus_supported >= args.coverage_target
                                 else 'incomplete'),
              'geometry_accuracy_validated': False,
              'note': 'Only tier 1 is measured. Tier 2 is a metric-prior estimate aligned '
                      'to measured depth, then checked against neighboring measured/prior maps '
                      'with a camera-baseline gate; agreement between learned priors does not '
                      'establish true depth. Tier 3 is inferred from global planes.',
              'elapsed_seconds': time.monotonic() - started}
    (ws / 'hybrid_dense_report.json').write_text(json.dumps(report, indent=2) + '\n')
    write_quality_page(ws, report)
    print(json.dumps({k: report[k] for k in ('coverage_mean', 'coverage_measured_plus_supported',
                                              'coverage_with_inferred', 'quality_status')}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--stage', choices=('prior', 'fuse', 'all'), default='all')
    p.add_argument('--prior-size', type=int, default=518, help='Long side for the prior (4 GB GPU: 448-644)')
    p.add_argument('--prior-model', default=PRIOR_MODEL)
    p.add_argument('--min-anchors', type=int, default=400)
    p.add_argument('--max-align-error', type=float, default=.06, help='Held-out median relative error')
    p.add_argument('--tol', type=float, default=.035, help='Relative depth agreement')
    p.add_argument('--min-agree', type=int, default=2)
    p.add_argument('--neighbours', type=int, default=6)
    p.add_argument('--voxel-fraction', type=float, default=.01, help='TSDF voxel / median depth')
    p.add_argument('--depth-trunc-factor', type=float, default=3.)
    p.add_argument('--measured-weight', type=int, default=2)
    p.add_argument('--max-planes', type=int, default=8)
    p.add_argument('--plane-min-fraction', type=float, default=.03)
    p.add_argument('--plane-margin-fraction', type=float, default=.5)
    p.add_argument('--extrapolation-px', type=float, default=150)
    p.add_argument('--coverage-target', type=float, default=.9)
    p.add_argument('--previews', type=int, default=12)
    args = p.parse_args()
    ws = args.workspace.resolve()
    if args.stage == 'prior':
        run_prior(ws, args.prior_size, args.prior_model)
        return
    if args.stage == 'all':
        # separate process so PatchMatch/prior GPU memory is fully released
        # Launch as a module: running this file directly from adaptive_uav/
        # places its types.py ahead of the standard-library types module.
        subprocess.run([sys.executable, '-m', 'adaptive_uav.hybrid_dense', '--workspace', str(ws),
                        '--stage', 'prior', '--prior-size', str(args.prior_size),
                        '--prior-model', args.prior_model], check=True)
    run_hybrid(ws, args)


if __name__ == '__main__':
    main()
