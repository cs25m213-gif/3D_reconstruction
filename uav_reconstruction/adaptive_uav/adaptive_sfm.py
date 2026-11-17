"""Bounded visual profiling and rule-based sparse reconstruction control."""
import math
import sqlite3
from pathlib import Path

import cv2
import numpy as np

PAIR_BASE = 2147483647


def read_profile_frame(cap, index, max_forward):
    """Avoid decoding the same GOP repeatedly for nearby increasing frame indices."""
    position = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    if index < position or index - position > max_forward:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    else:
        for _ in range(index - position):
            if not cap.grab():
                return False, None
    return cap.read()


def profile_video(video, analysis_width=360, rotation='none', samples=48):
    """Sample short windows across the video; all motion is in analysis pixels.

    SIFT counts and homography inliers are difficulty proxies, not evidence of
    triangulatable geometry. Sensor synchronization is deliberately not assumed.
    """
    cap = cv2.VideoCapture(str(video))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or count < 2:
            raise ValueError('Cannot profile video with invalid FPS/frame count')
        sift = cv2.SIFT_create(nfeatures=4000)
        records, flow, decay = [], [], {lag: [] for lag in (.2, .5, 1., 2.)}

        def read(index):
            ok, image = read_profile_frame(cap, index, max(8, round(2 * fps)))
            if not ok:
                return None
            if rotation in ('cw', 'ccw'):
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if rotation == 'cw'
                                   else cv2.ROTATE_90_COUNTERCLOCKWISE)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.resize(gray, (analysis_width, max(1, round(gray.shape[0]*analysis_width/gray.shape[1]))))

        for index in np.unique(np.linspace(0, max(0, count-1-2*fps), max(1, samples//5)).astype(int)):
            gray = read(int(index))
            if gray is None:
                continue
            keys, desc = sift.detectAndCompute(gray, None)
            records.append({'blur': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                            'features': len(keys), 'illumination': float(gray.mean()),
                            'texture': float(gray.std())})
            target_cache = {}
            for lag in decay:
                target = int(index + max(1, round(lag*fps)))
                if target >= count:
                    continue
                if target in target_cache:
                    other, score = target_cache[target]
                    decay[lag].append(score)
                    continue
                other = read(target)
                if other is None:
                    continue
                if lag == .2:
                    corners = cv2.goodFeaturesToTrack(gray, 350, .015, 8)
                    if corners is not None:
                        tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray, other, corners, None)
                        if tracked is not None and status.any():
                            good = status.ravel() == 1
                            flow.append(float(np.median(np.linalg.norm(tracked[good]-corners[good], axis=2))))
                k2, d2 = sift.detectAndCompute(other, None)
                score = 0.
                if desc is not None and d2 is not None and len(d2) >= 2:
                    matches = [a for pair in cv2.BFMatcher().knnMatch(desc, d2, k=2)
                               if len(pair) == 2 for a, b in [pair] if a.distance < .75*b.distance]
                    if len(matches) >= 8:
                        src = np.float32([keys[m.queryIdx].pt for m in matches])
                        dst = np.float32([k2[m.trainIdx].pt for m in matches])
                        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.)
                        if mask is not None:
                            score = float(mask.sum()/max(1, min(len(keys), len(k2))))
                target_cache[target] = (other, score)
                decay[lag].append(score)
        if not records:
            raise ValueError('No decodable profiling frames')
        return {'version': 1, 'analysis_width': analysis_width, 'fps': fps,
                'duration_seconds': count/fps, 'sampled_windows': len(records),
                'median_blur': float(np.median([r['blur'] for r in records])),
                'median_sift_features': float(np.median([r['features'] for r in records])),
                'median_texture': float(np.median([r['texture'] for r in records])),
                'illumination_std': float(np.std([r['illumination'] for r in records])),
                'median_flow_px': float(np.median(flow)) if flow else None,
                'p90_flow_px': float(np.percentile(flow, 90)) if flow else None,
                'matchability': {str(lag): float(np.median(values)) if values else None
                                 for lag, values in decay.items()},
                'gyro_used': False, 'geometry_validated': False}
    finally:
        cap.release()


def choose_settings(profile):
    curve = profile['matchability']
    reliable = [float(lag) for lag, score in curve.items() if score is not None and score >= .15]
    gap = min(2., max(.2, .8*max(reliable))) if reliable else .3
    features = profile['median_sift_features']
    weak = features < 600 or not reliable
    flow = profile.get('median_flow_px')
    return {'max_features': 16000 if weak else 8000,
            'min_motion_pixels': float(np.clip(.5*flow if flow is not None else 6., 2., 30.)),
            'max_keyframe_gap_seconds': gap,
            'sample_fps': float(min(12., max(3., math.ceil(2/gap)))),
            'min_blur_score': float(np.clip(.35*profile['median_blur'], 10., 45.)),
            'temporal_references': 18 if weak else 12,
            'retrieval_references': 8 if weak else 4,
            'guided_matching': True}


def match_graph(database, min_inliers=30):
    """Inspect verified COLMAP edges, including isolated database images."""
    database = Path(database)
    if not database.is_file():
        return {'available': False}
    with sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True) as db:
        images = dict(db.execute('SELECT image_id, name FROM images'))
        edges = list(db.execute('SELECT pair_id, rows FROM two_view_geometries WHERE rows >= ?', (min_inliers,)))
    neighbors = {i: set() for i in images}
    inliers = []
    for pair, rows in edges:
        a, b = divmod(pair, PAIR_BASE)
        if a in neighbors and b in neighbors:
            neighbors[a].add(b)
            neighbors[b].add(a)
            inliers.append(rows)
    components, labels = [], {}
    for node in neighbors:
        if node in labels:
            continue
        label, stack, members = len(components), [node], []
        labels[node] = label
        while stack:
            current = stack.pop()
            members.append(images[current])
            for other in neighbors[current]:
                if other not in labels:
                    labels[other] = label
                    stack.append(other)
        components.append(sorted(members))
    ordered = sorted(images, key=lambda i: images[i])
    return {'available': True, 'components': components, 'component_count': len(components),
            'image_count': len(images), 'strong_edges': len(inliers), 'min_inliers': min_inliers,
            'median_inliers': float(np.median(inliers)) if inliers else 0.,
            'temporal_breaks': [[images[a], images[b]] for a, b in zip(ordered, ordered[1:])
                                if labels[a] != labels[b]],
            'geometry_validated': False}


def quality_passes(metrics, graph):
    registration = metrics.get('registration_fraction', 0)
    error = metrics.get('median_reprojection_error_px')
    return (isinstance(registration, (int, float)) and math.isfinite(registration)
            and registration >= .8 and isinstance(error, (int, float))
            and math.isfinite(error) and 0 <= error <= 2
            and metrics.get('registered_frames', 0) >= 5
            and metrics.get('sparse_points', 0) >= 150
            and graph.get('available', False) and graph.get('component_count') == 1)


def recovery_settings(config, metrics):
    """Retry connectivity failures; high reprojection error requires calibration review."""
    error = metrics.get('median_reprojection_error_px')
    if isinstance(error, (float, int)) and (not math.isfinite(error) or error > 2):
        return None
    return {**config, 'max_keyframe_gap_seconds': max(.15, config['max_keyframe_gap_seconds']*.6),
            'min_motion_pixels': max(1., config['min_motion_pixels']*.6),
            'sample_fps': min(15., config['sample_fps']*1.5),
            'temporal_references': min(36, config['temporal_references']+8),
            'retrieval_references': min(16, config['retrieval_references']+4),
            'max_features': min(16000, config['max_features']*2)}
