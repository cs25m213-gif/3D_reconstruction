"""Measured per-frame contribution to the fused reconstruction, plus DINO novelty bins.

Contribution is not "this frame looks different": it is the change in fused-surface F-score against
the fixed LiDAR reference when the frame is added.

* shapley: truncated Monte Carlo Shapley value. Random frame orders (each paired with its reverse to
  cut variance); every frame is integrated in turn into one TSDF volume with fixed reference poses
  and the F-score gain is its marginal contribution. Redundant frames share credit instead of both
  looking worthless, which leave-one-out cannot do. shapley_se is its standard error: differences
  smaller than about 2 SE are not evidence.
* loo: leave-one-out, F(all) - F(all without frame).

Depth for every frame uses fixed dense-spacing sources (contract keyframe ratio, past-only), so a
frame's value is its own surface contribution. The effect a frame has as a depth source for other
frames is not included.

DINO bins (frozen dinov2_vits14 from embedding_cache): a frame joins the most similar existing bin
if cosine similarity to its protected prototype is >= tau, otherwise it opens a new bin. A bin's
prototype is its first accepted frame and is never rewritten; revisits rejoin old bins. Frames
failing the sharpness gate get bin -1 and never update a prototype.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .counterfactual_replay import Clip, calibration_scales, eligible, role_of
from .embedding_cache import Descriptors
from .hybrid_dense import integrate, make_volume
from .reconstruction_quality import ReferencePopulation, surface_metrics
from .student_inference import load_student
from .utility_contracts import load_contract, sha256_file, write_json

TAUS = (.80, .85, .90, .95)


def load_clip(root: Path, contract: dict, name: str, student, device: str, diagnostic: bool = False):
    """Clip with calibrated, fixed-context student depth and its LiDAR reference."""
    data = contract['data']
    gate_path = root / contract['geometry_gate']
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    cache_root = root / data['cache_root']
    area = name.rstrip('0123456789_')
    scales = calibration_scales(root, contract, sorted(p.name for p in cache_root.iterdir()
                                                       if p.name.startswith(area) and (p / 'selected_infos.json').exists()))
    clip = Clip(root, contract, name)
    status = eligible(gate, clip.scene, diagnostic)
    if status is None or clip.scene not in scales or not len(clip.points):
        raise ValueError(f'{name}: reference not eligible or no calibration')
    clip.apply_scale(scales[clip.scene])
    clip.choose_keyframes(contract['utility']['keyframe_baseline_ratio'])
    clip.use_student(student, device)
    for view in clip.views:  # same calibration fix as failure_attribution
        view.fusion_depth = view.fusion_depth * clip.depth_scale
        view.depth_m = view.depth_m * clip.depth_scale
    settings = contract['utility']
    reference = ReferencePopulation(clip.points, settings['tolerance_m'], status, settings['quality_protocol'])
    return clip, reference


def add(volume, view) -> None:
    integrate(volume, [{'depth': view.fusion_depth, 'tier': (view.fusion_depth > 0).astype(np.uint8), 'rgb': view.fusion_rgb,
                        'K': view.fusion_k, 'P': np.linalg.inv(view.reference_pose)}], [1], 250.)


def value(views: list, reference, voxel: float) -> float:
    volume = make_volume(voxel)
    for view in views:
        add(volume, view)
    return surface_metrics(np.asarray(volume.extract_point_cloud().points), reference)['f_score']


def shapley(views: list, reference, voxel: float, permutations: int, seed: int) -> tuple:
    n, rng = len(views), np.random.default_rng(seed)
    gains = []
    for _ in range(max(1, permutations // 2)):
        order = rng.permutation(n)
        for sequence in (order, order[::-1]):
            volume, previous, row = make_volume(voxel), 0., np.zeros(n)
            for i in sequence:
                add(volume, views[i])
                current = surface_metrics(np.asarray(volume.extract_point_cloud().points), reference)['f_score']
                row[i], previous = current - previous, current
            gains.append(row)
    gains = np.asarray(gains)
    return gains.mean(0), gains.std(0, ddof=1) / np.sqrt(len(gains)), len(gains)


def runtime_features(clip, descriptors: Descriptors) -> dict:
    """Only what a deployed selector can see: image, embedding, relative pose. No LiDAR."""
    views = clip.views
    emb = np.stack([descriptors.get(clip.name, v.index, v.rgb) for v in views]).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    grey = [cv2.cvtColor(v.rgb, cv2.COLOR_RGB2GRAY) for v in views]
    sharp = np.array([cv2.Laplacian(g, cv2.CV_64F).var() for g in grey])
    bright = np.array([g.mean() / 255. for g in grey])
    baseline = np.zeros(len(views))
    for i in range(1, len(views)):
        baseline[i] = clip.spacing(i - 1, i)
    novelty = np.array([1. - (emb[max(0, i - 5):i] @ emb[i]).max() if i else 1. for i in range(len(views))])
    return {'embedding': emb, 'sharpness': sharp, 'brightness': bright, 'baseline_prev': baseline, 'novelty': novelty}


def novelty_bins(emb: np.ndarray, sharp: np.ndarray, tau: float, gate_quantile: float = .1) -> np.ndarray:
    gate = np.quantile(sharp, gate_quantile)
    prototypes, bins = [], np.full(len(emb), -1)
    for i, e in enumerate(emb):
        if sharp[i] < gate:
            continue  # null slot: rejected frames never create or update a bin
        similarity = np.array([p @ e for p in prototypes]) if prototypes else np.array([])
        if len(similarity) and similarity.max() >= tau:
            bins[i] = int(similarity.argmax())
        else:
            prototypes.append(e)  # protected: first accepted frame, never rewritten
            bins[i] = len(prototypes) - 1
    return bins


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--clip', required=True)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--embeddings', type=Path, required=True, help='embedding_cache output folder (dinov2_vits14)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--permutations', type=int, default=24)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--diagnostic', action='store_true')
    a = p.parse_args()
    target = a.output / f'{a.clip}.npz'
    if target.exists():
        return
    started = time.monotonic()
    contract = load_contract(a.config, ('utility.quality_protocol',))
    if a.clip.endswith(contract['geometry']['calibration_batch_suffix']):
        raise SystemExit('Calibration clips fit the scale, so they cannot also be scored')
    clip, reference = load_clip(a.root.resolve(), contract, a.clip, load_student(a.student, a.device), a.device, a.diagnostic)
    voxel, views = contract['utility']['voxel_m'], clip.views
    full = value(views, reference, voxel)
    loo = np.array([full - value(views[:i] + views[i + 1:], reference, voxel) for i in range(len(views))])
    mean, se, draws = shapley(views, reference, voxel, a.permutations, a.seed)
    features = runtime_features(clip, Descriptors(a.embeddings))
    bins = {f'bins_tau{t:.2f}': novelty_bins(features['embedding'], features['sharpness'], t) for t in TAUS}
    a.output.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + '.partial.npz')
    np.savez_compressed(temporary, frame_index=np.array([v.index for v in views]), shapley=mean, shapley_se=se, loo=loo,
                        f_full=full, centres=np.array([v.reference_pose[:3, 3] for v in views]), **features, **bins)
    temporary.replace(target)
    write_json(a.output / f'{a.clip}.json', {'clip': a.clip, 'flight': clip.scene, 'split': role_of(clip.scene, contract),
               'frames': len(views), 'f_full': full, 'shapley_sum': float(mean.sum()), 'draws': draws,
               'median_se': float(np.median(se)), 'frames_above_2se': int((mean > 2 * se).sum()),
               'bins_per_tau': {k: int(v.max() + 1) for k, v in bins.items()}, 'descriptor': Descriptors(a.embeddings).name,
               'student_sha256': sha256_file(a.student), 'config_sha256': sha256_file(a.config),
               'depth': 'fixed dense-spacing past-only sources; source effect on other frames excluded',
               'poses': 'fixed reference poses', 'seconds': round(time.monotonic() - started, 1)})


if __name__ == '__main__':
    main()
