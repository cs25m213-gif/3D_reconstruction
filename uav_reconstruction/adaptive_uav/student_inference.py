"""Run a saved student on arbitrary views in float32, with its normalization contract."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .depth_student import DepthStudent

GRID = (192, 256)
MIN_ANCHORS = 30


def anchor_scale(anchors: np.ndarray) -> float:
    """An operational scale from RGB/SfM anchors, never teacher or measured depth."""
    if anchors is None:
        raise ValueError('Sparse RGB/SfM anchors are required; teacher depth is not a scale fallback')
    anchors = np.asarray(anchors)
    valid = anchors[np.isfinite(anchors) & (anchors > 0)]
    if valid.size < MIN_ANCHORS:
        raise ValueError(f'Need at least {MIN_ANCHORS} finite positive sparse anchors')
    return float(np.median(valid))


def load_student(path, device='cpu') -> DepthStudent:
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = DepthStudent(48, 24)
    model.load_state_dict(state['model'] if 'model' in state else state)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, views: list, scale: float, device='cpu', anchors=None) -> tuple:
    """views: (rgb, K, world_from_camera), reference first. Returns metric depth and confidence."""
    h, w = GRID
    if not views or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Prediction needs views and a finite positive normalization scale')
    rows = list(views) + [views[-1]] * (3 - len(views))
    images, intrinsics, poses = [], [], []
    for rgb, k, world in rows[:3]:
        ih, iw = rgb.shape[:2]
        images.append(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).transpose(2, 0, 1).astype(np.float32) / 255.)
        fitted = np.asarray(k, np.float32).copy()
        fitted[0] *= w / iw
        fitted[1] *= h / ih
        intrinsics.append(fitted)
        p = np.linalg.inv(world).astype(np.float32)
        p[:3, 3] /= scale
        poses.append(p)
    tensors = [torch.from_numpy(np.stack(v))[None].to(device) for v in (images, intrinsics, poses)]
    depth, confidence = model(*tensors)
    result = depth[0, 0].float().cpu().numpy() * scale
    probability = confidence[0, 0].float().cpu().numpy()
    if not np.isfinite(result).all() or (result <= 0).any() or not np.isfinite(probability).all():
        raise ValueError('Student returned invalid depth or confidence')
    if anchors is not None:
        from .build_cached_lidar_targets import downsample_sparse_min
        sparse = np.asarray(anchors, np.float32)
        sparse = np.where(np.isfinite(sparse) & (sparse > 0), sparse, 0)
        if sparse.shape != result.shape:
            sparse = downsample_sparse_min(sparse, result.shape)
        valid = sparse > 0
        if int(valid.sum()) < MIN_ANCHORS:
            raise ValueError('Insufficient sparse anchors on the prediction grid')
        # Same post-inference anchor fit used by train_student_update.evaluate.
        result = result * float(np.median(sparse[valid] / result[valid]))
    return result, probability
