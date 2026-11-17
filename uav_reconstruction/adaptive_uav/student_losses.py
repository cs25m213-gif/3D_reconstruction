"""Student loss, ordered by trust: LiDAR, map, multiview geometry, photometric, then teacher.

Every term is safe on an empty mask, and a lower-trust target only applies where no higher-trust
target exists, so the teacher can never overrule a measurement. Confidence is calibrated on real
targets only and cannot lower the depth loss by hiding hard pixels.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .train_depth_student import photometric_loss

DEFAULT_WEIGHTS = {'measured': 2., 'neighbour': 1., 'map': 1., 'anchor': .2, 'multiview': .2, 'photometric': .05,
                   'smooth': .01, 'teacher': .05, 'confidence': .05, 'visibility': .02}


def masked_log(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return prediction.sum() * 0
    return (prediction.clamp_min(1e-4).log() - target.clamp_min(1e-4).log()).abs()[mask].mean()


def edge_smoothness(depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    normalized = depth / depth.mean((2, 3), keepdim=True).clamp_min(1e-4)
    dx, dy = (normalized[..., 1:] - normalized[..., :-1]).abs(), (normalized[..., 1:, :] - normalized[..., :-1, :]).abs()
    gx = (rgb[..., 1:] - rgb[..., :-1]).abs().mean(1, keepdim=True)
    gy = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(1, keepdim=True)
    return (dx * torch.exp(-gx * 10)).mean() + (dy * torch.exp(-gy * 10)).mean()


def supervised_masks(batch: dict) -> dict:
    measured = batch['measured_valid']
    neighbour = batch['neighbour_valid'] & ~measured
    map_valid = batch['map_valid'] & ~measured & ~neighbour
    known = measured | neighbour | map_valid
    anchor = (batch['anchor'] > .25) & (batch['anchor'] < 4.) & ~known
    teacher = batch['teacher_valid'] & ~known & ~anchor
    return {'measured': measured, 'neighbour': neighbour, 'map': map_valid, 'anchor': anchor, 'teacher': teacher}


def student_loss(batch: dict, depth: torch.Tensor, confidence: torch.Tensor, weights: dict, prior=None, visibility=None) -> tuple:
    """batch tensors are [B,1,H,W] for one frame; rgb, K and P are [B,V,...]."""
    masks = supervised_masks(batch)
    parts = {name: masked_log(depth, batch[name], mask) for name, mask in masks.items()}
    parts['photometric'] = photometric_loss(batch['rgb'], batch['K'], batch['P'], depth) if weights['photometric'] else depth.sum() * 0
    parts['smooth'] = edge_smoothness(depth, batch['rgb'][:, 0]) if weights['smooth'] else depth.sum() * 0
    known = masks['measured'] | masks['neighbour'] | masks['map']
    if known.any():
        best = torch.where(masks['measured'], batch['measured'], torch.where(masks['neighbour'], batch['neighbour'], batch['map']))
        error = (depth.detach().clamp_min(1e-4).log() - best.clamp_min(1e-4).log()).abs()
        target = torch.exp(-5 * error).clamp(0, 1)
        probability = confidence[known].clamp(1e-5, 1 - 1e-5)
        parts['confidence'] = -(target[known] * probability.log() + (1 - target[known]) * (1 - probability).log()).mean()
    else:
        parts['confidence'] = confidence.sum() * 0
    parts['multiview'], parts['visibility'] = depth.sum() * 0, depth.sum() * 0
    if prior is not None:
        warped, warped_valid = prior
        agree = (warped_valid > 0) & ((depth.detach().clamp_min(1e-4).log() - warped.clamp_min(1e-4).log()).abs() < .2)
        parts['multiview'] = masked_log(depth, warped, agree)
        if visibility is not None:
            parts['visibility'] = F.binary_cross_entropy(visibility.clamp(1e-5, 1 - 1e-5), agree.float())
    total = sum(weights.get(name, 0.) * value for name, value in parts.items())
    logged = {name: float(value.detach()) for name, value in parts.items()}
    logged.update({f'pixels_{name}': int(mask.sum()) for name, mask in masks.items()})
    return total, logged
