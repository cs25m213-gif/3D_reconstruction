"""QualityStudent: the depth student with calibrated quality outputs and early exits.

Shared encoder (DepthStudent, 48 ch at 1/4 resolution)
  -> Exit 1  frame-quality head from the reference image features and source-camera geometry only
             (runs before the cost volume: no depth is computed for the candidate)
  -> 24-plane cost volume -> depth (unchanged DepthStudent path)
  -> per-pixel Laplace log-scale s of the log-depth error (replaces the old confidence logit;
     confidence = exp(-5 * exp(s)) keeps the previous meaning, the expected exp(-5 |log error|))
  -> Exit 2  frame-quality head refined with cost-volume entropy and pixel-uncertainty statistics

Each quality head predicts (mu, log variance) of y = log(median |log depth - log truth| + 1e-3),
the frame's own depth error, which drives how much the frame can help a fused map.
forward() keeps the DepthStudent interface (depth, confidence); forward_full() returns everything.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .depth_student import DepthStudent, conv2

N_GEOMETRY = 8   # per source: baseline / anchor depth, rotation (rad), used flag; then sources used / 2, any source


def source_geometry(P: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """P [B,V,4,4] world-to-camera relative to the reference (translations already / anchor scale)."""
    rows = []
    for s in range(1, P.shape[1]):
        relative = P[:, s].float() @ torch.linalg.inv(P[:, 0].float())
        rotation = relative[:, :3, :3]
        cos = ((rotation.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)
        # camera centre of the source in the reference frame
        centre = -(rotation.transpose(1, 2) @ relative[:, :3, 3:4]).squeeze(-1)
        used = weights[:, s - 1]
        rows += [centre.norm(dim=1) * used, torch.arccos(cos) * used, used]
    used = weights.sum(1)
    rows += [used / max(P.shape[1] - 1, 1), (used > 0).float()]
    return torch.stack(rows, 1)


def _pool(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x.mean((2, 3)), x.flatten(2).std(2)), 1)


class QualityStudent(DepthStudent):
    architecture_id = 'quality_student_48ch_24planes_exit12_v1'

    def __init__(self, channels=48, planes=24):
        super().__init__(channels, planes)
        self.exit1_tower = nn.Sequential(conv2(channels, 64, 2), conv2(64, 96, 2), conv2(96, 128, 2))
        self.exit1_head = nn.Sequential(nn.Linear(256 + N_GEOMETRY, 128), nn.SiLU(), nn.Dropout(.1),
                                        nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))
        self.exit2_head = nn.Sequential(nn.Linear(256 + N_GEOMETRY + 8, 128), nn.SiLU(), nn.Dropout(.1),
                                        nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))
        self.scale_bias = nn.Parameter(torch.tensor(-2.))
        self.entropy_weight = nn.Parameter(torch.tensor(1.))

    def exit1(self, features0: torch.Tensor, geometry: torch.Tensor):
        pooled = _pool(self.exit1_tower(features0))
        out = self.exit1_head(torch.cat((pooled, geometry), 1))
        return out[:, 0], out[:, 1].clamp(-8, 6), pooled

    def forward_full(self, rgb, K, P, source_weights=None):
        b, v, _, h, w = rgb.shape
        if v < 2:
            raise ValueError('At least two posed views are required')
        weights = torch.ones((b, v - 1), device=rgb.device) if source_weights is None else source_weights
        geometry = source_geometry(P, weights)
        features = self.encoder(rgb.reshape(b * v, 3, h, w)).reshape(b, v, -1, h // 4, w // 4)
        hf, wf = features.shape[-2:]
        mu1, logvar1, pooled = self.exit1(features[:, 0], geometry)
        k = K.clone(); k[:, :, 0] *= wf / w; k[:, :, 1] *= hf / h
        y, x = torch.meshgrid(torch.arange(hf, device=rgb.device), torch.arange(wf, device=rgb.device), indexing='ij')
        xy = torch.stack((x, y, torch.ones_like(x)), 0).float().reshape(1, 1, 3, hf * wf)
        rays = torch.linalg.solve(k[:, 0].float(), xy.expand(b, 1, -1, -1).squeeze(1))
        reference = features[:, 0]
        accum = torch.zeros((b, 3, self.planes, hf, wf), device=rgb.device, dtype=torch.float32)
        for source in range(1, v):
            relative = P[:, source].float() @ torch.linalg.inv(P[:, 0].float())
            xyz = rays[:, None] * self.depth_planes[None, :, None, None]
            xyz = relative[:, None, :3, :3] @ xyz + relative[:, None, :3, 3, None]
            projected = k[:, source, None].float() @ xyz
            z = projected[:, :, 2].clamp_min(1e-6)
            u = projected[:, :, 0] / z; vv = projected[:, :, 1] / z
            valid = ((xyz[:, :, 2] > 1e-6) & (u >= 0) & (u < wf) & (vv >= 0) & (vv < hf)).reshape(b, self.planes, hf, wf)
            grid = torch.stack((2 * u / max(wf - 1, 1) - 1, 2 * vv / max(hf - 1, 1) - 1), -1).reshape(b * self.planes, hf, wf, 2)
            src = features[:, source, None].expand(-1, self.planes, -1, -1, -1).reshape(b * self.planes, -1, hf, wf)
            warped = F.grid_sample(src.float(), grid.float(), align_corners=True, padding_mode='zeros').reshape(b, self.planes, -1, hf, wf)
            diff = reference[:, None].float() - warped.float()
            weight = weights[:, source - 1, None, None, None]
            accum[:, 0] += (-diff.square().mean(2) * valid).float() * weight
            accum[:, 1] += (reference[:, None].float() * warped.float()).mean(2) * valid * weight
            accum[:, 2] += valid.float() * weight
        accum = accum / weights.sum(1).clamp_min(1e-6)[:, None, None, None, None]
        logits = self.volume(accum.to(rgb.dtype)).squeeze(1).float()
        probabilities = logits.softmax(1)
        low = (probabilities * self.depth_planes[None, :, None, None]).sum(1, keepdim=True)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1, keepdim=True)
        guidance = torch.cat((reference.float(), F.interpolate(rgb[:, 0].float(), size=(hf, wf), mode='bilinear', align_corners=False), low), 1)
        residual = self.refine(guidance.to(rgb.dtype)).float()
        log_scale_low = residual[:, 1:2] + self.entropy_weight * entropy / math.log(self.planes) + self.scale_bias
        depth = F.interpolate(low * torch.exp(residual[:, :1].tanh() * .5), size=(h, w), mode='bilinear', align_corners=False)
        log_scale = F.interpolate(log_scale_low, size=(h, w), mode='bilinear', align_corners=False).clamp(-8, 4)
        confidence = torch.exp(-5 * torch.exp(log_scale))
        coverage = accum[:, 2].mean(1)  # fraction of source support per pixel
        stats = torch.stack((entropy.mean((1, 2, 3)) / math.log(self.planes), entropy.flatten(1).std(1) / math.log(self.planes),
                             log_scale_low.mean((1, 2, 3)), log_scale_low.flatten(1).std(1),
                             log_scale_low.flatten(1).quantile(.9, dim=1), probabilities.amax(1).mean((1, 2)),
                             coverage.mean((1, 2)), low.clamp_min(1e-4).log().flatten(1).std(1)), 1)
        out2 = self.exit2_head(torch.cat((pooled, geometry, stats.detach()), 1))
        return {'depth': depth, 'confidence': confidence, 'log_scale': log_scale,
                'q1': (mu1, logvar1), 'q2': (out2[:, 0], out2[:, 1].clamp(-8, 6)), 'geometry': geometry}

    def forward(self, rgb, K, P, source_weights=None):
        out = self.forward_full(rgb, K, P, source_weights)
        return out['depth'], out['confidence']
