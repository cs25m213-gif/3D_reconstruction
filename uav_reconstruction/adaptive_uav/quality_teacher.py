"""QualityTeacher: a larger multi-view depth + quality model built on a frozen pretrained backbone.

Purpose: train once with as much data as possible, then distil its depth, pixel uncertainty and
frame quality into the small QualityStudent for deployment (the backbone never runs at inference).

  each view ─► frozen DINOv2 ViT-B/14 (pretrained on diverse images: cross-area robustness)
            │     patch tokens 768-d ─► 1x1 projection ─► upsample to 1/4 resolution
            └► trainable CNN (32/64/96 ch) at 1/4 resolution
     fuse ─► 128-ch matching features
  reference + up to 4 past sources ─► 48-plane cost volume
     per plane: 8-group correlation + mean squared difference + validity (10 channels),
     weighted average over sources (dropped/missing sources have weight 0)
     each source's warp is gradient-checkpointed, so memory does not grow with sources x planes
  ─► 3D CNN regulariser ─► soft-argmin depth, entropy
  ─► 2D refinement guided by fused features ─► depth residual + per-pixel Laplace log-scale
  Exit 1: frame quality from DINO CLS/patch statistics + CNN pooling + source geometry (no depth)
  Exit 2: Exit-1 inputs + cost-volume entropy and uncertainty statistics

forward_full() returns the same dictionary as QualityStudent.forward_full(), so the same trainer,
losses and evaluation apply to both.
"""
from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .depth_student import conv2
from .quality_student import source_geometry

DINO_REPO = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
GROUPS = 8


class QualityTeacher(nn.Module):
    architecture_id = 'quality_teacher_dinov2b_48planes_v1'

    def __init__(self, planes=48, channels=128, backbone='dinov2_vitb14', dino_size=(392, 518)):
        super().__init__()
        self.planes, self.channels, self.dino_size = planes, channels, dino_size
        self.backbone = torch.hub.load(DINO_REPO, backbone, source='local')
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        width = self.backbone.embed_dim
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))
        self.cnn = nn.Sequential(conv2(3, 32, 2), conv2(32, 64, 2), conv2(64, 96), conv2(96, 96))
        self.project = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GroupNorm(8, 128), nn.SiLU())
        self.fuse = nn.Sequential(conv2(96 + 128, channels), conv2(channels, channels))
        self.volume = nn.Sequential(nn.Conv3d(GROUPS + 2, 32, 3, padding=1), nn.SiLU(),
                                    nn.Conv3d(32, 32, 3, padding=1), nn.SiLU(),
                                    nn.Conv3d(32, 32, 3, padding=2, dilation=2), nn.SiLU(),
                                    nn.Conv3d(32, 1, 3, padding=1))
        self.refine = nn.Sequential(conv2(channels + 4, 96), conv2(96, 96), conv2(96, 96), nn.Conv2d(96, 2, 3, padding=1))
        self.register_buffer('depth_planes', torch.exp(torch.linspace(-1.3862944, 1.3862944, planes)))
        self.scale_bias = nn.Parameter(torch.tensor(-2.))
        self.entropy_weight = nn.Parameter(torch.tensor(1.))
        self.n_geometry = None  # set lazily: depends on the number of sources
        self.exit1_head = None
        self.exit2_head = None
        self.pool_width = 2 * width + 2 * 96

    def _heads(self, n_geometry, device):
        if self.exit1_head is None:
            self.n_geometry = n_geometry
            make = lambda extra: nn.Sequential(nn.Linear(self.pool_width + n_geometry + extra, 256), nn.SiLU(), nn.Dropout(.1),
                                               nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 2)).to(device)
            self.exit1_head, self.exit2_head = make(0), make(8)

    def build(self, n_sources, device):
        """Create the size-dependent heads before the optimizer is made."""
        self._heads(3 * n_sources + 2, device)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _dino(self, images):
        x = F.interpolate(images, size=self.dino_size, mode='bilinear', align_corners=False)
        out = self.backbone.forward_features((x - self.mean) / self.std)
        h, w = self.dino_size[0] // 14, self.dino_size[1] // 14
        tokens = out['x_norm_patchtokens'].transpose(1, 2).reshape(len(images), -1, h, w)
        return tokens, out['x_norm_clstoken']

    def _warp_cost(self, reference, source_features, relative, k_ref_inv_rays, k_source, hf, wf):
        b = reference.shape[0]
        xyz = k_ref_inv_rays[:, None] * self.depth_planes[None, :, None, None]
        xyz = relative[:, None, :3, :3] @ xyz + relative[:, None, :3, 3, None]
        projected = k_source[:, None] @ xyz
        z = projected[:, :, 2].clamp_min(1e-6)
        u = projected[:, :, 0] / z; v = projected[:, :, 1] / z
        valid = ((xyz[:, :, 2] > 1e-6) & (u >= 0) & (u < wf) & (v >= 0) & (v < hf)).reshape(b, self.planes, hf, wf).float()
        grid = torch.stack((2 * u / max(wf - 1, 1) - 1, 2 * v / max(hf - 1, 1) - 1), -1).reshape(b * self.planes, hf, wf, 2)
        src = source_features[:, None].expand(-1, self.planes, -1, -1, -1).reshape(b * self.planes, -1, hf, wf)
        warped = F.grid_sample(src, grid, align_corners=True, padding_mode='zeros').reshape(b, self.planes, GROUPS, -1, hf, wf)
        ref = reference.reshape(b, 1, GROUPS, -1, hf, wf)
        corr = (ref * warped).mean(3) * valid[:, :, None]                                    # [B,D,G,H,W]
        sq = (ref - warped).square().mean((2, 3)) * valid                                   # [B,D,H,W]
        return torch.cat((corr.transpose(1, 2), -sq[:, None], valid[:, None]), 1)           # [B,G+2,D,H,W]

    def forward_full(self, rgb, K, P, source_weights=None):
        b, v, _, h, w = rgb.shape
        if v < 2:
            raise ValueError('At least two posed views are required')
        weights = torch.ones((b, v - 1), device=rgb.device) if source_weights is None else source_weights
        geometry = source_geometry(P, weights)
        self._heads(geometry.shape[1], rgb.device)
        images = rgb.reshape(b * v, 3, h, w).float()
        tokens, cls = self._dino(images)
        cnn = self.cnn(images)
        hf, wf = cnn.shape[-2:]
        dino = F.interpolate(self.project(tokens.to(cnn.dtype)), size=(hf, wf), mode='bilinear', align_corners=False)
        features = self.fuse(torch.cat((cnn, dino), 1)).float().reshape(b, v, -1, hf, wf)
        k = K.clone().float(); k[:, :, 0] *= wf / w; k[:, :, 1] *= hf / h
        y, x = torch.meshgrid(torch.arange(hf, device=rgb.device), torch.arange(wf, device=rgb.device), indexing='ij')
        xy = torch.stack((x, y, torch.ones_like(x)), 0).float().reshape(1, 3, hf * wf).expand(b, -1, -1)
        rays = torch.linalg.solve(k[:, 0], xy)
        reference = features[:, 0]
        cost = 0.
        for s in range(1, v):
            relative = P[:, s].float() @ torch.linalg.inv(P[:, 0].float())
            part = checkpoint(self._warp_cost, reference, features[:, s], relative, rays, k[:, s], hf, wf, use_reentrant=False)
            cost = cost + part * weights[:, s - 1, None, None, None, None]
        cost = cost / weights.sum(1).clamp_min(1e-6)[:, None, None, None, None]
        logits = self.volume(cost).squeeze(1).float()
        probabilities = logits.softmax(1)
        low = (probabilities * self.depth_planes[None, :, None, None]).sum(1, keepdim=True)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1, keepdim=True)
        guidance = torch.cat((reference, F.interpolate(rgb[:, 0].float(), size=(hf, wf), mode='bilinear', align_corners=False), low), 1)
        residual = self.refine(guidance).float()
        log_scale_low = residual[:, 1:2] + self.entropy_weight * entropy / math.log(self.planes) + self.scale_bias
        depth = F.interpolate(low * torch.exp(residual[:, :1].tanh() * .5), size=(h, w), mode='bilinear', align_corners=False)
        log_scale = F.interpolate(log_scale_low, size=(h, w), mode='bilinear', align_corners=False).clamp(-8, 4)
        confidence = torch.exp(-5 * torch.exp(log_scale))
        tok0 = tokens.reshape(b, v, *tokens.shape[1:])[:, 0].float()
        pooled = torch.cat((cls.reshape(b, v, -1)[:, 0].float(), tok0.flatten(2).std(2),
                            cnn.reshape(b, v, *cnn.shape[1:])[:, 0].float().mean((2, 3)),
                            cnn.reshape(b, v, *cnn.shape[1:])[:, 0].float().flatten(2).std(2)), 1)
        q1 = self.exit1_head(torch.cat((pooled, geometry), 1))
        coverage = cost[:, -1].mean(1)
        stats = torch.stack((entropy.mean((1, 2, 3)) / math.log(self.planes), entropy.flatten(1).std(1) / math.log(self.planes),
                             log_scale_low.mean((1, 2, 3)), log_scale_low.flatten(1).std(1),
                             log_scale_low.flatten(1).quantile(.9, dim=1), probabilities.amax(1).mean((1, 2)),
                             coverage.mean((1, 2)), low.clamp_min(1e-4).log().flatten(1).std(1)), 1)
        q2 = self.exit2_head(torch.cat((pooled, geometry, stats.detach()), 1))
        return {'depth': depth, 'confidence': confidence, 'log_scale': log_scale,
                'q1': (q1[:, 0], q1[:, 1].clamp(-8, 6)), 'q2': (q2[:, 0], q2[:, 1].clamp(-8, 6)), 'geometry': geometry}

    def forward(self, rgb, K, P, source_weights=None):
        out = self.forward_full(rgb, K, P, source_weights)
        return out['depth'], out['confidence']

    def trainable_state(self):
        """Checkpoint without the frozen backbone (it is reloaded from the hub cache)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith('backbone.')}
