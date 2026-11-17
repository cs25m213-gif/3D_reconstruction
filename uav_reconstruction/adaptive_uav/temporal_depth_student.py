"""Temporal memory around the existing student (DeepVideoMVS-style hidden-state warping).

A small ConvGRU sees the student's depth and earlier geometry warped into this camera, and predicts
a bounded residual plus a visibility map. The residual starts at zero, so an untrained branch equals
the student exactly and cannot make the baseline worse. Priors come only from earlier predictions
in the same run, and any discontinuity resets them. The student stays frozen in this stage.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE_ID = 'temporal_student_v1'


def warp_prior(depth: torch.Tensor, confidence: torch.Tensor, k: torch.Tensor, current_from_previous: torch.Tensor,
               previous_k: torch.Tensor | None = None) -> tuple:
    """Forward-warp [B,1,H,W] depth; nearest surface wins and holes stay zero."""
    b, _, h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing='ij')
    pixels = torch.stack((x, y, torch.ones_like(x)), 0).float().reshape(1, 3, -1).expand(b, -1, -1)
    points = torch.linalg.solve(k if previous_k is None else previous_k, pixels) * depth.reshape(b, 1, -1)
    moved = current_from_previous[:, :3, :3] @ points + current_from_previous[:, :3, 3:]
    projected = k @ moved
    z = moved[:, 2]
    u = (projected[:, 0] / z.clamp_min(1e-6)).round().long()
    v = (projected[:, 1] / z.clamp_min(1e-6)).round().long()
    valid = torch.isfinite(depth.reshape(b,-1)) & torch.isfinite(moved).all(1) & (depth.reshape(b, -1) > 0) & (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out, weight = torch.zeros_like(depth), torch.zeros_like(confidence)
    for i in range(b):
        where = valid[i].nonzero().squeeze(1)
        # Repeated-index assignment is not an ordered z-buffer on CUDA.
        destination = v[i, where]*w+u[i, where]
        nearest = torch.full((h*w,),float('inf'),device=depth.device,dtype=depth.dtype)
        nearest.scatter_reduce_(0,destination,z[i,where],reduce='amin',include_self=True)
        candidates = torch.where(z[i,where]==nearest[destination],where,torch.full_like(where,h*w))
        winner = torch.full((h*w,),h*w,device=depth.device,dtype=torch.long)
        winner.scatter_reduce_(0,destination,candidates,reduce='amin',include_self=True)
        occupied = winner<h*w
        out[i].view(-1)[occupied] = nearest[occupied]
        weight[i].view(-1)[occupied] = confidence[i].reshape(-1)[winner[occupied]]
    return out, weight, (out > 0).float()


def warp_hidden(hidden: torch.Tensor, depth: torch.Tensor, k: torch.Tensor, previous_from_current: torch.Tensor,
                previous_k: torch.Tensor | None = None) -> torch.Tensor:
    """Bring last frame's memory into this camera using this frame's depth."""
    b, _, h, w = hidden.shape
    y, x = torch.meshgrid(torch.arange(h, device=hidden.device), torch.arange(w, device=hidden.device), indexing='ij')
    pixels = torch.stack((x, y, torch.ones_like(x)), 0).float().reshape(1, 3, -1).expand(b, -1, -1)
    points = torch.linalg.solve(k, pixels) * F.interpolate(depth, size=(h, w), mode='nearest').reshape(b, 1, -1)
    moved = previous_from_current[:, :3, :3] @ points + previous_from_current[:, :3, 3:]
    projected = (k if previous_k is None else previous_k) @ moved
    z = projected[:, 2].clamp_min(1e-6)
    grid = torch.stack((2 * projected[:, 0] / z / max(w - 1, 1) - 1, 2 * projected[:, 1] / z / max(h - 1, 1) - 1), -1)
    valid = torch.isfinite(moved).all(1) & (moved[:,2]>0) & (F.interpolate(depth,size=(h,w),mode='nearest').reshape(b,-1)>0)
    safe_grid = torch.where(valid[...,None],grid,torch.full_like(grid,2.))
    return F.grid_sample(hidden, safe_grid.reshape(b, h, w, 2), align_corners=True, padding_mode='zeros')*valid.reshape(b,1,h,w)


class TemporalState:
    RESET_REASONS = ('component_change', 'tracking_lost', 'pose_correction', 'model_change')

    def __init__(self):
        self.depth = self.confidence = self.pose = self.hidden = None

    def reset(self, reason: str) -> None:
        if reason not in self.RESET_REASONS:
            raise ValueError(f'Unknown reset reason {reason}')
        self.depth = self.confidence = self.pose = self.hidden = None

    def update(self, depth, confidence, world_from_camera, hidden=None) -> None:
        self.depth, self.confidence, self.pose = depth.detach(), confidence.detach(), world_from_camera.detach()
        self.hidden = None if hidden is None else hidden.detach()


class ConvGRU(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(inputs + hidden, 2 * hidden, 3, padding=1)
        self.update = nn.Conv2d(inputs + hidden, hidden, 3, padding=1)

    def forward(self, x, state=None):
        state = torch.zeros(x.shape[0], self.hidden, *x.shape[2:], device=x.device, dtype=x.dtype) if state is None else state
        joined = torch.cat((x, state), 1)
        gate, reset = torch.sigmoid(self.gates(joined)).chunk(2, 1)
        fresh = torch.tanh(self.update(torch.cat((x, reset * state), 1)))
        return (1 - gate) * state + gate * fresh


class TemporalDepthStudent(nn.Module):
    architecture_id = ARCHITECTURE_ID

    def __init__(self, student: nn.Module, channels: int = 16, hidden: int = 16):
        super().__init__()
        self.student = student
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        self.encode = nn.Sequential(nn.Conv2d(5, channels, 3, padding=1), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU())
        self.memory = ConvGRU(channels, hidden)
        self.residual = nn.Conv2d(hidden, 1, 3, padding=1)
        self.visibility = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, rgb, k, p, prior=None, prior_confidence=None, prior_valid=None, hidden=None):
        with torch.no_grad():
            depth, confidence = self.student(rgb, k, p)
        if prior is None:
            prior, prior_confidence, prior_valid = torch.zeros_like(depth), torch.zeros_like(depth), torch.zeros_like(depth)
        ratio = torch.where(prior_valid > 0, (prior.clamp_min(1e-4) / depth.clamp_min(1e-4)).log().clamp(-1, 1), torch.zeros_like(prior))
        state = self.memory(self.encode(torch.cat((depth.log(), ratio, prior_confidence, prior_valid, confidence), 1)), hidden)
        # A prior can correct depth but never replace it: the residual is limited to about 28%.
        delta = self.residual(state).tanh() * .25
        return depth * torch.exp(delta), confidence, torch.sigmoid(self.visibility(state)), state
