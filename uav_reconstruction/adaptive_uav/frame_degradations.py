"""GPU frame degradations that make a drone frame a poor reconstruction candidate.

Applied to the reference frame during training. The quality heads are NOT given a synthetic
"bad" label: their target stays the frame's real depth error against LiDAR or map depth, so they
learn from measured outcomes how much each kind and strength of degradation actually hurts.

Kinds (each with a random strength in [0, 1]):
  gaussian_blur   defocus / soft focus             sigma 0.8 .. 4 px
  motion_blur     fast motion or vibration         line kernel 5 .. 25 px, random direction
  exposure        over- or under-exposure          gain 0.25 .. 0.6 or 1.8 .. 3.5, with clipping
  noise           low light sensor noise           Gaussian + signal-dependent, sigma 0.02 .. 0.12
  low_resolution  compression / digital zoom       down-sample 2 .. 8x and back
  haze            fog, glare, low contrast         blend towards bright grey 0.3 .. 0.8
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

KINDS = ('gaussian_blur', 'motion_blur', 'exposure', 'noise', 'low_resolution', 'haze')


def _depthwise(images: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """images [B,3,H,W], kernels [B,k,k] (one per image, applied to every channel)."""
    b, c, h, w = images.shape
    k = kernels.shape[-1]
    weight = kernels[:, None].expand(b, c, k, k).reshape(b * c, 1, k, k)
    x = F.pad(images.reshape(1, b * c, h, w), (k // 2,) * 4, mode='replicate')
    return F.conv2d(x, weight, groups=b * c).reshape(b, c, h, w)


def gaussian_blur(x, s):
    sigma = .8 + 3.2 * s
    k = 25
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(r[None] ** 2) / (2 * sigma[:, None] ** 2))
    kernel = g[:, :, None] * g[:, None, :]
    return _depthwise(x, kernel / kernel.sum((1, 2), keepdim=True))


def motion_blur(x, s):
    k = 25
    length = 5 + 20 * s
    angle = torch.rand_like(s) * math.pi
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    yy, xx = torch.meshgrid(r, r, indexing='ij')
    # distance of each tap from a centred line segment of the given length and direction
    dx, dy = torch.cos(angle)[:, None, None], torch.sin(angle)[:, None, None]
    along = xx[None] * dx + yy[None] * dy
    across = (-xx[None] * dy + yy[None] * dx).abs()
    kernel = ((across < .75) & (along.abs() <= length[:, None, None] / 2)).to(x.dtype) * (1 - across / .75).clamp_min(0) + 1e-6
    return _depthwise(x, kernel / kernel.sum((1, 2), keepdim=True))


def exposure(x, s):
    dark = torch.rand_like(s) < .5
    gain = torch.where(dark, .6 - .35 * s, 1.8 + 1.7 * s)
    return (x * gain[:, None, None, None]).clamp(0, 1)


def noise(x, s):
    sigma = (.02 + .1 * s)[:, None, None, None]
    return (x + torch.randn_like(x) * sigma * (.5 + x.sqrt())).clamp(0, 1)


def low_resolution(x, s):
    out = torch.empty_like(x)
    factor = (2 + 6 * s).round().int()
    h, w = x.shape[-2:]
    for f in factor.unique().tolist():
        sel = factor == f
        small = F.interpolate(x[sel], size=(max(h // f, 4), max(w // f, 4)), mode='area')
        out[sel] = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
    return out


def haze(x, s):
    a = (.3 + .5 * s)[:, None, None, None]
    return x * (1 - a) + .8 * a


FUNCTIONS = {'gaussian_blur': gaussian_blur, 'motion_blur': motion_blur, 'exposure': exposure,
             'noise': noise, 'low_resolution': low_resolution, 'haze': haze}


def degrade(images: torch.Tensor, kind: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """images [B,3,H,W] in [0,1]; kind [B] long index into KINDS (-1 = untouched); strength [B] in [0,1]."""
    out = images.clone()
    for i, name in enumerate(KINDS):
        sel = kind == i
        if sel.any():
            out[sel] = FUNCTIONS[name](images[sel].float(), strength[sel].float()).to(images.dtype)
    return out


# Fast motion and blur are the main reasons a drone frame should be rejected, so they dominate.
KIND_WEIGHTS = {'gaussian_blur': .25, 'motion_blur': .35, 'exposure': .1, 'noise': .1, 'low_resolution': .1, 'haze': .1}


def random_degradation(b: int, probability: float, device) -> tuple:
    weights = torch.tensor([KIND_WEIGHTS[k] for k in KINDS], device=device)
    kind = torch.multinomial(weights, b, replacement=True)
    kind = torch.where(torch.rand(b, device=device) < probability, kind, torch.full_like(kind, -1))
    return kind, torch.rand(b, device=device)
