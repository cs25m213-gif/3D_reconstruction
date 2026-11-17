"""Opt-in map-conditioned ADD/SWAP head; no runtime or checkpoint replacement.

Inputs must describe one causal, common map with the proposed occupant removed.
The shared frame scorer makes SWAP antisymmetric under exchanging the two frames.
This is an inductive bias, not a guarantee of accurate measured reconstruction gain.
The data adapter must enforce frame identities, timestamps and reference exclusion.
"""
from dataclasses import dataclass

import torch
from torch import nn

FRAME_FEATURES = (
    'sharpness', 'clipped_fraction', 'exposure_mean', 'texture_energy',
    'baseline_over_depth', 'rotation_from_last', 'verified_overlap',
    'tracked_fraction', 'depth_available', 'predicted_depth_error',
    'valid_depth_fraction', 'surface_agreement',
)
MAP_FEATURES = (
    'map_keyframes', 'map_coverage_fraction', 'map_redundancy', 'graph_edges',
    'graph_components', 'remaining_seconds', 'remaining_depth_calls',
    'remaining_matches', 'queue_load', 'tracking_ok', 'elapsed_seconds',
    'scale_anchor_support',
)
MEMORY_FEATURES = (
    'age_s', 'sharpness', 'verified_track_support', 'graph_degree', 'is_bridge',
    'same_tracking_component', 'valid_depth_fraction', 'predicted_depth_error',
)


@dataclass(frozen=True)
class ReconstructionSelectorConfig:
    visual_dim: int = 384
    width: int = 128
    heads: int = 4
    layers: int = 2
    max_memory: int = 16


class ReconstructionActionSelector(nn.Module):
    architecture_id = 'common_map_pairwise_selector_v1'

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ReconstructionSelectorConfig()
        c = self.config
        if min(c.width, c.heads, c.layers, c.max_memory, c.visual_dim) <= 0 or c.width % c.heads:
            raise ValueError('Positive dimensions and width divisible by attention heads required')
        for name, size in (('frame', len(FRAME_FEATURES)), ('query', len(MAP_FEATURES)),
                           ('memory', len(MEMORY_FEATURES))):
            self.register_buffer(name + '_mean', torch.zeros(size))
            self.register_buffer(name + '_std', torch.ones(size))
        self.visual = nn.Linear(c.visual_dim, c.width)
        self.frame = nn.Linear(len(FRAME_FEATURES), c.width)
        self.query = nn.Linear(len(MAP_FEATURES), c.width)
        self.memory = nn.Linear(len(MEMORY_FEATURES), c.width)
        self.roles = nn.Parameter(torch.randn(3, c.width) * .02)
        layer = nn.TransformerEncoderLayer(c.width, c.heads, c.width * 4,
                    dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.frame_value = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, 64),
                                         nn.GELU(), nn.Linear(64, 1))
        self.add = nn.Sequential(nn.LayerNorm(c.width * 2), nn.Linear(c.width * 2, 64),
                                 nn.GELU(), nn.Linear(64, 1))

    def _normalize(self, value, name):
        mean = getattr(self, name + '_mean')
        std = getattr(self, name + '_std').clamp_min(1e-4)
        return ((value - mean) / std).clamp(-6, 6)

    def forward(self, candidate, occupant, has_occupant, frame_features, query,
                memory_visual, memory_features, memory_valid):
        """Embeddings [B,D]; frames [B,2,12]; query [B,12]; memory [B,M,D/8].

        memory_valid [B,M] and has_occupant [B] are boolean, True means present.
        Depth feature availability is explicit; unavailable depth values are ignored.
        Fit normalization buffers on training data only before fitting this head.
        No reference errors, teacher predictions or future-neighbour depths are inputs.
        """
        c = self.config
        if candidate.ndim != 2:
            raise ValueError('Candidate must have shape [B,visual_dim]')
        b = candidate.shape[0]
        if memory_visual.ndim != 3:
            raise ValueError('Memory must have shape [B,M,visual_dim]')
        m = memory_visual.shape[1]
        expected = (
            (candidate, (b, c.visual_dim)), (occupant, (b, c.visual_dim)),
            (has_occupant, (b,)), (frame_features, (b, 2, len(FRAME_FEATURES))),
            (query, (b, len(MAP_FEATURES))), (memory_visual, (b, m, c.visual_dim)),
            (memory_features, (b, m, len(MEMORY_FEATURES))), (memory_valid, (b, m)),
        )
        if any(tuple(t.shape) != shape for t, shape in expected) or m > c.max_memory:
            raise ValueError('Input dimensions or bounded memory contract violated')
        if has_occupant.dtype != torch.bool or memory_valid.dtype != torch.bool:
            raise ValueError('Availability masks must be boolean')
        has = has_occupant
        occ = torch.where(has[:, None], occupant, torch.zeros_like(occupant))
        present = torch.stack((torch.ones_like(has), has), 1)
        frames = torch.where(present[..., None], frame_features, torch.zeros_like(frame_features))
        # Ignore unavailable deep features even when placeholders contain NaN.
        deep = torch.where(frames[..., 8:9] > .5, frames[..., 9:], torch.zeros_like(frames[..., 9:]))
        frames = torch.cat((frames[..., :9], deep), -1)
        mv = torch.where(memory_valid[..., None], memory_visual, torch.zeros_like(memory_visual))
        mf = torch.where(memory_valid[..., None], memory_features, torch.zeros_like(memory_features))
        pair = self.visual(torch.stack((candidate, occ), 1)) + self.frame(self._normalize(frames, 'frame'))
        memory = self.visual(mv) + self.memory(self._normalize(mf, 'memory'))
        # Identical frame roles and no positional embeddings preserve pair symmetry
        # and make the result independent of memory serialization order.
        tokens = torch.cat((self.query(self._normalize(query, 'query'))[:, None] + self.roles[0],
                            pair + self.roles[1], memory + self.roles[2]), 1)
        padding = torch.cat((torch.zeros((b, 2), dtype=torch.bool, device=candidate.device),
                             ~has[:, None], ~memory_valid), 1)
        encoded = self.context(tokens, src_key_padding_mask=padding)
        values = self.frame_value(encoded[:, 1:3]).squeeze(-1)
        swap = torch.where(has, values[:, 0] - values[:, 1], torch.zeros_like(values[:, 0]))
        add = self.add(torch.cat((encoded[:, 0], encoded[:, 1]), -1)).squeeze(-1)
        return {'gain': torch.where(has, swap, add), 'add_gain': add, 'swap_gain': swap}
