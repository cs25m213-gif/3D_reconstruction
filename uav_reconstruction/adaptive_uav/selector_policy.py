"""Frame-selection policies at an exactly matched keyframe budget.

A frame is chosen from each stretch of the clip, in time order, so every policy decides causally
and spends the same budget. The learned policy can also sample, which is what reconstruction-reward
fine-tuning needs. Coverage-gain selection is the non-neural yardstick: maximising a coverage
function greedily has classic guarantees, so a neural policy must beat it to justify itself.
"""
from __future__ import annotations

import numpy as np
import torch

from .keyframe_policy import KeyframePolicy
from .selector_features import MapState, as_pose


def collate(rows: list, device) -> tuple:
    stack = lambda key: torch.as_tensor(np.stack([r[key] for r in rows]), dtype=torch.float32, device=device)[:, None]
    tokens = stack('tokens')
    return (stack('visual'), stack('query'), stack('token_visual'), tokens, tokens[..., -1] > 0,
            torch.as_tensor([r['question'] for r in rows], device=device)[:, None])


def windows(clip, count: int) -> list:
    return [w for w in np.array_split(np.arange(len(clip.views)), count) if len(w)]


def uniform(clip, count: int, *_) -> list:
    return [clip.views[int(w[len(w) // 2])] for w in windows(clip, count)]


def random_policy(clip, count: int, descriptors=None, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [clip.views[int(rng.choice(w))] for w in windows(clip, count)]


def rule_selection(clip, threshold: float) -> list:
    policy, chosen = KeyframePolicy(.25, 1., threshold), []
    for view in clip.views:
        if policy.decide(view, as_pose(view, view.feature_pose))[0]:
            chosen.append(view)
    return chosen


def rule(clip, count: int, *_) -> list:
    """The real rule: tighten its parallax threshold to the budget, then bridge the longest gaps.

    Without the fill step the rule would be compared using fewer frames than everyone else.
    """
    where, thresholds = {id(v): i for i, v in enumerate(clip.views)}, np.geomspace(2., 600., 48)
    low, high, chosen = 0, len(thresholds) - 1, []
    while low <= high:  # Each try runs optical flow over the clip, so search instead of scanning.
        middle = (low + high) // 2
        trial = rule_selection(clip, float(thresholds[middle]))
        if len(trial) <= count:
            chosen, high = trial, middle - 1
        else:
            low = middle + 1
    positions, total = {where[id(v)] for v in chosen[:count]} or {0}, len(clip.views)
    while len(positions) < min(count, total):
        edges = [-1] + sorted(positions) + [total]
        wide = int(np.argmax(np.diff(edges)))
        positions.add((edges[wide] + edges[wide + 1]) // 2)
    return [clip.views[p] for p in sorted(positions)]


def coverage_gain(clip, count: int, descriptors) -> list:
    """Observable baseline: in each stretch, take the view the current map explains least."""
    state, chosen = MapState(), []
    for window in windows(clip, count):
        step = max(1, len(window) // 5)
        options = [clip.views[int(p)] for p in window[::step][:5]]
        if chosen:
            options.sort(key=lambda v: state.coverage.view_stats(v.K, v.feature_pose, v.rgb.shape[1], v.rgb.shape[0])[0])
        view = options[0]
        state.observe(view.frame_id, view.depth_m, view.K, view.feature_pose, view.confidence)
        chosen.append(view)
    return chosen


class UtilityPolicy:
    def __init__(self, model, data: dict, device):
        if data.get('selection_protocol_id'):
            raise ValueError('This checkpoint requires its versioned selection policy, not legacy UtilityPolicy')
        self.model, self.head, self.device = model, data['decision_head'], device

    def scores(self, rows: list) -> torch.Tensor:
        return self.model(*collate([{**r, 'question': self.head} for r in rows], self.device))['utility'][:, 0, 3]

    def run(self, clip, count: int, descriptors, temperature: float = 0.) -> tuple:
        """temperature 0 takes the best; above 0 it samples and returns log-probabilities."""
        state, chosen, log_probabilities, entropies = MapState(), [], [], []
        for position, window in enumerate(windows(clip, count)):
            step = max(1, len(window) // 5)
            options = [clip.views[int(p)] for p in window[::step][:5]]
            pick = 0
            if chosen and len(options) > 1:
                budget = {'seconds': 60. * (count - position), 'depth_calls': count - position}
                rows = [state.features(v, as_pose(v, v.feature_pose), budget, descriptors.get(clip.name, v.index, v.rgb)) for v in options]
                scores = self.scores(rows)
                if temperature > 0:
                    distribution = sampling_distribution(scores, temperature)
                    pick = int(distribution.sample())
                    log_probabilities.append(distribution.log_prob(torch.as_tensor(pick, device=scores.device)))
                    entropies.append(distribution.entropy())
                else:
                    pick = int(scores.argmax())
            view = options[pick]
            state.commit(view, as_pose(view, view.feature_pose), descriptors.get(clip.name, view.index, view.rgb))
            state.observe(view.frame_id, view.depth_m, view.K, view.feature_pose, view.confidence)
            chosen.append(view)
        return chosen, log_probabilities, entropies

    def __call__(self, clip, count: int, descriptors) -> list:
        with torch.no_grad():
            return self.run(clip, count, descriptors)[0]


def sampling_distribution(scores, temperature):
    """Preserve the sampling scale without cancelling two-candidate gradients."""
    if temperature <= 0:
        raise ValueError('Sampling temperature must be positive')
    scale = scores.std().detach() + 1e-6
    return torch.distributions.Categorical(logits=(scores - scores.mean()) / scale / temperature)
