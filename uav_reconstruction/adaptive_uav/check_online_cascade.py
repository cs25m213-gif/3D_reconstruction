"""Small behavioral checks for cascade causality and demand accounting (CPU only)."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import test_online_cascade as experiment


class FakeClip:
    def __init__(self):
        self.ctx = SimpleNamespace(n=4, emb=np.ones((4, 2), np.float32), F=lambda selected: len(selected) / 10.)
        self.sharp = np.ones(4)
        self.requested = set()
        self.accesses = []

    demand = experiment.Clip.demand

    def geometry(self, t, selected):
        assert all(s < t for s in selected)
        assert set(selected).issubset(self.requested)
        self.demand(t, t)
        self.accesses.append(t)
        return np.zeros(len(experiment.GEOMETRY), np.float32)


def main():
    experiment.CFG.tau = .9
    cheap = lambda t, a, x: (.9, .1, .5, .9)[t]
    deep = lambda t, a, x: .1
    with patch.object(experiment.bm, 'state_features', return_value=np.zeros(24, np.float32)):
        c = FakeClip()
        selected, _, cost = experiment.rollout(c, cheap, deep, margin=.1)
        assert selected == [3], selected
        assert cost['requested_depth_frames'] == [0, 2, 3], cost
        assert cost['swaps'] == 1 and cost['routed_count'] == 1, cost
        assert c.accesses == [2], c.accesses

        c = FakeClip()
        selected, _, cost = experiment.rollout(c, cheap, deep, margin=0.)
        assert not c.accesses
        assert cost['requested_depth_frames'] == [0, 2, 3]

        c = FakeClip()
        selected, _, cost = experiment.rollout(c, cheap, deep, margin=1.)
        assert selected == [] and cost['requested_depth_frames'] == [0, 1, 2, 3]

        c = FakeClip()
        selected, labels, cost = experiment.rollout(c, measure=True)
        assert len(labels) == 4  # All decisions, including zero-gain rejected swaps.
        assert cost['requested_depth_count'] == 4
        assert labels[1][-1] == 0.

    c = FakeClip()
    try:
        c.demand(2, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('Future-depth guard failed')
    print('PASS: routed rejection counted; swapped frames counted; cheap exits avoid candidate geometry; '
          'full-depth control counts all; rejected actions labelled; future depth rejected.')


if __name__ == '__main__':
    main()
