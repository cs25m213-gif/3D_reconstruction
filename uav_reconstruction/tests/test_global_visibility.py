import unittest
import numpy as np
from adaptive_uav.global_visibility import visibility_votes, guard_frames


def frame(name, depth, tx=0):
    d = np.full((32, 32), depth, np.float32)
    p = np.eye(4)
    p[0, 3] = tx
    return dict(name=name, depth=d.copy(), measured=d.copy(), P=p,
                K=np.array([[20., 0, 16], [0, 20., 16], [0, 0, 1]]),
                tier=np.ones(d.shape, np.uint8))


class VisibilityTests(unittest.TestCase):
    def test_stereo_holes_do_not_disable_free_space_checks(self):
        target=frame('target',2)
        other=frame('measured',4,.2)
        other['measured'][1::2]=0
        _,conflict=visibility_votes(target,[other])
        self.assertGreater(conflict.sum(),500)

    def test_measurement_boundaries_are_not_used_to_carve(self):
        target=frame('target',2);other=frame('edge',4,.2)
        other['measured'][1::2]=1.
        _,conflict=visibility_votes(target,[other])
        self.assertEqual(conflict.sum(),0)

    def test_reference_budget_is_reserved_for_measured_views(self):
        frames=[frame('unknown_'+str(i),0) for i in range(8)]+[frame('a',2),frame('b',2,.2)]
        report=guard_frames(frames,max_references=2)
        self.assertEqual(report['reference_frames'],['a','b'])
        self.assertGreater(report['mean_tested_fraction'],0)

    def test_free_space_conflict_and_occlusion_are_distinct(self):
        target = frame('target', 2)
        _, conflict = visibility_votes(target, [frame('far', 4, .2)])
        self.assertGreater(conflict.sum(), 500)
        _, conflict = visibility_votes(target, [frame('near', 1, .2)])
        self.assertEqual(conflict.sum(), 0)

    def test_matching_depth_supported(self):
        support, conflict = visibility_votes(frame('a', 2), [frame('b', 2, .2)])
        self.assertGreater(support.sum(), 500)
        self.assertEqual(conflict.sum(), 0)

    def test_unknown_depth_does_not_carve(self):
        _, conflict = visibility_votes(frame('a', 2), [frame('b', 0, .2)])
        self.assertEqual(conflict.sum(), 0)

    def test_repeated_contradiction_removes_measured_candidate(self):
        frames = [frame('a', 2), frame('b', 4, .2), frame('c', 4, -.2)]
        report = guard_frames(frames)
        self.assertGreater(report['per_frame'][0]['rejected_fraction'], .5)
        self.assertEqual(frames[0]['tier'][16, 16], 0)
        self.assertEqual(frames[0]['measured'][16, 16], 2)


if __name__ == '__main__':
    unittest.main()
