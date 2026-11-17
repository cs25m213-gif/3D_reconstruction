import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_uav.dense_coverage import (recovery_workspace, two_source_core,
                                         supported_reference_subset)


class DenseRecoveryTests(unittest.TestCase):
    def test_search_recovers_group_missed_by_temporal_anchors(self):
        good = {f'good{i}' for i in range(6)}
        isolated = {f'isolated{i}' for i in range(8)}
        graph = {name: good - {name} for name in good}
        graph.update({name: set() for name in isolated})
        self.assertEqual(two_source_core(graph, isolated), set())
        selected = supported_reference_subset(graph, 5, isolated)
        self.assertEqual(len(selected), 5)
        self.assertTrue(selected <= good)
        self.assertEqual(two_source_core(graph, selected), selected)

    def test_abundant_frames_without_parallax_do_not_become_stereo(self):
        graph = {str(i): set() for i in range(37)}
        self.assertEqual(supported_reference_subset(graph, 12), set())

    def test_sources_must_themselves_have_two_supported_sources(self):
        # A chain's internal views each have two neighbours, but its ends
        # cannot provide geometric depth. Pruning must propagate to the centre.
        graph = {str(i): {str(j) for j in (i-1, i+1) if 0 <= j < 8}
                 for i in range(8)}
        self.assertEqual(two_source_core(graph, graph), set())

    def test_recovery_reuses_poses_but_never_stale_geometric_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; output=root/'recovery'
            (source/'images').mkdir(parents=True); (source/'sparse').mkdir()
            for name in ('depth_maps','normal_maps'):
                folder=source/'stereo'/name; folder.mkdir(parents=True)
                (folder/'frame.jpg.photometric.bin').write_bytes(b'initialization')
                (folder/'frame.jpg.geometric.bin').write_bytes(b'stale-geometric-depth')
            with patch('adaptive_uav.dense_coverage.plan_coverage',return_value={'reference_views':29}) as planner:
                recovery_workspace(source,output,32,4)
            self.assertEqual((output/'sparse').resolve(),(source/'sparse').resolve())
            self.assertTrue((output/'stereo/depth_maps/frame.jpg.photometric.bin').is_file())
            self.assertFalse((output/'stereo/depth_maps/frame.jpg.geometric.bin').exists())
            self.assertTrue((output/'stereo/normal_maps').is_dir())
            self.assertTrue((output/'stereo/consistency_graphs').is_dir())
            self.assertEqual(planner.call_args.kwargs['min_angle'],3.)
            provenance=json.loads((output/'refinement_provenance.json').read_text())
            self.assertFalse(provenance['parallax_threshold_relaxed'])

    def test_existing_workspace_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output';output.mkdir()
            marker=output/'original.txt';marker.write_text('keep')
            with self.assertRaises(ValueError):
                recovery_workspace(Path(directory)/'source',output)
            self.assertEqual(marker.read_text(),'keep')


if __name__=='__main__':
    unittest.main()
