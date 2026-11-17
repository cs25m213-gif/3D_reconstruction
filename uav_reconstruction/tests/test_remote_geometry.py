"""Routing must preserve results and recover without failing active training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from adaptive_uav import remote_geometry as remote


class RemoteGeometryTests(unittest.TestCase):
    def setUp(self):
        self.old = remote._client, remote._disabled
        remote._client, remote._disabled = None, False

    def tearDown(self):
        remote._client, remote._disabled = self.old

    def test_absent_routing_preserves_local(self):
        with patch.object(remote, 'CONFIG', Path('/missing-geostream-routing')):
            local = Mock(return_value={'f_score': .3})
            self.assertEqual(remote.geometry_metrics([], None, .2, local), {'f_score': .3})
            local.assert_called_once()

    def test_remote_result_is_used_without_duplicate_local_work(self):
        client = Mock(); client.metrics.return_value = {'f_score': .5}
        remote._client = client
        local = Mock()
        self.assertEqual(remote.geometry_metrics([], None, .2, local), {'f_score': .5})
        local.assert_not_called()

    def test_failure_falls_back_and_stops_repeated_network_retries(self):
        client = Mock(); client.metrics.side_effect = TimeoutError('test timeout')
        remote._client = client
        local = Mock(return_value={'f_score': .4})
        for _ in range(2): self.assertEqual(remote.geometry_metrics([], None, .2, local), {'f_score': .4})
        self.assertEqual(client.metrics.call_count, 1)
        self.assertEqual(local.call_count, 2)
        client.close.assert_called_once()

    def test_disabled_route_does_not_create_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'routing.json'; path.write_text(json.dumps({'enabled': False}))
            with patch.object(remote, 'CONFIG', path), patch.object(remote, 'GeometryClient') as factory:
                self.assertEqual(remote.geometry_metrics([], None, .2, lambda: 7), 7)
                factory.assert_not_called()

    def test_single_rewards_can_stay_local_while_batches_are_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'routing.json'; path.write_text(json.dumps({'enabled': True, 'single_enabled': False}))
            with patch.object(remote, 'CONFIG', path), patch.object(remote, 'GeometryClient') as factory:
                self.assertEqual(remote.geometry_metrics([], None, .2, lambda: 8), 8)
                factory.assert_not_called()

    def test_batch_failure_falls_back_as_one_complete_result(self):
        client = Mock(); client.metrics.side_effect = EOFError('test disconnection')
        remote._client = client
        local = Mock(return_value=[{'f_score': .2}, {'f_score': .4}])
        self.assertEqual(remote.geometry_metrics_batch([[], []], None, .2, local), [{'f_score': .2}, {'f_score': .4}])
        local.assert_called_once(); self.assertTrue(remote._disabled)


if __name__ == '__main__': unittest.main()
