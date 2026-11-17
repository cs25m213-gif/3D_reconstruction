"""Parallel scheduler checks with fake workers only; no GPU or reconstruction."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from adaptive_uav import reconstruct_training_clips as pool


class ParallelReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def clip(self, name, ready=False):
        p = self.root / name; (p / 'session').mkdir(parents=True)
        (p / 'session/video.mp4').write_bytes(b'fixture')
        if ready:
            (p / 'reconstruction').mkdir()
            mesh = p / 'reconstruction/result.glb'; mesh.write_bytes(b'fixture')
            (p / 'reconstruction/run_report.json').write_text(json.dumps({'result_ready': True, 'mesh': str(mesh)}))
        return p

    def test_completed_assets_are_never_launched(self):
        p = self.clip('done', True)
        launch = mock.Mock(side_effect=AssertionError('must preserve completed reconstruction'))
        code = pool.run_pool([p], 2, self.root/'status.json', True, launch=launch, poll_seconds=0.)
        self.assertEqual(code, 0); launch.assert_not_called()

    def test_two_workers_overlap_and_thread_budget_is_forwarded(self):
        folders = [self.clip(str(i)) for i in range(3)]
        active = set(); peaks = []; launched = []

        class Worker:
            def __init__(self, folder):
                self.pid = 100 + len(launched); self.folder = folder; self.polls = 0
            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                active.discard(self.pid)
                return 0
            def wait(self):
                active.discard(self.pid); return 0

        def launch(command, **kwargs):
            folder = Path(command[2]).parent
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '4')
            worker = Worker(folder); launched.append(folder); active.add(worker.pid); peaks.append(len(active))
            return worker

        with mock.patch.object(pool, 'completed', side_effect=lambda p: p in launched):
            # Initial discovery sees none launched; completed checks succeed after fake workers exit.
            code = pool.run_pool(folders, 2, self.root/'status.json', True, threads=4,
                                 probe=lambda *a: None, launch=launch, poll_seconds=0.)
        self.assertEqual(code, 0)
        self.assertEqual(max(peaks), 2)
        self.assertEqual(len(launched), 3)

    def test_unrelated_gpu_job_blocks_admission_without_signals(self):
        p = self.clip('pending'); launch = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'unrelated'):
            pool.run_pool([p], 2, self.root/'status.json', False,
                          probe=lambda *a: 'unrelated GPU job', launch=launch, poll_seconds=0.)
        launch.assert_not_called()

    def test_unknown_gpu_status_fails_closed(self):
        p = self.clip('pending'); launch = mock.Mock()
        def failed_probe(*args):
            raise subprocess.CalledProcessError(1, 'nvidia-smi')
        with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
            pool.run_pool([p], 2, self.root/'status.json', True,
                          probe=failed_probe, launch=launch, poll_seconds=0.)
        launch.assert_not_called()

    def test_failure_cannot_mark_pool_complete(self):
        p = self.clip('bad')
        worker = mock.Mock(pid=123); worker.poll.return_value = 1
        code = pool.run_pool([p], 2, self.root/'status.json', True,
                             probe=lambda *a: None, launch=lambda *a, **kw: worker, poll_seconds=0.)
        status = json.loads((self.root/'status.json').read_text())
        self.assertEqual(code, 1); self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['failed_this_run'], ['bad'])

    def test_gpu_ancestry_only_exempts_our_descendants(self):
        self.assertTrue(pool.is_descendant(os.getpid(), os.getpid()))
        self.assertFalse(pool.is_descendant(os.getppid(), os.getpid()))
        self.assertFalse(pool.is_descendant(999999999, os.getpid()))

    def test_gpu_own_worker_is_allowed_but_foreign_pid_is_not(self):
        def response(command, **kwargs):
            if '--query-compute-apps=pid' in command:
                return subprocess.CompletedProcess(command, 0, str(os.getpid())+'\n', '')
            return subprocess.CompletedProcess(command, 0, '30000\n', '')
        with mock.patch.object(pool.subprocess, 'run', side_effect=response):
            self.assertIsNone(pool.admission_reason(os.getpid(), 10000, 1))
        with mock.patch.object(pool.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '999999999\n', '')):
            self.assertIn('unrelated', pool.admission_reason(os.getpid(), 10000, 1))

    def test_vram_reserve_rejects_additional_worker(self):
        responses = [subprocess.CompletedProcess([], 0, '', ''), subprocess.CompletedProcess([], 0, '500\n', '')]
        with mock.patch.object(pool.subprocess, 'run', side_effect=responses):
            self.assertIn('VRAM', pool.admission_reason(os.getpid(), 10000, 1))


if __name__ == '__main__':
    unittest.main()
