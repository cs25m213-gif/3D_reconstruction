"""Optional CPU fusion/scoring over SSH; retain identical local fallback.

Enabled by a run-local routing file, so existing processes are not modified.
Only explicit fixed-reference-pose rewards use this boundary. Neural models,
optimizers, datasets, and checkpoint writes stay in their original processes.
"""
import atexit
import hashlib
import io
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'runs/remote_cpu_20261104/routing.json'
_client = None
_disabled = False


def routing_provenance():
    return {'configured': CONFIG.is_file(), 'used_remote_worker': _client is not None,
            'fell_back_locally': _disabled,
            'configuration_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest() if CONFIG.is_file() else None,
            'call_log': str(_client.log) if _client is not None else None}


class GeometryClient:
    def __init__(self, settings):
        self.settings = settings
        self.process = None
        self.timeout = settings.get('timeout_seconds', 180)
        self.log = ROOT / 'runs/remote_cpu_20261104' / f'geometry_calls_{os.getpid()}.jsonl'
        self.known_frames = set()
        self.known_references = set()

    def record(self, **row):
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), utc_seconds=time.time(), **row)) + '\n')

    def start(self):
        if self.process is not None and self.process.poll() is None: return
        self.known_frames.clear(); self.known_references.clear()
        s = self.settings
        # Remote command is fixed and fully specified by the inspected routing file.
        self.error_stream = self.log.with_suffix('.stderr').open('a')
        self.process = subprocess.Popen(
            ['ssh', '-T', '-i', s['identity_file'], '-o', 'BatchMode=yes',
             '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
             s['host'], s['command']], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.error_stream, bufsize=0)

    def read_exact(self, size, deadline):
        data = bytearray(); fd = self.process.stdout.fileno()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError('Remote geometry worker timed out')
            block = os.read(fd, size-len(data))
            if not block: raise EOFError('Remote geometry worker closed its output')
            data.extend(block)
        return bytes(data)

    def write_all(self, payload, deadline):
        fd = self.process.stdin.fileno()
        view = memoryview(payload)
        os.set_blocking(fd, False)
        try:
            while len(view):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                    raise TimeoutError('Remote geometry upload timed out')
                try: written = os.write(fd, view[:65536])
                except BlockingIOError: continue
                view = view[written:]
        finally: os.set_blocking(fd, True)

    def metrics(self, chosen, reference, voxel, group_sizes=None):
        started = time.monotonic()
        self.start()
        arrays = dict(count=len(chosen), voxel=voxel,
                      tolerance=reference.tolerance_m, reference_status=reference.status, protocol=reference.protocol)
        if group_sizes is not None: arrays['group_sizes'] = np.asarray(group_sizes)
        frame_keys = []
        packed = set()
        for i, view in enumerate(chosen):
            values = (view.fusion_depth, view.fusion_rgb, view.fusion_k, np.linalg.inv(view.reference_pose))
            digest = hashlib.sha256()
            for value in values:
                digest.update(repr((value.shape, str(value.dtype))).encode())
                digest.update(np.ascontiguousarray(value).data)
            key = digest.hexdigest(); frame_keys.append(key)
            if key not in self.known_frames and key not in packed:
                arrays.update({f'{name}{i}': value for name, value in zip(('depth', 'rgb', 'K', 'P'), values)})
                packed.add(key)
        ref_key = hashlib.sha256(np.ascontiguousarray(reference.points).data).hexdigest()
        arrays.update(frame_keys=np.asarray(frame_keys), reference_key=ref_key)
        if ref_key not in self.known_references: arrays['reference'] = reference.points
        buffer = io.BytesIO(); np.savez(buffer, **arrays); payload = buffer.getvalue()
        if len(payload) > 512*1024*1024: raise ValueError('Remote geometry input exceeds memory bound')
        deadline = time.monotonic() + self.timeout
        self.write_all(struct.pack('!Q', len(payload)) + payload, deadline)
        size = struct.unpack('!Q', self.read_exact(8, deadline))[0]
        if size > 1024*1024: raise ValueError('Unexpected remote response size')
        reply = json.loads(self.read_exact(size, deadline))
        if not reply.get('ok'): raise RuntimeError(reply.get('error', 'Remote geometry failed'))
        if reply.get('reference_identity') != reference.identity(): raise ValueError('Remote reference identity mismatch')
        metrics = reply['metrics']
        required = ('precision', 'recall', 'f_score', 'supported_coverage', 'reference_points', 'predicted_points')
        rows = [metrics] if group_sizes is None else metrics
        if group_sizes is not None and len(rows) != len(group_sizes): raise ValueError('Wrong number of remote groups')
        for row in rows:
            if any(k not in row for k in required): raise ValueError('Incomplete remote geometry result')
            if any(not np.isfinite(row[k]) for k in required): raise ValueError('Nonfinite remote geometry result')
        self.known_frames.update(frame_keys); self.known_references.add(ref_key)
        self.known_frames.difference_update(reply.get('evicted_frames', []))
        self.known_references.difference_update(reply.get('evicted_references', []))
        self.record(backend='remote', views=len(chosen), input_bytes=len(payload), seconds=time.monotonic()-started,
                    worker_seconds=reply['worker_seconds'], remote_cache_bytes=reply.get('cache_bytes'))
        return metrics

    def close(self):
        if self.process is not None:
            # EOF terminates only this client's private worker; no other job is signalled.
            try: self.process.stdin.close()
            except OSError: pass
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
            self.process.stdout.close(); self.error_stream.close()
            self.process = None


def geometry_metrics(chosen, reference, voxel, local):
    global _client, _disabled
    if _disabled: return local()
    if _client is not None and not _client.settings.get('single_enabled', True): return local()
    if _client is None:
        if not CONFIG.is_file(): return local()
        settings = json.loads(CONFIG.read_text())
        if not settings.get('enabled') or not settings.get('single_enabled', True): return local()
        _client = GeometryClient(settings); atexit.register(_client.close)
    try:
        return _client.metrics(chosen, reference, voxel)
    except (OSError, ValueError, RuntimeError, EOFError, TimeoutError) as error:
        _client.record(backend='local_fallback', error=str(error))
        _client.close(); _disabled = True
        print(f'Remote geometry unavailable; continuing locally: {error}', flush=True)
        return local()


def geometry_metrics_batch(choices, reference, voxel, local):
    """Independent evaluation policies share one transfer and run on four CPU workers."""
    global _client, _disabled
    if _disabled: return local()
    if _client is None:
        if not CONFIG.is_file(): return local()
        settings = json.loads(CONFIG.read_text())
        if not settings.get('enabled'): return local()
        _client = GeometryClient(settings); atexit.register(_client.close)
    try:
        return _client.metrics([view for chosen in choices for view in chosen], reference, voxel,
                               group_sizes=[len(chosen) for chosen in choices])
    except (OSError, ValueError, RuntimeError, EOFError, TimeoutError) as error:
        _client.record(backend='local_fallback', error=str(error))
        _client.close(); _disabled = True
        print(f'Remote evaluation batch unavailable; continuing locally: {error}', flush=True)
        return local()
