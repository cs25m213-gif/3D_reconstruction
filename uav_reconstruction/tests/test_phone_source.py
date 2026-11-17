"""Phone HTTP capture preserves frames while the mapper is busy or stopping."""
import csv
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Lock, Thread
import time
import unittest

import cv2
import numpy as np

from adaptive_uav.imu import summarize_imu
from adaptive_uav.phone_source import PhoneAPISource, normalize_intrinsics


CALIBRATION = dict(width=160, height=120, fx=140., fy=139., cx=80., cy=60., distortion=[0.] * 5)


@contextmanager
def phone_server(capture_headers=True, telemetry_available=True):
    """A phone clock offset by five seconds, with genuine JPEG responses."""
    image = np.random.default_rng(12).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('JPEG encoder unavailable')
    jpeg = encoded.tobytes()
    counter = {'value': 0}
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            stamp = time.time() * 1000. - 5000.
            if self.path == '/snapshot.jpg':
                with lock:
                    sequence = counter['value']
                    counter['value'] += 1
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                if capture_headers:
                    self.send_header('X-Capture-Wall-Time-Ms', str(stamp))
                    self.send_header('X-Frame-Sequence', str(sequence))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path == '/camera_intrinsics.json':
                payload = CALIBRATION
            elif self.path == '/telemetry.json':
                if not telemetry_available:
                    self.send_error(404)
                    return
                payload = {'wall_time_ms': stamp, 'elapsed_ns': int(stamp * 1e6),
                           'state': 'IDLE', 'gps': {'has_fix': True, 'latitude': 13.6},
                           'imu': {'accel': [0., 9.81, 0.], 'gyro': [0., 0.1, 0.],
                                   'rates_hz': {'accel': 200, 'gyro': 200}}}
            else:
                self.send_error(404)
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def video_count(path):
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                return count
            count += 1
    finally:
        capture.release()


class PhoneSourceTests(unittest.TestCase):
    def test_capture_keeps_running_while_consumer_is_busy_and_drains_on_stop(self):
        with tempfile.TemporaryDirectory() as directory, phone_server() as url:
            source = PhoneAPISource(Path(directory), url, capture_fps=20, buffer_seconds=.1,
                                    max_frames=8, stream_endpoint='')
            source.prepare()
            signature = source.signature
            frames = source.frames()
            first = next(frames)
            source.thread.join(timeout=3)  # The consumer does no work during capture.
            self.assertFalse(source.thread.is_alive())
            self.assertEqual(source.received, 8)
            self.assertGreater(source.buffered, 0)
            self.assertEqual(len(records(source.acquisition_path)), 8)
            self.assertEqual(len(list(source.frames_dir.glob('*.jpg'))), 8)
            self.assertTrue(source.stop.is_set())
            captured = [first] + list(frames)
            self.assertEqual([frame.index for frame in captured], list(range(8)))
            self.assertEqual(source.processed, 8)
            self.assertEqual(source.signature, signature)
            self.assertEqual(len(records(source.manifest_path)), 8)
            self.assertEqual(video_count(Path(directory) / 'video.mp4'), 8)
            recording = json.loads((Path(directory) / 'recording.json').read_text())
            self.assertEqual(recording['frames'], recording['mp4_frames'])
            with (Path(directory) / 'video_frames.csv').open(newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            for frame in captured:
                self.assertEqual(frame.timing['clock_source'], 'phone_exposure_header')
                self.assertAlmostEqual(frame.sensor_timestamp_s * 1000,
                                       frame.timing['phone_capture_wall_ms'], places=3)
                self.assertIsNotNone(frame.timing['phone_frame_sequence'])
                self.assertEqual(frame.telemetry['gps']['latitude'], 13.6)
                self.assertLess(frame.timing['telemetry_age_ms'], 250)
            samples = [sample for frame in captured for sample in frame.imu_samples]
            self.assertTrue(samples)
            self.assertEqual({sample['sample_kind'] for sample in samples}, {'telemetry_latest'})

    def test_explicit_stop_flushes_pending_holdback_to_disk_and_mp4(self):
        with tempfile.TemporaryDirectory() as directory, phone_server() as url:
            source = PhoneAPISource(Path(directory), url, capture_fps=20, buffer_seconds=3.,
                                    stream_endpoint='')
            source.prepare()
            source.start()
            deadline = time.monotonic() + 3
            while source.received < 3 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertGreaterEqual(source.received, 3)
            self.assertEqual(source.buffered, 0)
            source.finish()
            self.assertFalse(source.thread.is_alive())
            self.assertEqual(len(records(source.manifest_path)), source.received)
            self.assertEqual(source.buffered, source.received)
            self.assertEqual(video_count(Path(directory) / 'video.mp4'), source.received)

    def test_headerless_capture_reports_estimated_clock_and_prepared_signature_is_stable(self):
        with tempfile.TemporaryDirectory() as directory, phone_server(capture_headers=False) as url:
            source = PhoneAPISource(Path(directory), url, capture_fps=20, buffer_seconds=.05,
                                    max_frames=3, stream_endpoint='')
            source.prepare()
            signature = source.signature
            captured = list(source.frames())
            self.assertEqual(source.signature, signature)
            for frame in captured:
                self.assertEqual(frame.timing['clock_source'], 'receiver_clock_estimate')
                self.assertIsNone(frame.timing['phone_capture_wall_ms'])
                self.assertIsNone(frame.timing['phone_frame_sequence'])
                self.assertLess(abs(frame.timing['receiver_wall_ms'] -
                                    frame.sensor_timestamp_s * 1000 - 5000.), 250)

    def test_resume_replays_cursor_and_max_frames_limits_only_new_capture(self):
        with tempfile.TemporaryDirectory() as directory, phone_server() as url:
            root = Path(directory)
            first = PhoneAPISource(root, url, capture_fps=20, buffer_seconds=.05,
                                   max_frames=3, stream_endpoint='')
            first.prepare()
            self.assertEqual([frame.index for frame in first.frames()], [0, 1, 2])
            resumed = PhoneAPISource(root, url, capture_fps=20, buffer_seconds=.05,
                                     max_frames=2, resume=True, stream_endpoint='')
            resumed.prepare()
            self.assertEqual(resumed.signature, first.signature)
            self.assertEqual([frame.index for frame in resumed.frames(after_index=0)], [1, 2, 3, 4])
            self.assertEqual(resumed.received, 5)
            self.assertEqual(video_count(root / 'video.mp4'), 5)
            self.assertEqual([row['index'] for row in records(resumed.manifest_path)], list(range(5)))
            with self.assertRaisesRegex(RuntimeError, 'Resume phone URL changed'):
                PhoneAPISource(root, 'http://127.0.0.1:1', resume=True)
            changed = root / 'changed_calibration.json'
            changed.write_text(json.dumps({**CALIBRATION, 'fx': 160.}))
            with self.assertRaisesRegex(RuntimeError, 'Resume calibration changed'):
                PhoneAPISource(root, url, resume=True, calibration=changed)

    def test_headerless_capture_without_clock_offset_has_no_inertial_pairing(self):
        with tempfile.TemporaryDirectory() as directory, phone_server(
                capture_headers=False, telemetry_available=False) as url:
            source = PhoneAPISource(Path(directory), url, capture_fps=20, buffer_seconds=.05,
                                    max_frames=2, stream_endpoint='')
            source.prepare()
            captured = list(source.frames())
            self.assertEqual(len(captured), 2)
            for frame in captured:
                self.assertEqual(frame.timing['clock_source'], 'receiver_unsynchronized')
                self.assertEqual(frame.imu_samples, [])
                self.assertEqual(frame.telemetry, {})
                self.assertIsNone(frame.timing['telemetry_age_ms'])

    def test_low_rate_telemetry_is_not_promoted_to_reported_hardware_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = PhoneAPISource(Path(directory), 'http://127.0.0.1:1')
            for index in range(11):
                stamp = 100000. + index * 100.
                source._add_telemetry({'wall_time_ms': stamp,
                    'imu': {'accel': [0., 9.81, 0.], 'gyro': [0., .2, 0.],
                            'rates_hz': {'accel': 200, 'gyro': 200}}}, stamp + 5000.)
            row = source._enrich({'sensor_timestamp_s': 101., 'previous_sensor_timestamp_s': 100.,
                                  'clock_source': 'phone_exposure_header'})
            features = summarize_imu(row['imu_samples'], 100., 101.)
            self.assertEqual(features['sample_count'], 11)
            self.assertAlmostEqual(features['effective_rate_hz'], 10.)
            self.assertTrue(features['low_sample_rate'])
            self.assertFalse(features['rotation_valid'])

    def test_resume_repairs_torn_journal_tail_before_appending_new_frames(self):
        with tempfile.TemporaryDirectory() as directory, phone_server() as url:
            root = Path(directory)
            first = PhoneAPISource(root, url, capture_fps=20, buffer_seconds=.05,
                                   max_frames=3, stream_endpoint='')
            first.prepare()
            list(first.frames())
            for path in (first.acquisition_path, first.manifest_path):
                with path.open('a') as stream:
                    stream.write('{"index":')  # Simulate interruption halfway through an append.
            resumed = PhoneAPISource(root, url, capture_fps=20, buffer_seconds=.05,
                                     max_frames=2, resume=True, stream_endpoint='')
            resumed.prepare()
            self.assertEqual([frame.index for frame in resumed.frames(after_index=2)], [3, 4])
            self.assertEqual([row['index'] for row in records(resumed.acquisition_path)], list(range(5)))
            self.assertEqual([row['index'] for row in records(resumed.manifest_path)], list(range(5)))
            self.assertEqual(video_count(root / 'video.mp4'), 5)

    def test_batched_imu_elapsed_clock_is_mapped_to_phone_wall_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            source = PhoneAPISource(Path(directory), 'http://127.0.0.1:1')
            samples = [{'elapsed_ns': 1_000_000_000 + index * 5_000_000,
                        'accel': [0., 9.81, 0.], 'gyro': [0., .2, 0.]}
                       for index in range(21)]
            source._add_telemetry({'wall_time_ms': 1100., 'elapsed_ns': 1_100_000_000,
                                   'imu': {'samples': samples}}, arrival_ms=6100.)
            row = source._enrich({'sensor_timestamp_s': 1.1, 'previous_sensor_timestamp_s': 1.,
                                  'clock_source': 'phone_exposure_header'})
            self.assertEqual(len(row['imu_samples']), 21)
            self.assertEqual({sample['sample_kind'] for sample in row['imu_samples']}, {'raw_batch'})
            features = summarize_imu(row['imu_samples'], 1., 1.1)
            self.assertAlmostEqual(features['effective_rate_hz'], 200., places=6)
            self.assertTrue(features['rotation_valid'])

    def test_normalize_intrinsics_rejects_invalid_principal_point(self):
        self.assertIsNone(normalize_intrinsics({**CALIBRATION, 'cx': 0., 'cy': 0.}))
        self.assertIsNone(normalize_intrinsics({**CALIBRATION, 'fx': float('nan')}))
        self.assertIsNone(normalize_intrinsics({**CALIBRATION, 'distortion': [0., 0.]}))
        self.assertEqual(normalize_intrinsics({'camera_intrinsics': CALIBRATION}), CALIBRATION)


if __name__ == '__main__':
    unittest.main()
