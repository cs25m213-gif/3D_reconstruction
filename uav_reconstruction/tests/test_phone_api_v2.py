"""Exercise the revised authenticated phone API across setup, capture and export."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
from threading import Thread
import time
import unittest
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

import cv2
import numpy as np

from adaptive_uav.phone_source import PhoneAPISource


@contextmanager
def updated_phone_server(pin='2468', images=None):
    image = np.random.default_rng(42).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, 'w') as archive:
        archive.writestr('video.mp4', b'mock phone video')
        archive.writestr('imu.csv', 'time,ax,ay,az\n')
        archive.writestr('gps.csv', 'time,latitude,longitude\n')
        archive.writestr('metadata.json', '{}')
    state = {'camera_id': '0', 'macro': False, 'quality': None, 'snapshots': 0,
             'events': [], 'recording': False, 'sessions': [{'id': 'old_session'}]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, content_type='application/json', headers=None):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.handle_request()

        def do_GET(self):
            self.handle_request()

        def handle_request(self):
            if self.headers.get('X-Auth-Token') != pin:
                self.send_error(401)
                return
            route = urlsplit(self.path)
            params = parse_qs(route.query)
            now_ms = time.time() * 1000
            state['events'].append((self.command, route.path, params))
            if route.path == '/camera/switch':
                state['camera_id'] = params['id'][0]
                self.respond({'status': 'switched', 'id': state['camera_id']})
            elif route.path == '/camera/macro':
                state['macro'] = params['enabled'][0] == 'true'
                self.respond({'status': 'ok', 'macro': state['macro']})
            elif route.path == '/stream_config':
                state['quality'] = params
                self.respond({'status': 'ok', 'resolution': params.get('res'), 'quality': params.get('q')})
            elif route.path == '/telemetry.json':
                self.respond({'wall_time_ms': now_ms, 'elapsed_ns': int(now_ms * 1e6),
                    'camera': {'current_id': state['camera_id'], 'current_name': 'Macro Camera',
                               'macro_mode': state['macro'], 'available': [
                                   {'id': '0', 'name': 'Rear Main'}, {'id': '2', 'name': 'Macro Camera'}]},
                    'imu': {'accel': [0, 9.81, 0], 'gyro': [0, 0, 0]},
                    'gps': {'has_fix': False}, 'sessions': state['sessions']})
            elif route.path == '/camera_intrinsics.json':
                self.send_error(404)  # The revised API does not list this endpoint.
            elif route.path == '/snapshot.jpg':
                number = state['snapshots']
                state['snapshots'] += 1
                frame = images[min(number * 2, len(images)-1)].read_bytes() if images else encoded.tobytes()
                self.respond(frame, 'image/jpeg',
                    {'X-Capture-Wall-Time-Ms': str(now_ms), 'X-Frame-Sequence': str(number)})
            elif route.path == '/record/start':
                state['recording'] = True
                self.respond({'status': 'started'})
            elif route.path == '/record/stop':
                state['recording'] = False
                state['sessions'].append({'id': 'new_session'})
                self.respond({'status': 'stopped'})
            elif route.path == '/download/new_session':
                self.respond(archive_buffer.getvalue(), 'application/zip')
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class RevisedPhoneAPITests(unittest.TestCase):
    def calibration(self, directory):
        path = Path(directory) / 'calibration.json'
        path.write_text(json.dumps({'width': 160, 'height': 120, 'fx': 140, 'fy': 140,
                                    'cx': 80, 'cy': 60, 'distortion': [0, 0, 0, 0, 0]}))
        return path

    def test_authenticated_setup_recording_and_download_preserve_camera_identity(self):
        with tempfile.TemporaryDirectory() as directory, updated_phone_server() as (url, phone):
            source = PhoneAPISource(Path(directory) / 'session', url, capture_fps=20,
                buffer_seconds=.05, max_frames=4, calibration=self.calibration(directory),
                stream_endpoint='', auth_token='2468', stream_resolution='1080p',
                jpeg_quality=100, camera_id='2', macro_mode=True,
                record_phone=True, download_phone_session=True)
            source.prepare()
            frames = list(source.frames())
            self.assertEqual(len(frames), 4)
            self.assertEqual(phone['quality'], {'res': ['1080p'], 'q': ['100']})
            self.assertFalse(phone['recording'])
            self.assertEqual(source.status()['camera']['current_id'], '2')
            self.assertTrue(source.status()['camera']['macro_mode'])
            self.assertEqual([frame.index for frame in frames], list(range(4)))
            self.assertEqual(source.remote_archive.name, 'phone_new_session.zip')
            with ZipFile(source.remote_archive) as archive:
                self.assertEqual(set(archive.namelist()), {'video.mp4', 'imu.csv', 'gps.csv', 'metadata.json'})
            calls = [(method, path) for method, path, _ in phone['events']]
            self.assertLess(calls.index(('GET', '/camera/switch')), calls.index(('GET', '/snapshot.jpg')))
            self.assertIn(('POST', '/record/start'), calls)
            self.assertIn(('POST', '/record/stop'), calls)
            self.assertNotIn('2468', source.meta_path.read_text())
            self.assertNotIn('2468', source.acquisition_path.read_text())
            self.assertNotIn('2468', source.telemetry_path.read_text())
            recording = json.loads((source.session / 'recording.json').read_text())
            self.assertEqual(recording['frames'], recording['mp4_frames'])
            self.assertEqual(recording['phone_archive'], str(source.remote_archive))
            with self.assertRaisesRegex(RuntimeError, 'settings changed'):
                PhoneAPISource(source.session, url, resume=True, calibration=self.calibration(directory),
                    auth_token='2468', camera_id='0', macro_mode=True,
                    stream_resolution='1080p', jpeg_quality=100)

    def test_bad_pin_fails_before_capture_and_query_pin_is_removed_from_metadata(self):
        with tempfile.TemporaryDirectory() as directory, updated_phone_server() as (url, phone):
            calibration = self.calibration(directory)
            bad = PhoneAPISource(Path(directory) / 'bad', url, calibration=calibration, auth_token='wrong')
            with self.assertRaisesRegex(PermissionError, 'PIN or token'):
                bad.prepare()
            self.assertEqual(phone['snapshots'], 0)
            good = PhoneAPISource(Path(directory) / 'good', url + '/?key=2468',
                                  calibration=calibration, max_frames=1, stream_endpoint='')
            good.prepare()
            self.assertEqual(len(list(good.frames())), 1)
            self.assertEqual(good.urls[0], url)
            self.assertNotIn('2468', good.meta_path.read_text())
            path_source = PhoneAPISource(Path(directory) / 'path', url + '/2468',
                calibration=calibration, max_frames=1, stream_endpoint='')
            path_source.prepare()
            self.assertEqual(len(list(path_source.frames())), 1)
            self.assertEqual(path_source.urls[0], url)
            self.assertNotIn('2468', path_source.meta_path.read_text())

    def test_camera_change_stops_session_before_mixing_lenses(self):
        with tempfile.TemporaryDirectory() as directory, updated_phone_server() as (url, _):
            source = PhoneAPISource(Path(directory) / 'session', url,
                calibration=self.calibration(directory), auth_token='2468')
            source.prepare()
            self.assertEqual(source.camera_identity['id'], '0')
            source._add_telemetry({'wall_time_ms': time.time() * 1000,
                'camera': {'current_id': '2', 'macro_mode': False}}, time.time() * 1000)
            self.assertTrue(source.stop.is_set())
            self.assertIn('lens or macro mode changed', source.fatal_error)


if __name__ == '__main__':
    unittest.main()
