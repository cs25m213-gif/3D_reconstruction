"""Phone API input for the causal COLMAP mapper, with a local live viewer."""
import argparse
import base64
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
from threading import Lock, Thread
import time

import numpy as np

from .online import OnlineReconstructor
from .phone_source import PhoneAPISource
from .session_reconstruct import read_model, save_json


class LiveMonitor:
    def __init__(self, source, runner):
        self.source, self.runner = source, runner
        self.last_frame = {}
        self.lock = Lock()
        self.cached_model = None
        self.map_data = {'points': [], 'trajectory': [], 'revision': -1}
        self.message = 'Capturing API frames and waiting for enough views'

    def progress(self, event):
        with self.lock:
            self.last_frame = event

    def status(self):
        online = self.runner.state.get('online', {})
        acquisition = self.source.status()
        updates = online.get('map_updates', [])
        accepted = [u for u in updates if u.get('accepted')]
        imu = self.last_frame.get('imu', {})
        return {**acquisition, 'processed_frames': online.get('sampled_frames', acquisition['processed_frames']),
            'message': self.message,
            'state': 'draining_buffer' if self.source.stop.is_set() else 'running',
            'backend': self.runner.config.get('sfm_backend', 'colmap'),
            'keyframes': len(online.get('keyframes', [])),
            'accepted_updates': len(accepted), 'rejected_updates': len(updates)-len(accepted),
            'registered_cameras': accepted[-1].get('registered_frames', 0) if accepted else 0,
            'sparse_points': accepted[-1].get('sparse_points', 0) if accepted else 0,
            'reprojection_error_px': accepted[-1].get('median_reprojection_error_px') if accepted else None,
            'source_timestamp_s': online.get('last_timestamp_s'),
            'last_selection_reason': self.last_frame.get('reason'),
            'imu_samples': imu.get('count', 0), 'imu_rate_hz': imu.get('effective_rate_hz'),
            'imu_window_valid': imu.get('valid', False),
            'rotation_compensated': self.last_frame.get('rotation_compensated', False),
            'frame_clock_source': self.last_frame.get('timing', {}).get('clock_source'),
            'coordinates': 'local_visual_units', 'dense_mode': 'optional final OpenMVS stage'}

    def map(self):
        model = self.runner.state.get('online', {}).get('best_text')
        with self.lock:
            if not model or model == self.cached_model:
                return self.map_data
            _, poses, points = read_model(Path(model))
            step = max(1, int(np.ceil(len(points)/30000)))
            xyzrgb = [p['xyz'] + p.get('rgb', [180, 180, 180]) for p in points[::step]]
            trajectory = []
            for pose in poses:
                w, x, y, z = pose['quaternion_wxyz']
                r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                              [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                              [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
                trajectory.append((-r.T @ np.asarray(pose['translation_world_to_camera'])).tolist())
            self.map_data = {'points': xyzrgb, 'trajectory': trajectory,
                             'revision': len(self.runner.state['online']['map_updates'])}
            self.cached_model = model
            return self.map_data


def dashboard_server(monitor, host='127.0.0.1', port=8090):
    page = Path(__file__).with_name('phone_dashboard.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split('?', 1)[0]
            if route in ('/', '/index.html'):
                body, content_type = page, 'text/html; charset=utf8'
            elif route == '/api/status':
                body, content_type = json.dumps(monitor.status()).encode(), 'application/json'
            elif route == '/api/map':
                body, content_type = json.dumps(monitor.map()).encode(), 'application/json'
            elif route == '/api/frame.jpg':
                body, content_type = monitor.source.latest_jpeg, 'image/jpeg'
                if body is None:
                    self.send_error(503, 'Waiting for first phone image')
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path != '/api/stop':
                self.send_error(404)
                return
            monitor.source.stop.set()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"stopping_capture_and_draining_buffer"}')

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def write_final_viewer(output, monitor):
    """Embed the accepted sparse map so the viewer works after the server exits."""
    status = {**monitor.status(), 'state': 'complete', 'message': 'Saved sparse reconstruction'}
    frame = monitor.source.latest_jpeg
    if frame is None:
        images = sorted(monitor.source.frames_dir.glob('*.jpg'))
        frame = images[-1].read_bytes() if images else None
    payload = {'FINAL_MAP': monitor.map(), 'FINAL_STATUS': status,
               'FINAL_FRAME': 'data:image/jpeg;base64,' + base64.b64encode(frame).decode() if frame else ''}
    def js(value):
        return json.dumps(value, separators=(',', ':')).replace('<', '\\u003c')
    prefix = '<script>\n' + ''.join(f'window.{name}={js(value)};\n' for name, value in payload.items()) + '</script>\n'
    template = Path(__file__).with_name('phone_dashboard.html').read_text()
    (output / 'viewer.html').write_text(template.replace('<script>\n', prefix + '<script>\n', 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phone-url', '--source-url', required=True)
    parser.add_argument('--session', type=Path, default=Path('runs')/('phone_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent.parent/'configs/cpu_online.json')
    parser.add_argument('--sfm-backend', choices=('pycolmap', 'colmap'), default='pycolmap')
    parser.add_argument('--colmap', type=Path, default=Path(shutil.which('colmap') or '/usr/bin/colmap'))
    parser.add_argument('--openmvs', type=Path, default=Path('apps/openmvs/bin/OpenMVS'))
    parser.add_argument('--calibration', type=Path, help='Explicit pixel intrinsics JSON; otherwise discover via API')
    parser.add_argument('--strict-intrinsics', action='store_true')
    parser.add_argument('--imu-calibration', type=Path, help='JSON with calibrated camera_from_imu 3x3 rotation')
    parser.add_argument('--save-imu-windows', action='store_true')
    parser.add_argument('--capture-fps', type=float, default=3.)
    parser.add_argument('--buffer-seconds', type=float, default=.5)
    parser.add_argument('--phone-pin', default=os.environ.get('GEOSTREAM_PHONE_PIN'),
                        help='Phone server PIN; GEOSTREAM_PHONE_PIN avoids exposing it in the command')
    parser.add_argument('--stream-resolution', choices=('1080p', '720p', '540p'))
    parser.add_argument('--jpeg-quality', type=int, choices=range(80, 101), metavar='80..100')
    parser.add_argument('--camera-id', help='Select a camera lens before capture; use a new session for another lens')
    parser.add_argument('--macro-mode', choices=('on', 'off'), help='Set macro focus before capture')
    parser.add_argument('--record-phone', action='store_true', help='Also record the phone MP4 and sensor files')
    parser.add_argument('--download-phone-session', action='store_true',
                        help='After --record-phone, download the newly created phone ZIP')
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finalize-dense', action='store_true', help='Run installed CPU OpenMVS after capture stops')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--snapshot-endpoint', default='/snapshot.jpg')
    parser.add_argument('--telemetry-endpoint', default='/telemetry.json')
    parser.add_argument('--stream-endpoint', default='/stream')
    parser.add_argument('--intrinsics-endpoint', default='/camera_intrinsics.json')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('max-frames must be >= 1')
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error('max-seconds must be positive')
    if args.download_phone_session and not args.record_phone:
        parser.error('--download-phone-session requires --record-phone')
    config = json.loads(args.config.read_text())
    config['threads'] = min(config['threads'], len(os.sched_getaffinity(0)))
    config['sfm_backend'] = args.sfm_backend
    config['mapping_timeout_seconds'] = config.get('mapping_timeout_seconds', 120)
    config['save_imu_windows'] = args.save_imu_windows
    if args.sfm_backend == 'pycolmap' and importlib.util.find_spec('pycolmap') is None:
        parser.error('Install requirements-live.txt to use PyCOLMAP')
    if args.sfm_backend == 'colmap' and not args.colmap.is_file():
        parser.error('COLMAP executable missing: provide --colmap')
    if args.finalize_dense:
        missing = [name for name in ('InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'TextureMesh')
                   if not (args.openmvs/name).is_file()]
        if missing or not args.colmap.is_file():
            parser.error('Final dense stage requires COLMAP CLI and CPU OpenMVS executables: ' + ', '.join(missing))
        for module in ('open3d', 'trimesh'):
            if importlib.util.find_spec(module) is None:
                parser.error(f'Final dense stage requires installed {module}')
    if args.imu_calibration:
        rotation = np.asarray(json.loads(args.imu_calibration.read_text())['camera_from_imu'], float)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
            parser.error('camera_from_imu must be a calibrated proper 3x3 rotation')
        config['camera_from_imu'] = rotation.tolist()
    source = PhoneAPISource(args.session, args.phone_url, args.capture_fps, args.buffer_seconds,
        args.max_seconds, args.max_frames, args.resume, args.calibration, args.strict_intrinsics,
        config['focal_ratio'], args.snapshot_endpoint, args.telemetry_endpoint,
        args.stream_endpoint, args.intrinsics_endpoint, args.phone_pin,
        args.stream_resolution, args.jpeg_quality, args.camera_id,
        None if args.macro_mode is None else args.macro_mode == 'on',
        args.record_phone, args.download_phone_session)
    source.prepare()
    config['camera_intrinsics'] = source.calibration
    config['calibration_exact'] = source.calibration_exact
    output = args.output or args.session/'reconstruction_online_cpu'
    runner = OnlineReconstructor(args.session, output, config, args.colmap, args.openmvs, args.resume)
    monitor = LiveMonitor(source, runner)
    runner.on_progress = monitor.progress
    server = dashboard_server(monitor, args.host, args.port)
    web = Thread(target=server.serve_forever, name='live-dashboard', daemon=True)
    web.start()
    previous_sigint = signal.signal(signal.SIGINT, lambda *_: source.stop.set())
    print(f'Live COLMAP dashboard: http://{args.host}:{server.server_port}/', flush=True)
    print(f'Capture cache: {source.root}', flush=True)
    started = time.monotonic()
    try:
        runner.run_online(finalize=args.finalize_dense, frame_source=source)
        monitor.message = 'Capture finalized; sparse model saved'
        save_json(output/'api_capture_status.json', monitor.status())
        save_json(output/'dashboard_map.json', monitor.map())
        write_final_viewer(output, monitor)
    except Exception as error:
        monitor.message = str(error)
        save_json(output/'online_failure.json', {'status':'failed', 'error':str(error),
            'seconds_current_invocation':time.monotonic()-started,
            'map_updates':runner.state.get('online', {}).get('map_updates', []), 'capture_cache':str(source.root)})
        raise
    finally:
        source.finish()
        signal.signal(signal.SIGINT, previous_sigint)
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
