"""Durable phone snapshot and telemetry acquisition, independent of SfM work."""
from collections import deque
from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
from queue import Queue, Empty
import re
from threading import Event, Lock, Thread
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

import cv2
import numpy as np


@dataclass
class SourceFrame:
    index: int
    timestamp_s: float
    image: np.ndarray
    decode_seconds: float = 0.
    sensor_timestamp_s: float | None = None
    telemetry: dict = field(default_factory=dict)
    imu_samples: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)


def _records(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A process may have died during its final journal append.
            continue
    return rows


def _repair_journal_tail(path):
    """Prevent a interrupted JSON append from swallowing the next record."""
    if not path.exists():
        return
    with path.open('rb+') as stream:
        data = stream.read()
        if not data or data.endswith(b'\n'):
            return
        boundary = data.rfind(b'\n') + 1
        try:
            json.loads(data[boundary:])
        except (ValueError, UnicodeDecodeError):
            stream.truncate(boundary)
        else:
            stream.seek(0, 2)
            stream.write(b'\n')


def normalize_intrinsics(payload, width=None, height=None):
    """Accept explicit pixel intrinsics; never mark a zero principal point exact."""
    for key in ('intrinsics', 'camera_intrinsics', 'calibration', 'camera'):
        if isinstance(payload.get(key), dict):
            result = normalize_intrinsics(payload[key], width, height)
            if result:
                return result
    if all(key in payload for key in ('fx', 'fy', 'cx', 'cy')):
        try:
            w, h = int(payload.get('width', width)), int(payload.get('height', height))
            values = [float(payload[key]) for key in ('fx', 'fy', 'cx', 'cy')]
            distortion = list(map(float, payload.get('distortion', [0.] * 5)))
            if w <= 0 or h <= 0 or not np.isfinite(values + distortion).all():
                return None
            fx, fy, cx, cy = values
            if min(fx, fy) <= 0 or not (0 < cx < w and 0 < cy < h):
                return None
            if len(distortion) not in (0, 4, 5, 8):
                return None
            return dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy,
                        distortion=distortion or [0.] * 5)
        except (ValueError, TypeError):
            return None
    return None


class PhoneAPISource:
    """Persist frames before yielding, and keep acquiring while COLMAP blocks.

    The disk journal is the buffer. The queue contains paths and small metadata,
    rather than decoded images. Telemetry histories are paired after a short
    holdback; exposure timestamps are only exact when supplied by the phone.
    """
    def __init__(self, session, phone_url, capture_fps=3., buffer_seconds=.5,
                 max_seconds=None, max_frames=None, resume=False, calibration=None,
                 strict_intrinsics=False, focal_ratio=.7666,
                 snapshot_endpoint='/snapshot.jpg', telemetry_endpoint='/telemetry.json',
                 stream_endpoint='/stream', intrinsics_endpoint='/camera_intrinsics.json',
                 auth_token=None, stream_resolution=None, jpeg_quality=None,
                 camera_id=None, macro_mode=None, record_phone=False, download_phone_session=False):
        self.session = Path(session).resolve()
        self.root = self.session / 'capture'
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.root / 'frames'
        self.frames_dir.mkdir(exist_ok=True)
        parsed = urlparse(phone_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('phone URL must start with http:// or https://')
        query = parse_qs(parsed.query)
        supplied_token = next((query[key][0] for key in ('key', 'auth', 'token', 'pwd') if query.get(key)), None)
        if parsed.query and not supplied_token:
            raise ValueError('phone URL query is unsupported; supply a base URL')
        path_parts = [part for part in parsed.path.split('/') if part]
        if len(path_parts) > 1:
            raise ValueError('phone URL path must be empty or contain a single PIN segment')
        path_token = unquote(path_parts[0]) if path_parts else None
        self.auth_token = auth_token or supplied_token or path_token
        clean_url = urlunparse((parsed.scheme, parsed.netloc, '', '', '', '')).rstrip('/')
        self.urls = [clean_url]
        if parsed.port == 8080:
            self.urls.append(self.urls[0].rsplit(':', 1)[0] + ':8081')
        if stream_resolution not in (None, '1080p', '720p', '540p'):
            raise ValueError('stream resolution must be 1080p, 720p, or 540p')
        if jpeg_quality is not None and not 80 <= jpeg_quality <= 100:
            raise ValueError('JPEG quality must be between 80 and 100')
        if macro_mode not in (None, True, False):
            raise ValueError('macro_mode must be true, false, or unspecified')
        self.requested_settings = {'stream_resolution': stream_resolution, 'jpeg_quality': jpeg_quality,
                                   'camera_id': str(camera_id) if camera_id is not None else None,
                                   'macro_mode': macro_mode}
        if download_phone_session and not record_phone:
            raise ValueError('download_phone_session requires record_phone')
        self.record_phone, self.download_phone_session = record_phone, download_phone_session
        self.remote_recording = False
        self.remote_record_error = None
        self.remote_archive = None
        self.prior_session_ids = set()
        if capture_fps <= 0 or buffer_seconds < 0:
            raise ValueError('capture fps must be positive; buffer seconds must be nonnegative')
        self.capture_fps, self.buffer_seconds = capture_fps, buffer_seconds
        self.max_seconds, self.max_frames = max_seconds, max_frames
        self.snapshot_endpoint, self.telemetry_endpoint = snapshot_endpoint, telemetry_endpoint
        self.stream_endpoint, self.intrinsics_endpoint = stream_endpoint, intrinsics_endpoint
        self.stop = Event()
        self.telemetry_ready = Event()
        self.lock = Lock()
        self.queue = Queue()
        self.history = deque(maxlen=12000)
        self.imu_history = deque(maxlen=24000)
        self.offsets = deque(maxlen=100)
        self.received = self.processed = self.decode_errors = 0
        self.error = None
        self.fatal_error = None
        self.connected_url = None
        self.latest_jpeg = None
        self.capture_size = None
        self.latest_packet = {}
        self.last_frame_monotonic = None
        self.thread = self.telemetry_thread = None
        self.started = None
        self._closed = False
        self._previous_sensor_time = None
        self.acquisition_path = self.root / 'acquisition.jsonl'
        self.manifest_path = self.root / 'frames.jsonl'
        self.telemetry_path = self.session / 'telemetry.jsonl'
        self.meta_path = self.root / 'source.json'
        for path in (self.acquisition_path, self.manifest_path, self.telemetry_path):
            _repair_journal_tail(path)
        acquisitions = _records(self.acquisition_path)
        enriched = _records(self.manifest_path)
        if acquisitions and not resume:
            raise RuntimeError('Capture exists: use --resume or a new --session folder')
        self._saved_rows = enriched
        self._acquisitions = acquisitions
        if acquisitions:
            first_image = cv2.imread(str(self.root / acquisitions[0]['image']))
            if first_image is None:
                raise RuntimeError('Cannot resume: first captured JPEG is missing or unreadable')
            self.capture_size = (first_image.shape[1], first_image.shape[0])
        self.next_index = max((r['index'] for r in acquisitions), default=-1) + 1
        self.origin_sensor_s = acquisitions[0]['sensor_timestamp_s'] if acquisitions else None
        self.origin_receiver_ms = acquisitions[0]['receiver_wall_ms'] if acquisitions else None
        self._previous_sensor_time = acquisitions[-1]['sensor_timestamp_s'] if acquisitions else None
        self.received = len(acquisitions)
        metadata = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else None
        if metadata and metadata['phone_url'] != self.urls[0]:
            raise RuntimeError('Resume phone URL changed; use a new session')
        if metadata and metadata.get('requested_settings', self.requested_settings) != self.requested_settings:
            raise RuntimeError('Resume camera or stream settings changed; use a new session')
        self.camera_identity = metadata.get('camera_identity') if metadata else None
        self.calibration = metadata.get('calibration') if metadata else None
        self.calibration_exact = metadata.get('calibration_exact', False) if metadata else False
        self.calibration_source = metadata.get('calibration_source') if metadata else None
        if calibration:
            raw = json.loads(Path(calibration).read_text())
            result = normalize_intrinsics(raw)
            if not result:
                raise ValueError('Calibration must provide valid width/height/fx/fy/cx/cy/distortion')
            if self.calibration and self.calibration != result:
                raise RuntimeError('Resume calibration changed; use a new session')
            self.calibration = result
            self.calibration_exact = True
            self.calibration_source = str(Path(calibration).resolve())
        self.strict_intrinsics, self.focal_ratio = strict_intrinsics, focal_ratio
        for row in _records(self.telemetry_path):
            self._add_telemetry(row['packet'], row['receiver_wall_ms'], persist=False)
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        # Recover JPEGs written before a crash interrupted enrichment.
        committed = {r['index'] for r in enriched}
        for row in acquisitions:
            if row['index'] not in committed:
                matured = self._enrich(row)
                with self.manifest_path.open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(matured) + '\n')
                self._saved_rows.append(matured)
        self._saved_rows.sort(key=lambda r: r['index'])

    @property
    def signature(self):
        digest = hashlib.sha256(json.dumps(self.calibration, sort_keys=True).encode()).hexdigest()
        return {'kind': 'phone_api', 'cache': str(self.root), 'phone_url': self.urls[0],
                'calibration_sha256': digest}

    @property
    def buffered(self):
        return self.queue.qsize()

    def status(self):
        age = time.monotonic() - self.last_frame_monotonic if self.last_frame_monotonic else None
        return {'phone_url': self.connected_url or self.urls[0], 'captured_frames': self.received,
                'processed_frames': self.processed, 'buffered_frames': self.buffered,
                'capture_stopped': self.stop.is_set(), 'stream_age_s': age,
                'error': self.fatal_error or self.error,
                'calibration_exact': self.calibration_exact, 'calibration_source': self.calibration_source,
                'gps': self.latest_packet.get('gps', {}), 'camera': self.latest_packet.get('camera', {}),
                'decode_errors': self.decode_errors, 'phone_recording': self.remote_recording,
                'phone_record_error': self.remote_record_error,
                'phone_archive': str(self.remote_archive) if self.remote_archive else None}

    def _open(self, base, endpoint, timeout=3, params=None, method='GET', headers=None):
        query = ('?' + urlencode(params)) if params else ''
        request_headers = {'Cache-Control': 'no-cache', **(headers or {})}
        if self.auth_token:
            request_headers['X-Auth-Token'] = self.auth_token
        request = Request(base + endpoint + query, headers=request_headers, method=method)
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as error:
            if error.code in (401, 403):
                raise PermissionError('Phone API rejected the PIN or token') from None
            raise

    def _bases(self):
        return list(dict.fromkeys(([self.connected_url] if self.connected_url else []) + self.urls))

    def _request_json(self, endpoint, params=None, method='GET', timeout=3):
        errors = []
        for base in self._bases():
            try:
                with self._open(base, endpoint, timeout=timeout, params=params, method=method) as response:
                    result = json.load(response)
                if not isinstance(result, dict) or result.get('status') in ('error', 'failed'):
                    raise ValueError('Phone API returned an unsuccessful control response')
                self.connected_url = base
                return result
            except PermissionError:
                raise
            except Exception as error:
                errors.append(str(error))
        raise ConnectionError(f'Phone API {endpoint} failed: ' + '; '.join(errors))

    def _observe_camera(self, packet):
        camera = packet.get('camera') if isinstance(packet, dict) else None
        if not isinstance(camera, dict) or camera.get('current_id') is None:
            return
        current = {'id': str(camera['current_id']), 'macro_mode': camera.get('macro_mode')}
        if self.camera_identity is None:
            self.camera_identity = current
            self._save_metadata()
        elif current['id'] != self.camera_identity['id'] or (
                current['macro_mode'] is not None and self.camera_identity.get('macro_mode') is not None
                and current['macro_mode'] != self.camera_identity['macro_mode']):
            self.fatal_error = 'Phone camera lens or macro mode changed during this session; start a new session'
            self.error = self.fatal_error
            self.stop.set()

    def prepare(self):
        settings = self.requested_settings
        if settings['camera_id'] is not None:
            response = self._request_json('/camera/switch', {'id': settings['camera_id']})
            if response.get('status') != 'switched' or str(response.get('id')) != settings['camera_id']:
                raise RuntimeError('Phone did not switch to the requested camera')
        if settings['macro_mode'] is not None:
            response = self._request_json('/camera/macro', {'enabled': str(settings['macro_mode']).lower()})
            if response.get('status') != 'ok' or response.get('macro') is not settings['macro_mode']:
                raise RuntimeError('Phone did not apply the requested macro mode')
        if settings['stream_resolution'] is not None or settings['jpeg_quality'] is not None:
            params = {}
            if settings['stream_resolution'] is not None:
                params['res'] = settings['stream_resolution']
            if settings['jpeg_quality'] is not None:
                params['q'] = settings['jpeg_quality']
            self._request_json('/stream_config', params)
        try:
            packet = self._request_json(self.telemetry_endpoint, timeout=2)
            self._observe_camera(packet)
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)
            self.latest_packet = packet
        except PermissionError:
            raise
        except RuntimeError:
            raise
        except ConnectionError:
            pass
        if self.calibration is None:
            for base in self._bases():
                for endpoint in (self.intrinsics_endpoint, self.telemetry_endpoint):
                    try:
                        with self._open(base, endpoint, timeout=2) as response:
                            raw = json.load(response)
                        result = normalize_intrinsics(raw)
                        if result:
                            self.calibration = result
                            self.calibration_exact = True
                            self.calibration_source = base + endpoint
                            break
                    except Exception:
                        pass
                if self.calibration:
                    break
        if self.calibration is None and self.strict_intrinsics:
            raise RuntimeError('No valid API intrinsics; provide --calibration or add /camera_intrinsics.json')
        if self.calibration is None:
            errors = []
            for base in self._bases():
                try:
                    with self._open(base, self.snapshot_endpoint, timeout=3) as response:
                        image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError('cannot decode calibration snapshot')
                    h, w = image.shape[:2]
                    self.calibration = dict(width=w, height=h, fx=self.focal_ratio*w,
                        fy=self.focal_ratio*w, cx=w/2, cy=h/2, distortion=[0.] * 5)
                    self.calibration_source = 'approximate_from_snapshot_resolution'
                    break
                except Exception as error:
                    errors.append(str(error))
            if self.calibration is None:
                raise ConnectionError('Phone API unavailable: ' + '; '.join(errors))
        if self.calibration:
            self._save_metadata()

    def _save_metadata(self):
        temporary = self.meta_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'phone_url': self.urls[0], 'calibration': self.calibration,
            'calibration_exact': self.calibration_exact, 'calibration_source': self.calibration_source,
            'snapshot_endpoint': self.snapshot_endpoint, 'telemetry_endpoint': self.telemetry_endpoint,
            'stream_endpoint': self.stream_endpoint, 'capture_fps': self.capture_fps,
            'camera_identity': self.camera_identity, 'requested_settings': self.requested_settings,
            'imu_rate_warning': 'rates_hz is sensor configuration, not the number of samples delivered'}, indent=2))
        temporary.replace(self.meta_path)

    def _add_telemetry(self, packet, arrival_ms, persist=True):
        if not isinstance(packet, dict):
            return
        self._observe_camera(packet)
        try:
            phone_ms = float(packet['wall_time_ms'])
        except (KeyError, TypeError, ValueError):
            return
        imu = packet.get('imu') or {}
        samples = imu.get('samples', packet.get('imu_samples', []))
        if not isinstance(samples, list):
            samples = []
        canonical = []
        for sample in samples or [imu]:
            if not isinstance(sample, dict):
                continue
            stamp = sample.get('wall_time_ms', sample.get('wall_ms'))
            if stamp is None and sample.get('elapsed_ns') is not None and packet.get('elapsed_ns') is not None:
                stamp = phone_ms + (float(sample['elapsed_ns']) - float(packet['elapsed_ns'])) / 1e6
            if stamp is None and sample is imu:
                stamp = phone_ms
            if stamp is None:
                continue
            try:
                accel, gyro = list(map(float, sample['accel'])), list(map(float, sample['gyro']))
                if len(accel) != 3 or len(gyro) != 3 or not np.isfinite(accel + gyro + [float(stamp)]).all():
                    continue
                row = {'timestamp_s': float(stamp) / 1000., 'accel': accel, 'gyro': gyro,
                       'sample_kind': 'raw_batch' if samples else 'telemetry_latest'}
                if 'orientation_quaternion_wxyz' in sample:
                    row['orientation_quaternion_wxyz'] = sample['orientation_quaternion_wxyz']
                canonical.append(row)
            except (KeyError, ValueError, TypeError):
                continue
        with self.lock:
            self.latest_packet = packet
            self.history.append((phone_ms / 1000., packet))
            self.imu_history.extend(canonical)
            self.offsets.append(arrival_ms - phone_ms)
            self.telemetry_ready.set()
        if persist:
            with self.telemetry_path.open('a', encoding='utf8') as stream:
                stream.write(json.dumps({'receiver_wall_ms': arrival_ms, 'packet': packet}) + '\n')

    def _telemetry_worker(self):
        while not self.stop.is_set():
            for base in self._bases():
                if self.stop.is_set():
                    break
                try:
                    if not self.stream_endpoint:
                        raise RuntimeError('SSE disabled')
                    with self._open(base, self.stream_endpoint, timeout=3,
                                    headers={'Accept': 'text/event-stream'}) as response:
                        if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                            raise ValueError('not SSE')
                        data = []
                        while not self.stop.is_set():
                            line = response.readline()
                            if not line:
                                break
                            line = line.decode('utf8').strip()
                            if line.startswith('data:'):
                                data.append(line[5:].strip())
                            elif not line and data:
                                try:
                                    self._add_telemetry(json.loads('\n'.join(data)), time.time() * 1000)
                                except (ValueError, TypeError):
                                    pass
                                data.clear()
                except PermissionError as error:
                    self.fatal_error = str(error)
                    self.stop.set()
                    break
                except Exception:
                    for _ in range(20):
                        if self.stop.is_set():
                            break
                        try:
                            with self._open(base, self.telemetry_endpoint, timeout=2) as response:
                                packet = json.load(response)
                            self._add_telemetry(packet, time.time() * 1000)
                        except PermissionError as error:
                            self.fatal_error = str(error)
                            self.stop.set()
                            break
                        except Exception:
                            break
                        if self.stop.wait(.1):
                            break
            self.stop.wait(.2)

    def _enrich(self, row):
        stamp = row['sensor_timestamp_s']
        start = row.get('previous_sensor_timestamp_s', stamp)
        with self.lock:
            packets = list(self.history)
            samples = list(self.imu_history)
        synchronized = row['clock_source'] != 'receiver_unsynchronized'
        nearest = min(packets, key=lambda p: abs(p[0] - stamp)) if packets and synchronized else None
        # Include bracketing samples so the consumer can interpolate its window.
        window = [s for s in samples if start - .15 <= s['timestamp_s'] <= stamp + .15] if synchronized else []
        return {**row, 'telemetry': nearest[1] if nearest else {}, 'imu_samples': window,
                'telemetry_age_ms': abs(nearest[0] - stamp) * 1000 if nearest else None}

    def _capture_worker(self):
        pending = deque()
        period = 1. / self.capture_fps
        fresh = 0
        try:
            self.telemetry_ready.wait(timeout=1)
            while not self.stop.is_set():
                began = time.monotonic()
                if self.max_seconds is not None and began - self.started >= self.max_seconds:
                    self.stop.set()
                    break
                for base in self._bases():
                    try:
                        with self._open(base, self.snapshot_endpoint, timeout=3) as response:
                            raw_header = response.headers.get('X-Capture-Wall-Time-Ms') or response.headers.get('X-Timestamp-Ms')
                            sequence = response.headers.get('X-Frame-Sequence')
                            jpeg = response.read()
                        if self.fatal_error:
                            break
                        arrival = time.time() * 1000
                        if not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):
                            raise ValueError('snapshot returned an incomplete JPEG')
                        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError('snapshot returned an unreadable JPEG')
                        h, w = image.shape[:2]
                        if self.capture_size is not None and self.capture_size != (w, h):
                            raise ValueError('snapshot resolution changed; keep one resolution per session')
                        self.capture_size = (w, h)
                        if self.calibration is None:
                            self.calibration = dict(width=w, height=h, fx=self.focal_ratio*w,
                                fy=self.focal_ratio*w, cx=w/2, cy=h/2, distortion=[0.] * 5)
                            self.calibration_source = 'approximate_from_snapshot_resolution'
                            self._save_metadata()
                        with self.lock:
                            offset = float(np.median(self.offsets)) if self.offsets else 0.
                            clock_known = bool(self.offsets)
                        phone_s = float(raw_header) / 1000. if raw_header else (arrival - offset) / 1000.
                        if self.origin_sensor_s is None:
                            self.origin_sensor_s = phone_s
                        if self.origin_receiver_ms is None:
                            self.origin_receiver_ms = arrival
                        path = self.frames_dir / f'{self.next_index:08d}.jpg'
                        path.write_bytes(jpeg)
                        row = {'index': self.next_index, 'image': str(path.relative_to(self.root)),
                            'timestamp_s': (arrival - self.origin_receiver_ms)/1000., 'sensor_timestamp_s': phone_s,
                            'previous_sensor_timestamp_s': self._previous_sensor_time or phone_s,
                            'receiver_wall_ms': arrival, 'phone_capture_wall_ms': float(raw_header) if raw_header else None,
                            'phone_frame_sequence': sequence, 'connected_url': base,
                            'clock_source': ('phone_exposure_header' if raw_header else
                                             'receiver_clock_estimate' if clock_known else 'receiver_unsynchronized'),
                            'capture_monotonic': began}
                        with self.acquisition_path.open('a', encoding='utf8') as stream:
                            stream.write(json.dumps(row) + '\n')
                        self._acquisitions.append(row)
                        self._previous_sensor_time = phone_s
                        self.next_index += 1
                        self.received += 1
                        fresh += 1
                        self.latest_jpeg = jpeg
                        self.last_frame_monotonic = time.monotonic()
                        self.connected_url, self.error = base, None
                        pending.append(row)
                        break
                    except PermissionError as error:
                        self.fatal_error = str(error)
                        self.stop.set()
                        break
                    except ValueError as error:
                        self.error = str(error)
                        if 'resolution changed' in str(error):
                            self.fatal_error = str(error)
                            self.stop.set()
                            break
                    except Exception as error:
                        self.error = str(error)
                while pending and time.monotonic() - pending[0]['capture_monotonic'] >= self.buffer_seconds:
                    self._commit(pending.popleft())
                if self.max_frames is not None and fresh >= self.max_frames:
                    self.stop.set()
                self.stop.wait(max(0., period - (time.monotonic() - began)))
        except Exception as error:
            self.fatal_error = f'Phone acquisition failed: {error}'
            self.stop.set()
        finally:
            try:
                while pending:
                    self._commit(pending.popleft())
            except Exception as error:
                self.fatal_error = f'Could not commit the capture journal: {error}'
                self.stop.set()
            if self.remote_recording:
                try:
                    result = self._request_json('/record/stop', method='POST', timeout=30)
                    if result.get('status') not in ('stopped', 'ok'):
                        raise ValueError('Phone did not confirm recording stopped')
                    self.remote_recording = False
                except Exception as error:
                    self.remote_record_error = f'Phone recording stop failed: {error}'

    def _commit(self, row):
        record = self._enrich(row)
        with self.manifest_path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(record) + '\n')
        self.queue.put(record)

    def start(self, after_index=-1):
        for record in self._saved_rows:
            if record['index'] > after_index:
                self.queue.put(record)
        if self.record_phone:
            try:
                previous = self._request_json(self.telemetry_endpoint)
                self.prior_session_ids = {str(row['id']) for row in previous.get('sessions', [])
                                          if isinstance(row, dict) and row.get('id') is not None}
            except ConnectionError:
                self.prior_session_ids = set()
            response = self._request_json('/record/start', method='POST', timeout=10)
            if response.get('status') not in ('started', 'ok'):
                raise RuntimeError('Phone did not confirm recording started')
            self.remote_recording = True
        self.started = time.monotonic()
        self.telemetry_thread = Thread(target=self._telemetry_worker, name='api-telemetry', daemon=True)
        self.thread = Thread(target=self._capture_worker, name='api-snapshots', daemon=True)
        self.telemetry_thread.start()
        self.thread.start()

    def frames(self, after_index=-1):
        self.start(after_index)
        try:
            while self.thread.is_alive() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=.2)
                except Empty:
                    continue
                began = time.monotonic()
                image = cv2.imread(str(self.root / record['image']))
                if image is None:
                    self.decode_errors += 1
                    raise RuntimeError(f"Cannot decode saved frame {record['index']}")
                self.processed += 1
                yield SourceFrame(record['index'], record['timestamp_s'], image,
                    time.monotonic() - began,
                    record['sensor_timestamp_s'] if record['clock_source'] != 'receiver_unsynchronized' else None,
                    record['telemetry'],
                    record['imu_samples'], {k: record[k] for k in ('clock_source', 'receiver_wall_ms',
                        'phone_capture_wall_ms', 'phone_frame_sequence', 'telemetry_age_ms')})
        finally:
            self.finish()
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)

    def finish(self):
        if self._closed:
            return
        self.stop.set()
        for thread in (self.thread, self.telemetry_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5)
        self._closed = True
        rows = _records(self.acquisition_path)
        if not rows:
            return
        intervals = np.diff([row['receiver_wall_ms'] for row in rows])
        intervals = intervals[intervals > 0]
        fps = 1000. / float(np.median(intervals)) if len(intervals) else self.capture_fps
        fps = max(.5, min(60., fps))
        writer = None
        written = 0
        with (self.session / 'video_frames.csv').open('w', newline='', encoding='utf8') as stream:
            keys = ('index', 'timestamp_s', 'receiver_wall_ms', 'phone_capture_wall_ms', 'phone_frame_sequence', 'image', 'clock_source')
            csv_writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
            csv_writer.writeheader()
            try:
                for row in rows:
                    csv_writer.writerow(row)
                    image = cv2.imread(str(self.root / row['image']))
                    if image is None:
                        raise RuntimeError('Saved JPEG is unreadable during video export')
                    if writer is None:
                        size = (image.shape[1], image.shape[0])
                        writer = cv2.VideoWriter(str(self.session / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                        if not writer.isOpened():
                            raise RuntimeError('Cannot create video.mp4')
                    if (image.shape[1], image.shape[0]) != size:
                        raise RuntimeError('Capture resolution changed within a session')
                    writer.write(image)
                    written += 1
            finally:
                if writer:
                    writer.release()
        (self.session / 'recording.json').write_text(json.dumps({'frames': len(rows), 'mp4_frames': written,
            'mp4_fps': fps, 'duration_wall_s': (rows[-1]['receiver_wall_ms'] - rows[0]['receiver_wall_ms']) / 1000.,
            'original_jpegs': str(self.frames_dir), 'precise_timing': str(self.session / 'video_frames.csv'),
            'timing_note': 'MP4 is constant-rate; use per-frame timestamps for sensor synchronization'}, indent=2))
        if self.download_phone_session and not self.remote_recording and not self.remote_record_error:
            try:
                self._download_phone_session()
            except Exception as error:
                self.remote_record_error = f'Phone archive download failed: {error}'
        recording_path = self.session / 'recording.json'
        recording = json.loads(recording_path.read_text())
        recording.update(phone_recording_still_active=self.remote_recording,
                         phone_archive=str(self.remote_archive) if self.remote_archive else None,
                         phone_record_error=self.remote_record_error)
        recording_path.write_text(json.dumps(recording, indent=2))

    def _download_phone_session(self):
        sessions = []
        for _ in range(15):
            packet = self._request_json(self.telemetry_endpoint, timeout=3)
            sessions = [row for row in packet.get('sessions', []) if isinstance(row, dict)
                        and row.get('id') is not None and str(row['id']) not in self.prior_session_ids]
            if sessions:
                break
            time.sleep(.5)
        if not sessions:
            raise RuntimeError('No newly recorded session appeared in phone telemetry')
        session_id = str(sessions[-1]['id'])
        safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', session_id).strip('._') or 'session'
        archive_path = self.session / f'phone_{safe_name}.zip'
        temporary = archive_path.with_suffix('.zip.part')
        endpoint = '/download/' + quote(session_id, safe='')
        base = self.connected_url or self.urls[0]
        with self._open(base, endpoint, timeout=60) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        with ZipFile(temporary) as archive:
            present = {Path(item.filename).name for item in archive.infolist()}
            if not {'video.mp4', 'gps.csv', 'imu.csv'}.issubset(present):
                raise ValueError('Phone ZIP is missing video.mp4, gps.csv or imu.csv')
            if archive.testzip() is not None:
                raise ValueError('Phone session ZIP failed its integrity check')
        temporary.replace(archive_path)
        self.remote_archive = archive_path


class RecordedVideoSource:
    def __init__(self, video, sample_fps, max_seconds=None):
        self.video, self.sample_fps = Path(video), sample_fps
        if sample_fps < 0 or max_seconds is not None and max_seconds <= 0:
            raise ValueError('sample_fps must be nonnegative and max_seconds must be positive')
        self.max_seconds = max_seconds
        self.signature = {'path': str(self.video), 'size': self.video.stat().st_size,
                          'mtime_ns': self.video.stat().st_mtime_ns,
                          'analysis_rate': 'every_frame' if sample_fps == 0 else sample_fps}
        if max_seconds is not None:
            self.signature['max_video_seconds'] = max_seconds

    def frames(self, after_index=-1):
        capture = cv2.VideoCapture(str(self.video))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open {self.video}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise RuntimeError('Video FPS unavailable')
        index, next_sample, pending_decode = -1, 0., 0.
        try:
            while True:
                began = time.monotonic()
                if not capture.grab():
                    break
                pending_decode += time.monotonic() - began
                index += 1
                stamp = index / fps
                if self.max_seconds is not None and stamp > self.max_seconds:
                    break
                if self.sample_fps and stamp + 1e-6 < next_sample:
                    continue
                if self.sample_fps:
                    next_sample = stamp + 1 / self.sample_fps
                if index <= after_index:
                    continue
                decode_start = time.monotonic()
                ok, image = capture.retrieve()
                if not ok:
                    raise RuntimeError(f'Cannot decode frame {index}')
                pending_decode += time.monotonic() - decode_start
                yield SourceFrame(index, stamp, image, pending_decode)
                pending_decode = 0.
        finally:
            capture.release()
