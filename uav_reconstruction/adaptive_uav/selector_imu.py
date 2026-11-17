"""Exactly synchronized gyro magnitude for conservative keyframe bridging.

Angular speed is invariant to the unknown camera/IMU axis rotation. It is used
only to request an earlier visual keyframe, never to invent translation or a
camera pose. Sessions without a recorded video start timestamp use no gyro.
"""
import csv
import json
from pathlib import Path

import numpy as np


class GyroHistory:
    def __init__(self, times, vectors):
        times = np.asarray(times, dtype=float)
        vectors = np.asarray(vectors, dtype=float)
        if times.ndim != 1 or vectors.shape != (len(times), 3):
            raise ValueError('Gyro requires one timestamp and three values per sample')
        valid = np.isfinite(times) & np.isfinite(vectors).all(axis=1)
        times, vectors = times[valid], vectors[valid]
        order = np.argsort(times)
        times, vectors = times[order], vectors[order]
        unique = np.r_[True, np.diff(times) > 0] if len(times) else np.empty(0, bool)
        self.times, self.vectors = times[unique], vectors[unique]
        self.sample_rate_hz = (1. / float(np.median(np.diff(self.times)))
                               if len(self.times) > 1 else 0.)

    @classmethod
    def from_session(cls, session):
        session = Path(session)
        if not (session / 'metadata.json').is_file():
            return None, 'metadata.json unavailable'
        metadata = json.loads((session / 'metadata.json').read_text())
        start = metadata.get('video_start_elapsed_ns')
        origin = metadata.get('session_start_elapsed_ns')
        if start is None or origin is None:
            return None, 'recorded video-to-IMU start timestamp unavailable'
        path = session / 'sensors.csv'
        if not path.is_file():
            return None, 'sensors.csv unavailable'
        times, vectors = [], []
        with path.open() as stream:
            reader = csv.DictReader(stream)
            if not {'elapsed_ns', 'type', 'x', 'y', 'z'}.issubset(reader.fieldnames or []):
                return None, 'gyro columns unavailable'
            for row in reader:
                try:
                    if int(row['type']) != 4:
                        continue
                    times.append((int(row['elapsed_ns']) - int(origin)) * 1e-9)
                    vectors.append([float(row[k]) for k in ('x', 'y', 'z')])
                except (ValueError, TypeError, KeyError):
                    continue
        if len(times) < 20:
            return None, 'fewer than 20 gyro samples'
        return cls(times, vectors), 'exact recorded video start timestamp'

    def motion(self, start, end):
        """Return measured angular motion or None for gaps/unsynced bounds."""
        if start is None or end is None or end <= start or len(self.times) < 2 or self.sample_rate_hz < 20.:
            return None
        if start < self.times[0] or end > self.times[-1]:
            return None
        left = np.searchsorted(self.times, start, side='right')
        right = np.searchsorted(self.times, end, side='left')
        stamps = np.r_[start, self.times[left:right], end]
        if np.diff(stamps).max() > .1:
            return None
        vectors = np.column_stack([np.interp(stamps, self.times, self.vectors[:, i]) for i in range(3)])
        speed = np.linalg.norm(vectors, axis=1)
        angle = float(np.trapezoid(speed, stamps))
        return {'angular_speed_rad_s': angle / (end - start),
                'angular_motion_rad': angle, 'gyro_samples': int(right - left),
                'source': 'recorded_synchronized_gyro'}
