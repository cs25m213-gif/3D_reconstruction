"""Select sharp, geometrically connected views from every decoded video frame."""

from pathlib import Path
import time

import cv2
import numpy as np

from .phone_source import SourceFrame


class BufferedQualityVideoSource:
    """Keep the best real camera image in each short window.

    Cheap focus/exposure checks run on every frame. ORB verification runs only
    on the best few candidates. The saved image is the unenhanced RGB frame.
    """

    def __init__(self, video, window_seconds=.25, max_seconds=None, rotation='none',
                 output_width=1080, analysis_width=540, candidate_count=3):
        self.video = Path(video).resolve()
        if window_seconds <= 0 or output_width < 100 or analysis_width < 100 or candidate_count < 1:
            raise ValueError('Invalid quality selection dimensions or window')
        if rotation not in ('none', 'cw', 'ccw', '180'):
            raise ValueError('rotation must be none, cw, ccw, or 180')
        self.window_seconds = float(window_seconds)
        self.max_seconds = max_seconds
        self.rotation = rotation
        self.output_width = int(output_width)
        self.analysis_width = int(analysis_width)
        self.candidate_count = int(candidate_count)
        stat = self.video.stat()
        self.signature = {'kind': 'quality_buffered_video', 'path': str(self.video),
                          'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                          'window_seconds': self.window_seconds, 'max_seconds': max_seconds,
                          'rotation': rotation, 'output_width': output_width,
                          'analysis_width': analysis_width, 'candidate_count': candidate_count}
        self.orb = cv2.ORB_create(nfeatures=1800, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _orient(self, frame):
        if self.rotation in ('cw', 'ccw'):
            width_before = round(self.output_width * frame.shape[1] / frame.shape[0])
            height_before = self.output_width
        else:
            width_before = self.output_width
            height_before = round(self.output_width * frame.shape[0] / frame.shape[1])
        frame = cv2.resize(frame, (width_before, height_before), interpolation=cv2.INTER_AREA)
        if self.rotation == 'cw':
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 'ccw':
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.rotation == '180':
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def _candidate(self, index, stamp, raw):
        image = self._orient(raw)
        h, w = image.shape[:2]
        small = cv2.resize(image, (self.analysis_width, round(h * self.analysis_width / w)),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = float(np.mean(gx * gx + gy * gy))
        clipped = float(np.mean((gray < 8) | (gray > 247)))
        focus = .55 * np.log1p(lap) + .45 * np.log1p(gradient)
        quality = float(focus - 3.0 * clipped)
        return {'index': index, 'stamp': stamp, 'image': image, 'gray': gray,
                'quality': quality, 'laplacian': lap, 'gradient': gradient,
                'clipped_fraction': clipped}

    def _features(self, candidate):
        if 'keypoints' not in candidate:
            candidate['keypoints'], candidate['descriptors'] = self.orb.detectAndCompute(
                candidate['gray'], None)
        return candidate['keypoints'], candidate['descriptors']

    def _geometry(self, previous, candidate):
        old_kp, old_desc = self._features(previous)
        new_kp, new_desc = self._features(candidate)
        if old_desc is None or new_desc is None or len(old_desc) < 20 or len(new_desc) < 20:
            return 0, 0., 0.
        matches = self.matcher.knnMatch(old_desc, new_desc, k=2)
        good = [a for pair in matches if len(pair) == 2
                for a, b in [pair] if a.distance < .78 * b.distance]
        if len(good) < 20:
            return 0, 0., 0.
        a = np.float32([old_kp[m.queryIdx].pt for m in good])
        b = np.float32([new_kp[m.trainIdx].pt for m in good])
        _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 2., .995)
        if mask is None:
            return 0, 0., 0.
        valid = mask.ravel().astype(bool)
        count = int(valid.sum())
        motion = float(np.median(np.linalg.norm(b[valid] - a[valid], axis=1))) if count else 0.
        return count, count / len(good), motion

    def _choose(self, candidates, previous):
        ranked = sorted(candidates, key=lambda item: item['quality'], reverse=True)
        shortlist = ranked[:self.candidate_count]
        if previous is None:
            chosen = shortlist[0]
            geometry = (0, 0., 0.)
        else:
            evaluated = [(self._geometry(previous, item), item) for item in shortlist]
            viable = [(g, item) for g, item in evaluated if g[0] >= 25 and g[1] >= .35]
            if viable:
                geometry, chosen = max(viable, key=lambda pair:
                    (pair[0][0] + 40 * pair[0][1] + 8 * pair[1]['quality']))
            else:
                geometry, chosen = max(evaluated, key=lambda pair: (pair[0][0], pair[1]['quality']))
        return chosen, geometry, len(candidates)

    def frames(self, after_index=-1):
        capture = cv2.VideoCapture(str(self.video))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open {self.video}')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise RuntimeError('Source video FPS unavailable')
        index, bucket, previous = -1, [], None
        bucket_start = 0.
        try:
            while True:
                began = time.monotonic()
                ok, raw = capture.read()
                if not ok:
                    break
                index += 1
                stamp = index / fps
                if self.max_seconds is not None and stamp > self.max_seconds:
                    break
                if bucket and stamp - bucket_start >= self.window_seconds:
                    chosen, geometry, analyzed = self._choose(bucket, previous)
                    previous = chosen
                    if chosen['index'] > after_index:
                        yield SourceFrame(chosen['index'], chosen['stamp'], chosen['image'],
                                          timing={'quality_buffer_frames': analyzed,
                                                  'quality_score': chosen['quality'],
                                                  'focus_laplacian': chosen['laplacian'],
                                                  'focus_gradient': chosen['gradient'],
                                                  'clipped_fraction': chosen['clipped_fraction'],
                                                  'geometry_inliers': geometry[0],
                                                  'geometry_inlier_ratio': geometry[1],
                                                  'geometry_motion_px': geometry[2]})
                    bucket, bucket_start = [], stamp
                candidate = self._candidate(index, stamp, raw)
                if not bucket:
                    bucket_start = stamp
                bucket.append(candidate)
                candidate['decode_seconds'] = time.monotonic() - began
            if bucket:
                chosen, geometry, analyzed = self._choose(bucket, previous)
                if chosen['index'] > after_index:
                    yield SourceFrame(chosen['index'], chosen['stamp'], chosen['image'],
                                      timing={'quality_buffer_frames': analyzed,
                                              'quality_score': chosen['quality'],
                                              'focus_laplacian': chosen['laplacian'],
                                              'geometry_inliers': geometry[0],
                                              'geometry_inlier_ratio': geometry[1],
                                              'geometry_motion_px': geometry[2]})
        finally:
            capture.release()
