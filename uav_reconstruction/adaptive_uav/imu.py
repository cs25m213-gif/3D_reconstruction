"""Timestamped CPU IMU features for the online frame selector.

Inputs use seconds, acceleration in m/s^2 and angular velocity in rad/s.
Acceleration statistics include gravity; this module never estimates translation
from acceleration. A telemetry snapshot rate is distinct from the phone sensor's
hardware rate: only samples actually received contribute to the measured rate.

These are engineered features, not a trained inertial or keyframe model.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


MIN_ROTATION_RATE_HZ = 20.0
MIN_COVERAGE = 0.8
MIN_ROTATION_COVERAGE = 0.95


def _window(start_s: float, end_s: float, fixed_samples: int) -> tuple[float, float, int]:
    start, end = float(start_s), float(end_s)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start or not np.isfinite(end - start):
        raise ValueError("IMU window requires finite end_s > start_s")
    if isinstance(fixed_samples, bool) or int(fixed_samples) != fixed_samples or fixed_samples < 2:
        raise ValueError("fixed_samples must be an integer >= 2")
    return start, end, int(fixed_samples)


def _samples(samples: list[dict], start: float, end: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep finite canonical samples, sort by time, and average duplicates.

    Samples outside the requested window are excluded, so an old latest packet
    cannot silently become a synchronized observation for a new frame.
    """
    rows: list[tuple[float, np.ndarray]] = []
    invalid, outside = 0, 0
    for sample in samples:
        try:
            if not isinstance(sample, Mapping):
                raise ValueError("sample must be a mapping")
            timestamp = float(sample["timestamp_s"])
            accel = np.asarray(sample["accel"], dtype=np.float64)
            gyro = np.asarray(sample["gyro"], dtype=np.float64)
            if accel.shape != (3,) or gyro.shape != (3,):
                raise ValueError("expected three acceleration and gyro axes")
            vector = np.concatenate((accel, gyro))
            if not np.isfinite(timestamp) or not np.all(np.isfinite(vector)):
                raise ValueError("nonfinite IMU sample")
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid += 1
            continue
        if start <= timestamp <= end:
            rows.append((timestamp, vector))
        else:
            outside += 1
    rows.sort(key=lambda row: row[0])
    times: list[float] = []
    vectors: list[np.ndarray] = []
    duplicates = 0
    index = 0
    while index < len(rows):
        stop = index + 1
        while stop < len(rows) and rows[stop][0] == rows[index][0]:
            stop += 1
        times.append(rows[index][0])
        vectors.append(np.mean([row[1] for row in rows[index:stop]], axis=0))
        duplicates += stop - index - 1
        index = stop
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(vectors, dtype=np.float64).reshape(-1, 6),
        {"invalid_samples": invalid, "outside_window_samples": outside, "duplicate_samples": duplicates},
    )


def _metadata(times: np.ndarray, start: float, end: float, fixed_samples: int, counts: dict) -> dict[str, Any]:
    duration = end - start
    observed = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    coverage = min(1.0, observed / duration)
    rate = (len(times) - 1) / observed if observed > 0.0 else 0.0
    max_gap = float(np.max(np.diff(times))) if len(times) >= 2 else duration
    start_gap = float(times[0] - start) if len(times) else duration
    end_gap = float(end - times[-1]) if len(times) else duration
    # A long missing interval must not be hidden by resampling to 64 entries.
    allowed_gap = max(0.05, min(0.25, 0.25 * duration))
    if len(times) < 2:
        reason = "fewer than two unique finite IMU samples"
    elif coverage < MIN_COVERAGE:
        reason = "insufficient IMU window coverage"
    elif max_gap > allowed_gap + 1e-12:
        reason = "IMU window contains a large sampling gap"
    else:
        reason = "ok"
    valid = reason == "ok"
    low_rate = rate < MIN_ROTATION_RATE_HZ
    rotation_gaps_ok = max_gap <= 0.1 + 1e-12 and max(start_gap, end_gap) <= 0.05 + 1e-12
    rotation_valid = valid and not low_rate and coverage >= MIN_ROTATION_COVERAGE and rotation_gaps_ok
    if not valid:
        rotation_reason = reason
    elif low_rate:
        rotation_reason = "received IMU sampling rate is below 20 Hz"
    elif coverage < MIN_ROTATION_COVERAGE:
        rotation_reason = "rotation compensation requires at least 95% window coverage"
    elif not rotation_gaps_ok:
        rotation_reason = "rotation compensation has an excessive sample or boundary gap"
    else:
        rotation_reason = "ok"
    return {
        "valid": bool(valid),
        "reason": reason,
        "rotation_valid": bool(rotation_valid),
        "rotation_reason": rotation_reason,
        "count": len(times),
        "sample_count": len(times),
        "start_timestamp_s": start,
        "end_timestamp_s": end,
        "duration_s": duration,
        "observed_duration_s": observed,
        "effective_rate_hz": float(rate),
        "coverage_fraction": float(coverage),
        "max_gap_s": max_gap,
        "allowed_gap_s": allowed_gap,
        "start_gap_s": start_gap,
        "end_gap_s": end_gap,
        "low_sample_rate": bool(low_rate),
        "minimum_rotation_rate_hz": MIN_ROTATION_RATE_HZ,
        "fixed_samples": fixed_samples,
        "boundary_hold_s": start_gap + end_gap if len(times) else duration,
        **counts,
    }


def resample_imu(samples: list[dict], start_s: float, end_s: float,
                 fixed_samples: int = 64) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``fixed_samples x 6`` [accel, gyro] and quality metadata.

    Linear interpolation uses only measurements within this time window. Edges
    without measurements hold the closest value and are reported in metadata.
    Empty windows produce zeros and single samples repeat their value; both are
    explicitly invalid and must not be treated as measured motion by a model.
    Resampling never increases the reported effective sensor sampling rate.
    """
    start, end, count = _window(start_s, end_s, fixed_samples)
    times, values, counts = _samples(samples, start, end)
    metadata = _metadata(times, start, end, count, counts)
    result = np.zeros((count, 6), dtype=np.float32)
    if len(times):
        targets = np.linspace(start, end, count, dtype=np.float64)
        for axis in range(6):
            result[:, axis] = np.interp(targets, times, values[:, axis])
    return result, metadata


def summarize_imu(samples: list[dict], start_s: float, end_s: float,
                  fixed_samples: int = 64) -> dict[str, Any]:
    """JSON features computed from original samples, before resampling.

    Gyro integration covers only the observed interval, with trapezoidal
    quadrature. ``gyro_integrated_rad`` is a first order rotation vector; it is
    not an inertial pose estimate, and biases are not automatically calibrated.
    ``rotation_valid`` is more strict than statistical ``valid`` so low rate
    telemetry can inform the selector without unsafe image compensation.
    """
    start, end, count = _window(start_s, end_s, fixed_samples)
    times, values, counts = _samples(samples, start, end)
    features = _metadata(times, start, end, count, counts)
    integrated = np.zeros(3, dtype=np.float64)
    path = 0.0
    if len(times) >= 2:
        intervals = np.diff(times)
        gyro = values[:, 3:]
        integrated = np.sum(0.5 * (gyro[:-1] + gyro[1:]) * intervals[:, None], axis=0)
        speed = np.linalg.norm(gyro, axis=1)
        path = float(np.sum(0.5 * (speed[:-1] + speed[1:]) * intervals))
    if len(times):
        accel, gyro = values[:, :3], values[:, 3:]
        accel_norm = np.linalg.norm(accel, axis=1)
        angular_speed = np.linalg.norm(gyro, axis=1)
        accel_mean = np.mean(accel, axis=0)
        accel_variance = np.var(accel, axis=0)
        gyro_mean = np.mean(gyro, axis=0)
        gyro_rms = float(np.sqrt(np.mean(angular_speed ** 2)))
        gyro_max = float(np.max(angular_speed))
        norm_mean = float(np.mean(accel_norm))
        norm_std = float(np.std(accel_norm))
    else:
        accel_mean = accel_variance = gyro_mean = np.zeros(3, dtype=np.float64)
        gyro_rms = gyro_max = norm_mean = norm_std = 0.0
    features.update({
        "gyro_integrated_rad": integrated.tolist(),
        "angular_displacement_rad": float(np.linalg.norm(integrated)),
        "angular_path_rad": path,
        "gyro_mean_rad_s": gyro_mean.tolist(),
        "gyro_rms_rad_s": gyro_rms,
        "gyro_max_rad_s": gyro_max,
        "accel_mean_mps2": accel_mean.tolist(),
        "accel_variance_mps4": accel_variance.tolist(),
        "accel_norm_mean_mps2": norm_mean,
        "accel_norm_std_mps2": norm_std,
        "integrated_duration_s": features["observed_duration_s"],
        "gyro_bias_corrected": False,
        "acceleration_includes_gravity": True,
    })
    return features


def rotation_homography(features: dict, camera_k: np.ndarray,
                        camera_from_imu: np.ndarray | None) -> np.ndarray | None:
    """Predict previous-image to current-image rotation flow, or return None.

    ``camera_from_imu`` must be an explicitly calibrated, proper 3x3 rotation.
    Positive gyro rotates camera axes in the world; stationary scene rays in the
    current camera therefore rotate by the negative integrated vector. This
    small-motion approximation cannot recover translation, rolling shutter or
    gyro bias, and should support (never replace) visual geometric verification.
    """
    if not features.get("valid") or not features.get("rotation_valid") or camera_from_imu is None:
        return None
    try:
        intrinsic = np.asarray(camera_k, dtype=np.float64)
        extrinsic = np.asarray(camera_from_imu, dtype=np.float64)
        vector = np.asarray(features["gyro_integrated_rad"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or extrinsic.shape != (3, 3) or vector.shape != (3,):
            return None
        if not all(np.all(np.isfinite(value)) for value in (intrinsic, extrinsic, vector)):
            return None
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            return None
        if not np.allclose(extrinsic.T @ extrinsic, np.eye(3), atol=1e-5, rtol=0):
            return None
        if not np.isclose(np.linalg.det(extrinsic), 1.0, atol=1e-5, rtol=0):
            return None
        camera_vector = -(extrinsic @ vector)
        angle = float(np.linalg.norm(camera_vector))
        if angle < 1e-12:
            rotation = np.eye(3)
        else:
            x, y, z = camera_vector / angle
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        homography = intrinsic @ rotation @ np.linalg.inv(intrinsic)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    return homography
