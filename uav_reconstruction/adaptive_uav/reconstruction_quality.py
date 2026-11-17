"""Paired reconstruction quality on a fixed reference population.

The reference is chosen before reconstruction and never from what a branch produced,
so a missing surface lowers recall instead of disappearing from the denominator.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib

import numpy as np
from scipy.spatial import cKDTree

from .utility_contracts import THREADS

DEFAULT_WEIGHTS = {'version': 'train_calibrated_v1_uncalibrated_defaults', 'f_score': 1., 'coverage': 0., 'surface_error': .25,
                   'pose_error': .25, 'connectivity': .1, 'cost_per_budget_fraction': .1,
                   'surface_error_scale_m': 1., 'pose_error_scale_m': 1.}


@dataclass(frozen=True)
class ReferencePopulation:
    points: np.ndarray
    tolerance_m: float
    status: str
    protocol: str = 'reference_roi_v2'

    def identity(self) -> str:
        return hashlib.sha256(np.round(self.points, 2).tobytes() + repr(self.tolerance_m).encode()).hexdigest()[:16]

    @cached_property
    def tree(self) -> cKDTree:
        return cKDTree(self.points)

    @cached_property
    def cell(self) -> float:
        return max(4 * self.tolerance_m, 2.)

    @cached_property
    def cells(self) -> np.ndarray:
        return np.unique(cell_keys(self.points, self.cell))


def cell_keys(points: np.ndarray, cell: float) -> np.ndarray:
    keys = np.floor(points[:, :2] / cell).astype(np.int64)
    return (keys[:, 0] + 2 ** 30) * 2 ** 31 + keys[:, 1] + 2 ** 30


def surface_metrics(predicted: np.ndarray, reference: ReferencePopulation) -> dict:
    """The ROI is the reference footprint in plan view, so a floater over it counts as wrong.

    Dropping far-away points instead would hide the worst errors and reward them over small ones.
    """
    target = reference.points
    if not len(target):
        raise ValueError('Reference population is empty')
    tolerance = reference.tolerance_m
    empty = {'precision': 0., 'recall': 0., 'f_score': 0., 'accuracy_m': None, 'completeness_m': None,
             'supported_coverage': 0., 'predicted_points': 0, 'outside_roi_points': len(predicted), 'reference_points': len(target)}
    if not len(predicted):
        return empty
    surface = predicted[np.isin(cell_keys(predicted, reference.cell), reference.cells)]
    if not len(surface):
        return empty
    to_reference = reference.tree.query(surface, workers=THREADS)[0]
    to_prediction = cKDTree(surface).query(target, workers=THREADS)[0]
    precision = float((to_reference <= tolerance).mean())
    recall = float((to_prediction <= tolerance).mean())
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'precision': precision, 'recall': recall, 'f_score': f_score, 'accuracy_m': float(np.median(to_reference)),
            'completeness_m': float(np.median(to_prediction)), 'supported_coverage': recall, 'predicted_points': len(surface),
            'outside_roi_points': len(predicted) - len(surface), 'reference_points': len(target)}


def depth_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & (target > 0) & (prediction > 0) & np.isfinite(prediction)
    if not valid.any():
        return {'pixels': 0, 'absrel': None, 'mae_m': None, 'rmse_m': None, 'bias_m': None, 'p95_abs_m': None}
    delta = prediction[valid] - target[valid]
    return {'pixels': int(valid.sum()), 'absrel': float((np.abs(delta) / target[valid]).mean()),
            'mae_m': float(np.abs(delta).mean()), 'rmse_m': float(np.sqrt((delta ** 2).mean())),
            'bias_m': float(delta.mean()), 'p95_abs_m': float(np.percentile(np.abs(delta), 95))}


def pose_error(estimated: dict, reference: dict) -> dict:
    """Without refitting: a branch's drift has to count against it."""
    shared = sorted(set(estimated) & set(reference))
    if not shared:
        return {'frames': 0, 'median_translation_m': None, 'median_rotation_deg': None}
    translation, rotation = [], []
    for name in shared:
        a, b = np.asarray(estimated[name]), np.asarray(reference[name])
        translation.append(np.linalg.norm(a[:3, 3] - b[:3, 3]))
        trace = np.trace(a[:3, :3].T @ b[:3, :3])
        rotation.append(np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1))))
    return {'frames': len(shared), 'median_translation_m': float(np.median(translation)),
            'median_rotation_deg': float(np.median(rotation))}


def quality(metrics: dict, pose: dict | None, weights: dict) -> float:
    """Higher is better. No surface takes the full error penalty, so empty never beats poor."""
    value = weights['f_score'] * metrics['f_score'] + weights['coverage'] * metrics['supported_coverage']
    error = 1. if metrics['accuracy_m'] is None else min(metrics['accuracy_m'] / weights['surface_error_scale_m'], 1.)
    value -= weights['surface_error'] * error
    if pose and pose.get('median_translation_m') is not None:
        value -= weights['pose_error'] * min(pose['median_translation_m'] / weights['pose_error_scale_m'], 1.)
    return float(value)


def utility(control: float, intervened: float, connectivity_gain: float, cost_seconds: float,
            remaining_seconds: float, weights: dict) -> float:
    """Near an exhausted budget, cost is bounded rather than divided by nearly zero."""
    pressure = min(cost_seconds / max(remaining_seconds, 1e-3), 10.)
    return float(intervened - control + weights['connectivity'] * connectivity_gain
                 - weights['cost_per_budget_fraction'] * pressure)


def cluster_interval(gains: dict, draws: int = 2000, seed: int = 0) -> dict:
    """Mean gain with a bootstrap over flights: correlated clips are not independent samples."""
    clusters = sorted(gains)
    if not clusters:
        return {'mean': None, 'ci95': None, 'clusters': 0}
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(gains[c]) for c in clusters])
    resampled = means[rng.integers(0, len(means), (draws, len(means)))].mean(1)
    return {'mean': float(means.mean()), 'ci95': np.percentile(resampled, [2.5, 97.5]).tolist(), 'clusters': len(clusters)}
