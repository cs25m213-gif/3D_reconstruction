"""Small, dependency-light weighted geometry routines."""
import numpy as np


def weighted_umeyama(source, target, weights=None, estimate_scale=True):
    """Return scale, rotation, translation mapping source onto target.

    Raises ValueError for insufficient or degenerate correspondences.  This is
    deliberately NumPy-only so CPU execution is always available.
    """
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("source and target must be Nx3 with N >= 3")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(x),) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and non-zero")
    w = w / w.sum()
    mx, my = (w[:, None] * x).sum(0), (w[:, None] * y).sum(0)
    xc, yc = x - mx, y - my
    cov = (w[:, None] * yc).T @ xc
    u, singular, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1
    r = u @ d @ vt
    var = (w[:, None] * xc * xc).sum()
    if var <= 1e-12:
        raise ValueError("degenerate source geometry")
    scale = float((singular * np.diag(d)).sum() / var) if estimate_scale else 1.0
    t = my - scale * r @ mx
    return scale, r, t


def residuals(source, target, transform):
    scale, rotation, translation = transform
    predicted = scale * (rotation @ np.asarray(source).T).T + translation
    return np.linalg.norm(predicted - np.asarray(target), axis=1)
