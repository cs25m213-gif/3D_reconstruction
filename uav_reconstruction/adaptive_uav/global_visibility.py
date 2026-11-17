"""Bounded cross-sequence free-space checks, not a learned VisFusion model.

Measured depth is evidence, not ground truth. Repeated contradictions are
reported as possible pose/depth failure; they must not be filled with a prior.
"""
import numpy as np


def visibility_votes(frame, references, tolerance=.08):
    d = frame['depth']
    k, p = frame['K'], frame['P']
    y, x = np.indices(d.shape)
    camera = np.stack(((x-k[0, 2])*d/k[0, 0],
                       (y-k[1, 2])*d/k[1, 1], d), -1).reshape(-1, 3)
    world = (camera-p[:3, 3]) @ p[:3, :3]
    valid = np.isfinite(d.ravel()) & (d.ravel() > 0)
    support = np.zeros(d.size, np.uint16)
    conflict = np.zeros(d.size, np.uint16)
    center = -p[:3, :3].T @ p[:3, 3]
    scale = np.median(d.ravel()[valid]) if valid.any() else 1.
    for other in references:
        if other is frame or other.get('name') == frame.get('name'):
            continue
        q, ko = other['P'], other['K']
        if np.linalg.norm(center + q[:3, :3].T @ q[:3, 3]) < .025*scale:
            continue
        pc = world @ q[:3, :3].T + q[:3, 3]
        z = np.maximum(pc[:, 2], 1e-9)
        u = np.rint(pc[:, 0]/z*ko[0, 0]+ko[0, 2]).astype(np.int64)
        v = np.rint(pc[:, 1]/z*ko[1, 1]+ko[1, 2]).astype(np.int64)
        m = other['measured']
        h, w = m.shape
        ix = np.flatnonzero(valid & (pc[:, 2] > 0) & (u >= 1) &
                            (v >= 1) & (u < w-1) & (v < h-1))
        # Sparse stereo filtering leaves holes. Require three real samples
        # with consistent depths; requiring all nine disabled this gate on
        # otherwise usable measurements. The nearest observed depth protects
        # object boundaries against false free-space carving.
        samples = np.stack([m[v[ix]+dy, u[ix]+dx]
                            for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        observed = np.isfinite(samples) & (samples > 0)
        near = np.where(observed,samples,np.inf).min(axis=0)
        far = np.where(observed,samples,-np.inf).max(axis=0)
        known = (observed.sum(axis=0)>=3) & (far<=near*(1+tolerance))
        support[ix] += known & (pc[ix, 2] >= near*(1-tolerance)) & (pc[ix, 2] <= far*(1+tolerance))
        conflict[ix] += known & (pc[ix, 2] < near*(1-tolerance))
        # Behind observed geometry is occluded: no negative vote.
    return support.reshape(d.shape), conflict.reshape(d.shape)


def guard_frames(frames, max_references=24):
    """Use frozen measurements; decisions do not depend on processing order."""
    measured_frames=[f for f in frames if np.any(f['measured']>0)]
    indices = np.unique(np.linspace(0, len(measured_frames)-1, min(max_references, len(measured_frames)), dtype=int)) if measured_frames else []
    references = [measured_frames[i] for i in indices]
    masks, rows = [], []
    for f in frames:
        support, conflict = visibility_votes(f, references)
        reject = (conflict >= 2) & (conflict > support)
        active = f['depth'] > 0
        tested = active & ((support+conflict) > 0)
        rows.append({'frame': f['name'], 'tested_pixels': int(tested.sum()),
                     'active_pixels':int(active.sum()),
                     'rejected_pixels': int((reject & active).sum()),
                     'rejected_fraction': float((reject & active).sum()/max(1, active.sum()))})
        masks.append(reject)
    for f, reject in zip(frames, masks):
        f['depth'] = np.where(reject, 0, f['depth'])
        f['tier'] = np.where(reject, 0, f['tier']).astype(np.uint8)
    medians = [float(np.median(f['measured'][f['measured'] > 0])) for f in frames
               if np.any(f['measured'] > 0)]
    return {'reference_frames': [f['name'] for f in references], 'per_frame': rows,
            'measured_reference_candidates':len(measured_frames),
            'mean_tested_fraction':float(np.mean([r['tested_pixels']/max(1,r['active_pixels']) for r in rows])),
            'minimum_neighbourhood_observations':3,
            'depth_median_ratio_p95_p05': float(np.percentile(medians, 95)/np.percentile(medians, 5)) if medians else None,
            'mean_rejected_fraction': float(np.mean([r['rejected_fraction'] for r in rows])),
            'note': 'Global contradictions can indicate pose drift or erroneous depth. Depth range alone is not proof of drift.'}
