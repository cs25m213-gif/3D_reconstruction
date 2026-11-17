"""Audit sparse-model conditioning rather than judging quality from counts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .preview_mesh import rotation
from .session_reconstruct import read_model, save_json


def audit(text_model, logs=None):
    cameras,poses,points=read_model(text_model)
    centers={p['id']:-rotation(p['quaternion_wxyz']).T @ np.asarray(p['translation_world_to_camera']) for p in poses}
    angles=[]
    for line in (text_model/'points3D.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        values=line.split()
        xyz=np.asarray(list(map(float,values[1:4])))
        views=[centers[int(i)] for i in values[8::2] if int(i) in centers]
        if len(views)<2:
            continue
        rays=xyz-np.asarray(views)
        norm=np.linalg.norm(rays,axis=1)
        if np.any(norm<1e-12):
            continue
        rays/=norm[:,None]
        angles.append(float(np.degrees(np.arccos(np.clip((rays@rays.T).min(),-1,1)))))
    errors=np.asarray([p['error_px'] for p in points])
    warnings={}
    if logs and logs.exists():
        paths=[logs] if logs.is_file() else sorted(logs.glob('*.log'))
        for path in paths:
            count=path.read_text(errors='replace').count('Linear solver failure')
            if count:
                warnings[path.name]=count
    result={'registered_cameras':len(poses),'sparse_points':len(points),'cameras':cameras,
            'median_reprojection_error_px':float(np.median(errors)) if errors.size else None,
            'p95_reprojection_error_px':float(np.quantile(errors,.95)) if errors.size else None,
            'median_max_triangulation_angle_deg':float(np.median(angles)) if angles else None,
            'fraction_points_max_angle_below_2_deg':float(np.mean(np.asarray(angles)<2)) if angles else None,
            'linear_solver_failure_count':sum(warnings.values()),'solver_failure_logs':warnings,
            'solver_logs_checked':logs is not None and logs.exists(),
            'metric_accuracy_validated':False,'surface_quality_validated':False}
    result['warnings']=[]
    if warnings:
        result['warnings'].append('Numerical solver failures occurred; registration count alone is not an acceptance criterion.')
    if angles and np.mean(np.asarray(angles)<2)>.25:
        result['warnings'].append('Many landmarks have weak triangulation support; depth may be poorly constrained.')
    return result


def quality_gate(result, config):
    """Conservative sparse conditioning gate, not a surface-quality guarantee."""
    if result['linear_solver_failure_count'] > config.get('max_linear_solver_failures',20):
        return False, 'repeated numerical solver failures'
    angle=result['median_max_triangulation_angle_deg']
    if angle is None or not np.isfinite(angle):
        return False, 'triangulation support unavailable'
    if angle < config.get('min_median_triangulation_angle_deg',3.):
        return False, 'weak median triangulation support'
    fraction=result['fraction_points_max_angle_below_2_deg']
    if fraction is None or fraction > config.get('max_low_angle_point_fraction',.25):
        return False, 'too many weakly triangulated landmarks'
    return True, 'sparse conditioning checks passed; surface quality still unvalidated'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('text_model',type=Path)
    p.add_argument('--logs',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.text_model,args.logs)
    save_json(args.output,result)
    print(json.dumps(result,indent=2))
