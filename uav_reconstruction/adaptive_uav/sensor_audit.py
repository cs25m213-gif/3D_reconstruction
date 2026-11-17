"""Independent gyro rotation audit; does not assume camera-IMU calibration."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .gyro_initialization import gyro_history


def compare_rotations(timestamps, camera, times, gyro, offsets, minimum_overlap=.8):
    """Compare only overlapping clock ranges; never fabricate endpoint samples."""
    best=None
    for offset in offsets:
        shifted=timestamps+offset
        valid=(shifted>=times[0])&(shifted<=times[-1])
        if valid.sum()<min(9,len(timestamps)) or valid.mean()<minimum_overlap:continue
        indices=np.flatnonzero(valid)
        idx=np.searchsorted(times,shifted[valid]).clip(0,len(times)-1)
        visual=(camera[indices[0]].inv()*camera[indices]).magnitude()*180/np.pi
        angles=(gyro[idx[0]].inv()*gyro[idx]).magnitude()*180/np.pi
        error=np.abs(visual-angles);score=float(np.median(error))
        if best is None or score<best['median_rotation_error_deg']:
            best={'median_rotation_error_deg':score,'diagnostic_video_offset_seconds':float(offset),
                  'errors':error,'visual_angles':visual,'gyro_angles':angles,'indices':indices,
                  'sensor_overlap_fraction':float(valid.mean())}
    return best


def audit_gyro(session, model, fps):
    import pycolmap
    session = Path(session)
    metadata = json.loads((session / 'metadata.json').read_text())
    try:times,gyro=gyro_history(session)
    except (ValueError,OSError,KeyError) as error:return {'usable':False,'reason':str(error)}
    reconstruction = pycolmap.Reconstruction(model)
    images = sorted((im for im in reconstruction.images.values() if im.has_pose), key=lambda im: im.name)
    timestamps = np.array([int(im.name.split('_')[-1].split('.')[0]) / fps for im in images])
    if len(images)<2:return {'usable':False,'reason':'Fewer than two checked cameras'}
    camera = Rotation.from_matrix([im.cam_from_world().rotation.matrix() for im in images])
    explicit=metadata.get('video_start_elapsed_ns')
    offsets=([(int(explicit)-int(metadata['session_start_elapsed_ns']))*1e-9] if explicit is not None
             else np.arange(0,2.51,.05))
    best=compare_rotations(timestamps,camera,times,gyro,offsets)
    if best is None:return {'usable':True,'passed':False,'reason':'Recorded gyro does not cover at least 80% of checked cameras at a valid clock offset'}
    score=best['median_rotation_error_deg'];offset=best['diagnostic_video_offset_seconds'];error=best['errors']
    visual,angle,indices=best['visual_angles'],best['gyro_angles'],best['indices']
    p90 = float(np.quantile(error, .9))
    return {'usable': True, 'passed': p90 < 20 and float(np.mean(error > 30)) < .1,
            'median_rotation_error_deg': score, 'p90_rotation_error_deg': p90,
            'fraction_error_above_30_deg': float(np.mean(error > 30)),
            'diagnostic_video_offset_seconds': offset,
            'max_gyro_gap_seconds': float(np.diff(times).max()),
            'sensor_overlap_fraction':best['sensor_overlap_fraction'],
            'compared_camera_frames':len(indices),'excluded_camera_frames':len(images)-len(indices),
            'offset_source':'recorded_video_start_elapsed_ns' if explicit is not None else 'diagnostic_search',
            'camera_imu_extrinsics_calibrated':False,
            'note': 'Offset search is diagnostic. Relative rotation angle is invariant to camera-body rotation; exact sensor synchronization is not claimed.',
            'examples': [{'frame': images[indices[j]].name, 'visual_rotation_deg': float(visual[j]),
                          'gyro_rotation_deg': float(angle[j])}
                         for j in np.unique(np.linspace(0, len(indices) - 1, min(8,len(indices)), dtype=int))]}
