"""Fit learned scale on training pixels; qualify completion on held-out pixels."""
from dataclasses import replace
import numpy as np
from .multiview_dense import fit_group_scale,correct_local_scale


def refine_depth(frame,measured):
    measured=np.asarray(measured,np.float32).copy()
    if measured.shape!=frame.depth.shape:raise ValueError('Stereo/depth raster mismatch')
    measured[~np.isfinite(measured)|(measured<=0)|frame.excluded]=0
    scale,checks=fit_group_scale([frame.depth],[measured]);check=checks[0]
    qualified=check['scale_fit_valid'] and check['pixels']>=100 and check['median'] is not None and check['median']<=.2
    corrected=np.zeros_like(frame.depth);local={'used':False,'reason':'unqualified_prior'}
    if qualified:
        corrected,local=correct_local_scale(frame.depth*scale,measured)
        corrected[frame.excluded]=0
    depth=np.where(measured>0,measured,corrected).astype(np.float32)
    evidence=np.where(measured>0,1,np.where(corrected>0,3,0)).astype(np.uint8)
    return (replace(frame,depth=depth,evidence=evidence,confidence=np.where(measured>0,1.,frame.confidence).astype(np.float32),source='stereo_plus_qualified_learned_depth',
        validation={**frame.validation,'stereo_scale_fit':check,'stereo_scale':scale,'local_correction':local,
            'learned_completion_qualified':bool(qualified),'measurement_source':'COLMAP stereo; correlated with supplied poses'}),
        {'scale':scale,'held_out_check':check,'learned_completion_qualified':bool(qualified),
         'measured_pixels':int((measured>0).sum()),'retained_pixels':int((depth>0).sum())})
