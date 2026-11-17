"""Preserve actual COLMAP tracks when adapting the offline CPU/GPU baseline."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from .runtime_metrics import save_json


def copy_tracked_workspace(directory,reconstruction,image_directory):
    directory=Path(directory).resolve();images=Path(image_directory).resolve()
    if directory in (images,images.parent):raise ValueError('Choose a separate output workspace')
    if any(c.model_name not in ('PINHOLE','SIMPLE_PINHOLE') for c in reconstruction.cameras.values()):
        raise ValueError('Undistort the original camera model before copying it for depth inference')
    for folder in ('sparse','images','stereo/depth_maps','stereo/normal_maps','stereo/consistency_graphs'):
        (directory/folder).mkdir(parents=True,exist_ok=True)
    save_json(directory/'pose_provenance.json',{'status':'writing'})
    reconstruction.write(directory/'sparse')
    for image in reconstruction.images.values():
        source=images/image.name;target=directory/'images'/image.name
        if not source.is_file():raise ValueError('Missing original tracked RGB: '+image.name)
        if not target.exists():target.symlink_to(source)
    save_json(directory/'pose_provenance.json',{'status':'complete','real_feature_tracks_exported':True,
        'poses':reconstruction.num_reg_images(),'native_point_count':reconstruction.num_points3D(),
        'metric_scale_validated':False,'source_images':str(images),'evidence':'Actual original feature observations and triangulated COLMAP landmarks'})
    return reconstruction


def align_sparse_depth(frame,workspace,reconstruction):
    from .multiview_dense import validation_anchors,fit_group_scale
    images={im.name:im for im in reconstruction.images.values() if im.has_pose}
    image=images.get(frame.frame_id)
    if image is None:return frame
    measured,anchors,origin=validation_anchors(Path(workspace),frame.frame_id,reconstruction,image,frame.K,frame.depth.shape)
    anchors[frame.excluded]=0
    scale,checks=fit_group_scale([frame.depth],[anchors],sparse=origin=='triangulated_sfm_tracks')
    check=checks[0];qualified=check['scale_fit_valid'] and check['pixels']>=10
    depth=frame.depth*scale if check['scale_fit_valid'] else frame.depth
    # These dense pixels remain learned predictions. Sparse feature anchors do
    # not turn the entire image into a measured depth map.
    return replace(frame,depth=depth,source='sparse_track_aligned_learned_prediction',
        validation={**frame.validation,'sparse_anchor_check':check,'anchor_source':origin,
            'median':check['median'] if qualified else None,'independent_depth_ground_truth':False})
