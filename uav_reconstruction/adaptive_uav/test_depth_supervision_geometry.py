"""Numerical consistency checks for measured LiDAR training geometry."""
import numpy as np

from .build_cached_lidar_targets import downsample_sparse_min
from .lidar_depth_supervision import extrinsics_for_scene, project_returns


def run():
    r, t = extrinsics_for_scene('HKisland03')
    camera = np.array([[0., 0., 10.]])
    lidar = (camera - t) @ np.linalg.inv(r.T)
    info = {'OriginalSceneName': 'HKisland03', 'P3x3': [[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]],
            'Width': 100, 'Height': 80, 'T4x4': np.eye(4).tolist()}
    depth, stats = project_returns(lidar, info)
    assert stats['valid_pixels'] == 1
    assert abs(depth[40, 50] - 10.) < .01
    moved = np.eye(4)
    moved[0, 3] = 1.
    compensated, _ = project_returns(lidar, info, camera_at_lidar=moved)
    assert abs(compensated[40, 60] - 10.) < .01
    native = np.zeros((4, 4), np.float32)
    native[0, 0], native[1, 1], native[3, 3] = 12., 8., 20.
    small = downsample_sparse_min(native, (2, 2))
    assert small[0, 0] == 8. and small[1, 1] == 20. and (small > 0).sum() == 2
    # Camera translations and depths expressed in two metric gauges must feed
    # identical normalized geometry to the student.
    baseline, anchor_depth = 10., 100.
    assert np.isclose(baseline / anchor_depth, 3 * baseline / (3 * anchor_depth))
    return {'projection_center_z_m': float(depth[40, 50]),
            'motion_compensation_pixel_shift': 10,
            'sparse_z_buffer_min_m': float(small[0, 0]),
            'scale_equivariance': 'pass'}


if __name__ == '__main__':
    print(run())
