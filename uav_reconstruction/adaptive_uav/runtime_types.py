"""Validated geometry/evidence contracts for incremental reconstruction."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


def transform(value):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(4,4) or not np.isfinite(value).all():
        raise ValueError('Pose must be a finite 4x4 matrix')
    if not np.allclose(value[3],[0,0,0,1],atol=1e-7):raise ValueError('Invalid homogeneous pose')
    r=value[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-4) or np.linalg.det(r)<.999:
        raise ValueError('Pose rotation must be proper and orthonormal')
    return value.copy()


def intrinsic(value):
    k=np.asarray(value,dtype=np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('Intrinsics must be finite with positive focal lengths')
    if not np.allclose(k[2],[0,0,1]):raise ValueError('Invalid pinhole intrinsic matrix')
    return k.copy()


@dataclass
class FramePacket:
    frame_id: str
    index: int
    timestamp: float
    rgb: np.ndarray
    K: np.ndarray
    source: str=''
    clock: str='video_pts'
    calibration_id: str='estimated'
    pixel_transform: np.ndarray=field(default_factory=lambda:np.eye(3))
    sensor_timestamp: float|None=None
    calibration_exact: bool=False

    def __post_init__(self):
        self.K=intrinsic(self.K)
        if not np.isfinite(self.timestamp) or self.index<0:raise ValueError('Invalid frame timestamp/index')
        if self.rgb.ndim!=3 or self.rgb.shape[2]!=3:raise ValueError('RGB image required')


@dataclass
class PoseEstimate:
    frame_id: str
    timestamp: float
    T_world_from_camera: np.ndarray
    backend: str
    state: str='tracking'
    version: int=0
    submap_id: str='0000'
    scale_status: str='arbitrary_visual_units'
    diagnostics: dict=field(default_factory=dict)

    def __post_init__(self):
        self.T_world_from_camera=transform(self.T_world_from_camera)
        if self.state not in ('initializing','tracking','degraded','lost','relocalizing'):
            raise ValueError('Unknown tracking state')

    @property
    def cam_from_world(self):return np.linalg.inv(self.T_world_from_camera)

    def to_dict(self):
        return {**self.__dict__,'T_world_from_camera':self.T_world_from_camera.tolist()}


@dataclass
class DepthFrame:
    frame_id: str
    timestamp: float
    depth: np.ndarray
    rgb: np.ndarray
    K: np.ndarray
    T_world_from_camera: np.ndarray
    observation_id: str
    pose_version: int=0
    source: str='learned_prediction'
    scale_status: str='arbitrary_visual_units'
    excluded: np.ndarray|None=None
    confidence: np.ndarray|None=None
    context_ids: tuple=()
    context_poses: dict=field(default_factory=dict)
    validation: dict=field(default_factory=dict)
    age: int=0
    submap_id: str='0000'
    evidence: np.ndarray|None=None

    def __post_init__(self):
        self.K=intrinsic(self.K);self.T_world_from_camera=transform(self.T_world_from_camera)
        self.depth=np.asarray(self.depth,np.float32).copy()
        if self.depth.ndim!=2 or self.rgb.shape[:2]!=self.depth.shape:raise ValueError('RGB/depth shape mismatch')
        self.depth[~np.isfinite(self.depth)|(self.depth<=0)]=0
        if self.excluded is None:self.excluded=np.zeros(self.depth.shape,bool)
        self.excluded=np.asarray(self.excluded,bool)
        if self.excluded.shape!=self.depth.shape:raise ValueError('Mask/depth shape mismatch')
        if self.confidence is None:self.confidence=np.ones(self.depth.shape,np.float32)
        if self.confidence.shape!=self.depth.shape:raise ValueError('Confidence/depth shape mismatch')
        if self.evidence is None:
            kind=1 if self.source in ('sensor_measured','simulated_ground_truth') else 4 if self.source=='warped_previous_estimate' else 3
            self.evidence=np.where(self.depth>0,kind,0).astype(np.uint8)
        if self.evidence.shape!=self.depth.shape:raise ValueError('Evidence/depth shape mismatch')

    @property
    def valid(self):return (self.depth>0)&~self.excluded&np.isfinite(self.confidence)&(self.confidence>0)


@dataclass
class PoseCorrection:
    version: int
    poses: dict
    reason: str='local_optimization'


@dataclass
class MapUpdate:
    sequence: int
    added: tuple=()
    removed: tuple=()
    rebuilding: tuple=()
    pose_version: int=0
