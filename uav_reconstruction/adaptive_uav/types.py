from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class Frame:
    frame_id: str
    path: Path
    timestamp: Optional[float] = None
    gps: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    quality: float = 1.0
    novelty: float = 0.0
    uncertainty: float = 1.0
    feature_count: int = 0


@dataclass
class PointObservation:
    xyz: np.ndarray
    confidence: float = 1.0
    visibility: float = 1.0
    uncertainty: float = 0.0
    frame_id: str = ""
    semantic: Optional[int] = None

    @property
    def weight(self) -> float:
        return max(1e-8, self.confidence * self.visibility / (1.0 + self.uncertainty))


class ReconstructionDecision(str, Enum):
    SKIP = "skip"
    CHEAP_UPDATE = "cheap_update"
    LOCAL_3R = "local_3r"
    LOCAL_3R_HIGH_CONTEXT = "local_3r_high_context"
    REFINE = "refine"
    REJECT = "reject"


@dataclass
class Decision:
    action: ReconstructionDecision
    reason: str
    references: int = 0
    resolution: int = 0


@dataclass
class MapCell:
    xyz: np.ndarray
    weight: float
    uncertainty: float
    observations: int = 1
    frames: set[str] = field(default_factory=set)
    semantic: Optional[int] = None
