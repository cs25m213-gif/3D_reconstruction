"""CPU-first adaptive UAV reconstruction primitives.

The package intentionally contains no GPU or deep-learning import.  A model
adapter can be added later without changing the scheduler or map contracts.
"""

from .types import Frame, PointObservation, ReconstructionDecision
from .scheduler import AdaptiveScheduler, ComputeBudget
from .geometry import weighted_umeyama
from .fusion import VoxelMap
from .pipeline import AdaptiveReconstruction, ReconstructionResult, ThreeRBackend

__all__ = ["Frame", "PointObservation", "ReconstructionDecision",
           "AdaptiveScheduler", "ComputeBudget", "weighted_umeyama", "VoxelMap",
           "AdaptiveReconstruction", "ReconstructionResult", "ThreeRBackend"]
