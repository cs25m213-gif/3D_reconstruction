"""End-to-end adaptive reconstruction orchestration.

The 3R backend is intentionally an injected adapter.  This makes the control
logic executable on CPU now and prevents a model implementation from forcing
CUDA or a particular accelerator into the architecture.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable, Optional

from .fusion import VoxelMap
from .scheduler import AdaptiveScheduler
from .types import Decision, Frame, PointObservation, ReconstructionDecision


@dataclass
class ReconstructionResult:
    accepted: bool
    observations: list[PointObservation] = field(default_factory=list)
    residual: float = 0.0
    reason: str = ""


class ThreeRBackend:
    """Contract for DUSt3R/MASt3R/VGGT/etc. adapters.

    Implementations may use CPU or another runtime, but must return local
    points with confidence and visibility; the pipeline does not inspect model
    internals or assume a batch size.
    """
    def reconstruct(self, frame: Frame, references: list[Frame], resolution: int) -> ReconstructionResult:
        raise NotImplementedError


class AdaptiveReconstruction:
    def __init__(self, backend: Optional[ThreeRBackend] = None,
                 scheduler: Optional[AdaptiveScheduler] = None,
                 voxel_size: float = 0.25):
        self.backend = backend
        self.scheduler = scheduler or AdaptiveScheduler()
        self.map = VoxelMap(voxel_size)
        self.frames: list[Frame] = []
        self.events: list[dict] = []

    @staticmethod
    def cheap_geometric_check(frame: Frame, previous: Optional[Frame]):
        """Estimate novelty from metadata and a stable file fingerprint.

        This fallback never decodes every pixel.  A production feature adapter
        can replace it with ORB/AKAZE or a CPU descriptor implementation.
        """
        if previous is None:
            return 1.0, 0.0
        distance = 0.0
        if frame.gps is not None and previous.gps is not None:
            distance = sum((float(a) - float(b)) ** 2 for a, b in zip(frame.gps, previous.gps)) ** 0.5
        try:
            same = sha256(Path(frame.path).read_bytes()).digest() == sha256(Path(previous.path).read_bytes()).digest()
        except OSError:
            same = False
        novelty = min(1.0, distance / 5.0) if distance else (0.0 if same else 0.5)
        overlap = 0.95 if same else max(0.0, 1.0 - novelty)
        return novelty, overlap

    def process(self, frame: Frame, references: Optional[list[Frame]] = None) -> Decision:
        previous = self.frames[-1] if self.frames else None
        frame.novelty, overlap = self.cheap_geometric_check(frame, previous)
        decision = self.scheduler.evaluate(frame, self.map.uncertainty(), overlap,
                                           len(references or self.frames[-8:]))
        event = {'frame_id': frame.frame_id, 'action': decision.action.value,
                 'reason': decision.reason, 'novelty': frame.novelty,
                 'map_uncertainty_before': self.map.uncertainty()}
        if decision.action in (ReconstructionDecision.LOCAL_3R,
                               ReconstructionDecision.LOCAL_3R_HIGH_CONTEXT):
            if self.backend is None:
                decision = Decision(ReconstructionDecision.REJECT, '3R backend not configured')
            else:
                result = self.backend.reconstruct(frame, references or self.frames[-decision.references:], decision.resolution)
                if result.accepted:
                    for observation in result.observations:
                        self.map.insert(observation)
                else:
                    decision = Decision(ReconstructionDecision.REJECT, result.reason or 'local validation failed')
        elif decision.action == ReconstructionDecision.CHEAP_UPDATE:
            # Metadata-only updates do not invent geometry.
            pass
        self.frames.append(frame)
        event['action_final'] = decision.action.value
        event['map_uncertainty_after'] = self.map.uncertainty()
        self.events.append(event)
        return decision

    def refine_uncertain(self, refiner: Optional[Callable[[VoxelMap], int]] = None,
                         threshold: float = 0.65) -> int:
        """Refine only when the map uncertainty justifies it."""
        if self.map.uncertainty() < threshold or refiner is None:
            return 0
        return int(refiner(self.map))

    def run(self, frames: Iterable[Frame]):
        for frame in frames:
            yield self.process(frame)
