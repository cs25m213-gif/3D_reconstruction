from dataclasses import dataclass
from .types import Decision, Frame, ReconstructionDecision


@dataclass(frozen=True)
class ComputeBudget:
    """Runtime budget, not a hardware-specific batch-size prescription."""
    max_references: int = 8
    base_resolution: int = 512
    max_resolution: int = 1024
    uncertainty_refine: float = 0.65
    novelty_reconstruct: float = 0.25
    minimum_quality: float = 0.20


class AdaptiveScheduler:
    def __init__(self, budget=None):
        self.budget = budget or ComputeBudget()

    def evaluate(self, frame: Frame, map_uncertainty=1.0, overlap=0.0,
                 references_available=0, quality=None):
        q = frame.quality if quality is None else quality
        if q < self.budget.minimum_quality:
            return Decision(ReconstructionDecision.SKIP, "insufficient frame quality")
        if frame.uncertainty >= self.budget.uncertainty_refine or map_uncertainty >= self.budget.uncertainty_refine:
            refs = min(self.budget.max_references, max(2, references_available))
            return Decision(ReconstructionDecision.LOCAL_3R_HIGH_CONTEXT,
                            "high geometric uncertainty", refs, self.budget.max_resolution)
        if frame.novelty >= self.budget.novelty_reconstruct and overlap < 0.75:
            refs = min(self.budget.max_references, max(1, references_available))
            return Decision(ReconstructionDecision.LOCAL_3R, "new geometric content", refs,
                            self.budget.base_resolution)
        if overlap > 0.90 and map_uncertainty < 0.35:
            return Decision(ReconstructionDecision.SKIP, "geometry already reliable")
        return Decision(ReconstructionDecision.CHEAP_UPDATE, "low-cost observation update")
