from collections import defaultdict
import numpy as np
from .types import MapCell, PointObservation


class VoxelMap:
    """Sparse confidence/visibility-weighted map; no full-scene allocation."""
    def __init__(self, voxel_size=0.25):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.cells = {}

    def _key(self, xyz):
        return tuple(np.floor(np.asarray(xyz) / self.voxel_size).astype(np.int64))

    def insert(self, observation: PointObservation):
        key = self._key(observation.xyz)
        old = self.cells.get(key)
        if old is None:
            self.cells[key] = MapCell(np.asarray(observation.xyz, float).copy(), observation.weight,
                                      observation.uncertainty, 1, {observation.frame_id}, observation.semantic)
            return
        total = old.weight + observation.weight
        old.xyz = (old.xyz * old.weight + observation.xyz * observation.weight) / total
        old.weight = total
        old.uncertainty = (old.uncertainty * old.observations + observation.uncertainty) / (old.observations + 1)
        old.observations += 1
        if observation.frame_id:
            old.frames.add(observation.frame_id)

    def uncertainty(self):
        if not self.cells:
            return 1.0
        return float(np.mean([c.uncertainty / max(1.0, c.observations) for c in self.cells.values()]))

    def points(self):
        return np.asarray([c.xyz for c in self.cells.values()], dtype=float)
