# Adaptive UAV prototype

For the current phone API capture and incremental CPU COLMAP mapping, see
`PHONE_API.md`. The live command is `python -m adaptive_uav.phone_live` with
`requirements-live.txt` installed. It saves a sparse map and standalone
viewer. Dense meshing needs the optional COLMAP/OpenMVS/Open3D runtime.

The modules described below are research prototypes and are not the live
reconstruction backend.

CPU-first primitives for adaptive keyframe scheduling, weighted Sim(3)
alignment, and sparse confidence/visibility-aware voxel fusion.  The package
has no PyTorch, CUDA, Open3D, or GPU runtime dependency.  Future 3R models
must implement an adapter around these contracts rather than changing the
scheduler or map.
