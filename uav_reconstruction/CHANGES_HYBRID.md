# Hybrid reconstruction integration

The optional `./reconstruct.sh --hybrid VIDEO` path runs GPU COLMAP pose estimation and PatchMatch
depth, then `hybrid_dense.py` for aligned prior depth and CPU TSDF fusion. It
stops the COLMAP dense stage after depth maps, avoiding OpenMVS meshing. The
colored mesh is `scene_hybrid_mesh.glb`; `scene_hybrid_provenance.ply` marks
measured and prior-derived surfaces; `scene_hybrid_planes.glb` holds separate
inferred planes. `hybrid_dense_report.json` records all three pixel tiers and
keeps `geometry_accuracy_validated` false.

The learned depth cache records model ID and inference size. A different model
or size requires a new workspace. A prior stage failure stops the run and
records the error in `run_report.json`.

Cached 128-view evaluation of `session_20261011_202810` found 32.9% measured
depth pixels and 16.4% additional prior-consistent pixels. The result remains
incomplete relative to the 90% target. The spatial holdout audit found 3.6%
median alignment error over held-out measured regions, with 62/128 frames
having a 90th percentile error above 10%. That audit cannot validate blank
surfaces. Prior-prior cross-view agreement is not independent depth evidence.
