# CPU reconstruction prototype

This page documents the original offline VM workflow. For the phone's HTTP
API on the current machine, use `PHONE_API.md` and the PyCOLMAP live entry
point. The offline paths and installed binaries below refer to the original VM.

The tested entry point is `adaptive_uav.session_reconstruct`. It uses the
existing **COLMAP build without CUDA**, OpenMVS, OpenCV, Open3D and Trimesh in
the `drone3d` environment. It installs nothing and does not import PyTorch.
The older `pipeline.py` is an experimental model-adapter scaffold, not the
entry point used to generate this session's mesh.

From `/home/vishnu/eswar`:

```bash
/home/vishnu/miniconda3/envs/drone3d/bin/python \
  -m adaptive_uav.session_reconstruct session_20261004_175422
```

For the existing run, append `--resume`. For a separate experiment use
`--output /absolute/path/to/new/output` and optionally `--config config.json`.
External executable locations can be supplied with `--colmap` and `--openmvs`.

For bounded CPU work, add `--profile fast`. It writes to a separate
`reconstruction_adaptive_cpu_fast/` directory. The fast profile uses 1280-pixel
frames, up to 300 keyframes spread over the sequence, fewer matching references
and four-view dense reconstruction. It rejects SfM registration below 70%.
The budget limits computation, but cannot guarantee scene coverage or a
15-minute deadline for arbitrary 10-minute footage. Fast motion can require
more frames; use balanced or a larger custom budget in that case.

## Implemented path

1. Stream video decoding to disk; never hold the video in RAM.
2. Analyze thumbnails for blur, exposure and tracked pixel displacement.
3. Select useful frames with a bounded maximum observation gap.
4. Retrieve temporal neighbors and visually similar non-local references.
5. Cache SIFT features in the COLMAP SQLite database; geometrically verify
   candidate pairs on CPU with guided matching.
6. Estimate poses/triangulate with incremental SfM and bundle adjustment.
   Start with a fixed approximate focal prior, then allow bounded calibration
   refinement if registration is poor. At most two mapping attempts.
7. Reject insufficient camera/point coverage before densification.
8. Undistort; convert zero-distortion SIMPLE_RADIAL/SIMPLE_PINHOLE cameras to
   equivalent PINHOLE cameras for OpenMVS compatibility.
9. Estimate multi-view depth, geometrically validate and visibility-fuse it.
10. Reconstruct/texture a surface; check minimum point/triangle counts.
11. Center the mesh and export GLB with embedded texture. Reload the GLB and
    verify finite vertices and preserved triangle count.

The texture pass uses the original full-resolution MVS camera scene. Seam
leveling is disabled because the installed build produced dark/color-corrupted
patches with it enabled; the corrected mesh was visually inspected.

`open3d_cpu.py` loads the existing CPU extension directly and checks its linked
libraries. This bypasses the installed mixed Open3D package's automatic CUDA
probe; no CUDA binding is imported.

## Outputs and inspection

The session output is `reconstruction_adaptive_cpu/`. It contains:

- `reconstruction.glb`: embedded-texture mesh, relative scale.
- `global_point_cloud.ply`, `global_mesh.ply`: geometry in original SfM coordinates.
- `camera_poses.json`: COLMAP world-to-camera quaternions/translations.
- `reconstruction_metrics.json`: measured coverage, errors, point/face counts
  and the transform between SfM coordinates and the centered export.
- `confidence_map.json`, `uncertainty_map.json`: sparse point quality heuristics
  based on observation counts and reprojection residuals, **not calibrated probabilities**.
- `frame_decisions.json`, `pairs.txt`, `descriptors.npy`, `database.db`: decisions,
  reference pairs and reusable features.
- `checkpoint.json`, `runtime_metrics.json`, `logs/`: stage status/timing and
  exact external commands. Resume skips completed stages; use a new directory
  if source input or configuration changes. Do not edit input while resuming.
- `viewer.html`: interactive viewer; its model-viewer script needs internet.

Generate a CPU raster preview:

```bash
/home/vishnu/miniconda3/envs/drone3d/bin/python \
  -m adaptive_uav.preview_mesh session_20261004_175422/reconstruction_adaptive_cpu
```

## Selective volumetric experiment

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
/home/vishnu/miniconda3/envs/drone3d/bin/python -m adaptive_uav.selective_tsdf \
  session_20261004_175422/reconstruction_adaptive_cpu --max-blocks 3
```

This reuses the bundled OpenMVS `MvsUtils.loadDMAP` parser and selects spatial
blocks from sparse-point uncertainty. It masks depth to those bounds, removes
the bottom confidence quartile per view, integrates into local sparse Open3D
TSDF volumes and exports candidate patches plus measured integration counts.
It does not replace the global mesh without seam and quality validation.

## Scope and limitations

This implements the first practical stages of the requested incremental
research plan using a **classical CPU SfM/MVS backend**. It is not a VisFusion
or On-the-Fly3R implementation. No neural 3R model/checkpoint, learned visibility
network, confidence calibration, semantics, online local-block fusion or
standalone pose-graph optimizer was installed or benchmarked. Bundle adjustment
provides global consistency for the offline model. The local TSDF pass reuses
existing depths; it does not create new evidence for missing surfaces.

Metadata/GPS/IMU inputs are preserved but are not treated as camera poses. This
indoor session's phone GPS is inadequate for metric georeferencing, and
IMU-to-camera extrinsics/video time alignment are not calibrated. The focal
ratio in the configuration is an approximate recorded-camera prior, not a
universal calibration. Supply a suitable prior for other cameras.

Unobserved backsides and textureless/reflective surfaces can have holes or
artifacts. Reprojection error alone is not ground-truth surface accuracy.
Memory/CPU utilization and cross-hardware equivalence are not claimed; stage
wall times are measured. Explicit thread limits avoid using all logical CPUs.

Verify geometry, retrieval and parser behavior:

```bash
/home/vishnu/miniconda3/envs/drone3d/bin/python -m unittest discover -s tests -v
```
