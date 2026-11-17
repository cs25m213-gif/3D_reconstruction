# Automatic reconstruction policy — 12 October 2026

Run `./reconstruct.sh /path/to/video.mp4` for the measured OpenMVS path, or `./reconstruct.sh --hybrid /path/to/video.mp4` for the **experimental hybrid path**. A phone session directory containing `video.mp4` is also accepted. Hybrid uses GPU COLMAP poses and PatchMatch depth, then an aligned depth prior and CPU TSDF; it skips OpenMVS meshing. This processes saved captures, not a live phone stream. If this machine cannot bind a viewer port, pass `--no-serve` and open the generated `dense/mesh_viewer.html` through your own local HTTP server.

Hybrid outputs are `dense/scene_hybrid_mesh.glb`, `dense/scene_hybrid_provenance.ply`, `dense/scene_hybrid_planes.glb`, and `dense/hybrid_dense_report.json`. Green provenance means near measured TSDF points; blue means added prior geometry. Inferred planes are a separate optional viewer layer. The report's measured, prior-consistent, and inferred percentages describe image pixel coverage, not validated 3D accuracy. A failed hybrid stage marks the run failed and records the error.

The stages and metrics below document the measured OpenMVS baseline. Hybrid is an alternative final meshing path.

## Automatic stages

1. Buffered frame selection favors sharp, useful views, preserving original RGB for output.
2. GPU COLMAP estimates camera poses and refines uncertain focal length/distortion. Registration and reprojection gates detect failed maps. Recorded gyro provides a rotation diagnostic when usable, not precisely synchronized pose constraints.
3. Dense source selection considers actual point parallax and image-cell coverage, reserving views for supporting geometry. Budget: up to 128 depth reference cameras, six sources each, 960-pixel long side, three PatchMatch iterations.
4. Local inverse-depth planes estimate normals only on sufficiently consistent patches. Depth values remain unchanged. An automatic cross-view check accepts the correction only when sampled normal agreement does not decrease.
5. Fusion keeps the existing 1% depth, 2-pixel reprojection and 10-degree normal tolerances, with three supporting observations. OpenMVS receives visibility data. Unsupported mesh surfaces are trimmed; RGB colors are exported to PLY/GLB.
6. A low-cost planar completion step segments smooth neutral surfaces, fits planes from measured depth, checks held-out anchors and nearby views, then writes a **separate optional inferred layer**. It does not alter measured depth or mesh. The viewer toggle starts off. Areas without supporting depth are left empty.
7. A coverage report identifies missing surfaces and low-texture regions. Inferred planes do not raise the measured coverage score. Finished computation and adequate coverage have separate statuses.

Input, policy and implementation manifests prevent accidental reuse of mismatched caches. The viewer follows `runs/latest_reconstruction`. Logs and diagnostic artifacts remain available without asking the user to inspect each stage manually. Failed quality gates stop with a recorded reason; the system does not guarantee recovery from every capture.

## Diagnosed causes and corrected run

The previous source planner concentrated on repeated textured regions and accepted view pairs with inadequate useful point parallax. Noisy depth normals then rejected many samples whose positions agreed. The revised spatial planner and checked normal estimation address those two measured losses. Earlier failures also included pose drift and missing OpenMVS visibility information; those now have explicit checks.

Result: `runs/session_20261011_202810_corrected/dense_automatic`.

- 677,410 fused points; 672,741 colored mesh faces.
- Same-view projected mesh occupancy: bed 13.63% → 43.07%; laptop 43.93% → 57.41%; doorway 7.74% → 11.01%; wardrobe 14.30% → 19.44%; ceiling 8.65% → 15.00%.
- The final automatic quality status is **incomplete**. Image occupancy is not whole-room completeness or geometric accuracy. No 90% coverage has been demonstrated.
- About 43% of analyzed image pixels have very low local texture; substantial wall and ceiling surfaces still lack reliable depth. Sharpening cannot recover absent image evidence or missing viewpoints.
- Geometric depth refinement took 780.1 seconds using cached camera poses and photometric initialization. This is not a measured fresh end-to-end runtime.
- The planar step checked 64 reference views in about 2.2 seconds on this saved result. It accepted 11 small planes (4,528 faces), mainly walls and floor, with measured anchors and other-view checks. The ceiling still lacked sufficient corroborating depth. The inferred layer is available with **Show estimated planes** in the colored mesh viewer and is not a claim of 3D accuracy.

Validation: 46 unit tests ran, 44 passed and two skipped. The corrected dense path completed on the recorded session and its colored result and planar previews were visually inspected. The new wrapper was checked separately; a fresh full reconstruction through that wrapper has not yet been benchmarked. Broader capture validation remains necessary before claiming consistent quality across arbitrary videos.

## Adaptive sparse controller — 13 October 2026

The automatic wrapper now profiles short frame windows across each video before
GPU SfM. At the selector's 360-pixel analysis width it measures sharpness,
texture, illumination variation, SIFT feature counts, short-window optical flow,
and verified homography match ratios at 0.2, 0.5, 1 and 2 seconds. These are
visual difficulty proxies; homography support does not establish 3D parallax.
Gyro is not used by the profiler because video/sensor synchronization is unknown;
the existing post-SfM gyro audit remains in place.

Rules select feature budget, motion threshold, blur threshold, maximum frame
gap and temporal/retrieval budgets. The sharpness buffer is shortened with the
chosen gap so it does not discard the requested temporal coverage. Feature
counts are capped CPU SIFT counts at analysis resolution, not predictions of
full-resolution GPU feature yield.

Before dense processing, the controller requires at least 80% registration,
at least five cameras and 150 sparse points, finite median reprojection error
between 0 and 2 pixels, and one connected verified-match graph including all
database images. Strong graph edges require 30 inliers. Graph connectivity alone
does not certify correct geometry. Calibration and sensor gates still apply.

Connectivity failures can trigger two recovery attempts with denser frames and
larger bounded matching/feature budgets. Recovery outputs are isolated in
`online_recovery_1` and `online_recovery_2`; dense processing uses the first
attempt that passes all sparse gates. High or nonfinite reprojection error stops
recovery for calibration/geometry review. Exhausted recovery marks the run
failed and does not proceed to dense.

Artifacts: `video_profile.json`, `sparse_config_N.json`, and
`adaptive_sfm_history.json` record context, settings, graph components, temporal
breaks, metrics and runtime. Policy version 4 includes the controller source in
the wrapper's cache identity. Existing finished runs retain their artifacts.

This first version retries the full sparse pipeline. It locates temporal breaks
but does not yet insert frames or rematch only those regions. Temporal reference
counts are selected per attempt, not expanded per frame until verified support
is sufficient. Learned features, model training and contextual bandits are not
implemented. Thresholds are initial heuristics requiring varied-capture
validation; no fresh end-to-end GPU reconstruction has been benchmarked for
this change.

## Combined measured / learned / visibility run

`reconstruct_combined.sh` reuses an adaptive sparse run for the same input video,
checks its identity, selects a camera interval, computes fresh CUDA PatchMatch
maps for those poses, runs pose-conditioned pretrained DA3, and compares baseline
TSDF fusion with the optional cross-sequence visibility filter. It retains both
outputs and selects a result by consistency, measured-depth agreement and then
coverage. Neither a completed run nor this ranking establishes external accuracy.
This uses DA3 and classical visibility checks, not the trained VisFusion network.

Example on this machine:

```bash
./reconstruct_combined.sh session_20261004_175217   --sparse-run runs/session_20261004_175217_automatic_3e4f63136e   --output runs/session_20261004_175217_combined
```

The selector checks existing models and one startup-frame exclusion. Every
candidate, rejected audit and accepted audit is retained in `camera_selection.json`.
Intrinsics, reprojection, sparse support, at least 80% registration within the
selected interval, and the existing gyro thresholds must pass. The report also
records original selected-frame counts and the fraction of capture time retained,
so interval registration is never presented as full-video registration. Using
gyro for selection means its accepted audit is not an independent held-out test.
Partial intervals are permitted and clearly labelled. Unknown metric scale is
retained as COLMAP visual units. The selected result includes separate supported
and completed learned-surface meshes; inferential completion is not measured
geometry. A visibility pass is a comparison and may reduce coverage.

For this session, camera selection retained 94 poses from approximately 25.1 to
51.2 seconds. The full-video fixed-intrinsic model remains rejected. CUDA depth
processing was confirmed running; the combined dense result has not yet finished.
Watch `runs/session_20261004_175217_combined.log` and
`runs/session_20261004_175217_combined/run_report.json` for progress.
