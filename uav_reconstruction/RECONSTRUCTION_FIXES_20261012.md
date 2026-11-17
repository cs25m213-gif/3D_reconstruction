# Reconstruction diagnosis and corrected trial

## Confirmed failures

1. The old CUDA dense path loaded plain `scene_dense.ply` into `ReconstructMesh`.
   Its `scene_dense.ply.vis` file was present but unused. The mesher requires the
   per-point visibility lists to build its camera support and free-space votes.
   Reimporting `fused.ply` and `fused.ply.vis` through `InterfaceCOLMAP` generated
   an OpenMVS PLY containing `view_indices`. Meshing that file succeeded without
   the prior SIGSEGV. The old code silently fell back to Poisson; it now reports
   failure instead of changing the requested geometry algorithm.
2. The previous 276-view CUDA preview computed depth for every view against up
   to 20 source images, with five iterations and two consistency passes. Its
   measured depth runtime was 6,844.9 seconds. Sparse camera coverage and dense
   reference count had not been separated. The new plan keeps full sparse poses
   but bounds dense references and source views by shared landmarks and parallax.
3. Incremental mapping combined its own 120-second runtime budget with an
   external cancellation timer. The timer converted usable saved partial maps
   into rejected updates. Five updates were rejected; a single 320-second retry
   registered all 276 frames. The external timer has been removed. The built-in
   mapping budget still applies and candidate validation still runs.
4. Uncertain phone intrinsics were fixed during bundle adjustment. Android sensor
   intrinsics do not establish video intrinsics without crop, stabilization and
   resize information. The new phone reports principal point (1632,1224) and
   pixel array (4096,3072), suggesting different reference sizes. The new trial
   explicitly treats its focal seed as approximate and refines focal/radial
   distortion rather than treating the metadata as exact video calibration.
5. Meshing can span unsupported gaps. The new surface filter checks each
   triangle centroid and three edge midpoints against the fused measurements.
   It preserves the raw mesh and reports the retained faces and area. This
   prevents filling empty space merely to produce a connected mesh.

## What remains uncertain

There is no geometric ground truth for either phone session. Low reprojection
error and registering every camera cannot prove shape accuracy. White walls,
repeating doors, shiny floors, fast turns and moving people weaken visual depth.
The new recording shows a room, a corridor and a person: a disconnected model
or sparse gate failure should be reported, not hidden by adding fake surfaces.

The maintained checkout does not contain VisFusion weights or a running
VisFusion model. Earlier VisFusion folders were prepared inputs. The successful
TUM comparison used COLMAP poses and OpenMVS dense reconstruction and texturing.

Sensor logs exist, but the recordings do not provide the video frame PTS to
monotonic timestamp relationship. A gyro comparison can diagnose timing and
rotation disagreement; imposing uncalibrated IMU pose constraints would be
unjustified. Adjacent rotation magnitudes in the old map initially appeared
consistent, but integrating the gyro exposed accumulated drift: at frame 867,
the camera map had rotated 157 degrees from the first view while the gyro
indicated 18 degrees. Local reprojection checks had missed this. The corrected
saved-session path audits accumulated rotation before dense processing.

On the new session the median rotation disagreement is 1.87 degrees and its
90th percentile is 3.14 degrees over the inferred timing-offset range. This
supports consistent rotations, without proving translations or surface shape.

## New trial

`adaptive_uav.fast_session` processes the saved recording causally for frame
selection, then performs one buffered camera solve to avoid repeated large
bundle adjustments. GPU SIFT and matching are enabled. It uses 96 dense
reference views, at most six source views with useful parallax, and three
iterations at maximum image size 960. These are explicit compute limits;
runtime and geometric accuracy are measured separately.

The existing CPU and API entry points remain available. No source video or
older output is deleted. The new trial output is
`runs/session_20261011_202810_corrected` in the Documents workspace.

## Completed trial and limitations

The recording is 211.88 seconds long. Selected frames: 218; registered: 207.
Median reprojection error: 0.71 px. CUDA depth took 817.71 seconds (13.63 minutes).
Measured processing across the initial attempt and cached recovery took 972.21
seconds (16.20 minutes). These numbers are from this input and configuration;
they are not a controlled speed comparison against the older recording.

Fusion produced 80,050 RGB points. Visibility-based meshing, support filtering,
and native topology cleanup produced 47,129 vertices and 90,754 faces. Native
cleanup uses no hole filling or smoothing. Image-atlas export on the initial
attempt crashed after support trimming; native cleanup repairs that topology.
The fast preset now skips the atlas and exports measured vertex RGB to PLY and
GLB. Cached recovery reused all existing depth maps and fused measurements.

The colored mesh was rendered from five registered camera poses and compared
with their actual undistorted source images. Visible mesh coverage was 10.3%,
2.5%, 9.3%, 36.2%, and 5.1%. Textured objects such as the laptop have recognizable
partial surfaces, but large wall and ceiling areas are absent. This output is
**incomplete**, not a verified accurate room mesh. The five samples do not
measure global completeness or ground-truth accuracy. Low-texture walls are
visible in the input; exact video calibration and translation accuracy remain
unverified. Increasing mesh density alone is not a demonstrated remedy.

See `dense/render_check/source_vs_mesh.jpg`, `dense/render_check/coverage.json`,
`gyro_audit.json`, and `run_report.json` for the actual evidence. The viewer
displays an incomplete-reconstruction warning. Forty tests ran successfully
with two skipped tests for optional dependencies.

## Coverage recovery from the same cached depth maps

Depth-map auditing found that the median fraction of positive, finite geometric
depth pixels was 22.76% across 96 references. Photometric proposals occupied a
median 99.68%, but these proposals have not passed geometric consistency and
cannot be treated as reliable measurements.

The five-observation fusion threshold discarded additional supported geometry.
A controlled cached comparison changed only `min_num_pixels` from five to three
while retaining geometric depth input, 2 px reprojection tolerance, 1% relative
depth tolerance, and 10 degree normal tolerance. It produced 262,823 fused points
and 241,203 cleaned colored mesh faces. Sampled camera-view coverage increased
from [10.3%, 2.5%, 9.3%, 36.2%, 5.1%] to
[13.6%, 7.7%, 14.3%, 43.9%, 8.6%]. Camera poses and GPU depth maps did not change.
Lower support can also admit more noise; this is a measured completeness gain,
not proof of increased geometric accuracy. Blank surfaces still largely lack
consistent geometric depth. Support trimming and topology cleanup remain active.

The comparison output is `dense_coverage`, with `coverage_comparison.json` at
the run root. The original `dense` output is preserved. New fusion manifests
reject attempts to reuse a cloud with different fusion thresholds.
