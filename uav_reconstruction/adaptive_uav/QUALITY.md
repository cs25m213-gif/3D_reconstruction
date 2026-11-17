# CPU quality-reconstruction path

This path repairs an existing sparse model before spending time on dense
reconstruction. It supports one shared SIMPLE_RADIAL camera, not arbitrary
multi-camera datasets. SciPy is required; it is already installed in drone3d.
No GPU backend is imported and no package installation is performed.

```bash
cd /home/vishnu/eswar
/home/vishnu/miniconda3/envs/drone3d/bin/python \
  -m adaptive_uav.quality_reconstruct \
  /home/vishnu/eswar/session_20261006_230626
```

Input defaults to SESSION/reconstruction_online_cpu (best_model.json, its text
model, and images/). Override it with `--source /path/to/reconstruction`.
Output defaults to SESSION/reconstruction_refined_cpu, separate from the input.
Override it with `--output /path/to/new/output`.
Use `--sparse-only` to test cameras without dense meshing; `--resume` continues
a completed repair into dense reconstruction or reuses completed dense stages.
Do not use the old online `--resume` command to request camera repair.

## What changed

- Sparse CPU trust-region/LSMR optimization with soft-L1 residuals, independent
  of the CHOLMOD solver that failed in the original map.
- One fixed camera pose plus one translation coordinate remove the seven
  monocular gauge degrees of freedom. Scale remains arbitrary.
- Shared focal/radial-distortion refinement with bounded parameters; principal
  point remains fixed. This is self-calibration, not measured calibration.
- A configurable point budget with explicit camera coverage (default 3,000)
  bounds optimization cost. It does not optimize every original sparse point;
  all remaining eligible tracks are re-triangulated or revalidated with the
  refined cameras. The budget does not restrict final map coverage.
- Track pruning requires positive depth, <=2.5 px reprojection error, >=3
  observations and >=3 degrees maximum triangulation angle. Cameras with fewer
  than 20 retained observations are removed; dependent tracks are rechecked.
- Candidate acceptance requires solver convergence, lower robust objective,
  >=150 landmarks and retention of at least 70% of the source registered views.
- Quality dense defaults use full selected-frame resolution, eight candidate
  views, four-view fusion, and three geometric-consistency iterations. Stricter
  fusion may leave more holes rather than invent missing geometry.
- Existing maps/GLBs are preserved. Each repair attempt has a separate candidate
  directory. Reports include actual measured time and rejected camera names.
- A CPU ray-casting audit samples nine registered views after dense export.
  At least 60% must have surface hits across at least 35% of image rays. Failing
  this heuristic marks the run failed and retains the GLB as a rejected candidate.
  A passing ray-coverage check still does not certify accurate geometry.
- Online mapper calibration is now refined once enough cameras are registered,
  rather than permanently freezing an approximate prior. Its quality gates
  still reject repeatedly failing mapper solutions. A bounded CPU repair is
  attempted on a numerically/ geometrically weak update; it must pass both
  conditioning checks and retention of previously accepted camera identities.
  Failed repair retains the previous global map instead of promoting the update.
  The updated online configuration requires a fresh output directory; it cannot
  resume an older checkpoint created with different configuration values.

## Limits and validation

This cannot recover unobserved walls or guarantee metric accuracy. Dense and
textured results need visual checks and, for accuracy claims, measured ground
truth. Robust optimization can converge to an incorrect local solution.
Removing unsupported views reduces coverage, which is reported explicitly.
The repair does not add new feature matches or register previously missing
cameras. Hardware and capture geometry determine runtime and achievable quality.

Reports: quality_audit.json, robust_candidate_*/optimization_report.json,
surface_audit.json, quality_run.json, runtime_metrics.json and per-stage logs/.
Early point-budget-only repair checkpoints cannot resume with this version;
use a new output folder to preserve the previous candidate.

Algorithm references: [SciPy least_squares](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html),
[COLMAP calibration guidance](https://colmap.github.io/faq.html).
