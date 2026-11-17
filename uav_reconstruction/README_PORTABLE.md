# Adaptive UAV reconstruction package

## Recommended saved-video command on this machine

```bash
cd /path/to/3D_reconstruction
./reconstruct.sh /path/to/new_video.mp4
# Experimental depth-prior path, with no OpenMVS meshing:
./reconstruct.sh --hybrid /path/to/new_video.mp4
```

A phone session folder containing `video.mp4` is also accepted. The default
keeps the measured OpenMVS path. `--hybrid` runs GPU COLMAP camera poses and
PatchMatch depth, aligns a learned depth prior, then fuses a colored mesh with
CPU TSDF. It skips OpenMVS meshing. The viewer separates prior provenance and
inferred planes, and the report does not claim geometric accuracy. Repeat the
same command to reuse matching artifacts. Add `--no-serve` if this machine
cannot bind an HTTP port.
See the workspace `AUTOMATIC_RECONSTRUCTION.md` for measured results and
limits. The manual commands below are legacy specialized workflows.

This source package supports phone API capture with incremental CPU COLMAP sparse mapping. It also contains the original offline COLMAP/OpenMVS pipeline and research prototypes. See [adaptive_uav/PHONE_API.md](adaptive_uav/PHONE_API.md) for the live API protocol, timing requirements, outputs and limits.

This is the single maintained source tree for this workspace. The automatic
colored mesh path is `adaptive_uav.reconstruct`; the live dashboard displays a sparse map during
capture. The CPU camera-repair and surface-audit modules from the later
prototype are retained as optional tools in `adaptive_uav/QUALITY.md`. They
currently require a `SIMPLE_RADIAL` camera and do not replace the `PINHOLE`
saved-video path.

For the new pose-conditioned pretrained depth and coloured completion path,
run `./reconstruct.sh /path/to/video.mp4 --multiview` from the workspace root.
See [HIGH_COVERAGE_RECONSTRUCTION.md](../HIGH_COVERAGE_RECONSTRUCTION.md) for
cached-workspace commands, supported/completed outputs, and measured limitations.

## Run the live sparse mapper

```bash
cd /path/to/uav_reconstruction
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-live.txt
.venv/bin/python -m adaptive_uav.phone_live \
  --phone-url http://PHONE_IP:8080 \
  --session runs/phone_capture_1 --capture-fps 3 --buffer-seconds 1
```

Open `http://127.0.0.1:8090` while it runs. Ctrl+C or the dashboard Stop button ends capture and drains saved frames. Reuse the same session and output with `--resume` after interruption. Provide exact camera intrinsics with `--calibration /path/to/intrinsics.json --strict-intrinsics` whenever possible. The phone adapter saves every snapshot and produces `video.mp4`, `video_frames.csv`, raw telemetry, sparse model snapshots, a camera path and timing files.

The revised phone server supports `GEOSTREAM_PHONE_PIN`, `--stream-resolution 1080p --jpeg-quality 100`, `--camera-id ID`, and `--macro-mode on|off`. Optional `--record-phone --download-phone-session` also saves the phone's original ZIP. See `adaptive_uav/PHONE_API.md` for capture setup and lens calibration rules.

`requirements-live.txt` installs NumPy, OpenCV and CPU PyCOLMAP. The live `--finalize-dense` path still requires CPU COLMAP CLI, OpenMVS, Open3D and Trimesh. For a saved video, `adaptive_uav.refine_recorded` and `adaptive_uav.dense_recorded` provide a separate OpenMVS mesh path without the COLMAP CLI or Open3D. Live map updates are sparse, with arbitrary scale.
After the command ends, open `<session>/reconstruction_online_cpu/viewer.html` to inspect the saved sparse map without restarting the server.

## Recorded video and offline tools

For a saved phone session on this Linux machine with the installed CUDA package,
run the bounded GPU trial below. It selects sharp RGB frames from the entire
recording, estimates and refines uncertain intrinsics, and computes depth for at
most 96 reference views with up to six geometrically supported source views.
It imports COLMAP point visibility for OpenMVS meshing and colors surfaces
supported by the fused measurements. Sparse and recorded-gyro gate failures stop
the dense stage; passing them does not establish complete or accurate surfaces.

```bash
cd /path/to/3D_reconstruction/uav_reconstruction
PYTHONPATH=/path/to/3D_reconstruction/.gpu_pycolmap \
  .venv/bin/python -u -m adaptive_uav.fast_session \
  /path/to/3D_reconstruction/session_20261011_202810 \
  --output /path/to/3D_reconstruction/runs/session_20261011_202810_corrected \
  --fusion-min-pixels 5
```

Progress is printed to the terminal. Outputs are `online/viewer.html`,
`dense/mesh_viewer.html`, `dense/scene_mesh_colored.ply` and `dense/scene_colored.glb`.
The fast preset exports measured vertex RGB; an image atlas can be generated
with `adaptive_uav.dense_cuda` by omitting `--skip-texture`.
`run_report.json` records measured timings, calibration assumptions and failures.
New `fast_session` runs default to three agreeing geometric-depth observations
per fused point; the command above explicitly preserves the existing five-observation
cache. Reprojection, depth and normal consistency checks still apply. Changing
fusion settings requires a separate output directory. The cached three-observation
comparison is saved in `runs/session_20261011_202810_corrected/dense_coverage`.
This saved-video trial buffers the sequence before the final camera solve; it
does not promise real-time mesh reconstruction. Recorded sensor logs are retained,
but camera-IMU timing needs a video-start monotonic timestamp before imposing IMU
pose constraints. The detailed diagnosis is in `RECONSTRUCTION_FIXES_20261012.md`.

For the recorded-video online entry point, provide a saved MP4 directly:

```bash
.venv/bin/python -m adaptive_uav.online /path/to/video.mp4 \
  --sfm-backend pycolmap --sparse-only --max-video-seconds 40
```

Omit `--max-video-seconds` to process the full video. The original offline `adaptive_uav.session_reconstruct` and final OpenMVS workflow need the extra CPU binaries and Python packages described in [adaptive_uav/RUNNING.md](adaptive_uav/RUNNING.md). The bundled `apps/openMVS-source/scripts/python/MvsUtils.py` has its upstream license. The `drone_telemetry/scripts` folder is legacy reference code.

Use `--sample-fps 0` to decode and cheaply inspect every frame of a saved video before selecting keyframes, as recommended in the visual-inertial architecture. The default still samples 3 FPS. This option cannot recover frames absent from an existing low-FPS phone capture and does not provide visual-inertial SLAM by itself.

To refine that saved run and make a dense mesh, follow [adaptive_uav/ONLINE.md](adaptive_uav/ONLINE.md#dense-mesh-from-a-saved-video). Mesh quality depends on connected camera coverage and enough parallax in the capture.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

`MANIFEST.json` records source hashes for a packaged ZIP. This project does not include trained VisFusion, 3R, SLAM3R, MASt3R or semantic model weights. It does not yet produce a validated dense online mesh or GPS metric alignment. Some prior corridor reconstructions have visible holes and floating geometry. Accepted camera registration and a low reprojection error alone do not prove complete surface accuracy.
