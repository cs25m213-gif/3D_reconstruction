# Online CPU reconstruction

The live phone entry point is described in [PHONE_API.md](PHONE_API.md). The recorded-video entry point reads any saved MP4 sequentially and publishes validated sparse map snapshots while it reads. Both paths use `configs/cpu_online.json` for keyframe and map update thresholds.

```bash
.venv/bin/python -m adaptive_uav.online /absolute/path/to/video.mp4 \
  --sfm-backend pycolmap --sparse-only \
  --sample-fps 3 --image-width 960 --max-video-seconds 40 \
  --output /absolute/path/to/recorded_online_output
```

`--max-video-seconds` is optional; omit it to process the complete saved video. A session directory containing `video.mp4` also works as input. Resume an interrupted run with the same options plus `--resume`. The video file identity, config and time limit must remain unchanged for resume. CPU PyCOLMAP is included in `requirements-live.txt`. `--sfm-backend colmap --colmap /path/to/cpu/colmap` remains available for the original CLI backend.

## Architecture actually implemented

1. Decode each incoming image and measure blur, exposure, optical flow, and optional synchronized IMU motion.
2. Select sharp keyframes when visual motion or the coverage interval calls for them. Every source image remains stored by the phone adapter, including unselected frames.
3. Retrieve only earlier temporal and visual neighbors for matching.
4. Run CPU SIFT matching and incremental COLMAP mapping in blocks. The phone path defaults to PyCOLMAP; recorded video can use PyCOLMAP or COLMAP CLI.
5. Validate candidate models for registered cameras, point support, reprojection errors, and retention of the previous map's cameras. Publish only accepted candidates; retain accepted snapshots.
6. Optionally run a local OpenMVS dense/mesh pipeline **after** capture. The separate `dense_recorded` command can use PyCOLMAP for image undistortion, so it does not require the COLMAP CLI.

The system is causal but mapping can run slower than capture. Phone acquisition persists all incoming JPEGs to disk while mapping blocks; the dashboard distinguishes captured, processed, buffered, selected, registered and accepted counts. Its 3D view renders the current sparse point cloud and camera path. It does not show a dense mesh during capture.

## Dense mesh from a saved video

The sparse online map can be improved by adding wider image matches and rerunning global SfM. This refinement uses the entire saved image set, so it is a final offline step rather than a causal online map update. Dense reconstruction then reads the **original MP4 frames** at higher resolution, rescales the camera intrinsics, and runs OpenMVS depth estimation and meshing. Results cover only views with valid, connected camera poses.

```bash
.venv/bin/python -m adaptive_uav.refine_recorded /path/to/online_output \
  --output /path/to/refined
.venv/bin/python -m adaptive_uav.dense_recorded /path/to/video.mp4 \
  /path/to/refined/best_model --output /path/to/refined/dense_mesh \
  --openmvs /path/to/OpenMVS/bin --width 1920 --resolution-level 0 \
  --three-assets /path/to/local_three_modules
```

`dense_recorded` writes `scene_dense.ply`, `scene_mesh.ply`, `scene_textured_glb.glb`, a companion texture PNG, logs for each stage, and `dense_metrics.json`. The optional `--three-assets` writes `mesh_viewer.html` and `dense_mesh_bundle.zip`; the bundle keeps the GLB with its companion PNG and local browser modules. Use `--runtime /path/to/local_runtime` only for an OpenMVS binary that needs a separately unpacked glibc runtime; never replace the system glibc. A mesh cannot recover surfaces outside the registered camera coverage, and a high vertex count is not proof of geometric accuracy.

On memory-limited runs, OpenMVS may fail while fusing full-resolution depth maps.
Retry the same output with `--threads 2 --geometric-iters 0` to fuse cached maps;
this skips geometric consistency refinement and may reduce surface quality. Do not
change `--width` or `--resolution-level` in an existing output folder: use a new
folder for those settings. If texture atlas generation is too slow, create a
vertex-colored PLY from the completed cloud and mesh:

```bash
.venv/bin/python -m adaptive_uav.color_mesh dense_mesh/scene_dense.ply \
  dense_mesh/scene_mesh.ply dense_mesh/scene_mesh_colored.ply
.venv/bin/python -m adaptive_uav.mesh_viewer dense_mesh \
  --three-assets /path/to/local_three_modules
```

The fallback assigns each mesh vertex the RGB of its nearest dense point. It
does not reproduce the original-image texture atlas. The viewer uses this PLY
when no textured GLB exists.

## Reprojection and timing

`online_events.jsonl` records each processed frame's timestamp, reason for selection, IMU quality, earlier references, and processing duration. Map update records include timing, acceptance and reprojection error. `current_map.json` points to the latest accepted immutable snapshot. `online_timing.json`, `frame_timing.csv`, `map_update_timing.csv`, and `TIME_REPORT.md` summarize a completed run. `checkpoint.json` stores the source cursor and database/model progress for resume. A rejected candidate does not replace an accepted map.

Sensor alignment depends on phone exposure timestamps and actual raw IMU samples. The supplied 10 Hz telemetry is not equivalent to a 200 Hz IMU stream. Gyro rotation compensation requires a measured camera/IMU extrinsic and enough timed samples. GPS is logged for later alignment; there is no validated metric GPS pose constraint in the online solver. No trained neural selector, semantic model, online dense TSDF, VisFusion, or real-time performance guarantee is present. Sparse coordinates have arbitrary scale. Inspect a map's geometry and camera path before treating it as a reconstruction of the scene.
