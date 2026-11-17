# Live phone API and online COLMAP

This entry point consumes the phone's `/snapshot.jpg`, `/stream` (with `/telemetry.json` polling fallback), and optional `/camera_intrinsics.json`. It does not need the MJPEG feed. Snapshot acquisition and telemetry run in background threads while CPU PyCOLMAP computes sparse maps. Each JPEG is written to disk before the mapper sees it. A holdback pairs delayed telemetry with captured images; the consumer drains every saved frame after capture stops. Mapping may lag behind capture.

The updated phone API adds PIN authentication, camera details in telemetry, `/camera/switch`, and `/camera/macro`. The adapter applies lens, macro and `/stream_config` settings **before** recording the first frame. It displays the current and available lenses and stops a capture if telemetry reports a lens or macro change later. Different lenses and focus modes need separate sessions and camera calibration.

## Run here

```bash
cd /home/mtp26/Documents/3d_reconstruction/uav_reconstruction
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-live.txt
.venv/bin/python -m adaptive_uav.phone_live \
  --phone-url http://PHONE_IP:8080 \
  --session runs/phone_capture_1 \
  --capture-fps 3 \
  --buffer-seconds 1
```

Open `http://127.0.0.1:8090` while the command runs. The phone API's 8081 port is tried if 8080 fails. `--max-seconds 120` or `--max-frames 360` creates a finite run. Ctrl+C or the dashboard Stop button stops acquisition, drains the saved frame queue, exports the MP4, and publishes a final sparse update. Wait for the process to exit. To continue after an interruption, use the **same** `--session` and `--output` with `--resume`; source URL, calibration, and config must match. Use a new session directory for different camera settings.

For a PIN-protected server and highest configured stream quality:

```bash
export GEOSTREAM_PHONE_PIN='your-phone-pin'
.venv/bin/python -m adaptive_uav.phone_live \
  --phone-url http://PHONE_IP:8080 --session runs/wide_1080p \
  --stream-resolution 1080p --jpeg-quality 100 \
  --camera-id 0 --macro-mode off --capture-fps 3
```

`--camera-id` uses an ID from `camera.available` in `/telemetry.json`. Choose a new session folder for a different lens. The PIN is sent as `X-Auth-Token` on every API request, including SSE and downloads; it is not written to capture metadata. The phone URL may also include `?key=...` or a single PIN path segment such as `http://PHONE_IP:8080/PIN`; the adapter sends that PIN as a header and strips it from saved URLs. The environment variable avoids putting the PIN in a process command line.

To retain the phone's native video and sensor files as well as the local snapshot recording, add `--record-phone --download-phone-session`. The adapter calls `/record/start` before capture, `/record/stop` when acquisition ends, and downloads the new `/download/{sessionId}` ZIP when it appears in telemetry. A failed phone archive download is reported in `recording.json`; locally saved frames and reconstruction remain available. The downloaded ZIP is kept intact.

Run `.venv/bin/python -m unittest discover -s tests -v` to check the installation. PyCOLMAP 4.2 runs on CPU. The current machine has no installed COLMAP CLI, OpenMVS executables or Open3D runtime, so the standard command produces validated **sparse point clouds and camera poses**. Final dense mesh and texture require those tools and `--finalize-dense --colmap /path/to/colmap --openmvs /path/to/bin`. This dense stage runs after capture.

## Calibration and timing

For best geometry, supply exact pixel intrinsics for the snapshot resolution:

```json
{"width":1920,"height":1080,"fx":1450.0,"fy":1450.0,"cx":960.0,"cy":540.0,"distortion":[0,0,0,0,0]}
```

These numbers are an **example**; measure your phone's actual values for the selected lens and macro mode. Use `--calibration /path/to/calibration.json --strict-intrinsics`. The updated API catalog does not list `/camera_intrinsics.json`, although the adapter accepts it if available, or valid intrinsics nested in telemetry. If none are available, it uses an approximate focal ratio and reports `calibration_exact: false`; this limits reconstruction quality. Existing Android `lens_intrinsic_calibration` data with zero principal point is not silently treated as an exact pixel calibration.

The current API's 10 Hz telemetry contains the *latest* IMU vector, not the raw 100–200 Hz samples that `rates_hz` advertises. IMU features are computed from the samples actually received. Gyro rotation compensation only activates for a sufficiently sampled window with an explicit calibrated camera-to-IMU rotation:

```json
{"camera_from_imu":[[1,0,0],[0,1,0],[0,0,1]]}
```

That identity matrix is a format example, **not** a calibration. Supply a measured transform via `--imu-calibration /path/to/imu_extrinsics.json`; `--save-imu-windows` saves selected 64×6 accelerometer/gyro windows for later model training. There is no trained neural selector in this package.

For synchronized capture, expose an `X-Capture-Wall-Time-Ms` header on `/snapshot.jpg`, representing the actual exposure time in the same clock as telemetry `wall_time_ms`. `X-Frame-Sequence` is useful for gap detection. With no exposure header, the adapter estimates phone time from telemetry arrival and labels it as estimated; with no clock offset it records frames as unsynchronized and does not invent IMU alignment. Ideally `/stream` should include **all raw** IMU samples since the prior packet as `imu.samples` or `imu_samples`, each with `wall_time_ms` or `elapsed_ns`, `accel:[ax,ay,az]` in m/s², and `gyro:[gx,gy,gz]` in rad/s. The relative `elapsed_ns` is anchored to the packet's own `elapsed_ns` and `wall_time_ms`.

## Saved outputs

| Path | Meaning |
|---|---|
| `<session>/capture/frames/*.jpg` | Every original saved snapshot; independent of keyframe selection |
| `<session>/capture/acquisition.jsonl` | Durable acquisition timestamps and image paths |
| `<session>/capture/frames.jsonl` | Images paired with nearest telemetry and received IMU window |
| `<session>/telemetry.jsonl` | Raw received API telemetry and receiver arrival times |
| `<session>/video.mp4`, `video_frames.csv` | Constant-rate review video plus original per-frame times |
| `<session>/phone_<sessionId>.zip` | Optional original phone recording, GPS, IMU and metadata archive |
| `<session>/reconstruction_online_cpu/current_map.json` | Latest accepted sparse map pointer |
| `<session>/reconstruction_online_cpu/blocks/*/sparse_map.ply` | Accepted map snapshots |
| `<session>/reconstruction_online_cpu/camera_poses.json` | COLMAP camera poses at arbitrary scale |
| `<session>/reconstruction_online_cpu/viewer.html` | Standalone saved sparse-map viewer with embedded last photo |
| `<session>/reconstruction_online_cpu/online_events.jsonl` | Per-frame selection, timing, IMU window and mapping events |
| `<session>/reconstruction_online_cpu/online_timing.json` | Final counters and timing |

The dashboard reports captured versus processed frames, buffer depth, selected keyframes, registered cameras, sparse points and reprojection error. A selected keyframe is processed by COLMAP; other snapshots remain saved for review. The map has arbitrary scale until a reliable metric alignment is implemented. This code does not perform online dense depth fusion, semantic segmentation, VisFusion, ORB-SLAM3, MASt3R-SLAM or TSDF meshing.
