# External student dataset downloader

Created 3 November 2026. The default is a bounded KITTI development pilot, chosen
for paired RGB, raw LiDAR, timestamps, calibration and GPS/IMU. Existing UAVScenes
and MARS assets are reused by their current workflows, not downloaded again here.
Other suggested datasets are not part of the default acquisition.

The script and manifest are deployed together in `/home/vishnu/datasets/` on
`10.23.20.60`. Python 3 and curl are required; it installs nothing and uses no GPU.
The default `plan` mode writes nothing and downloads nothing.

Default selection: three synced+rectified drives, plus calibration for both dates.

| Drive | Provisional role |
|---|---|
| `2011_09_26_drive_0001_sync` | External development training |
| `2011_09_26_drive_0005_sync` | External development training |
| `2011_09_30_drive_0016_sync` | External development validation |

This is **2,238,819,967 bytes (2.24 decimal GB) compressed**. Extraction needs
additional disk space; the script checks the ZIP member sizes before extraction.
These are provisional sequence/date roles, not proof of independent locations.
Do not use this pilot to claim an untouched final evaluation.

Official sources:

- https://www.cvlibs.net/datasets/kitti/raw_data.php
- https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
- Publisher downloader archive, used to check the URL convention:
  https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip

The public publisher S3 URLs were checked directly with HEAD. If a provider
changes access or returns an error, the script fails and retains partial data;
it does not bypass authentication or quotas.

## Run on the storage server

```bash
cd /home/vishnu/datasets

# Show exactly what would be downloaded; no transfer.
python3 download_student_datasets.py --mode plan

# Start only the default 2.24 GB pilot, keep running after disconnect.
tmux new-session -s geostreamfusion-student-data
nice -n 19 ionice -c 3 python3 -u download_student_datasets.py \
  --mode download --limit-mib 10 \
  2>&1 | tee -a student_datasets_download.log
```

Detach with Ctrl-B then D. Reconnect with:

```bash
tmux attach -t geostreamfusion-student-data
```

Monitor separately:

```bash
tail -f /home/vishnu/datasets/student_datasets_download.log
cat /home/vishnu/datasets/student_training_v1/_state/status.json
```

The script runs one download at a time, caps transfer bandwidth at 10 MiB/s by
default, uses low CPU priority and preserves archives. Idle I/O priority comes
from the `ionice` wrapper. A lock prevents duplicate instances writing the same
destination. Existing unrelated dataset and training jobs are never stopped.
Rerun the same command after an interruption: curl resumes supported `.part`
files; verified matching archives/extracted files are reused. A mismatching
existing archive or extracted file raises an error instead of being overwritten.

## Optional derived-depth archive

Raw LiDAR is included in the default drives, so the larger benchmark depth archive
is optional. To include it explicitly:

```bash
nice -n 19 ionice -c 3 python3 -u download_student_datasets.py \
  --mode download --with-depth --max-download-gb 20 --limit-mib 10 \
  2>&1 | tee -a student_datasets_download.log
```

Total selected compressed bytes become **16.48 GB**, including existing reusable
pilot archives. The optional source archive covers the wider depth benchmark;
only entries for selected drives are extracted. No guarantee is made that every
pilot drive belongs to the benchmark's annotated subset. Derived/accumulated depth
and raw LiDAR must retain distinct provenance in a future dataset adapter.

## Verify and inspect

```bash
nice -n 19 ionice -c 3 python3 download_student_datasets.py --mode verify
```

Include `--with-depth --max-download-gb 20` to verify the optional archive too.
Verification checks expected sizes, full ZIP CRC and extracted-file CRC, and records
a local SHA-256. The manifest has no publisher SHA-256, so local hashes are a
recorded identity, not independent publisher checksum confirmation. S3 multipart
ETags are not treated as MD5 or SHA hashes. None of these checks certify geometry.

Output:

```text
/home/vishnu/datasets/student_training_v1/
  archives/kitti/         original publisher ZIPs, retained
  extracted/kitti/        date/drive RGB, LiDAR, OXTS and calibration
  _state/status.json     current asset, completion/error and selected scope
  _state/*.source.json   source size, ETag and URL for safe resume
  _state/*.verified.json full ZIP CRC, extracted count and local hash
  _state/manifest_snapshot.json
  _state/source_register.json
```

Calibration-only smoke (8,141 bytes total), in an isolated directory:

```bash
nice -n 19 ionice -c 3 python3 download_student_datasets.py --mode download \
  --calibration-only --root /home/vishnu/datasets/student_download_smoke_20261103
```

Downloading does **not** add KITTI to GeoStreamFusion training. An adapter must
first validate pixel/depth conventions, camera-LiDAR time association, calibrated
projection, pose/scale compatibility, duplicates and physical-location splits.
`training_ready` remains false in acquisition status. Existing production
checkpoints, splits and queued training are unchanged.
