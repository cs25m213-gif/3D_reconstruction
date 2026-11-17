# Required dataset acquisition

## Active scope: required only (user correction)

The user selected **required only: 27 files, 307.009 GB before reuse**.
The expanded `--tier all` process was stopped; existing optional files/partials
are preserved and are not resumed. Current tmux session: `geostreamfusion-required`.
Run arguments match v3 below except `--tier required`. Current log:

```bash
tail -f /home/vishnu/datasets/vnext_downloads/_state/setup_required.log
```

The pinned label mirrors remain enabled. Google Drive-only files can still be
quota-blocked. This scope change does not imply completion or lifted quotas.

## Current execution: mirror-aware v3

The sleeping v2 job was stopped on 31 October 2026. Its files/logs remain.
Current detached session: `geostreamfusion-download-v3` on `10.23.20.60`.

```bash
tail -f /home/vishnu/datasets/vnext_downloads/_state/setup_v3.log
```

Executed on that host:

```bash
python3 -u geostreamfusion_setup_datasets_v3.py \
  --root /home/vishnu/datasets \
  --manifest /home/vishnu/datasets/geostreamfusion_download_datasets_v3.sh \
  --mirrors /home/vishnu/datasets/download_mirrors_v3.json \
  --tier all --jobs 2 --passes 8 --retry-hours 24
```

Selected manifest: 199 files, 979.599 decimal GB before existing-file reuse.
Available HF files run first: MatrixCity block 3, NTU VIRAL, UAVScenes labels.
Other listed UAVScenes/MARS/MUN-FRL files retain their original Drive sources.
Quota restrictions remain real blockers, not successful transfers. No guarantee
that the entire manifest will complete while a provider denies access.

Two interval-1 label files use the community `LevelsQ/UAVscenes` mirror, with
pinned revision, matching manifest sizes and its LFS SHA-256 hashes. Original
publisher hashes were unavailable, so mirror integrity is not independent proof
of publisher equivalence. The source sidecars preserve that qualification.
Official interval-5 label archives were also added; they do not replace full-rate
RGB/LiDAR streams. Downloading test assets does not authorize using them to tune.

Progress updates contain received bytes and transfer rate. `summary.json` now
refreshes during work, approximately every 30 seconds, and distinguishes running,
waiting, incomplete and complete. ZIPs are extracted; TAR archives and ROS bags
remain files for later dataset preparation. Six offline response/source tests
passed; remote dry-run parsed the full selection; real MatrixCity JSON downloads
completed and archive bytes were observed growing after launch. No GPU work.

## Historical v2 execution

31 October 2026: the remote Google Drive responses were inspected and explicitly
reported `Quota exceeded`. A successful tiny range probe did not imply that an
archive-sized request was available. Do not bypass provider quotas.

The copied manifest selects 27 required UAVScenes/full-rate and MARS-LVIG files,
307.009 GB before reuse. Extracted content requires extra space. Optional datasets
are not started. These overlapping flights are not 27 independent scenes.

Remote root: `vishnu@10.23.20.60:/home/vishnu/datasets`.
Deployed script: `geostreamfusion_setup_datasets_v2.py`.
Detached tmux session: `geostreamfusion-download-v2`.

Executed remotely:

```bash
python3 -u geostreamfusion_setup_datasets_v2.py \
  --root /home/vishnu/datasets --jobs 2 --passes 8 --retry-hours 24
```

Watch on the remote host:

```bash
tail -f /home/vishnu/datasets/vnext_downloads/_state/setup_v2.log
```

Per-file states: `vnext_downloads/_state/setup_*.json`.
Pass summary: `vnext_downloads/_state/summary.json` (written after each full pass).
Archives: `vnext_downloads/{uavscenes,mars_lvig}/`.
Extracted ZIPs: `vnext_downloads/extracted/uavscenes/`.
ROS bags remain files; extraction does not convert them into training samples.

The previous downloader process was stopped and its original source/log preserved.
Existing completed archives and partial files remain reusable. The new script has
a common filesystem lock preventing simultaneous setup/download writers.
Quota pages are classified separately and are never appended to an archive.
Ordinary transient failures retain bounded retries; blocked files are retried on
the next daily pass for at most eight passes. Completion is not guaranteed while
the provider denies access. Inspect per-file states after any exhausted run.

Verification is expected size plus ZIP CRC, supplied SHA-256 where present, or
ROS bag header. A bag header/size check is not full semantic bag validation.
Extracted markers certify extraction completion, not camera/LiDAR alignment.
No training, reconstruction, dependency installation or GPU workload was started.

Offline response tests:

```bash
python3 scripts/datasets/test_dataset_download_responses.py
```

Four tests passed: exact range, wrong range rejection, quota detection and HTML
confirmation rejection. Local and remote dry runs parsed the 27-file selection.

The official UAVScenes repository also lists OneDrive as a full-rate mirror;
the inspected link redirected to Microsoft sign-in, so no unattended mirror
fallback was asserted. The official Hugging Face mirror supplies interval-5,
not a drop-in replacement for these interval-1 files.
