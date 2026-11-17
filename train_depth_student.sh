#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/uav_reconstruction:$ROOT/tools/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
DATA_ROOT="${GEOSTREAM_STUDENT_DATA:-$ROOT/runs/uavscenes_30_batch}"
OUTPUT="${GEOSTREAM_STUDENT_OUTPUT:-$ROOT/runs/depth_student_20261028}"
exec "$ROOT/.venv/bin/python" -u -m adaptive_uav.train_depth_student \
  --data-root "$DATA_ROOT" --output "$OUTPUT" "$@"
