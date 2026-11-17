#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GEOSTREAM_TOOLS="${GEOSTREAM_TOOLS:-$ROOT/tools}"
export PYTHONPATH="$ROOT/.gpu_pycolmap:$ROOT/uav_reconstruction:$ROOT/tools/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
exec "$ROOT/.venv/bin/python" -u -m adaptive_uav.final_reconstruct "$@"
