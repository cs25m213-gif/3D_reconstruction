#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GEOSTREAM_PYTHON:-$ROOT/.venv/bin/python}"
export GEOSTREAM_WORKSPACE="$ROOT"
export GEOSTREAM_TOOLS="${GEOSTREAM_TOOLS:-$ROOT/tools}"
export PYTHONPATH="$ROOT:$ROOT/uav_reconstruction:$ROOT/.gpu_pycolmap:$ROOT/tools/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m "$@"
