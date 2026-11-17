#!/bin/bash
# Run one reconstruction-driven selector pipeline end to end.
# Usage: ./run_stage.sh <module: reconstruction_selector|bin_memory_selector> <work_dir> <workers> [extra args]
# Paths come from the environment (defaults are the 10.23.20.60 layout):
#   ROOT     repo root (contains uav_reconstruction/)
#   PY       python with torch, open3d, numpy, scipy
#   CONFIG   utility contract json
#   STUDENT  depth student checkpoint
#   LABELS   contribution_labels output     GREEDY  greedy oracle output
#   LOGTAG   optional suffix for the log name (lets two runs share one work dir)
# Optional: SELECTOR_CONF=1 adds student-confidence features (off by default; see report 10.11).
ROOT=${ROOT:-/home/vishnu/geostreamfusion_attribution_20261108}
PY=${PY:-/home/vishnu/miniconda3/envs/drone3d/bin/python}
CONFIG=${CONFIG:-runs/contract_vishnu.json}
STUDENT=${STUDENT:-/home/vishnu/geostreamfusion_more_data_20261104/data/student.pt}
LABELS=${LABELS:-runs/contribution_20261109/labels}
GREEDY=${GREEDY:-runs/contribution_20261109/greedy}
cd "$ROOT" || exit 1
MOD=$1; WORK=$2; W=$3; shift 3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$WORK"
LOG=$WORK/$MOD${LOGTAG:+_$LOGTAG}.log
"$PY" -m uav_reconstruction.adaptive_uav.$MOD all --work "$WORK" --workers "$W" \
  --labels "$LABELS" --greedy "$GREEDY" --config "$CONFIG" --student "$STUDENT" "$@" \
  > "$LOG" 2>&1 || echo "$(date +%T) $MOD $LOGTAG FAILED" >> "$WORK/FAILED.txt"
