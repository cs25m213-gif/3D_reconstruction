#!/usr/bin/env bash
# Reconstruct a session with what training proved, falling back to what already works.
#
#   ./reconstruct_best.sh SESSION OUTPUT            full-quality offline path
#   ./reconstruct_best.sh SESSION OUTPUT --stream   incremental path with the learned selector
#
# Lens distortion refinement is always on, because it fixes bent trajectories on nadir flights.
# The student replaces DA3 only if its training report says it beat DA3, and the learned selector
# only runs if evaluation promoted it. Otherwise this uses DA3 and the rule policy.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 2 ] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
SESSION=$1 OUTPUT=$2 MODE=${3:-offline}
PY=.venv/bin/python
STUDENT=runs/vnext_student_update_v1
SELECTOR=runs/vnext_selector_seq_v1/best.pt; [ -f runs/vnext_selector_seq_finetuned_v1/best.pt ] && SELECTOR=runs/vnext_selector_seq_finetuned_v1/best.pt
promoted() { "$PY" -m uav_reconstruction.adaptive_uav.promotion_gate --report "$1" --checkpoint "$2" --kind "$3"; }
export GEOSTREAM_STUDENT_PRECISION=float32
export GEOSTREAM_WORKSPACE="$PWD" GEOSTREAM_TOOLS="${GEOSTREAM_TOOLS:-$PWD/tools}"
export PYTHONPATH="$PWD/.gpu_pycolmap:$PWD/uav_reconstruction:$PWD/tools/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = --stream ]; then
  EXTRA=()
  if promoted runs/vnext_video_evaluation_seq_v1/promotion.json "$SELECTOR" selector; then EXTRA=(--selector-policy utility --utility-checkpoint "$SELECTOR"); echo 'Using the learned selector.'; else echo 'The learned selector has no matching full-video validation, so the rule policy runs.'; fi
  exec "$PY" -u -m adaptive_uav.runtime "$SESSION" --mode balanced --output "$OUTPUT" "${EXTRA[@]}"
fi
EXTRA=(--da3-model large)
if promoted "$STUDENT/report.json" "$STUDENT/best.pt" student; then EXTRA+=(--depth-backend student --student-checkpoint "$STUDENT/best.pt"); echo 'Using the updated student.'; else echo 'The student has not passed its matched gate, so DA3-LARGE runs.'; fi
exec ./reconstruct_final.sh "$SESSION" --mode legacy --detail high --refine-distortion --output "$OUTPUT" "${EXTRA[@]}"
