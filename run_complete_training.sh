#!/usr/bin/env bash
# GeoStreamFusion complete training pipeline, from calibration to an evaluated selector.
#
#   ./run_complete_training.sh --list               show the steps
#   ./run_complete_training.sh --to targets         CPU calibration and corrected data only
#   ./run_complete_training.sh                      everything
#   ./run_complete_training.sh --from student       resume at a step
#
# Options:
#   --threads N          CPU threads for each step (default 8)
#   --reconstruction-jobs N  concurrent new-clip reconstructions (default 1; bounded 1-4)
#   --own-sessions "A B" depth_cache folders of your drone sessions, for self-supervised training
#   --embeddings-cpu     compute DINO on CPU when the GPU is busy
#   --temporal           also train the temporal branch (slower, and optional)
#   --diagnostic         continue on an unresolved reference; nothing it produces can be promoted
#   --allow-busy-gpu     run GPU steps even if other jobs are using the GPU
#   --wait-for-gpu       wait and recheck instead of exiting while other users use the GPU
#
# Each finished step leaves a signed runs/vnext_logs/<step>.done artifact; changed inputs
# invalidate it. Logs append under runs/vnext_logs/. A finished step can still
# fail its quality gate: the pipeline stops there, and status.json says why.
set -euo pipefail
cd "$(dirname "$0")"
CONFIG=uav_reconstruction/configs/vnext_contract_v3.json
PY=.venv/bin/python; M=uav_reconstruction.adaptive_uav
THREADS=8 RECONSTRUCTION_JOBS=1 FROM='' TO='' OWN='' TEMPORAL=0 DIAGNOSTIC=0 BUSY=0 WAIT=0 LIST=0 EMBED_DEVICE=cuda
while [ $# -gt 0 ]; do
  case "$1" in
    --threads) THREADS=$2; shift 2 ;;
    --reconstruction-jobs) RECONSTRUCTION_JOBS=$2; shift 2 ;;
    --from) FROM=$2; shift 2 ;;
    --to) TO=$2; shift 2 ;;
    --own-sessions) OWN=$2; shift 2 ;;
    --embeddings-cpu) EMBED_DEVICE=cpu; shift ;;
    --temporal) TEMPORAL=1; shift ;;
    --diagnostic) DIAGNOSTIC=1; shift ;;
    --allow-busy-gpu) BUSY=1; shift ;;
    --wait-for-gpu) WAIT=1; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
done
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS VNEXT_THREADS=$THREADS
LOGS=runs/vnext_logs; mkdir -p "$LOGS"
case "$THREADS" in ''|*[!0-9]*|0) echo '--threads must be a positive integer'; exit 2 ;; esac
case "$RECONSTRUCTION_JOBS" in 1|2|3|4) ;; *) echo '--reconstruction-jobs must be between 1 and 4'; exit 2 ;; esac
# Isolate new sequential labels/checkpoints from historical independent-decision experiments.
DATA=runs/vnext_dataset_v2 LABELS=runs/vnext_utility_seq_v1 EMBED=runs/vnext_memory_v1/embeddings
STUDENT=runs/vnext_student_update_v1 SELECTOR=runs/vnext_selector_seq_v1 TUNED=runs/vnext_selector_seq_finetuned_v1
DIAG=(); ALLOW=(); ALLOW_S=()
[ "$DIAGNOSTIC" = 1 ] && { DIAG=(--diagnostic); ALLOW=(--allow-diagnostic); ALLOW_S=(--allow-unresolved); }

py() { "$PY" -u -m "$M.$1" "${@:2}"; }

gpu_free() {
  [ "$BUSY" = 1 ] && return 0
  local apps
  while true; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || { echo 'Cannot verify GPU availability; refusing GPU work.'; exit 3; }
    [ -z "$apps" ] && return 0
    if [ "$WAIT" != 1 ]; then
      echo 'GPU is occupied. This step will not start; use --wait-for-gpu to wait without disturbing other users.'
      exit 3
    fi
    echo "$(date '+%F %T') GPU occupied; preserving all jobs and rechecking in 60 seconds."
    sleep 60
  done
}

finished() {  # Same test as the batch that built the first 30 clips.
  "$PY" - "$1/reconstruction/run_report.json" <<'P'
import json, sys
from pathlib import Path
try:
    report = json.loads(Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if report.get('result_ready') and Path(report.get('mesh', '')).is_file() else 1)
P
}

gate() {  # gate <python expression over g> <message>
  "$PY" - "$1" "$2" <<'P'
import json, sys
g = json.load(open('runs/vnext_dataset_v2/geometry_gate.json'))
if not eval(sys.argv[1]):
    sys.exit(sys.argv[2])
P
}

s_freeze()    { py freeze_reconstruction_baseline --config uav_reconstruction/configs/vnext_contract_v2.json --output runs/vnext_baseline_v2 --reuse-existing; }
s_clips()     { py prepare_uavscenes_clips link --config "$CONFIG"; }
s_newclips()  { local extra=()
                [ "$WAIT" = 1 ] && extra+=(--wait-for-gpu)
                [ "$BUSY" = 1 ] && extra+=(--allow-busy-gpu)
                py reconstruct_training_clips --jobs "$RECONSTRUCTION_JOBS" --threads "$THREADS" "${extra[@]}"; }
s_scales()    { mkdir -p "$DATA"; py calibrate_cache_scales --cache-root runs/vnext_clips_v1 --output "$DATA/provisional_scale_calibration.json"; }
s_geometry()  { [ -f runs/vnext_dataset_v1/geometry_gate.json ] || py validate_reference_geometry --config uav_reconstruction/configs/vnext_contract_v1.json --output runs/vnext_dataset_v1 --workers 4; }
s_calibrate() { py register_lidar_to_map --config "$CONFIG" --output "$DATA"; }
s_dedome()    { py sfm_dedome --config "$CONFIG" --output "$DATA/poses" --threads "$THREADS"; }
s_targets()   { py build_calibrated_targets --config "$CONFIG" --output "$DATA/targets"
                [ "$DIAGNOSTIC" = 1 ] || gate "g['status_counts']['QUALIFIED'] + g['status_counts']['VERIFIED'] > 0" 'No flight passed the geometry gate. Read runs/vnext_dataset_v2/calibration.json, or use --diagnostic.'; }
s_records()   { py trusted_sequences build --config "$CONFIG" --output "$DATA"; }
s_controls()  { gpu_free; py reproduce_controls --config uav_reconstruction/configs/vnext_contract_v1.json --output runs/vnext_baseline_v2 --device cuda; }
s_student()   { gpu_free
                local extra=(); [ -n "$OWN" ] && extra=(--unlabelled $OWN)
                py train_student_update --config "$CONFIG" --output "$STUDENT" --workers 4 --resume "${ALLOW_S[@]}" "${extra[@]}"; }
s_temporal()  { [ "$TEMPORAL" = 1 ] || { echo 'skipped: pass --temporal to train it'; return 0; }
                gpu_free
                py train_student_update --config "$CONFIG" --output runs/vnext_temporal_student_v1 --init-checkpoint "$STUDENT/best.pt" --temporal --sequence 3 --epochs 8 --workers 4 --resume "${ALLOW_S[@]}"; }
s_embed()     { [ "$EMBED_DEVICE" = cpu ] || gpu_free; py embedding_cache --config "$CONFIG" --output "$EMBED" --device "$EMBED_DEVICE"; }
s_memory()    { py memory_graph replay --config "$CONFIG" --output runs/vnext_memory_v1; }
s_labels()    { py counterfactual_replay --config "$CONFIG" --output "$LABELS" --student-checkpoint "$STUDENT/best.pt" --embeddings "$EMBED" --replay-mode sequential --device cpu "${DIAG[@]}"; }
s_sources()   { py source_view_labels --config "$CONFIG" --output "$LABELS" --student-checkpoint "$STUDENT/best.pt" --embeddings "$EMBED" "${DIAG[@]}"; }
s_dataset()   { py build_utility_dataset --config "$CONFIG" --output "$LABELS"; }
s_selector()  { gpu_free; py train_utility_selector --config "$CONFIG" --output "$SELECTOR" --labels "$LABELS" --resume "${ALLOW[@]}"; }
s_finetune()  { gpu_free; py finetune_selector --config "$CONFIG" --checkpoint "$SELECTOR/best.pt" --output "$TUNED" --student-checkpoint "$STUDENT/best.pt" --embeddings "$EMBED" "${ALLOW[@]}"; }
s_evaluate()  { local best="$SELECTOR/best.pt"; [ -f "$TUNED/best.pt" ] && best="$TUNED/best.pt"
                py evaluate_vnext --config "$CONFIG" --output runs/vnext_video_evaluation_seq_v1 --utility-checkpoint "$best" --student-checkpoint "$STUDENT/best.pt" --embeddings "$EMBED"; }

STEPS=(freeze clips newclips scales geometry calibrate dedome targets records controls student temporal embed memory labels sources dataset selector finetune evaluate)
if [ "$LIST" = 1 ]; then printf '%s\n' "${STEPS[@]}"; exit 0; fi
for bound in "$FROM" "$TO"; do
  if [ -n "$bound" ] && ! printf '%s\n' "${STEPS[@]}" | grep -qx "$bound"; then echo "unknown step: $bound"; exit 2; fi
done
exec 9>"$LOGS/pipeline.lock"
flock -n 9 || { echo 'This pipeline already has an active launcher; preserving its process.'; exit 3; }
OPTIONS=$("$PY" - "$TEMPORAL" "$DIAGNOSTIC" "$EMBED_DEVICE" "$OWN" "$RECONSTRUCTION_JOBS" "$THREADS" "${STEPS[@]}" <<'P'
import json,sys
print(json.dumps(dict(temporal=sys.argv[1],diagnostic=sys.argv[2],embedding_device=sys.argv[3],own_sessions=sys.argv[4],reconstruction_jobs=sys.argv[5],threads=sys.argv[6],steps=sys.argv[7:])))
P
)
stamp() { py stage_cache "$1" --stage "$name" --config "$CONFIG" --logs "$LOGS" --options "$OPTIONS"; }
started=$([ -z "$FROM" ] && echo 1 || echo 0)
for name in "${STEPS[@]}"; do
  [ "$name" = "$FROM" ] && started=1
  if [ "$started" = 1 ]; then
    if [ "$name" = temporal ] && [ "$TEMPORAL" != 1 ]; then
      echo '== temporal: disabled; no completion marker written'
    elif stamp check; then
      echo "== $name: already done"
    else
      echo "== $name: $(date '+%F %T')"
      "s_$name" 2>&1 | tee -a "$LOGS/$name.log"
      stamp write
    fi
  fi
  [ "$name" = "$TO" ] && break
done
echo "Finished. Development evaluation: runs/vnext_video_evaluation_seq_v1/promotion.json; online deployment requires a separate full-video gate."
