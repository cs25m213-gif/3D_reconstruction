#!/usr/bin/env bash
set -Eeuo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE"
PY="$BASE/.venv/bin/python"
RUN="${1:-$BASE/runs/neural_fusion_20261028}"
mkdir -p "$RUN"
LOG="$RUN/pipeline.log"
STATUS="$RUN/status.json"

stage() {
    local name="$1"
    local detail="${2:-}"
    "$PY" - "$STATUS" "$name" "$detail" <<'PY'
import json,sys,time
from pathlib import Path
path=Path(sys.argv[1]);path.write_text(json.dumps({'stage':sys.argv[2],
    'detail':sys.argv[3],'updated_unix':time.time()},indent=2)+'\n')
PY
    printf '[%s] %s %s\n' "$(date -Is)" "$name" "$detail" >> "$LOG"
}

failure() { stage failed "line $1; inspect $LOG"; }
trap 'failure "$LINENO"' ERR

stage preflight
test -x "$PY"
test -f "$BASE/runs/depth_student_20261028/best.pt"
test -f "$BASE/datasets/uavscenes/interval5_CAM_LIDAR.zip"
test -f "$BASE/runs/uavscenes_hkisland03_benchmark/session/video.mp4"
test -f "$BASE/runs/uavscenes_hkisland03_benchmark/reconstruction/sparse_run/pipeline_manifest.json"

stage overlap_training
if [[ ! -s "$BASE/runs/overlap_attention/report.json" ]]; then
    "$PY" -m uav_reconstruction.adaptive_uav.train_overlap_attention \
        --output "$RUN/overlap_attention" >> "$LOG" 2>&1
else
    stage overlap_training_reused "$BASE/runs/overlap_attention/report.json"
fi

stage depth_student_training
DEPTH_DIR="$RUN/depth_student"
DEPTH_ARGS=()
if [[ -f "$DEPTH_DIR/latest.pt" ]]; then DEPTH_ARGS+=(--resume); fi
if [[ ! -f "$DEPTH_DIR/status.json" ]] || ! "$PY" - "$DEPTH_DIR/status.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['status']=='complete'
PY
then
    if ! "$PY" -m uav_reconstruction.adaptive_uav.train_depth_student \
        --data-root "$BASE/runs/uavscenes_30_batch" --output "$DEPTH_DIR" \
        --epochs 12 --patience 3 "${DEPTH_ARGS[@]}" >> "$LOG" 2>&1; then
        stage depth_student_training_failed "original validated checkpoint retained"
    fi
fi

SELECTED="$BASE/runs/depth_student_20261028/best.pt"
if [[ -f "$DEPTH_DIR/best.pt" ]]; then
    stage depth_student_validation
    if "$PY" -m uav_reconstruction.adaptive_uav.select_depth_student \
        --data-root "$BASE/runs/uavscenes_30_batch" \
        --original "$BASE/runs/depth_student_20261028/best.pt" \
        --candidate "$DEPTH_DIR/best.pt" --output "$RUN" >> "$LOG" 2>&1; then
        SELECTED="$(cat "$RUN/selected_depth_checkpoint.txt")"
    else
        stage depth_student_validation_failed "original checkpoint retained"
    fi
fi
printf '%s\n' "$SELECTED" > "$RUN/selected_depth_checkpoint.txt"

stage recurrent_training
RECURRENT_DIR="$RUN/recurrent"
if [[ ! -f "$RECURRENT_DIR/report.json" ]]; then
    if ! "$PY" -m uav_reconstruction.adaptive_uav.train_recurrent_depth \
        --data-root "$BASE/runs/uavscenes_30_batch" \
        --student-checkpoint "$SELECTED" --output "$RECURRENT_DIR" \
        --epochs 8 >> "$LOG" 2>&1; then
        stage recurrent_training_failed "recurrent checkpoint disabled; inspect loss and data"
    fi
fi

RECURRENT_ARGS=()
if [[ -f "$RECURRENT_DIR/report.json" ]]; then
    if "$PY" - "$RECURRENT_DIR/report.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['passes_internal_baseline']
PY
    then
        RECURRENT_ARGS=(--recurrent-checkpoint "$RECURRENT_DIR/best.pt")
        stage recurrent_internal_gate_passed
    else
        stage recurrent_internal_gate_failed "updater kept offline; existing depth fusion retained"
    fi
fi

stage reconstruction
BENCH="$BASE/runs/uavscenes_hkisland03_benchmark"
OUTPUT="$RUN/hkisland03_neural"
if [[ ! -f "$OUTPUT/run_report.json" ]] || ! "$PY" - "$OUTPUT/run_report.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1])).get('result_ready')
PY
then
    if ! "$PY" -m uav_reconstruction.adaptive_uav.final_reconstruct \
        "$BENCH/session" --sparse-run "$BENCH/reconstruction/sparse_run" \
        --output "$OUTPUT" --neural-fusion --student-checkpoint "$SELECTED" \
        --reuse-workspace "$BENCH/reconstruction/dense_recovery" \
        --dense-recovery-views 36 --detail balanced --da3-model large \
        --semantic-filter off "${RECURRENT_ARGS[@]}" >> "$LOG" 2>&1; then
        stage cached_workspace_rejected "retrying with fresh stereo workspace"
        OUTPUT="$RUN/hkisland03_neural_fresh_stereo"
        "$PY" -m uav_reconstruction.adaptive_uav.final_reconstruct \
            "$BENCH/session" --sparse-run "$BENCH/reconstruction/sparse_run" \
            --output "$OUTPUT" --neural-fusion --student-checkpoint "$SELECTED" \
            --dense-recovery-views 36 --detail balanced --da3-model large \
            --semantic-filter off "${RECURRENT_ARGS[@]}" >> "$LOG" 2>&1
    fi
fi

stage lidar_evaluation
PYTHONPATH="$BASE/uav_reconstruction:${PYTHONPATH:-}" "$PY" \
    "$BENCH/evaluate_lidar.py" --reconstruction "$OUTPUT" \
    --mesh "$OUTPUT/multiview/scene_mesh.ply" \
    --output "$RUN/hkisland03_lidar_metrics.json" >> "$LOG" 2>&1

"$PY" - "$RUN" "$BENCH/lidar_metrics_stereo.json" "$OUTPUT" "$BENCH/reconstruction" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);baseline=json.loads(Path(sys.argv[2]).read_text())
current=json.loads((out/'hkisland03_lidar_metrics.json').read_text())
neural_manifest=json.loads((Path(sys.argv[3])/'multiview/manifest.json').read_text())
baseline_manifest=json.loads((Path(sys.argv[4])/'multiview/manifest.json').read_text())
new=current['summary']['map_calibration'];old=baseline['summary']['map_calibration']
result={'baseline_mesh':baseline['mesh'],'neural_mesh':current['mesh'],
    'baseline_coverage':old['coverage'],'neural_coverage':new['coverage'],
    'baseline_absrel':old['mean_view_abs_rel'],'neural_absrel':new['mean_view_abs_rel'],
    'baseline_mae_m':old['mean_view_mae_m'],'neural_mae_m':new['mean_view_mae_m'],
    'same_camera_poses':neural_manifest['workspace']==baseline_manifest['workspace'],
    'same_selected_views':neural_manifest['names']==baseline_manifest['names'],
    'same_dataset_scene':True,
    'matched_compute_budget':False,'scale_fit':'even camera centres; LiDAR on odd frames'}
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
stage complete "$OUTPUT/multiview/scene_colored.glb"
