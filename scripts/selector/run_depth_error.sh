#!/bin/bash
cd /home/vishnu/geostreamfusion_attribution_20261108
PY=/home/vishnu/miniconda3/envs/drone3d/bin/python
OUT=runs/contribution_20261109/depth_error
W=${1:-40}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p $OUT/locks $OUT/logs
echo "$(date +%T) tables start" >> $OUT/progress.txt
worker() {
  for c in $(cat runs/contribution_20261109/clips.txt); do
    mkdir $OUT/locks/$c 2>/dev/null || continue
    $PY depth_error_selector.py table --clip $c > $OUT/logs/$c.log 2>&1 || echo "$c FAILED" >> $OUT/failures.txt
    echo "$(date +%T) w$1 $c" >> $OUT/progress.txt
  done
}
for w in $(seq 1 $W); do worker $w & done
wait
echo "$(date +%T) tables done $(ls $OUT/AM*.npz | wc -l)" >> $OUT/progress.txt
$PY depth_error_selector.py train > $OUT/train.log 2>&1 || echo "train FAILED" >> $OUT/failures.txt
echo DONE >> $OUT/progress.txt
