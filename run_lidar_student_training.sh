#!/usr/bin/env bash
set -euo pipefail
cd /home/iit-t/GeoStreamFusion_GPU_20261013/GeoStreamFusion_GPU
mkdir -p runs/depth_student_lidar
exec .venv/bin/python -u -m uav_reconstruction.adaptive_uav.train_depth_student_lidar \
  --cache-root runs/uavscenes_30_batch \
  --sidecars runs/uavscenes_cached_lidar_targets \
  --original runs/depth_student_20261028/best.pt \
  --output runs/depth_student_lidar \
  --epochs 12 --patience 3 --batch-size 2 --workers 4 "$@"
