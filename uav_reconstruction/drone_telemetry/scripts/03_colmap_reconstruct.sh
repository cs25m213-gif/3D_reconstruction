#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage:"
    echo "$0 IMAGES MASKS WORKDIR"
    exit 1
fi

IMAGES="$1"
MASKS="$2"
WORK="$3"

DB="$WORK/database.db"
SPARSE="$WORK/sparse"

mkdir -p "$WORK"

if [ -e "$DB" ]; then
    echo "ERROR: database already exists:"
    echo "$DB"
    echo
    echo "Remove it manually if you really want a fresh run."
    exit 1
fi

rm -rf "$SPARSE"
mkdir -p "$SPARSE"

echo "========================================"
echo "FEATURE EXTRACTION"
echo "========================================"

colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --ImageReader.mask_path "$MASKS" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --FeatureExtraction.use_gpu 0

echo
echo "========================================"
echo "EXHAUSTIVE CPU MATCHING"
echo "========================================"

colmap exhaustive_matcher \
    --database_path "$DB" \
    --FeatureMatching.use_gpu 0

echo
echo "========================================"
echo "MAPPING"
echo "========================================"

colmap mapper \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --output_path "$SPARSE"

echo
echo "========================================"
echo "MODELS"
echo "========================================"

for d in "$SPARSE"/*; do

    [ -d "$d" ] || continue

    echo
    echo "----------------------------------------"
    echo "$d"
    echo "----------------------------------------"

    colmap model_analyzer \
        --path "$d" 2>&1 | \
        grep -E \
        "Registered images|Points|Observations|Mean track length|Mean reprojection"

done
