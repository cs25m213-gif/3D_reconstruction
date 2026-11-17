#!/usr/bin/env bash
set -euo pipefail

source /home/vishnu/eswar/drone_telemetry/scripts/env.sh

if [ $# -lt 3 ]; then
    echo "Usage:"
    echo "$0 IMAGES COLMAP_MODEL OUTPUT_DIR"
    exit 1
fi

IMAGES="$1"
MODEL="$2"
OUT="$3"

UNDIST="$OUT/undistorted"
MVS="$OUT/openmvs"

mkdir -p "$OUT"

echo "========================================"
echo "COLMAP UNDISTORT"
echo "========================================"

rm -rf "$UNDIST"

colmap image_undistorter \
    --image_path "$IMAGES" \
    --input_path "$MODEL" \
    --output_path "$UNDIST" \
    --output_type COLMAP

echo
echo "Registered undistorted images:"
find "$UNDIST/images" -maxdepth 1 -type f | wc -l

echo
echo "========================================"
echo "COLMAP -> OPENMVS"
echo "========================================"

rm -rf "$MVS"
mkdir -p "$MVS"

InterfaceCOLMAP \
    -i "$UNDIST" \
    -o "$MVS/scene.mvs"

cd "$MVS"

echo
echo "========================================"
echo "DENSE POINT CLOUD"
echo "========================================"

DensifyPointCloud \
    -i scene.mvs \
    -o scene_dense.mvs \
    --max-threads 16

echo
echo "========================================"
echo "MESH"
echo "========================================"

ReconstructMesh \
    -i scene_dense.mvs \
    -o scene_mesh.mvs \
    --max-threads 16

echo
echo "========================================"
echo "TEXTURE"
echo "========================================"

TextureMesh \
    -i scene_dense.mvs \
    -m scene_mesh.ply \
    -o scene_textured.obj \
    --export-type obj \
    --max-threads 16

echo
echo "========================================"
echo "FINISHED"
echo "========================================"

ls -lh \
    scene_dense* \
    scene_mesh* \
    scene_textured*
