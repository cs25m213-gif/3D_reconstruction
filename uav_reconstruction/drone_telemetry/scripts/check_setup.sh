#!/usr/bin/env bash

source /home/vishnu/eswar/drone_telemetry/scripts/env.sh

echo
echo "========================================"
echo "DRONE 3D SYSTEM CHECK"
echo "========================================"

echo
echo "----- Executables -----"

for cmd in \
    python \
    ffmpeg \
    colmap \
    InterfaceCOLMAP \
    DensifyPointCloud \
    ReconstructMesh \
    TextureMesh \
    blender
do
    printf "%-20s : " "$cmd"

    if command -v "$cmd" >/dev/null 2>&1; then
        command -v "$cmd"
    else
        echo "MISSING"
    fi
done

echo
echo "----- Python packages -----"

python - <<'PY'
packages = [
    "numpy",
    "scipy",
    "open3d",
    "trimesh",
    "torch",
    "transformers",
    "PIL",
    "cv2",
]

for pkg in packages:
    try:
        m = __import__(pkg)
        version = getattr(m, "__version__", "installed")
        print(f"{pkg:15} OK   {version}")
    except Exception as e:
        print(f"{pkg:15} MISSING   {e}")

try:
    import torch

    print()
    print("CUDA available :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU            :", torch.cuda.get_device_name(0))
        print("CUDA version   :", torch.version.cuda)

except Exception as e:
    print("Torch CUDA check failed:", e)
PY

echo
echo "----- Project directories -----"

for d in \
    reconstruction_1fps \
    reconstruction_2fps_nomask \
    reconstruction_2fps_human_animal_mask
do

    printf "%-45s : " "$d"

    if [ -d "$PROJECT/$d" ]; then
        echo "FOUND"
    else
        echo "MISSING"
    fi
done

echo
echo "========================================"
echo "CHECK COMPLETE"
echo "========================================"
