#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$ROOT/.venv/bin/python"
ENV="$ROOT/.venv_visfusion"
"$BASE" -m pip install -r "$ROOT/uav_reconstruction/requirements_multiview.txt" 'open3d==0.19.0'
if [[ ! -x "$ENV/bin/python" ]]; then
  "$BASE" -m venv "$ENV"
fi
GEOSTREAM_SETUP_ROOT="$ROOT" "$BASE" - <<'PY'
import os
from pathlib import Path
import sysconfig
root=Path(os.environ['GEOSTREAM_SETUP_ROOT'])
base=Path(sysconfig.get_path('purelib'))
for directory in (root/'.venv_visfusion/lib').glob('python*/site-packages'):
    (directory/'geostream_base.pth').write_text(str(base)+'\n')
PY
"$ENV/bin/python" -m pip install -r "$ROOT/uav_reconstruction/requirements_visfusion.txt" setuptools wheel ninja
if [[ ! -f "$ROOT/tools/sparsehash/include/google/dense_hash_map" ]]; then
  cd "$ROOT/tools/sparsehash-src"
  ./configure --prefix="$ROOT/tools/sparsehash"
  make install
fi
cd "$ROOT"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CPLUS_INCLUDE_PATH="$ROOT/tools/sparsehash/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
"$ENV/bin/python" -m pip install --no-build-isolation --no-deps "$ROOT/tools/torchsparse-src"

GPU_WHEEL="$ROOT/tools/pycolmap-4.2.1+cuda-cp312-cp312-manylinux_2_35_x86_64.whl"
if [[ -f "$GPU_WHEEL" ]]; then
  "$BASE" -m pip install --no-deps --upgrade --target "$ROOT/.gpu_pycolmap" "$GPU_WHEEL"
  PYTHONPATH="$ROOT/.gpu_pycolmap${PYTHONPATH:+:$PYTHONPATH}" "$BASE" "$ROOT/tools/verify_gpu_ba.py"     --output "$ROOT/.gpu_pycolmap/gpu_ba_verified.json"
fi
