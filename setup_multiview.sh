#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/tools/Depth-Anything-3"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$REPO"
  git -C "$REPO" checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
fi
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/uav_reconstruction/requirements_multiview.txt"
echo 'Multiview dependencies ready. Run ./reconstruct.sh /path/to/video.mp4 --multiview'
