#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 VIDEO OUTPUT_DIR [FPS] [WIDTH] [HEIGHT]"
    exit 1
fi

VIDEO="$1"
OUT="$2"
FPS="${3:-2}"
WIDTH="${4:-1600}"
HEIGHT="${5:-900}"

mkdir -p "$OUT"

echo "Video : $VIDEO"
echo "Output: $OUT"
echo "FPS   : $FPS"
echo "Size  : ${WIDTH}x${HEIGHT}"

ffmpeg -y \
    -i "$VIDEO" \
    -vf "fps=${FPS},scale=${WIDTH}:${HEIGHT}:flags=lanczos" \
    -q:v 2 \
    "$OUT/frame_%06d.jpg"

echo
echo "Frames created:"
find "$OUT" -maxdepth 1 -name '*.jpg' | wc -l
