#!/bin/bash
# Setup (first run) and launch DeepFaceLive on macOS.
# Usage: build/macos/run.sh [userdata_dir]
set -e
cd "$(dirname "$0")/../.."
USERDATA_DIR="${1:-$HOME/DeepFaceLive_userdata}"

if [ ! -x .venv/bin/python ]; then
    echo "Creating Python 3.13 virtualenv..."
    if command -v uv >/dev/null; then
        uv venv --python 3.13 .venv
        uv pip install --python .venv/bin/python -r requirements-macos.txt
    else
        python3.13 -m venv .venv
        .venv/bin/python -m pip install -r requirements-macos.txt
    fi
fi

if [ ! -f modelhub/onnx/S3FD/S3FD.onnx ]; then
    echo "Merging split model files..."
    .venv/bin/python main.py dev merge_large_files
fi

mkdir -p "$USERDATA_DIR"
exec .venv/bin/python main.py run DeepFaceLive --userdata-dir "$USERDATA_DIR"
