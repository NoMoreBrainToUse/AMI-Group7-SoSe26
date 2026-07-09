#!/usr/bin/env bash
# One-command setup: virtualenv + dependencies (+ optional sample sequence).
#
#   ./setup.sh             install everything
#   ./setup.sh --sample    also download FRED seq 40 (~2.3 GB) for a test run
#
# Picks CUDA or CPU PyTorch wheels automatically (nvidia-smi presence).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "ERROR: python3 not found"; exit 1; }

if [ ! -d .venv ]; then
  echo "==> Creating virtualenv (.venv)"
  "$PY" -m venv .venv
fi
PIP=".venv/bin/pip"
"$PIP" install --quiet --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected - installing CUDA 12.4 PyTorch"
  "$PIP" install --quiet torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> No NVIDIA GPU - installing CPU PyTorch"
  "$PIP" install --quiet torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Installing remaining dependencies"
"$PIP" install --quiet -r requirements.txt

if [ "${1:-}" = "--sample" ]; then
  if [ -d dataset/40/PADDED_RGB ]; then
    echo "==> Sample sequence dataset/40 already present"
  else
    echo "==> Downloading FRED seq 40 (~2.3 GB)"
    mkdir -p dataset
    .venv/bin/gdown "1W4OmPq8UQB-HVDBwRcRXNN5CLh4cE4AJ" -O dataset/40.zip
    echo "==> Extracting"
    .venv/bin/python -m zipfile -e dataset/40.zip dataset/40/
    rm dataset/40.zip
  fi
fi

echo
echo "Setup complete. Next steps:"
echo "  .venv/bin/python run_gui.py            # web interface -> http://localhost:8501"
echo "  .venv/bin/python run_pipeline.py dataset/40   # or run the pipeline headless"
