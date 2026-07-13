# syntax=docker/dockerfile:1
#
# Hybrid Vision drone detection — CPU image.
# Mirrors setup.sh, but installs into the system Python (no venv needed:
# the container *is* the isolated environment) and skips the sample-data
# download (mount dataset/ as a volume or upload zips through the GUI).
#
# Build:  docker build -t hybrid-vision .
# Run:    docker run --rm -p 8501:8501 -v "$PWD/dataset:/app/dataset" \
#             -v "$PWD/outputs:/app/outputs" -v "$PWD/processed:/app/processed" \
#             hybrid-vision
# Then open http://localhost:8501

# Pin to the interpreter the project was verified with (venv was 3.12.3).
FROM python:3.12-slim

# --- OS-level dependencies --------------------------------------------------
# opencv-python needs libGL (libgl1) and libglib2.0-0 at runtime even for
# headless use; without them "import cv2" fails with a missing-.so error.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ----------------------------------------------------
# Copy only requirements first so this layer is cached and not rebuilt every
# time the source code changes (Docker caches layers top-to-bottom and
# invalidates from the first changed line down).
COPY requirements.txt .

# Don't cache wheels (keeps the image smaller) and don't let pip phone home
# about upgrades on every layer.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install the CPU builds of torch/torchvision from PyTorch's CPU wheel index
# first (same versions as setup.sh's CPU branch). Doing this before the rest
# means requirements.txt sees torch/torchvision already satisfied and won't
# pull the default (CUDA) wheels from PyPI.
RUN pip install --upgrade pip && \
    pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# --- Application code + weights --------------------------------------------
# .dockerignore keeps dataset/, .venv/, .git/ etc. out of this copy.
# weights/ (~90 MB) IS copied — it's committed and needed to run.
COPY . .

# Point Ultralytics at a writable base for its config dir (its default under
# $HOME triggers a "not writable" warning on every run). Ultralytics appends
# "/Ultralytics" itself, so this becomes /tmp/Ultralytics.
ENV YOLO_CONFIG_DIR=/tmp

# uvicorn/FastAPI GUI listens here.
EXPOSE 8501

# Bind to 0.0.0.0 so the port is reachable from outside the container
# (the default 127.0.0.1 would only be visible inside it).
CMD ["python", "run_gui.py", "--host", "0.0.0.0", "--port", "8501"]
