# OpenEB Installation (uv + Python 3.12)

Setup guide for the OpenEB / Metavision SDK used to decode FRED .raw event files.

## Prerequisite: OpenEB Installation (uv + Python 3.12)

Installed OpenEB under `external/openeb` using the official Linux guide:
https://github.com/prophesee-ai/openeb

I highly recommend using UV instead of pip and venv as its so much quicker. Follow the guide below to install openeb, do not use the official guide. 

### 1. Clone OpenEB (official release branch)

```bash
cd external
git clone https://github.com/prophesee-ai/openeb.git --branch 5.2.0
cd openeb
```

### 2. Create uv virtual environment (Python 3.12)

```bash
uv venv --python 3.12 .venv
```

### 3. Install OpenEB Python requirements

OpenEB provides the requirement files in `utils/python/`.

```bash
uv pip install --index-strategy unsafe-best-match -p .venv/bin/python \
	-r utils/python/requirements_openeb.txt \
	-r utils/python/requirements_pytorch_cpu.txt
```

### 4. Configure and build OpenEB

```bash
mkdir -p build
cd build
cmake .. -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=$(pwd)/../.venv/bin/python
cmake --build . --config Release -- -j 4
```

### 5. Setup runtime environment and verify Python bindings

```bash
source utils/scripts/setup_env.sh
../.venv/bin/python -c "import metavision_sdk_base, metavision_sdk_core, metavision_hal, metavision_sdk_stream; print('OpenEB Python bindings import OK')"
```

Expected output:

```text
OpenEB Python bindings import OK
```

## Step 1: Preprocessing

First, create /data folder, and decompress all wanted FRED dataset into it. 

Activate the UV enviroment and execute the following command

```
python3 scripts/preprocessing/preprocess_fred_dataset_event.py --dataset-root data/raw --output-root data/preprocessed --train-seqs 1,11,101,102,103 --val-seqs 10 --test-seqs 34,110 --rgb-dir PADDED_RGB --annotation-files interpolated_coordinates.txt,coordinates.txt --materialize hardlink --overwrite
```

The event raw txt file will be in /eventRaw folder and matched .txt to frame files will be in /eventMatched

# Viewer
To view the frames reconstructed from the raw .txt file, use 
```
python3 scripts/presentation/view_event_matched_raw.py --seq 1 --split train
```
