# AMI Drone Detection Pipeline — v4

Event-YOLO proposals → RGB+Event verifier → Late fusion → Web JSON output.

## Prerequisites

```bash
pip install ultralytics torch torchvision efficientnet_pytorch opencv-python gdown
```

`ffmpeg` is required only for `--make-video`.

## Quick start — Blind test evaluation

```bash
./run_blind_test_v4.sh                  # full pipeline (blind seqs 40, 43, 46, 49)
./run_blind_test_v4.sh --no-download    # skip download; data already on disk / Docker mount
./run_blind_test_v4.sh --make-video     # also render result video
./run_blind_test_v4.sh --overwrite      # force recompute all phases
```

## Pipeline phases

| Phase | Script | Output |
|---|---|---|
| 1 Download | gdown | `dataset/{40,43,46,49}/` |
| 2 Preprocess | `src/preprocessing/prepare_fred_yolo.py` | `processed/fred_blind_test_v4/` |
| 3 Proposals | `src/verifier/export_proposals.py` | `runs/proposals/fred_blind_test_v4/detections_conf0.20_*.jsonl` |
| 4 Crops | `src/verifier/extract_crops.py` | `processed/.../crops/*/crop_manifest_*.jsonl` |
| 5 Score | `src/verifier/eval_verifier.py` | `runs/verifier/rgb_v4/blind_v4/scored_*.jsonl` |
| 6 Fusion | `src/verifier/compute_fusion_metrics.py` | `outputs/web/fusion_{manifest,detections}_blind_test_v4.{json,jsonl}` |
| 7 Video *(optional)* | `src/visualization/render_fusion_video.py` | `outputs/videos/*.mp4` |

Each phase is skipped if its outputs already exist. Pass `--overwrite` to rerun.

## Training(Documentation only, no need to run again, best weights already computed=

> Pre-trained weights are already committed in `runs/`. Follow these steps only if retraining from scratch.

### 1 — Preprocess a training split

```bash
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root  processed/fred_subset_v4 \
  --train-seqs   10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,31,32,33,34,35,36,37,38,39 \
  --val-seqs     26,27,28,29 \
  --test-seqs    40,41,42
```

Note: YOLO training configs (`configs/`) are not committed because they contain
hardcoded absolute dataset paths. Regenerate them locally after preprocessing.

### 2 — Train Event YOLO

```bash
python src/training/train_event_yolo.py
# Weights → runs/event_yolo/<run-name>/weights/best.pt
```

### 3 — Train RGB YOLO (optional baseline)

```bash
python src/training/train_rgb_yolo.py
# Weights → runs/rgb_yolo/<run-name>/weights/best.pt
```

### 4 — Generate proposals for verifier training data

```bash
python src/verifier/export_proposals.py \
  --model        runs/event_yolo/fred_subset_v3_event_yolo11m/weights/best.pt \
  --event-images processed/fred_subset_v4/event_yolo/images \
  --event-labels processed/fred_subset_v4/event_yolo/labels \
  --split        train --output runs/proposals/fred_subset_v4
# Repeat for --split val and --split test
```

### 5 — Extract crops for verifier training

```bash
# RGB crops
python src/verifier/extract_crops.py \
  --detections runs/proposals/fred_subset_v4/detections_conf0.20_train.jsonl \
  --modality rgb --images-dir processed/fred_subset_v4/rgb_yolo/images \
  --output-dir processed/fred_subset_v4/crops --split train

# Event crops (same command, --modality event)
# Repeat for val/test splits
```

### 6 — Train RGB verifier

```bash
python src/verifier/train_rgb_verifier.py \
  --manifest     processed/fred_subset_v4/crops/rgb/crop_manifest_rgb_train_conf0.20.jsonl \
  --val-manifest processed/fred_subset_v4/crops/rgb/crop_manifest_rgb_val_conf0.20.jsonl \
  --modality rgb --output runs/verifier/rgb_v4
# Weights → runs/verifier/rgb_v4/efficientnet_b0/best.pt
```

### 7 — Train Event verifier

```bash
python src/verifier/train_rgb_verifier.py \
  --manifest     processed/fred_subset_v4/crops/event/crop_manifest_event_train_conf0.20.jsonl \
  --val-manifest processed/fred_subset_v4/crops/event/crop_manifest_event_val_conf0.20.jsonl \
  --modality event --output runs/verifier/rgb_v4
# Weights → runs/verifier/rgb_v4/event_efficientnet_b0/best.pt
```

### 8 — Produce fusion threshold

```bash
# Score the val set with each verifier (if not done)
python src/verifier/eval_verifier.py \
  --model    runs/verifier/rgb_v4/efficientnet_b0/best.pt \
  --manifest processed/fred_subset_v4/crops/rgb/crop_manifest_rgb_val_conf0.20.jsonl \
  --output   runs/verifier/rgb_v4/efficientnet_b0

python src/verifier/eval_verifier.py \
  --model    runs/verifier/rgb_v4/event_efficientnet_b0/best.pt \
  --manifest processed/fred_subset_v4/crops/event/crop_manifest_event_val_conf0.20.jsonl \
  --output   runs/verifier/rgb_v4/event_efficientnet_b0

# Sweep lambda, select best FP reduction at >= 95 % recall
python src/verifier/eval_fusion.py \
  --rgb-scored   runs/verifier/rgb_v4/efficientnet_b0/scored_rgb_val_conf0.20.jsonl \
  --event-scored runs/verifier/rgb_v4/event_efficientnet_b0/scored_event_val_conf0.20.jsonl \
  --recall-target 0.95 \
  --output runs/verifier/rgb_v4/fusion_results_rgb_val_conf0.20.json
```

Result: `best_lambda=1.0`, `best_threshold=0.9579`, stored in
`runs/verifier/rgb_v4/fusion_results_rgb_val_conf0.20.json`.
`run_blind_test_v4.sh` reads this file automatically.

## Output files

| File | Committed | Description |
|---|---|---|
| `outputs/web/fusion_manifest_blind_test_v4.json` | **yes** | Metrics, threshold, per-split breakdown |
| `outputs/web/fusion_detections_blind_test_v4.jsonl` | **yes** |7.1 MB: 10 762 frames with detections/scores; frames without detections use detections:[] |
| `outputs/confusion_matrices_blind_test_v4.png` | **yes** | 3-system comparison plot |
| `WEB_INTERFACE_HANDOFF.md` | **yes** | Full format spec and regeneration guide |
| `processed/fred_blind_test_v4/` | NO | Raw images (generated by Phase 2) |
| `runs/verifier/rgb_v4/blind_v4/` | NO | Scored JSONL (generated by Phase 5) |
| `runs/proposals/fred_blind_test_v4/` | NO | Proposal JSONL (generated by Phase 3) |
| `outputs/videos/` | NO | MP4 files (generated with `--make-video`) |

## Web GUI (Streamlit)

A Streamlit app (`gui.py`) visualizes the pipeline results in the browser.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-gui.txt
./run_gui.sh                       # open http://localhost:8501
```

**Workflow** (sidebar): Step 1 — upload a sequence zip → Step 2 — press
*Process* (extract + align RGB/event frames into `processed/<name>/`) →
Step 3 — pick a visualization tab.

| Tab | Content |
|---|---|
| **Preview** | Client-side frame player for the processed dataset: play/pause, seekable progress bar, adjustable fps, drag on the image to wipe between RGB and Event |
| **Tracking** | Kalman-filter tracking visualization for the processed dataset (see below) |
| **Fusion** | Blind-test model detections, event and RGB side by side, with GT / kept / rejected boxes |
| **confusion matrices** | Styled per-split confusion matrices + P/R/F1 from the fusion manifest |

### Tracking tab

`CA_Meas_kalman_new.py` implements a 6-state constant-acceleration Kalman
filter with a simple multi-drone tracker (Hungarian association, warmup and
miss handling). The Tracking tab runs it over the blind-test detections of
the currently processed dataset's sequence and renders overlay frames
(detection boxes, current GT position, GT future path, tentative tracks,
active tracks with predicted trajectory + speed) into
`processed/tracking_viz/<seq>/{event,rgb}/`. The overlays are then shown in
the same wipe player as Preview.

Notes:
- The sequence id is inferred from the dataset name (`..._40` → `seq40`).
- Tracking needs detections, which the GUI does not compute itself — they
  come from `outputs/web/fusion_detections_blind_test_v4.jsonl`, so only the
  blind-test sequences (seq40, seq43, seq46, seq49) can be visualized.
- First render per sequence takes about half a minute; results are cached on
  disk and shown instantly afterwards. Overlays are regenerable, not committed.

## Web Interface

Three committed files are ready for the web interface team:

| File | Purpose |
|---|---|
| `outputs/web/fusion_manifest_blind_test_v4.json` | **Start here** — metrics, threshold, split breakdown, paths |
| `outputs/web/fusion_detections_blind_test_v4.jsonl` | One JSON line per frame (10 762 frames total) |
| `outputs/confusion_matrices_blind_test_v4.png` | Pre-rendered 3-system comparison plot |
| `WEB_INTERFACE_HANDOFF.md` | Full format spec, GT-box conversion, regeneration guide |

### Detection record format

Each line in the JSONL file is one frame:

```json
{
  "stem": "seq40_011766549",
  "split": "train",
  "sequence": "seq40",
  "frame_index": 42,
  "timestamp_sec": 1.4,
  "rgb_image":   "processed/fred_blind_test_v4/rgb_yolo/images/train/seq40_011766549.jpg",
  "event_image": "processed/fred_blind_test_v4/event_yolo/images/train/seq40_011766549.png",
  "gt_boxes_norm": [[0.983, 0.088, 0.032, 0.043]],
  "detections": [
    {
      "crop_id": "train_seq40_011766549_det0000001",
      "bbox_xyxy": [1228.91, 47.58, 1277.33, 79.37],
      "detector_score": 0.708,
      "rgb_verifier_score": 0.387,
      "event_verifier_score": 0.998,
      "fusion_score": 0.996,
      "kept": true,
      "iou_with_gt": 0.821,
      "label": 1
    }
  ]
}
```

**Key fields:**
- `frame_index` — 0-based global position in the combined video (train → val → test, sorted by stem); use with `video.currentTime = frame_index / fps`
- `timestamp_sec` — `frame_index / 30.0`; seek target for a 30 fps MP4 (range: 0 – 358.7 s)
- `kept: true` — passed fusion threshold (0.9579); pipeline final verdict
- `kept: false` — YOLO proposal rejected by the verifier
- `bbox_xyxy` — pixel coordinates `[x1, y1, x2, y2]`
- `gt_boxes_norm` — normalized YOLO `[cx, cy, w, h]`; multiply by `[1280, 720, 1280, 720]` for pixels
- `"detections": []` — frame with no proposals

**Render suggestion:** GT → green, `kept: true` → cyan, `kept: false` → red.

### Image paths

`rgb_image` and `event_image` paths are relative to the repository root and point to
`processed/fred_blind_test_v4/`. These images are **not committed** — they must be
generated by running the pipeline (`./run_blind_test_v4.sh --no-download`) or
mounted as a Docker volume (see below).

## Data and image folder requirements

Raw sequences go into `dataset/` and preprocessed images are written to `processed/`.
Both directories are excluded from git — only a `.gitkeep` placeholder and
`dataset/README.md` are tracked. See [`dataset/README.md`](dataset/README.md) for details.

| Path | Committed | Source |
|---|---|---|
| `dataset/` | `.gitkeep` + `README.md` only | — |
| `dataset/{40,43,46,49}/` | NO | Downloaded from Google Drive or Docker-mounted |
| `processed/` | `.gitkeep` only | — |
| `processed/fred_blind_test_v4/` | NO | Output of Phase 2 (preprocessing, ~5 min) |
| `processed/fred_blind_test_v4/rgb_yolo/images/` | NO | Source for JSONL `rgb_image` paths |
| `processed/fred_blind_test_v4/event_yolo/images/` | NO | Source for JSONL `event_image` paths |
| `processed/.../crops/` | NO | Output of Phase 4 (crop extraction) |
| `runs/proposals/fred_blind_test_v4/` | NO | Output of Phase 3 (event YOLO proposals) |
| `runs/verifier/rgb_v4/blind_v4/` | NO | Output of Phase 5 (verifier scoring) |

The JSONL `rgb_image` and `event_image` paths are relative to the repo root and
point into `processed/fred_blind_test_v4/`. Regenerate them with:

```bash
./run_blind_test_v4.sh --no-download   # if dataset/ is already populated
./run_blind_test_v4.sh                 # downloads missing sequences first
```

## Docker usage

Mount `dataset/` (raw sequences) and optionally `processed/` (to persist
intermediate results across container runs), then pass `--no-download`:

```bash
docker run --rm \
  -v /path/to/fred-data:/app/dataset \
  -v /path/to/processed:/app/processed \
  my-image \
  bash -c "./run_blind_test_v4.sh --no-download"
```

The pipeline uses CPU-compatible defaults throughout (libx264 for video encoding).
CUDA is not required — all PyTorch inference runs on CPU.
For NVIDIA GPU video encoding: `python src/visualization/render_fusion_video.py --codec h264_nvenc`
(falls back to libx264 automatically if `h264_nvenc` is unavailable).

## What is not committed and why

| Not committed | Reason |
|---|---|
| `dataset/` | Tens of GB; download via gdown or mount externally |
| `processed/` | 2–10 GB of images; fully regenerable in ~5 min |
| `runs/proposals/`, `runs/.../blind_v4/` | Hundreds of MB; regenerable in ~10 min |
| `outputs/videos/` | Up to 500 MB; regenerable with `--make-video` |
| `configs/` | Contain hardcoded absolute dataset paths; regenerate after preprocessing |
| `.venv/` | User/environment-specific; run `pip install` |
| `__pycache__/`, `*.pyc` | Auto-generated by Python |
| `ami-project/` | Legacy v3 code subdirectory; superseded by root-level files |
| `yolo-fred/` | Unrelated coco8/YOLOv11 experiments |
