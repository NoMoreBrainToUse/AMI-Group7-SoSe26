# AMI 2026 — Group 7 · Hybrid Vision drone detection

Hybrid RGB + event-camera drone detection on the
[FRED dataset](https://miccunifi.github.io/FRED/): two YOLO11 detectors
(event leads, RGB backs it up), two EfficientNet-B0 crop verifiers, and a
calibrated log-odds fusion that decides which detections survive. Detections
are bounding boxes with class label (`drone`) and a fusion confidence.

## Quick start

```bash
./setup.sh --sample     # venv + all deps (CUDA/CPU auto) + FRED seq 40 (~2.3 GB)
.venv/bin/python run_gui.py
# open http://localhost:8501  →  select "seq 40"  →  Run pipeline
```

The first pipeline run takes a few minutes on GPU (longer on CPU: two YOLO
passes + two verifiers over ~2,100 frames). When it finishes, the Preview /
Detections / Tracking / Metrics tabs light up. Already have FRED zips? Skip
`--sample` and drag any sequence zip into the sidebar instead.

Headless (no GUI):

```bash
.venv/bin/python run_pipeline.py dataset/40            # results → outputs/40/
.venv/bin/python run_pipeline.py dataset/40 --device cpu
```

## What the pipeline does

```
FRED sequence (PADDED_RGB + Event/Frames + coordinates)
        │  align.py — time-align RGB↔event, shared stems & coords
        ▼
  event-YOLO (conf ≥ 0.20) ─┐
  RGB-YOLO  (conf ≥ 0.60)  ─┴─► merge.py — union, cross-modality dedup
        │                       source: event | rgb | both
        ▼
  verifier.py — RGB + event EfficientNet-B0 crop scores (in memory)
        ▼
  fusion.py — σ(logit(s_rgb) + λ·logit(s_event)) ≥ τ  → kept
        ▼
  outputs/<seq>/web/fusion_{detections,manifest}_<seq>.{jsonl,json}
```

Why two detectors: an event camera sees brightness *change* — a hovering
drone produces almost no events (measured on FRED seq31: event-YOLO covers
96% of GT while the drone flies fast, 47% while it hovers). The RGB detector
is gated hard so it only adds proposals it is very sure about, and every
proposal still passes the verifiers.

Note: the V6 YOLO RGB weights are too big so it is being excluded from the repo.
It can be downloaded under https://drive.google.com/file/d/1FRDlhGFMlxmB_cx17capjHpIAZJwIE4-/view?usp=drive_link

## Web interface

- **Sidebar** — sequences on disk with their state, zip drag & drop,
  Run pipeline with live progress log.
- **Preview** — RGB ↔ event wipe player (drag the divider).
- **Detections** — RGB/event side by side (or single view); ground truth
  green, kept detections yellow (`drone 0.97`), RGB-sourced keeps amber
  dashed, rejected proposals red; zoom crops below each pane.
- **Tracking** — Kalman multi-drone tracks with predicted trajectories
  (computed server-side in ~1 s, drawn on canvas).
- **Metrics** — confusion cards per system, true ↔ proposal-conditional
  toggle, proposal recall ceiling.
- Player transport: space = play, ←/→ = frame, shift+←/→ = ×10; the strip
  above the seek bar marks every frame green (drone found), red (missed) or
  amber (false positive) — click it to jump.

## Required input format

A sequence is one directory (or zip; the GUI extracts it), as shipped by
FRED:

```
<seq>/
  PADDED_RGB/                     RGB frames, 1280×720 JPEG, padded to the
                                  event sensor's field of view (shared
                                  coordinate space with the event frames)
      Video_<seq>_<H>_<M>_<S.frac>.jpg    wall-clock naming; frame i is
                                          placed at (i+1) × 1/30 s
  Event/Frames/                   event frames, 1280×720 PNG
      *_<time_us>.png             timestamp in µs since sequence start
  interpolated_coordinates.txt    annotations, one box per line:
      <t_seconds>: x1, y1, x2, y2[, track_id]
  (coordinates.txt is used as fallback when the interpolated file is absent)
```

Alignment pairs each annotation timestamp with the nearest RGB and event
frame within 40 ms; pairs are written with a shared stem so cross-modality
crops and proposal merging are exact. Multiple boxes per timestamp
(multiple drones) are supported.

## Metrics — read this before quoting numbers

`fusion_manifest_*.json → metrics` is **GT-box level over all frames**: a
ground-truth drone that no kept detection matches at IoU ≥ 0.5 counts as a
false negative, *including frames where the detectors proposed nothing*.
`metrics_conditional` is the older proposal-only view kept for reference — it
ignores such misses entirely (it once reported 97% F1 at a true recall of
82%; don't quote it as system performance).

## Layout

| Path | What |
|---|---|
| `setup.sh` | one-command install (+ `--sample` test sequence) |
| `run_gui.py` / `run_pipeline.py` | web interface / headless CLI |
| `src/hybrid_vision/config.py` | every path + threshold (one dataclass) |
| `src/hybrid_vision/align.py` | FRED sequence → paired frames, YOLO layout |
| `src/hybrid_vision/detect.py` | YOLO proposals, one low-conf pass per modality |
| `src/hybrid_vision/merge.py` | hybrid union + dedup + recall-ceiling stats |
| `src/hybrid_vision/verifier.py` | EfficientNet crop scoring (in memory) |
| `src/hybrid_vision/fusion.py` | log-odds fusion, honest metrics, web outputs |
| `src/hybrid_vision/pipeline.py` | phase orchestration + caching |
| `gui/` | FastAPI server + canvas single-page app |
| `training/` | detector / verifier training, crop export, fusion calibration |
| `weights/` | committed model weights + fusion operating point |

## Weights

| File | Model | Notes |
|---|---|---|
| `event_yolo11m.pt` | YOLO11m, event frames, imgsz 640 | mAP50 0.86 (v5, 54 seqs) |
| `rgb_yolo11m.pt` | YOLO11m, RGB frames, imgsz 640 | v5 — weak (mAP50 0.25); v6 retrain in progress |
| `verifier_{rgb,event}_effb0.pt` | EfficientNet-B0, 96 px crops | drone vs background |
| `fusion_config.json` | λ=1.0, τ=0.034 | calibrated on an 11-seq diverse pool; see file for why not λ=2.0/τ=0.002 |

To evaluate different RGB weights without touching the defaults:
`run_pipeline.py ... --rgb-model <best.pt> --rgb-imgsz 1280` (imgsz must
match how the weights were trained).

## Training (reproduce weights)

```bash
# 1. align training sequences into one root (repeat per sequence)
#    split ∈ train/val — see src/hybrid_vision/align.py
# 2. detectors
python training/train_detector.py processed/train_set/event_yolo/data.yaml --name event_yolo11m --imgsz 640
python training/train_detector.py processed/train_set/rgb_yolo/data.yaml   --name rgb_yolo11m   --imgsz 1280
# 3. verifier crops + training
python training/export_crops.py outputs/<seq>/proposals_event.jsonl \
    --images-dir processed/<seq>/rgb_yolo/images --modality rgb --output-dir processed/crops
python training/train_verifier.py --train-manifest ... --val-manifest ... --output runs/verifier_rgb
# 4. fusion operating point (n_missed_gt makes the recall target TRUE recall)
python training/calibrate_fusion.py --rgb-scored ... --event-scored ... \
    --n-missed-gt <merge_stats.json:n_missed_gt> --output weights/fusion_config.json
```
