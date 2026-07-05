# AMI 2026 — Hybrid Vision: Raw-Event Video Reconstruction & Detection

Branch `reconstruction_raw`: event-based video reconstruction from the FRED
raw event stream, and drone detection (YOLO) on the reconstructed videos.
This provides the "model trained on intensity images generated with
event-based video reconstruction" comparison required by the project
description, with two reconstruction methods (E2VID and E2VID++), plus
optional HyperE2VID and ET-Net.

## Pipeline

```
FRED zips          data/datasets/<seq>.zip
   |  1. decompress
   v
raw sequences      data/raw/<seq>/  (PADDED_RGB jpgs, Event/events.raw,
   |                                 Event/Frames pngs, coordinates.txt)
   |  2. preprocessing               scripts/preprocessing/
   v
preprocessed       data/preprocessed_all_val/preprocessed_fred_<seq>/
   |                 paired/manifest_val.csv   time-aligned RGB/event samples
   |                 matched/ labels/          images + YOLO labels
   |                 eventRaw/events.txt       decoded raw events (x,y,p,t_us)
   |                 eventMatched/             33ms event window per sample
   |  3. reconstruction inference    scripts/reconstruction/
   v
intensity videos   E2VID   -> external/rpg_e2vid/output_tests/all_120fps/
   |               E2VID++ -> external/V2V/generated_tests/  (EVBIRD)
   |               (optional: HyperE2VID, ET-Net -> artifacts/etnet/)
   |  4. YOLO detection              scripts/detection/
   v
detection          artifacts/yolo/<dataset>/   datasets built from recon
   |               artifacts/yolo/runs/        training runs + metrics
   |  5. presentation                scripts/presentation/
   v
videos & figures   sequence videos, prediction videos, comparison stills
```

## Setup

- **OpenEB / Metavision** (decodes `.raw` event files): see
  [docs/SETUP_OPENEB.md](docs/SETUP_OPENEB.md). Installed under
  `external/openeb` with its own uv venv.
- **Reconstruction + YOLO venv**: `external/V2V/.venv` (torch 2.4 cu124,
  ultralytics, h5py, pandas). All commands below run from the repo root with
  this interpreter unless stated otherwise.
- **Checkpoints**: `external/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar`,
  V2V/EVBIRD weights inside `external/V2V`, `external/ET-Net/pretrained/etnet.pth`
  (gdown id `1V7vj3YkbhAmgzyf6rqrkeF0HSSNwyGkO`), `yolo11m.pt` at repo root.

## Stage 1 — Decompress

Unzip each FRED sequence from `data/datasets/<seq>.zip` into `data/raw/<seq>/`.

## Stage 2 — Preprocessing

See [scripts/preprocessing/README.md](scripts/preprocessing/README.md) for the
full explanation (alignment, labels, manifest format, denoising rationale).

```bash
python scripts/preprocessing/prepare_fred_yolo.py ...   # per-sequence outputs
```

Produces, per sequence, the `preprocessed_fred_<seq>` layout shown above,
including `eventRaw/events.txt` (via OpenEB) and per-sample 33 ms event
windows. Note the manifest timing columns: labels sit at `event_time_s`;
`rgb_time_s` differs by ±5–6 ms per sequence (sign varies!) and the drone can
move >10 px/ms — relevant for any consumer of the labels.

## Stage 3 — Reconstruction inference

**E2VID** (rpg_e2vid, fixed 1/120 s windows over the full sequence):

```bash
external/V2V/.venv/bin/python scripts/reconstruction/run_e2vid_all_120fps.py
```

Output frames land in
`external/rpg_e2vid/output_tests/all_120fps/seq<N>_120fps/`.
**Important:** the output filenames `frame_<K>.png` are *event indices* (the
last event of each window), not timestamps. Map to real time by looking up
event `K`'s timestamp in `eventRaw/events.txt`.

**E2VID++** (V2V framework, one frame per manifest sample):

```bash
external/V2V/.venv/bin/python scripts/reconstruction/run_v2v_e2vidpp_all_prepared.py
```

Output: `external/V2V/generated_tests/seq<N>_<M>frames/results/EVBIRD/...`,
where frame `i` corresponds to manifest row `i+1` (rows sorted by
`event_time_s`).

**Optional methods**: `run_hypere2vid_*.py` (HyperE2VID), and ET-Net via

```bash
external/V2V/.venv/bin/python scripts/reconstruction/convert_fred_to_monash_h5.py \
    --seq 31 --downscale 2   # ET-Net needs <=8000 tokens -> half resolution
cd external/ET-Net && ../V2V/.venv/bin/python inference.py \
    --checkpoint_path pretrained/etnet.pth \
    --events_file_path ../../artifacts/etnet/fred_seq31_ds2.h5 \
    --output_folder ../../artifacts/etnet/seq31_out --num_encoder 3
```

ET-Net caveats: fixed 8000-token positional table forces 640×360 input for
1280×720 data, and it costs ~0.4 s/frame (~50× E2VID). It also computes
LPIPS/MSE/SSIM against the real RGB frames along the way.

## Stage 4 — YOLO detection on reconstructions

Datasets sample every fully reconstructed sequence on a shared 12.5 fps time
grid with a sequence-level split, one held-out flight per scene/lighting
group — train: 18, 26, 33, 36, 40, 46, 49; val: 19 (wall/day),
31 (courtyard/dusk), 34 (near-dark), 43 (garden/bright):

```bash
# datasets (8595 train / 4618 val each, sample-for-sample paired)
external/V2V/.venv/bin/python scripts/detection/prepare_e2vid_yolo_multiseq.py --overwrite
external/V2V/.venv/bin/python scripts/detection/prepare_rpge2vid_yolo_multiseq.py --overwrite

# training (YOLO11m, 50 epochs, batch 8 for an 8GB GPU)
external/V2V/.venv/bin/python scripts/detection/train_yolo.py \
    --data artifacts/yolo/e2vid_multiseq/data.yaml --name e2vid_multiseq_yolo11m

# validation, overall + per sequence
external/V2V/.venv/bin/python scripts/detection/val_yolo_per_sequence.py \
    --weights artifacts/yolo/runs/e2vid_multiseq_yolo11m/weights/best.pt \
    --dataset artifacts/yolo/e2vid_multiseq
```

(`prepare_e2vid_yolo_seq18_31.py` is the deprecated first attempt — train
seq18 / val seq31 only; kept for the report's ablation: its cross-scene split
scores mAP50 0.18 vs 0.67+ for the sequence-level split.)

### Results (YOLO11m best.pt, identical recipe on both datasets)

| Val sequence | Condition | E2VID++ (V2V) mAP50 | E2VID (rpg) mAP50 |
|---|---|---|---|
| **All** | | 0.674 | **0.751** |
| 19 | wall, daylight | 0.793 | **0.885** |
| 31 | courtyard, dusk | **0.641** | 0.544 |
| 34 | courtyard, near-dark | 0.664 | **0.836** |
| 43 | garden, bright | **0.702** | 0.678 |

Failure modes differ (seq 31 analysis, `artifacts/yolo/runs/seq31_failure_recs.json`):
E2VID fails by *missing* (smooth fog swallows the low-contrast drone; 7
sustained ≥1 s miss windows), E2VID++ fails by *hallucinating* (43% more
false positives on drone-sized background texture). Both share the same
blackout windows during hover (event starvation) — the modality floor.

## Stage 5 — Presentation

```bash
# reconstruction sequence videos
python scripts/presentation/make_v2v_sequence_videos.py
# prediction video (cyan = predictions, green = GT), 12.5 fps = real time
external/V2V/.venv/bin/python scripts/detection/predict_yolo_video.py \
    --weights .../best.pt --dataset artifacts/yolo/e2vid_multiseq \
    --seq 31 --out artifacts/yolo/runs/pred_seq31.mp4
```

Viewer utilities for raw/preprocessed data are also in
`scripts/presentation/` (`view_*.py`, `visualize_events_window.py`).

The **hybrid RGB+event composite** prototype (RGB background + event-gated
reconstruction injection — sharp background by construction, drone from
events) is in [`../hybrid/`](../hybrid/); see its README.

## Repository layout

```
data/            datasets/ (FRED zips) -> raw/ -> preprocessed_all_val/
external/        openeb, rpg_e2vid (E2VID), V2V (E2VID++), HyperE2VID, ET-Net
scripts/
  preprocessing/ FRED -> aligned samples + labels + raw events
  reconstruction/ E2VID / E2VID++ / HyperE2VID runners, ET-Net H5 converter
  detection/     YOLO dataset prep, train, per-sequence val, prediction video
  presentation/  sequence/prediction videos, data viewers
  analysis/      motion / optical-flow analysis of raw RGB
src/             event-guided RGB enhancement pilots
artifacts/
  yolo/          datasets + runs (weights, curves, metrics, videos)
  etnet/         ET-Net H5 inputs + seq31 reconstructions
logs/            long-running reconstruction/training logs
docs/            SETUP_OPENEB.md
```
