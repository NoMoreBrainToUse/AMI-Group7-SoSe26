# Event-Based Video Reconstruction for Drone Detection

Video-reconstruction workstream of the **AMI 2026 Hybrid Vision** project from group 7:
combining RGB and event-camera data for drone detection on the
[FRED dataset](https://miccunifi.github.io/FRED/) (Florence RGB-Event Drone
Dataset — synchronized 1280×720 RGB, event stream, and bounding-box
annotations).

Event cameras see fast, low-light motion that blinds RGB sensors, but produce
no image. This folder collects three complementary approaches to turning the
FRED event stream (plus RGB) back into videos a detector can use, with a full
YOLO evaluation of the reconstruction-based route.

## The three approaches

| Folder | Approach | Input |
|---|---|---|
| [`raw_based/`](raw_based/README.md) | **Reconstruction pipeline**: raw event stream → E2VID / E2VID++ (optional HyperE2VID, ET-Net) intensity video → YOLO11m drone detection. Full pipeline docs, stage-organized scripts, and results. | raw events (+ RGB for labels) |
| [`image_based/`](image_based/README.md) | **Event-guided RGB enhancement**: learned enhancement of RGB frames guided by preprocessed event frames (gated residual U-Net pilots). | event PNGs + RGB |
| [`hybrid/`](hybrid/README.md) | **RGB-anchored hybrid composite**: keep the sharp RGB background by construction, inject event/E2VID detail only where events indicate the drone (activity-mask + detail-max fusion). No training required. | RGB + events + E2VID output |

![Hybrid composite example: RGB-anchored background with event/E2VID-injected drone detail](hybrid/hybrid_result_example.gif)

*Hybrid composite on a dusk sequence — the drone is barely visible in raw RGB, but the event-gated E2VID injection makes it clearly detectable while the background stays sharp, real RGB.*

## Headline results (YOLO11m on reconstructed video, held-out flights)

Sequence-level split over 11 FRED sequences (train: 18, 26, 33, 36, 40, 46,
49 — val: 19 day / 31 dusk / 34 near-dark / 43 bright):

| Val sequence | E2VID++ mAP50 | E2VID mAP50 |
|---|---|---|
| **All** | 0.674 | **0.751** |
| 19 — wall, daylight | 0.793 | **0.885** |
| 31 — courtyard, dusk | **0.641** | 0.544 |
| 34 — courtyard, near-dark | 0.664 | **0.836** |
| 43 — garden, bright | **0.702** | 0.678 |

The two methods fail differently: E2VID *misses* (its smooth reconstruction
swallows the low-contrast drone), E2VID++ *hallucinates* (false positives on
drone-sized background texture). Both go blind when the drone hovers — no
events, no evidence — which is the motivation for the hybrid composite: its
background is real RGB, so hallucinated-background false positives disappear
by construction. Training curves and per-run configs are in
`raw_based/results/`; the failure-mode analysis is
`raw_based/results/seq31_failure_recs.json`.

## What is *not* in this repository

To keep it lightweight, the following are excluded and must be
fetched/regenerated (see `raw_based/README.md` for exact commands):

- **FRED dataset** — download from the [FRED site](https://miccunifi.github.io/FRED/)
  into `raw_based/data/datasets/`, then run the decompress + preprocessing
  stages.
- **Third-party repos** — [rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid)
  (E2VID), V2V (E2VID++/EVBIRD),
  [HyperE2VID](https://github.com/ercanburak/HyperE2VID),
  [ET-Net](https://github.com/WarranWeng/ET-Net), and
  [OpenEB](https://github.com/prophesee-ai/openeb) (raw event decoding — see
  `raw_based/docs/SETUP_OPENEB.md`). Clone each into `raw_based/external/`.
- **Model weights** — pretrained reconstruction checkpoints (links in the
  stage docs) and the trained YOLO weights (reproducible via
  `raw_based/scripts/detection/train_yolo.py`; training curves included here).
- **Rendered videos** — per-sequence example results (~500 MB per sequence)
  are regenerated with `raw_based/scripts/presentation/build_example_result.py`.

## Quick orientation

1. Read [`raw_based/README.md`](raw_based/README.md) — the end-to-end
   pipeline (decompress → preprocess → reconstruct → detect → present) with
   per-stage commands and the pitfalls we hit (e2vid frame naming, RGB/label
   timing offsets, ET-Net resolution limits).
2. Read [`hybrid/README.md`](hybrid/README.md) — the training-free composite
   that addresses both reconstruction failure modes; `hybrid/composite.py`
   is self-contained (numpy + OpenCV).
3. [`image_based/`](image_based/README.md) documents the learned
   RGB-enhancement pilots from the event-frame-only track.
