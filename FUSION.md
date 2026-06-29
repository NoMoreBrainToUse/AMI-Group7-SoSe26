# Fusion — Design, History, and Recalibration

## What the fusion stage does

After the Event-YOLO detector proposes bounding boxes, two independent verifier networks
(one RGB, one event-frame) each score every crop with a confidence in `[0, 1]`.
The fusion stage combines those two scores into a single `fusion_score`, then applies a
threshold to decide whether a detection is kept (`kept: true`) or discarded (`kept: false`).

```
Event-YOLO proposals
  ├─► RGB crop  → EfficientNet-B0 (RGB)   → rgb_score  ─┐
  └─► Event crop → EfficientNet-B0 (Event) → event_score ─┴► fusion_score → threshold → kept?
```

---

## The fusion formula

Scores are combined in **logit space** (log-odds space):

```
logit(p)     = log(p / (1 - p))
fusion_score = sigmoid( logit(rgb_score) + λ · logit(event_score) )
```

With `λ = 1.0` both modalities contribute equally.
`λ < 1` down-weights the event verifier; `λ > 1` up-weights it.

This is principled Bayesian fusion under conditional independence: each verifier
produces a likelihood ratio, and the log-odds add linearly before being projected
back to a probability.

The threshold `τ` is calibrated to achieve a target recall (default 95 %) on a held-out
val set. All crops with `fusion_score ≥ τ` are kept.

---

## Approaches we tried and why we moved on

### Attempt 1 — Score averaging (rejected)

```
fusion_score = (rgb_score + event_score) / 2
```

Simple but flawed: averaging in probability space does not respect the multiplicative
nature of independent evidence. A score of 0.6 from both modalities averages to 0.6,
but in log-odds it corresponds to strong joint evidence (~0.9 combined).
In practice, precision gains were modest and the threshold had no principled basis.

### Attempt 2 — Score product (rejected)

```
fusion_score = rgb_score · event_score
```

Equivalent to logical AND — both must be confident. Too strict: any uncertain modality
kills the detection. Recall dropped significantly on sequences where one modality was
noisy (e.g., dark scenes where the RGB verifier was uncertain).

### Attempt 3 — Single-modality RGB verifier only (v1/v2, MobileNetV3-Small)

Early experiments used only the RGB verifier (MobileNetV3-Small) to filter proposals.
Performance was reasonable on training sequences but the model was overconfident —
it learned sequence-specific textures rather than drone features, so scores collapsed
on unseen data.

### Attempt 4 — Dual-modality logit fusion, MobileNetV3-Small verifiers (v3)

Added the event-frame verifier alongside RGB. Logit-space fusion produced a 71 %
FP reduction at 95 % recall on the training-set val split.

**Problem:** blind test on sequences 50, 120, 150, 200 gave **17.9 % recall** — near
total collapse. Root cause: MobileNetV3-Small is shallow and became overconfident on
the training distribution; the calibrated threshold of `1.0` was unreachable on unseen
data where both verifiers scored true drones lower.

### Current — Dual-modality logit fusion, EfficientNet-B0 verifiers (v4) ✓

Retraining both verifiers with EfficientNet-B0 on **32 sequences** (vs 20 before),
50 epochs, with stronger augmentation:

- Better calibrated probabilities — EfficientNet does not saturate to extremes as easily.
- Threshold recalibrated to `λ = 1.0`, `τ = 0.9579`.
- Blind test v4 (seq 40, 43, 46, 49): **Precision 98.4 % / Recall 93.7 % / F1 96.0 %**.

The logit-space fusion formula itself was unchanged; the improvement came entirely from
better verifier calibration and more diverse training data.

---

## Known limitation — threshold generalization

The threshold `τ = 0.9579` was calibrated on **val sequences 9, 10, 15, 19, 20, 150**.
On sequences with different characteristics (far-away drones, cluttered backgrounds,
unusual lighting) the verifiers tend to assign lower confidence to true drones, so
the same threshold over-rejects and recall drops.

Example: seq 140 (harder conditions) with the default threshold:
**Precision 72.1 % / Recall 72.7 % / F1 72.4 %** — well below the v4 numbers.

The fix is to recalibrate the threshold on a more diverse val set (see below).
The verifier weights themselves do **not** need to be retrained for recalibration.

---

## Recalibrating the threshold for a new deployment

Recalibration requires no retraining. You only need to:
1. Score new calibration sequences through the existing verifiers.
2. Combine those scored files with the existing val scored files.
3. Re-run `eval_fusion.py` on the combined pool to find a new threshold.

### How many sequences and which ones?

**Target: 10–20 diverse calibration sequences on top of the existing 6.**

"Diverse" means covering the axes that affect verifier confidence:
- **Distance** — drone far (small bbox) vs. close (large bbox)
- **Background** — sky only, trees, urban rooftops, mixed
- **Lighting** — bright midday, overcast, golden hour, low-contrast
- **Drone size / type** — fixed-wing vs. multirotor if present in dataset

You do not need to open every zip to check. A practical approach:

```bash
# For each candidate sequence, check frame count and bbox-size distribution:
python3 - <<'EOF'
import os, pathlib, statistics

for seq_dir in sorted(pathlib.Path("dataset").iterdir()):
    label_dir = seq_dir / "RGB_YOLO"
    if not label_dir.exists():
        continue
    sizes = []
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                sizes.append(float(parts[3]))   # bbox width (normalized)
    if sizes:
        print(f"{seq_dir.name:>6}  frames={len(list(label_dir.glob('*.txt'))):>5}  "
              f"median_w={statistics.median(sizes):.3f}  "
              f"min_w={min(sizes):.3f}  max_w={max(sizes):.3f}")
EOF
```

Sequences with small median bbox width = drone is far away (harder for verifier).
Sequences with large width = drone is close (easier). Pick a mix.

### Step-by-step recalibration

```bash
# 1. Preprocess new calibration sequences (as test split, no train/val labels needed)
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root  processed/calib \
  --train-seqs "" --val-seqs "" \
  --test-seqs  "5,22,37,55,80,99,110,130,155,180"   # ← your diverse selection

# 2. Export proposals
python src/verifier/export_proposals.py \
  --model        runs/event_yolo/fred_subset_v3_event_yolo11m/weights/best.pt \
  --event-images processed/calib/event_yolo/images \
  --event-labels processed/calib/event_yolo/labels \
  --split test --output runs/proposals/calib

# 3. Extract crops
python src/verifier/extract_crops.py \
  --detections runs/proposals/calib/detections_conf0.20_test.jsonl \
  --modality rgb  --images-dir processed/calib/rgb_yolo/images \
  --output-dir processed/calib/crops --split test

python src/verifier/extract_crops.py \
  --detections runs/proposals/calib/detections_conf0.20_test.jsonl \
  --modality event --images-dir processed/calib/event_yolo/images \
  --output-dir processed/calib/crops --split test

# 4. Score with existing verifiers (no retraining)
python src/verifier/eval_verifier.py \
  --model    runs/verifier/rgb_v4/efficientnet_b0/best.pt \
  --manifest processed/calib/crops/rgb/crop_manifest_rgb_test_conf0.20.jsonl \
  --output   runs/verifier/rgb_v4/calib

python src/verifier/eval_verifier.py \
  --model    runs/verifier/rgb_v4/event_efficientnet_b0/best.pt \
  --manifest processed/calib/crops/event/crop_manifest_event_test_conf0.20.jsonl \
  --output   runs/verifier/rgb_v4/calib

# 5. Pool new scored files with existing val scored files
#    (cat preserves JSONL format — one record per line)
cat runs/verifier/rgb_v4/efficientnet_b0/scored_rgb_val_conf0.20.jsonl \
    runs/verifier/rgb_v4/calib/scored_rgb_test_conf0.20.jsonl \
  > runs/verifier/rgb_v4/calib/pooled_rgb.jsonl

cat runs/verifier/rgb_v4/event_efficientnet_b0/scored_event_val_conf0.20.jsonl \
    runs/verifier/rgb_v4/calib/scored_event_test_conf0.20.jsonl \
  > runs/verifier/rgb_v4/calib/pooled_event.jsonl

# 6. Re-run fusion calibration on pooled data
python src/verifier/eval_fusion.py \
  --rgb-scored   runs/verifier/rgb_v4/calib/pooled_rgb.jsonl \
  --event-scored runs/verifier/rgb_v4/calib/pooled_event.jsonl \
  --recall-target 0.95 \
  --output runs/verifier/rgb_v4/fusion_results_rgb_val_conf0.20.json
```

The new `best_threshold` is written back to
`runs/verifier/rgb_v4/fusion_results_rgb_val_conf0.20.json` and
`run_blind_test_v4.sh` picks it up automatically on the next run.

### How to know the recalibration worked

Run the blind test on a held-out harder sequence (e.g. seq 140) and compare F1 before
and after. Target: recall should stay above 90 % while precision stays above 85 %.

---

## Files at a glance

| File | Role |
|---|---|
| `src/verifier/eval_fusion.py` | Sweeps λ, finds best threshold, writes calibration JSON |
| `src/verifier/eval_verifier.py` | Scores crops with one verifier model, writes scored JSONL |
| `src/verifier/compute_fusion_metrics.py` | Reads calibration JSON, applies threshold, produces web outputs |
| `runs/verifier/rgb_v4/fusion_results_rgb_val_conf0.20.json` | Active calibration (threshold + λ) |
| `runs/verifier/rgb_v4/efficientnet_b0/best.pt` | RGB verifier weights |
| `runs/verifier/rgb_v4/event_efficientnet_b0/best.pt` | Event verifier weights |
