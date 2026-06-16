# AMI Group 7 – Drone Detection Pipeline (SoSe 2026)

Event-camera + RGB late-fusion pipeline for drone detection using the **FRED** dataset
(Florence RGB-Event Drone Dataset).

---

## Pipeline Overview

```
Event camera frames
       │
       ▼
  Event YOLO (yolo11m)          ← proposal generator
       │  bbox proposals (conf ≥ 0.20)
       ├──────────────────────────────────┐
       ▼                                  ▼
  Crop RGB image              Crop Event image
  at proposal bbox            at proposal bbox
       │                                  │
       ▼                                  ▼
  RGB Verifier                    Event Verifier
  (EfficientNet-B0)               (EfficientNet-B0)
       │  P(drone | RGB crop)            │  P(drone | event crop)
       └──────────────┬──────────────────┘
                      ▼
              Late Fusion
       sigmoid( logit(RGB) + λ · logit(Event) )
              λ = 1.0, threshold = 0.958
                      │
                      ▼
              Final detections
```

---

## Dataset – FRED (Florence RGB-Event Drone Dataset)

Sequences are numbered 0–230. Each zip contains paired event camera PNGs,
padded RGB JPGs, and YOLO-format ground-truth labels.

**Download source:** Google Drive folder
`https://drive.google.com/drive/folders/1pISIErXOx76xmCqkwhS3-azWOMlTKZMp`

### v4 Dataset Splits (current)

| Split | Sequences | # Frames |
|-------|-----------|----------|
| **Train** | 0–8, 11–13, 21–25, 30–39, 50, 101, 102, 103, 120 | ~82 000 |
| **Val**   | 9, 10, 15, 19, 20, 150 | ~17 000 |
| **Test**  | 14, 16, 17, 18, **100**, 200 | ~17 000 |

> Sequence 110 excluded: spatial label misalignment (~104 px offset).
> Sequence 100 contains a bird — used to validate false-positive rejection.

### v3 Dataset Splits (previous)

| Split | Sequences | # Frames |
|-------|-----------|----------|
| Train | 0–8, 11–13, 21–25, 101–103 | ~54 500 |
| Val   | 9, 10, 15, 19, 20 | ~14 200 |
| Test  | 14, 16, 17, 18, 100 | ~13 750 |

---

## Trained Models

| Model | Path | Performance |
|-------|------|-------------|
| Event YOLO v3 | `runs/event_yolo/fred_subset_v3_event_yolo11m/weights/best.pt` | mAP50 = 0.9335 |
| RGB Verifier v3 | `runs/verifier/rgb_v3/mobilenet_v3_small/best.pt` | Val AUC = 0.8316 |
| Event Verifier v3 | `runs/verifier/rgb_v3/event_mobilenet_v3_small/best.pt` | Val AUC = 0.9068 |
| **RGB Verifier v4** | `runs/verifier/rgb_v4/efficientnet_b0/best.pt` | Val FP-reduc @95% rec = 45% |
| **Event Verifier v4** | `runs/verifier/rgb_v4/event_efficientnet_b0/best.pt` | Val FP-reduc @95% rec = 56% |

---

## Results

### v4 Test Set (seq 14, 16, 17, 18, 100, 200)

| System | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Event YOLO conf ≥ 0.25 | 96.1% | 99.3% | 97.7% |
| Event YOLO conf ≥ 0.50 | 97.5% | 94.4% | 95.9% |
| **Fusion v4 (λ=1.0, t=0.958)** | **98.1%** | **94.9%** | **96.5%** |

Confusion matrices: `outputs/confusion_matrices_test_v4.png`
Result video: `outputs/test_v4_fusion_result_compressed.mp4`

### Blind Test v4 — Truly Unseen Sequences (seq 40, 43, 46, 49)

| System | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Event YOLO conf ≥ 0.25 | 91.7% | 98.7% | 95.1% |
| **Fusion v4 (λ=1.0, t=0.958)** | **98.4%** | **93.7%** | **96.0%** |

Confusion matrices: `outputs/confusion_matrices_blind_test_v4.png`
Result video: `outputs/blind_test_v4_fusion_result_compressed.mp4`

### Blind Test v3 — Why We Retrained (seq 50, 120, 150, 200)

After training v3, we tested on 4 sequences the model had never seen.
The fusion collapsed to **17.9% recall** while Event YOLO alone gave 97.5%.

**Root cause:** The v3 MobileNetV3-Small verifiers were overconfident — scores were
clustered near 1.0 on training sequences, so the calibrated threshold of 1.0 barely
let anything through on unseen data where scores naturally spread lower.

This motivated the v4 retraining:

| System | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Event YOLO conf ≥ 0.25 | 86.4% | 97.5% | 91.6% |
| Fusion v3 (λ=1.5, t=1.0) | 95.5% | **17.9%** | **30.1%** ← collapsed |

Confusion matrices: `outputs/confusion_matrices_blind_test.png`

---

## What Changed from v3 to v4 and Why

### 1. Bigger verifier backbone: MobileNetV3-Small → EfficientNet-B0
MobileNetV3-Small (~2.5M params) was too small to learn a well-generalised
drone vs background representation. EfficientNet-B0 (~5.3M params) has a better
feature extractor with compound scaling, improving calibration on unseen data.

### 2. More training sequences: 20 → 32
The v3 verifiers only saw sequences 0–25 (low-numbered, filmed in similar conditions).
We added seq 30–39 (more variety) plus seq 50 and 120 from the blind test set,
giving the verifier exposure to a wider range of environments and flight conditions.

### 3. Recalibrated fusion parameters: λ=1.5, t=1.0 → λ=1.0, t=0.958
The old threshold of exactly 1.0 was effectively a ceiling value that became
too strict on unseen data. The new sweep found λ=1.0 gives 71% FP reduction
at 95% recall, with a calibrated threshold of 0.958 that generalises well.

### 4. More training epochs: 30 → 50
Paired with the larger model and more data to ensure convergence without
increasing the risk of overfitting (early stopping patience=30 still active).

---

## False Positive Rejection Demo

Sequence 100 contains a bird that the Event YOLO sometimes detects.
The fusion verifier correctly rejects it — all fusion scores ≈ 0.000.

GIF: `outputs/bird_fp_comparison.gif`

---

## Reproducing the Pipeline

```bash
# 1. Create venv and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics torch torchvision opencv-python matplotlib scikit-learn tqdm gdown

# 2. Run v4 full pipeline (downloads seqs, preprocesses, trains verifiers)
bash run_verifier_v4.sh

# 3. Run blind inference on new sequences
bash run_blind_test_v4.sh
```

The Event YOLO weights are frozen from v3 — only the verifiers are retrained in v4.
