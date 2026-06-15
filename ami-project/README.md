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
  (MobileNetV3-Small)             (MobileNetV3-Small)
       │  P(drone | RGB crop)            │  P(drone | event crop)
       └──────────────┬──────────────────┘
                      ▼
              Late Fusion
       sigmoid( logit(RGB) + λ · logit(Event) )
              λ = 1.5, threshold = 1.0
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

### Sequences used

| Split | Sequences | # Frames |
|-------|-----------|----------|
| **Train** | 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 21, 22, 23, 24, 25, 101, 102, 103 | ~54 500 |
| **Val**   | 9, 10, 15, 19, 20 | ~14 200 |
| **Test**  | 14, 16, 17, 18, **100** | ~13 750 |

> Sequence 110 was excluded due to a spatial label misalignment (~104 px offset
> between annotation coordinates and event frame positions).
> Sequence 100 contains a bird false positive used to demonstrate fusion robustness.

---

## Trained Models

All weights are in `runs/`.

| Model | Path | mAP50 / Val AUC |
|-------|------|-----------------|
| Event YOLO v3 (best) | `runs/event_yolo/fred_subset_v3_event_yolo11m/weights/best.pt` | mAP50 = 0.9335 |
| RGB Verifier v3 | `runs/verifier/rgb_v3/mobilenet_v3_small/best.pt` | Val AUC = 0.8316 |
| Event Verifier v3 | `runs/verifier/rgb_v3/event_mobilenet_v3_small/best.pt` | Val AUC = 0.9068 |

Architecture: **YOLO11m** for detection, **MobileNetV3-Small** for both verifiers.

---

## Test Set Results (seq 14, 16, 17, 18, 100)

| System | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Event YOLO conf >= 0.25 | 96.4 % | 99.6 % | 98.0 % |
| Event YOLO conf >= 0.50 | 97.6 % | 95.9 % | 96.7 % |
| RGB Verifier only | 97.4 % | 97.8 % | 97.6 % |
| **RGB + Event Fusion (lambda=1.5)** | **98.6 %** | 92.7 % | 95.5 % |

Confusion matrices: `outputs/confusion_matrices_test_v3.png`
Result video (RGB background + event boxes): `outputs/test_v3_fusion_result_compressed.mp4`

---

## Reproducing the Pipeline

```bash
# 1. Create venv and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics torch torchvision opencv-python matplotlib scikit-learn tqdm gdown

# 2. Download sequences (example: seq 0-25, 100-103)
#    See run_pipeline_v3.sh for the gdown commands

# 3. Preprocess
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root processed/fred_subset_v3 \
  --train-seqs 0,1,2,3,4,5,6,7,8,11,12,13,21,22,23,24,25,101,102,103 \
  --val-seqs   9,10,15,19,20 \
  --test-seqs  14,16,17,18,100

# 4. Train Event YOLO
python src/training/train_event_yolo.py \
  --data processed/fred_subset_v3/event_yolo/data.yaml --epochs 100

# 5. Run verifier pipeline (proposals -> crops -> verifiers -> fusion)
bash run_verifier_pipeline_v3.sh
```
