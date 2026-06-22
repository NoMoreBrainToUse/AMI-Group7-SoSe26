# Web Interface Handoff — AMI Blind Test v4

## Files to use

| File | Size | Purpose |
|---|---|---|
| `outputs/web/fusion_manifest_blind_test_v4.json` | 3 KB | Start here: metrics, thresholds, paths |
| `outputs/web/fusion_detections_blind_test_v4.jsonl` | 7.1 MB | One JSON line per frame — all 10 762 frames |
| `outputs/confusion_matrices_blind_test_v4.png` | 64 KB | Pre-rendered confusion matrix (3 systems) |

Images are served from the two directories below (not committed — see [Regenerating outputs](#regenerating-outputs)):
```
processed/fred_blind_test_v4/rgb_yolo/images/{train,val,test}/*.jpg    1280×720 px
processed/fred_blind_test_v4/event_yolo/images/{train,val,test}/*.png  1280×720 px
```

---

## Path anchor

**All relative paths in the JSONL are relative to the repository root** (`ami-project/`).

Example: `"rgb_image": "processed/fred_blind_test_v4/rgb_yolo/images/train/seq40_011733216.jpg"`
resolves to `<repo_root>/processed/fred_blind_test_v4/rgb_yolo/images/train/seq40_011733216.jpg`.

In a Docker container, mount the repository root at the same relative offset.

---

## JSONL record structure

Each line is a JSON object for one frame. Frames with no detections are included with `"detections": []`.

```jsonc
{
  "stem":     "seq40_011766549",       // unique frame identifier
  "split":    "train",                 // train | val | test
  "sequence": "seq40",                 // which blind sequence

  // relative paths to the images (anchor: repo root)
  "rgb_image":   "processed/fred_blind_test_v4/rgb_yolo/images/train/seq40_011766549.jpg",
  "event_image": "processed/fred_blind_test_v4/event_yolo/images/train/seq40_011766549.png",

  // ground-truth boxes: normalized YOLO format [cx, cy, w, h]
  // multiply cx/w by image width (1280) and cy/h by image height (720) to get pixels
  "gt_boxes_norm": [[0.983, 0.088, 0.032, 0.043]],

  "detections": [
    {
      "crop_id":              "train_seq40_011766549_det0000001",
      "bbox_xyxy":            [1228.91, 47.58, 1277.33, 79.37], // pixels, x1 y1 x2 y2
      "detector_score":       0.708616,   // raw Event YOLO confidence
      "rgb_verifier_score":   0.386575,   // RGB EfficientNet-B0 sigmoid score
      "event_verifier_score": 0.997748,   // Event EfficientNet-B0 sigmoid score
      "fusion_score":         0.996431,   // combined score (see below)
      "kept":                 true,       // whether this detection passes the fusion threshold
      "iou_with_gt":          0.8211,     // IoU with nearest ground-truth box
      "label":                1           // 1 = drone (TP candidate), 0 = background (FP candidate)
    }
  ]
}
```

### What `kept` means

`kept = fusion_score >= 0.9579`

The fusion score combines the RGB and event verifier scores in logit space:

```
fusion_score = sigmoid(logit(rgb_score) + 1.0 × logit(event_score))
```

The threshold (0.9579) and lambda (1.0) were tuned on the held-out validation set to maximise FP reduction at 95% recall. A detection with `kept: true` is the pipeline's final "drone present" verdict.

**Render suggestion:** draw `kept: true` boxes in cyan, `kept: false` boxes in red, `gt_boxes_norm` in green.

---

## Sequence → split mapping

| Split | Sequence | Frames |
|---|---|---|
| `train` | `seq40` | 2 147 |
| `val` | `seq43` | 2 135 |
| `test` | `seq46`, `seq49` | 6 480 |

All four sequences were **never seen during training**.

---

## Performance summary (from `fusion_manifest_blind_test_v4.json`)

| System | Precision | Recall | F1 |
|---|---|---|---|
| Event YOLO conf ≥ 0.25 | 91.7 % | 98.7 % | 95.1 % |
| Event YOLO conf ≥ 0.50 | 95.9 % | 88.7 % | 92.2 % |
| **Fusion v4 (this pipeline)** | **98.4 %** | **93.7 %** | **96.0 %** |

Per-split metrics are available under `metrics.train`, `metrics.val`, `metrics.test` in the manifest.

---

## Converting ground-truth boxes to pixels

```python
def gt_norm_to_px(cx, cy, w, h, img_w=1280, img_h=720):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2
```

---

## Minimal Python to iterate the JSONL

```python
import json
from pathlib import Path

REPO_ROOT = Path("/path/to/ami-project")
JSONL     = REPO_ROOT / "outputs/web/fusion_detections_blind_test_v4.jsonl"

with JSONL.open() as fh:
    for line in fh:
        frame = json.loads(line)
        rgb_path  = REPO_ROOT / frame["rgb_image"]
        kept_dets = [d for d in frame["detections"] if d["kept"]]
        gt_boxes  = frame["gt_boxes_norm"]   # list of [cx, cy, w, h]
        # ... render here
```

---

## Regenerating outputs

All outputs are produced by:

```bash
cd ami-project
./run_blind_test_v4.sh            # full pipeline, no video
./run_blind_test_v4.sh --make-video   # also renders the mp4
./run_blind_test_v4.sh --no-download  # skip Google Drive (data already mounted)
./run_blind_test_v4.sh --overwrite    # force recompute everything
```

Phases and what they produce:

| Phase | Script | Skipped if… |
|---|---|---|
| 1 — Download | `gdown` | sequence dir exists |
| 2 — Preprocess | `src/preprocessing/prepare_fred_yolo.py` | always runs |
| 3 — Proposals | `src/verifier/export_proposals.py` | `runs/proposals/fred_blind_test_v4/detections_conf0.20_*.jsonl` exist |
| 4 — Crops | `src/verifier/extract_crops.py` | `processed/.../crops/*/crop_manifest_*.jsonl` exist |
| 5 — Score | `src/verifier/eval_verifier.py` | `runs/verifier/rgb_v4/blind_v4/scored_*.jsonl` exist |
| 6 — Fusion + web | `src/verifier/compute_fusion_metrics.py` | `outputs/web/fusion_*` exist |
| 7 — Video | `src/visualization/render_fusion_video.py` | only runs with `--make-video` |

Pass `--overwrite` to rerun any phase that would otherwise be skipped.
