#!/usr/bin/env bash
set -e
cd /home/simonwesenick/Projects/ami-project
source .venv/bin/activate

EVENT_MODEL="runs/event_yolo/fred_subset_v3_event_yolo11m/weights/best.pt"
DATA="processed/fred_subset_v3"
PROPOSALS="runs/proposals/fred_subset_v3"
CROPS="processed/fred_subset_v3/crops"
VERIFIER_OUT="runs/verifier/rgb_v3"
CONF="0.20"

echo "============================================"
echo "PHASE 1: Export proposals (skipped - already done)"
echo "============================================"
echo "Proposals already exist in $PROPOSALS"

echo ""
echo "============================================"
echo "PHASE 2: Extract crops (RGB + Event, all splits)"
echo "============================================"
for SPLIT in train val test; do
  # export_proposals names files: detections_conf0.20_train.jsonl
  # extract_crops appends split internally — pass parent dirs only
  PROPOSALS_FILE="$PROPOSALS/detections_conf${CONF}_${SPLIT}.jsonl"
  echo "--- RGB crops $SPLIT ---"
  python src/verifier/extract_crops.py \
    --detections "$PROPOSALS_FILE" \
    --modality rgb \
    --images-dir "$DATA/rgb_yolo/images" \
    --output-dir "$CROPS" \
    --split "$SPLIT"

  echo "--- Event crops $SPLIT ---"
  python src/verifier/extract_crops.py \
    --detections "$PROPOSALS_FILE" \
    --modality event \
    --images-dir "$DATA/event_yolo/images" \
    --output-dir "$CROPS" \
    --split "$SPLIT"
done

echo ""
echo "============================================"
echo "PHASE 3: Train RGB verifier"
echo "============================================"
python src/verifier/train_rgb_verifier.py \
  --train-manifest "$CROPS/rgb/crop_manifest_rgb_train_conf${CONF}.jsonl" \
  --val-manifest   "$CROPS/rgb/crop_manifest_rgb_val_conf${CONF}.jsonl" \
  --output "$VERIFIER_OUT" \
  --name mobilenet_v3_small \
  --epochs 30

echo ""
echo "============================================"
echo "PHASE 4: Train Event verifier"
echo "============================================"
python src/verifier/train_rgb_verifier.py \
  --train-manifest "$CROPS/event/crop_manifest_event_train_conf${CONF}.jsonl" \
  --val-manifest   "$CROPS/event/crop_manifest_event_val_conf${CONF}.jsonl" \
  --output "$VERIFIER_OUT" \
  --name event_mobilenet_v3_small \
  --epochs 30

echo ""
echo "============================================"
echo "PHASE 5: Score val crops (to find best lambda)"
echo "============================================"
python src/verifier/eval_verifier.py \
  --model "$VERIFIER_OUT/mobilenet_v3_small/best.pt" \
  --manifest "$CROPS/rgb/crop_manifest_rgb_val_conf${CONF}.jsonl" \
  --output "$VERIFIER_OUT/mobilenet_v3_small"

python src/verifier/eval_verifier.py \
  --model "$VERIFIER_OUT/event_mobilenet_v3_small/best.pt" \
  --manifest "$CROPS/event/crop_manifest_event_val_conf${CONF}.jsonl" \
  --output "$VERIFIER_OUT/event_mobilenet_v3_small"

echo ""
echo "============================================"
echo "PHASE 6: Score TEST crops"
echo "============================================"
python src/verifier/eval_verifier.py \
  --model "$VERIFIER_OUT/mobilenet_v3_small/best.pt" \
  --manifest "$CROPS/rgb/crop_manifest_rgb_test_conf${CONF}.jsonl" \
  --output "$VERIFIER_OUT/mobilenet_v3_small"

python src/verifier/eval_verifier.py \
  --model "$VERIFIER_OUT/event_mobilenet_v3_small/best.pt" \
  --manifest "$CROPS/event/crop_manifest_event_test_conf${CONF}.jsonl" \
  --output "$VERIFIER_OUT/event_mobilenet_v3_small"

echo ""
echo "============================================"
echo "PHASE 7: Find best lambda on val, then apply to test"
echo "============================================"
python src/verifier/eval_fusion.py \
  --rgb-scored   "$VERIFIER_OUT/mobilenet_v3_small/scored_rgb_val_conf${CONF}.jsonl" \
  --event-scored "$VERIFIER_OUT/event_mobilenet_v3_small/scored_event_val_conf${CONF}.jsonl" \
  --output       "$VERIFIER_OUT"

echo ""
echo "============================================"
echo "PHASE 8: Confusion matrices + fusion video"
echo "============================================"
python3 - << 'PYEOF'
import json, math, cv2, numpy as np, re
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/simonwesenick/Projects/ami-project")
VERIFIER_OUT = ROOT / "runs/verifier/rgb_v3"
CROPS = ROOT / "processed/fred_subset_v3/crops"
CONF = "0.20"

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def sigmoid(x): return 1/(1+math.exp(-x))
def logit(p, eps=1e-7):
    p = max(eps, min(1-eps, p))
    return math.log(p/(1-p))

# Find best lambda/threshold from val fusion results
# eval_fusion names it: fusion_results_rgb_val_conf0.20.json
fusion_file = next(VERIFIER_OUT.glob("fusion_results_rgb_val_conf*.json"))
val_results = json.loads(fusion_file.read_text())
best_lambda = float(val_results["best_lambda"])
best_thresh = float(val_results["best_threshold"])
print(f"Best lambda={best_lambda}, threshold={best_thresh:.4f}")

# Load test scored data
rgb_test  = load_jsonl(VERIFIER_OUT/"mobilenet_v3_small/scored_rgb_test_conf0.20.jsonl")
evt_test  = load_jsonl(VERIFIER_OUT/"event_mobilenet_v3_small/scored_event_test_conf0.20.jsonl")

# Build per-detection fusion scores for test
test_dets = []
for r, e in zip(rgb_test, evt_test):
    fs = sigmoid(logit(r["verifier_score"]) + best_lambda * logit(e["verifier_score"]))
    test_dets.append({
        "stem": r["stem"], "bbox": r["bbox_xyxy"],
        "det_score": r["detector_score"], "iou": r["iou_with_gt"],
        "label": r["label"], "fusion_score": fs,
        "kept": fs >= best_thresh,
        "rgb_score": r["verifier_score"], "evt_score": e["verifier_score"],
    })

# ---- Confusion matrices (4 systems) ----
def compute_cm(dets, use_kept=False, thresh=0.5):
    TP=FP=FN=0
    for d in dets:
        if use_kept:
            pred_pos = d["kept"]
        else:
            pred_pos = d["det_score"] >= thresh
        gt_pos = d["label"] == 1
        if pred_pos and gt_pos:  TP += 1
        elif pred_pos and not gt_pos: FP += 1
        elif not pred_pos and gt_pos: FN += 1
    prec = TP/(TP+FP) if TP+FP>0 else 0
    rec  = TP/(TP+FN) if TP+FN>0 else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    return {"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1}

systems = {
    "Event YOLO\n(conf≥0.25)": compute_cm(test_dets, use_kept=False, thresh=0.25),
    "Event YOLO\n(conf≥0.50)": compute_cm(test_dets, use_kept=False, thresh=0.50),
    "RGB Verifier\nonly":       compute_cm(
        [{**d,"kept": d["rgb_score"]>=0.5} for d in test_dets], use_kept=True),
    f"Fusion\n(λ={best_lambda}, t={best_thresh:.3f})": compute_cm(test_dets, use_kept=True),
}

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("v3 Test Set — Confusion Matrices (seq14,16,17,18,100)", fontsize=13, fontweight="bold")
for ax, (name, s) in zip(axes, systems.items()):
    mat = np.array([[s["TP"], s["FN"]], [s["FP"], 0]])
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred Pos","Pred Neg"])
    ax.set_yticklabels(["GT Pos","GT Neg"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i,j]), ha="center", va="center",
                    fontsize=14, color="white" if mat[i,j]>mat.max()*0.5 else "black")
    ax.set_title(f"{name}\nPrec={s['precision']:.2%} Rec={s['recall']:.2%} F1={s['f1']:.2%}",
                 fontsize=9)
plt.tight_layout()
out_cm = ROOT/"outputs/confusion_matrices_test_v3.png"
plt.savefig(out_cm, dpi=150, bbox_inches="tight")
print(f"Confusion matrices saved: {out_cm}")

# Print summary table
print("\n=== Test Set Results (v3) ===")
for name, s in systems.items():
    n = name.replace('\n',' ')
    print(f"  {n:<35} TP={s['TP']:4d} FP={s['FP']:4d} FN={s['FN']:3d} "
          f"Prec={s['precision']:.1%} Rec={s['recall']:.1%} F1={s['f1']:.1%}")

PYEOF

echo ""
echo "============================================"
echo "PHASE 9: Fusion result video (RGB background + event boxes)"
echo "============================================"
python3 - << 'PYEOF'
import json, math, cv2, numpy as np, re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/simonwesenick/Projects/ami-project")
VERIFIER_OUT = ROOT / "runs/verifier/rgb_v3"

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def sigmoid(x): return 1/(1+math.exp(-x))
def logit(p, eps=1e-7):
    p = max(eps, min(1-eps, p))
    return math.log(p/(1-p))

fusion_file = next(VERIFIER_OUT.glob("fusion_results_rgb_val_conf*.json"))
val_results = json.loads(fusion_file.read_text())
best_lambda = float(val_results["best_lambda"])
best_thresh = float(val_results["best_threshold"])

rgb_test = load_jsonl(VERIFIER_OUT/"mobilenet_v3_small/scored_rgb_test_conf0.20.jsonl")
evt_test = load_jsonl(VERIFIER_OUT/"event_mobilenet_v3_small/scored_event_test_conf0.20.jsonl")

stem_dets = defaultdict(list)
for r, e in zip(rgb_test, evt_test):
    fs = sigmoid(logit(r["verifier_score"]) + best_lambda * logit(e["verifier_score"]))
    stem_dets[r["stem"]].append({
        "bbox": r["bbox_xyxy"], "fusion_score": fs, "kept": fs >= best_thresh,
        "det_score": r["detector_score"],
    })

rgb_dir   = ROOT/"processed/fred_subset_v3/rgb_yolo/images/test"
label_dir = ROOT/"processed/fred_subset_v3/event_yolo/labels/test"

GT_COLOR   = (40, 220, 40)
KEPT_COLOR = (20, 200, 255)
REJ_COLOR  = (0, 0, 220)

def read_gt(p, W, H):
    if not p.is_file(): return []
    boxes = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5: continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        boxes.append((int((cx-bw/2)*W), int((cy-bh/2)*H),
                      int((cx+bw/2)*W), int((cy+bh/2)*H)))
    return boxes

def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)]

all_stems = sorted(rgb_dir.glob("*.jpg"), key=natural_key)
print(f"Test frames: {len(all_stems)} across seqs {set(p.stem.split('_')[0] for p in all_stems)}")

out_path = ROOT/"outputs/test_v3_fusion_result.mp4"
out_path.parent.mkdir(exist_ok=True)

first = cv2.imread(str(all_stems[0]))
H, W = first.shape[:2]
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

for img_path in all_stems:
    frame = cv2.imread(str(img_path))
    if frame is None: continue
    stem = img_path.stem

    for det in stem_dets.get(stem, []):
        x1,y1,x2,y2 = [int(v) for v in det["bbox"]]
        color = KEPT_COLOR if det["kept"] else REJ_COLOR
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        if det["kept"]:
            cv2.putText(frame, f"DRONE {det['fusion_score']:.2f}",
                        (x1, max(y1-4,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEPT_COLOR, 1, cv2.LINE_AA)

    for x1,y1,x2,y2 in read_gt(label_dir/(stem+".txt"), W, H):
        cv2.rectangle(frame, (x1,y1),(x2,y2), GT_COLOR, 2)

    seq_label = stem.split("_")[0]
    cv2.putText(frame, seq_label, (W-90, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
    writer.write(frame)

writer.release()
print(f"Raw video: {out_path}")

# Compress
import subprocess
compressed = ROOT/"outputs/test_v3_fusion_result_compressed.mp4"
subprocess.run([
    "ffmpeg", "-y", "-i", str(out_path),
    "-vcodec", "libx264", "-crf", "28", "-preset", "fast",
    "-acodec", "copy", str(compressed)
], check=True, capture_output=True)
out_path.unlink()
import os
print(f"Compressed: {compressed} ({os.path.getsize(compressed)//1024//1024} MB)")
PYEOF

echo ""
echo "=== ALL DONE ==="
