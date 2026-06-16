#!/usr/bin/env bash
set -e
cd /home/simonwesenick/Projects/ami-project
source .venv/bin/activate

# File IDs from Google Drive listing
declare -A FILE_IDS=(
  [50]="1FgGrBusnem8wzMW1AE1m_GHU6prH5Ewc"
  [120]="1rZ7GpDp1W1FDDdo64_IjuVbaVRPnjFX-"
  [150]="1ZFC7kzdUgwCV7ljZVyY2qQ33s0Ow7BwW"
  [200]="1uioY3PEFeyZaAxSoSCNjogmqbEZKFvS_"
)

EVENT_MODEL="runs/event_yolo/fred_subset_v3_event_yolo11m/weights/best.pt"
RGB_VER="runs/verifier/rgb_v3/mobilenet_v3_small/best.pt"
EVT_VER="runs/verifier/rgb_v3/event_mobilenet_v3_small/best.pt"
OUTROOT="processed/fred_blind_test"
PROPOSALS="runs/proposals/fred_blind_test"
CROPS="processed/fred_blind_test/crops"
LAMBDA=1.5
CONF="0.20"

echo "=== PHASE 1: Download sequences ==="
for seq in 50 120 150 200; do
  if [ -d "dataset/${seq}" ]; then
    echo "seq${seq} already exists, skipping"
  else
    echo "Downloading seq${seq}..."
    gdown "${FILE_IDS[$seq]}" -O "dataset/downloads/${seq}.zip" --quiet
    unzip -n -q "dataset/downloads/${seq}.zip" -d "dataset/${seq}"
    rm "dataset/downloads/${seq}.zip"
    echo "seq${seq} done."
  fi
done

echo ""
echo "=== PHASE 2: Preprocess (test-only, no train/val) ==="
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root "$OUTROOT" \
  --train-seqs 50 \
  --val-seqs   120 \
  --test-seqs  150,200 \
  --overwrite

# We treat ALL sequences as "test" via a hack:
# re-run with all four as test by symlinking or just running proposals on each split
# Actually easier: use train/val/test splits then run proposals on all three splits

echo ""
echo "=== PHASE 3: Export proposals (all splits) ==="
for SPLIT in train val test; do
  echo "--- proposals $SPLIT ---"
  python src/verifier/export_proposals.py \
    --model "$EVENT_MODEL" \
    --event-images "$OUTROOT/event_yolo/images" \
    --event-labels "$OUTROOT/event_yolo/labels" \
    --split "$SPLIT" \
    --output "$PROPOSALS"
done

echo ""
echo "=== PHASE 4: Extract crops (all splits) ==="
for SPLIT in train val test; do
  PROPOSALS_FILE="$PROPOSALS/detections_conf${CONF}_${SPLIT}.jsonl"
  python src/verifier/extract_crops.py \
    --detections "$PROPOSALS_FILE" \
    --modality rgb \
    --images-dir "$OUTROOT/rgb_yolo/images" \
    --output-dir "$CROPS" \
    --split "$SPLIT"
  python src/verifier/extract_crops.py \
    --detections "$PROPOSALS_FILE" \
    --modality event \
    --images-dir "$OUTROOT/event_yolo/images" \
    --output-dir "$CROPS" \
    --split "$SPLIT"
done

echo ""
echo "=== PHASE 5: Score all crops with v3 verifiers ==="
for SPLIT in train val test; do
  python src/verifier/eval_verifier.py \
    --model "$RGB_VER" \
    --manifest "$CROPS/rgb/crop_manifest_rgb_${SPLIT}_conf${CONF}.jsonl" \
    --output "runs/verifier/rgb_v3/blind"

  python src/verifier/eval_verifier.py \
    --model "$EVT_VER" \
    --manifest "$CROPS/event/crop_manifest_event_${SPLIT}_conf${CONF}.jsonl" \
    --output "runs/verifier/rgb_v3/blind"
done

echo ""
echo "=== PHASE 6: Fusion + metrics + video ==="
python3 - << 'PYEOF'
import json, math, cv2, numpy as np, re
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path("/home/simonwesenick/Projects/ami-project")
BLIND  = ROOT / "runs/verifier/rgb_v3/blind"
CROPS  = ROOT / "processed/fred_blind_test/crops"
CONF   = "0.20"
LAMBDA = 1.5
THRESH = 1.0

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def sigmoid(x): return 1/(1+math.exp(-x))
def logit(p, eps=1e-7):
    p = max(eps, min(1-eps, p))
    return math.log(p/(1-p))

# Merge all splits into one scored list
all_rgb, all_evt = [], []
for split in ["train", "val", "test"]:
    rp = BLIND / f"scored_rgb_{split}_conf{CONF}.jsonl"
    ep = BLIND / f"scored_event_{split}_conf{CONF}.jsonl"
    if rp.exists() and ep.exists():
        all_rgb += load_jsonl(rp)
        all_evt += load_jsonl(ep)

print(f"Total detections: {len(all_rgb)}")

# Compute fusion scores
stem_dets = defaultdict(list)
for r, e in zip(all_rgb, all_evt):
    fs = sigmoid(logit(r["verifier_score"]) + LAMBDA * logit(e["verifier_score"]))
    stem_dets[r["stem"]].append({
        "bbox": r["bbox_xyxy"], "fusion_score": fs, "kept": fs >= THRESH,
        "det_score": r["detector_score"], "label": r["label"],
        "iou": r["iou_with_gt"],
    })

# --- Metrics ---
def compute_cm(dets_flat, use_kept=False, thresh=0.5):
    TP=FP=FN=0
    for d in dets_flat:
        pred = d["kept"] if use_kept else d["det_score"] >= thresh
        gt   = d["label"] == 1
        if pred and gt:   TP += 1
        elif pred and not gt: FP += 1
        elif not pred and gt: FN += 1
    prec = TP/(TP+FP) if TP+FP else 0
    rec  = TP/(TP+FN) if TP+FN else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return {"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1}

flat = [d for dets in stem_dets.values() for d in dets]
systems = {
    "Event YOLO\n(conf≥0.25)":            compute_cm(flat, use_kept=False, thresh=0.25),
    "Event YOLO\n(conf≥0.50)":            compute_cm(flat, use_kept=False, thresh=0.50),
    "RGB Verifier\nonly":                  compute_cm([{**d,"kept":d["det_score"]>=0.5} for d in flat], use_kept=True),
    f"Fusion\n(λ={LAMBDA}, t={THRESH})":  compute_cm(flat, use_kept=True),
}

fig, axes = plt.subplots(1, 4, figsize=(18,5))
fig.suptitle("Blind Test — seq50, seq120, seq150, seq200 (never seen during training)", fontsize=12, fontweight="bold")
for ax, (name, s) in zip(axes, systems.items()):
    mat = np.array([[s["TP"], s["FN"]], [s["FP"], 0]])
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred Pos","Pred Neg"])
    ax.set_yticklabels(["GT Pos","GT Neg"])
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(mat[i,j]),ha="center",va="center",
                    fontsize=13,color="white" if mat[i,j]>mat.max()*0.5 else "black")
    ax.set_title(f"{name}\nPrec={s['precision']:.1%} Rec={s['recall']:.1%} F1={s['f1']:.1%}", fontsize=9)
plt.tight_layout()
out_cm = ROOT/"outputs/confusion_matrices_blind_test.png"
plt.savefig(out_cm, dpi=150, bbox_inches="tight")
print(f"Confusion matrices: {out_cm}")

print("\n=== Blind Test Results ===")
for name, s in systems.items():
    n = name.replace('\n',' ')
    print(f"  {n:<38} TP={s['TP']:5d} FP={s['FP']:4d} FN={s['FN']:4d}  "
          f"Prec={s['precision']:.1%} Rec={s['recall']:.1%} F1={s['f1']:.1%}")

# --- Result video ---
GT_COLOR   = (40, 220, 40)
KEPT_COLOR = (20, 200, 255)
REJ_COLOR  = (0, 0, 220)
rgb_dir    = ROOT / "processed/fred_blind_test/rgb_yolo/images"
label_dir  = ROOT / "processed/fred_blind_test/event_yolo/labels"

def read_gt(p, W, H):
    if not p.is_file(): return []
    boxes = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5: continue
        _, cx,cy,bw,bh = map(float, parts[:5])
        boxes.append((int((cx-bw/2)*W), int((cy-bh/2)*H),
                      int((cx+bw/2)*W), int((cy+bh/2)*H)))
    return boxes

def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)]

all_imgs = []
for split in ["train","val","test"]:
    all_imgs += list((rgb_dir/split).glob("*.jpg"))
all_imgs = sorted(all_imgs, key=natural_key)
print(f"\nVideo frames: {len(all_imgs)}")

out_raw = ROOT/"outputs/blind_test_fusion.mp4"
first = cv2.imread(str(all_imgs[0]))
H, W = first.shape[:2]
writer = cv2.VideoWriter(str(out_raw), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

for img_path in all_imgs:
    frame = cv2.imread(str(img_path))
    if frame is None: continue
    stem = img_path.stem
    split = img_path.parent.name

    for det in stem_dets.get(stem, []):
        x1,y1,x2,y2 = [int(v) for v in det["bbox"]]
        color = KEPT_COLOR if det["kept"] else REJ_COLOR
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        if det["kept"]:
            cv2.putText(frame,f"DRONE {det['fusion_score']:.2f}",
                        (x1,max(y1-4,12)),cv2.FONT_HERSHEY_SIMPLEX,0.5,KEPT_COLOR,1,cv2.LINE_AA)

    for x1,y1,x2,y2 in read_gt((label_dir/split/(stem+".txt")), W, H):
        cv2.rectangle(frame,(x1,y1),(x2,y2),GT_COLOR,2)

    seq_label = stem.split("_")[0]
    cv2.putText(frame, seq_label, (W-90,H-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1,cv2.LINE_AA)
    writer.write(frame)

writer.release()

import subprocess, os
out_comp = ROOT/"outputs/blind_test_fusion_compressed.mp4"
subprocess.run(["ffmpeg","-y","-i",str(out_raw),
                "-vcodec","libx264","-crf","28","-preset","fast",
                str(out_comp)], check=True, capture_output=True)
out_raw.unlink()
print(f"Video: {out_comp} ({os.path.getsize(out_comp)//1024//1024} MB)")
PYEOF

echo ""
echo "=== ALL DONE ==="
