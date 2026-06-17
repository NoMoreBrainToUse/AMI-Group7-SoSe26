#!/usr/bin/env bash
set -e
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
source .venv/bin/activate

declare -A FILE_IDS=(
  [40]="1W4OmPq8UQB-HVDBwRcRXNN5CLh4cE4AJ"
  [43]="18ltdRfkG0PJ1d76ZD1Kirr3YZQ7-bKRY"
  [46]="19X9QIaBO_z-MXxcu0blep9Il1fiZkowu"
  [49]="12rBd8eRLPw07WIRKnu_LY2AeeP46FAzU"
)

EVENT_MODEL="runs/event_yolo/fred_subset_v3_event_yolo11m/weights/best.pt"
RGB_VER="runs/verifier/rgb_v4/efficientnet_b0/best.pt"
EVT_VER="runs/verifier/rgb_v4/event_efficientnet_b0/best.pt"
FUSION_JSON="runs/verifier/rgb_v4/fusion_results_rgb_val_conf0.20.json"
OUTROOT="processed/fred_blind_test_v4"
PROPOSALS="runs/proposals/fred_blind_test_v4"
CROPS="processed/fred_blind_test_v4/crops"
CONF="0.20"

echo "=== PHASE 1: Download blind sequences (seq40,43,46,49) ==="
for seq in 40 43 46 49; do
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
echo "=== PHASE 2: Preprocess (all seqs as test) ==="
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root "$OUTROOT" \
  --train-seqs 40 \
  --val-seqs   43 \
  --test-seqs  46,49 \
  --overwrite

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
echo "=== PHASE 4: Extract crops ==="
for SPLIT in train val test; do
  DETS="$PROPOSALS/detections_conf${CONF}_${SPLIT}.jsonl"
  python src/verifier/extract_crops.py \
    --detections "$DETS" --modality rgb \
    --images-dir "$OUTROOT/rgb_yolo/images" \
    --output-dir "$CROPS" --split "$SPLIT"
  python src/verifier/extract_crops.py \
    --detections "$DETS" --modality event \
    --images-dir "$OUTROOT/event_yolo/images" \
    --output-dir "$CROPS" --split "$SPLIT"
done

echo ""
echo "=== PHASE 5: Score all crops with v4 verifiers ==="
for SPLIT in train val test; do
  python src/verifier/eval_verifier.py \
    --model "$RGB_VER" \
    --manifest "$CROPS/rgb/crop_manifest_rgb_${SPLIT}_conf${CONF}.jsonl" \
    --output "runs/verifier/rgb_v4/blind_v4"
  python src/verifier/eval_verifier.py \
    --model "$EVT_VER" \
    --manifest "$CROPS/event/crop_manifest_event_${SPLIT}_conf${CONF}.jsonl" \
    --output "runs/verifier/rgb_v4/blind_v4"
done

echo ""
echo "=== PHASE 6: Fusion metrics + video ==="
python3 - << 'PYEOF'
import json, math, cv2, numpy as np, re, subprocess, os
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT         = Path(os.environ["SCRIPT_DIR"])
VERIFIER_OUT = ROOT / "runs/verifier/rgb_v4"
BLIND_OUT    = VERIFIER_OUT / "blind_v4"
CROPS        = ROOT / "processed/fred_blind_test_v4/crops"
CONF         = "0.20"

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def sigmoid(x): return 1/(1+math.exp(-x))
def logit(p, eps=1e-7):
    p = max(eps, min(1-eps, p))
    return math.log(p/(1-p))

fusion_cfg = json.loads((VERIFIER_OUT/"fusion_results_rgb_val_conf0.20.json").read_text())
LAMBDA = float(fusion_cfg["best_lambda"])
THRESH = float(fusion_cfg["best_threshold"])
print(f"Using λ={LAMBDA}, threshold={THRESH:.4f}")

# Collect all splits
all_rgb, all_evt = [], []
for split in ["train","val","test"]:
    rgb_f = BLIND_OUT / f"scored_rgb_{split}_conf{CONF}.jsonl"
    evt_f = BLIND_OUT / f"scored_event_{split}_conf{CONF}.jsonl"
    if rgb_f.exists() and evt_f.exists():
        all_rgb += load_jsonl(rgb_f)
        all_evt += load_jsonl(evt_f)

print(f"Total detections across all splits: {len(all_rgb)}")

stem_dets = defaultdict(list)
for r, e in zip(all_rgb, all_evt):
    fs = sigmoid(logit(r["verifier_score"]) + LAMBDA * logit(e["verifier_score"]))
    stem_dets[r["stem"]].append({
        "bbox": r["bbox_xyxy"], "fusion_score": fs, "kept": fs >= THRESH,
        "det_score": r["detector_score"], "label": r["label"],
    })

flat = [d for dets in stem_dets.values() for d in dets]

def compute_cm(dets, use_kept=False, thresh=0.5):
    TP=FP=FN=0
    for d in dets:
        pred = d["kept"] if use_kept else d["det_score"] >= thresh
        gt   = d["label"] == 1
        if pred and gt:       TP += 1
        elif pred and not gt: FP += 1
        elif not pred and gt: FN += 1
    prec = TP/(TP+FP) if TP+FP else 0
    rec  = TP/(TP+FN) if TP+FN else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return {"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1}

systems = {
    "Event YOLO\n(conf≥0.25)":               compute_cm(flat, use_kept=False, thresh=0.25),
    "Event YOLO\n(conf≥0.50)":               compute_cm(flat, use_kept=False, thresh=0.50),
    f"Fusion v4\n(λ={LAMBDA}, t={THRESH:.3f})": compute_cm(flat, use_kept=True),
}

fig, axes = plt.subplots(1, 3, figsize=(14,5))
fig.suptitle("Blind Test v4 — seq40,43,46,49 (NEVER seen during training)", fontsize=13, fontweight="bold")
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
    ax.set_title(f"{name}\nPrec={s['precision']:.1%} Rec={s['recall']:.1%} F1={s['f1']:.1%}",fontsize=9)
plt.tight_layout()
out_cm = ROOT/"outputs/confusion_matrices_blind_test_v4.png"
plt.savefig(out_cm, dpi=150, bbox_inches="tight")
print(f"Confusion matrices: {out_cm}")

print("\n=== Blind Test v4 Results (seq40,43,46,49) ===")
for name, s in systems.items():
    n = name.replace('\n',' ')
    print(f"  {n:<45} TP={s['TP']:5d} FP={s['FP']:4d} FN={s['FN']:4d}  "
          f"Prec={s['precision']:.1%} Rec={s['recall']:.1%} F1={s['f1']:.1%}")

# Build video from RGB images
GT_COLOR=(40,220,40); KEPT_COLOR=(20,200,255); REJ_COLOR=(0,0,220)
rgb_dir   = ROOT/"processed/fred_blind_test_v4/rgb_yolo/images"
label_dir = ROOT/"processed/fred_blind_test_v4/event_yolo/labels"

def read_gt(p, W, H):
    if not p.is_file(): return []
    boxes=[]
    for line in p.read_text().splitlines():
        parts=line.split()
        if len(parts)<5: continue
        _,cx,cy,bw,bh=map(float,parts[:5])
        boxes.append((int((cx-bw/2)*W),int((cy-bh/2)*H),
                      int((cx+bw/2)*W),int((cy+bh/2)*H)))
    return boxes

def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)',p.name)]

all_imgs = []
for split in ["train","val","test"]:
    all_imgs += list((rgb_dir/split).glob("*.jpg"))
all_imgs = sorted(all_imgs, key=natural_key)
print(f"\nVideo frames: {len(all_imgs)}")

first = cv2.imread(str(all_imgs[0]))
H, W = first.shape[:2]
out_raw = ROOT/"outputs/blind_test_v4_fusion_result.mp4"
writer = cv2.VideoWriter(str(out_raw), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W,H))

for img_path in all_imgs:
    frame = cv2.imread(str(img_path))
    if frame is None: continue
    stem = img_path.stem
    split = img_path.parent.name
    for det in stem_dets.get(stem,[]):
        x1,y1,x2,y2=[int(v) for v in det["bbox"]]
        color = KEPT_COLOR if det["kept"] else REJ_COLOR
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        if det["kept"]:
            cv2.putText(frame,f"DRONE {det['fusion_score']:.2f}",
                        (x1,max(y1-4,12)),cv2.FONT_HERSHEY_SIMPLEX,0.5,KEPT_COLOR,1,cv2.LINE_AA)
    lbl_path = label_dir/split/(stem+".txt")
    for x1,y1,x2,y2 in read_gt(lbl_path,W,H):
        cv2.rectangle(frame,(x1,y1),(x2,y2),GT_COLOR,2)
    cv2.putText(frame,stem.split("_")[0],(W-90,H-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1,cv2.LINE_AA)
    writer.write(frame)
writer.release()

out_comp = ROOT/"outputs/blind_test_v4_fusion_result_compressed.mp4"
subprocess.run(["ffmpeg","-y","-i",str(out_raw),
                "-vcodec","libx264","-crf","28","-preset","fast",
                str(out_comp)], check=True, capture_output=True)
out_raw.unlink()
print(f"Video: {out_comp} ({os.path.getsize(out_comp)//1024//1024} MB)")
PYEOF

echo ""
echo "=== ALL DONE ==="
