#!/usr/bin/env bash
set -e
cd /home/simonwesenick/Projects/ami-project
source .venv/bin/activate

declare -A FILE_IDS=(
  [30]="1SlOmHo1SVo5HTiG_Cvz9Ss5jvk_dkor9"
  [31]="1vFVfDlSxWtMJS9ybyB2wK1I044A_uS6k"
  [32]="1mDiTt0h3cfPOhFZgIygJhh9ip01yAjPL"
  [33]="18QZGgU6XyMK9ROII0H6P80_srXJWWVdf"
  [34]="1JhKpE3Fq1YRH_Ao6M7nfO7ZPW5HlawWB"
  [35]="1Y7MAGqdInOgRWCrsNgNZmr_Zv1uhDlrv"
  [36]="10gDpV6o_07IqyZmxe_D-VXwy1h76Ep08"
  [37]="1MpJGzPzg7YPx4Bt9LTmmWXzqxvhdcZtT"
  [38]="1SDLKYteSLrf6LE0zEcIKwivLLCEl-s-7"
  [39]="1uS1_JwH8EYbKzfYsd00k3hTDo9ITCi9p"
)

EVENT_MODEL="runs/event_yolo/fred_subset_v3_event_yolo11m/weights/best.pt"
OUTROOT="processed/fred_subset_v4"
PROPOSALS="runs/proposals/fred_subset_v4"
CROPS="processed/fred_subset_v4/crops"
VERIFIER_OUT="runs/verifier/rgb_v4"
CONF="0.20"

echo "=== PHASE 1: Download seq30-39 ==="
for seq in 30 31 32 33 34 35 36 37 38 39; do
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
echo "=== PHASE 2: Preprocess v4 dataset ==="
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root "$OUTROOT" \
  --train-seqs 0,1,2,3,4,5,6,7,8,11,12,13,21,22,23,24,25,30,31,32,33,34,35,36,37,38,39,50,101,102,103,120 \
  --val-seqs   9,10,15,19,20,150 \
  --test-seqs  14,16,17,18,100,200 \
  --overwrite

echo ""
echo "=== PHASE 3: Export proposals (train / val / test) ==="
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
echo "=== PHASE 5: Train RGB verifier (EfficientNet-B0, 50 epochs) ==="
python src/verifier/train_rgb_verifier.py \
  --train-manifest "$CROPS/rgb/crop_manifest_rgb_train_conf${CONF}.jsonl" \
  --val-manifest   "$CROPS/rgb/crop_manifest_rgb_val_conf${CONF}.jsonl" \
  --output "$VERIFIER_OUT" --name efficientnet_b0 --epochs 50

echo ""
echo "=== PHASE 6: Train Event verifier (EfficientNet-B0, 50 epochs) ==="
python src/verifier/train_rgb_verifier.py \
  --train-manifest "$CROPS/event/crop_manifest_event_train_conf${CONF}.jsonl" \
  --val-manifest   "$CROPS/event/crop_manifest_event_val_conf${CONF}.jsonl" \
  --output "$VERIFIER_OUT" --name event_efficientnet_b0 --epochs 50

echo ""
echo "=== PHASE 7: Score val + test crops ==="
for SPLIT in val test; do
  python src/verifier/eval_verifier.py \
    --model "$VERIFIER_OUT/efficientnet_b0/best.pt" \
    --manifest "$CROPS/rgb/crop_manifest_rgb_${SPLIT}_conf${CONF}.jsonl" \
    --output "$VERIFIER_OUT/efficientnet_b0"
  python src/verifier/eval_verifier.py \
    --model "$VERIFIER_OUT/event_efficientnet_b0/best.pt" \
    --manifest "$CROPS/event/crop_manifest_event_${SPLIT}_conf${CONF}.jsonl" \
    --output "$VERIFIER_OUT/event_efficientnet_b0"
done

echo ""
echo "=== PHASE 8: Find best lambda on val ==="
python src/verifier/eval_fusion.py \
  --rgb-scored   "$VERIFIER_OUT/efficientnet_b0/scored_rgb_val_conf${CONF}.jsonl" \
  --event-scored "$VERIFIER_OUT/event_efficientnet_b0/scored_event_val_conf${CONF}.jsonl" \
  --output       "$VERIFIER_OUT"

echo ""
echo "=== PHASE 9: Confusion matrices + video ==="
python3 - << 'PYEOF'
import json, math, cv2, numpy as np, re, subprocess, os
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT         = Path("/home/simonwesenick/Projects/ami-project")
VERIFIER_OUT = ROOT / "runs/verifier/rgb_v4"
CROPS        = ROOT / "processed/fred_subset_v4/crops"
CONF         = "0.20"

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def sigmoid(x): return 1/(1+math.exp(-x))
def logit(p, eps=1e-7):
    p = max(eps, min(1-eps, p))
    return math.log(p/(1-p))

fusion_file = next(VERIFIER_OUT.glob("fusion_results_rgb_val_conf*.json"))
val_res     = json.loads(fusion_file.read_text())
LAMBDA      = float(val_res["best_lambda"])
THRESH      = float(val_res["best_threshold"])
print(f"Best lambda={LAMBDA}, threshold={THRESH:.4f}")

rgb_test = load_jsonl(VERIFIER_OUT/"efficientnet_b0/scored_rgb_test_conf0.20.jsonl")
evt_test = load_jsonl(VERIFIER_OUT/"event_efficientnet_b0/scored_event_test_conf0.20.jsonl")

stem_dets = defaultdict(list)
for r, e in zip(rgb_test, evt_test):
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
    "Event YOLO\n(conf≥0.25)":           compute_cm(flat, use_kept=False, thresh=0.25),
    "Event YOLO\n(conf≥0.50)":           compute_cm(flat, use_kept=False, thresh=0.50),
    "EfficientNet-B0\nRGB only":         compute_cm([{**d,"kept":d["det_score"]>=0.5} for d in flat], True),
    f"Fusion\n(λ={LAMBDA}, t={THRESH:.3f})": compute_cm(flat, use_kept=True),
}

fig, axes = plt.subplots(1, 4, figsize=(18,5))
fig.suptitle("v4 Test Set — seq14,16,17,18,100,200", fontsize=13, fontweight="bold")
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
out_cm = ROOT/"outputs/confusion_matrices_test_v4.png"
plt.savefig(out_cm, dpi=150, bbox_inches="tight")
print(f"Confusion matrices: {out_cm}")

print("\n=== v4 Test Results ===")
for name, s in systems.items():
    n = name.replace('\n',' ')
    print(f"  {n:<40} TP={s['TP']:5d} FP={s['FP']:4d} FN={s['FN']:4d}  "
          f"Prec={s['precision']:.1%} Rec={s['recall']:.1%} F1={s['f1']:.1%}")

# Video
GT_COLOR=(40,220,40); KEPT_COLOR=(20,200,255); REJ_COLOR=(0,0,220)
rgb_dir   = ROOT/"processed/fred_subset_v4/rgb_yolo/images/test"
label_dir = ROOT/"processed/fred_subset_v4/event_yolo/labels/test"

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

all_imgs = sorted(rgb_dir.glob("*.jpg"), key=natural_key)
first = cv2.imread(str(all_imgs[0]))
H, W = first.shape[:2]
out_raw = ROOT/"outputs/test_v4_fusion_result.mp4"
writer = cv2.VideoWriter(str(out_raw), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W,H))

for img_path in all_imgs:
    frame = cv2.imread(str(img_path))
    if frame is None: continue
    stem = img_path.stem
    for det in stem_dets.get(stem,[]):
        x1,y1,x2,y2=[int(v) for v in det["bbox"]]
        color = KEPT_COLOR if det["kept"] else REJ_COLOR
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        if det["kept"]:
            cv2.putText(frame,f"DRONE {det['fusion_score']:.2f}",
                        (x1,max(y1-4,12)),cv2.FONT_HERSHEY_SIMPLEX,0.5,KEPT_COLOR,1,cv2.LINE_AA)
    for x1,y1,x2,y2 in read_gt(label_dir/(stem+".txt"),W,H):
        cv2.rectangle(frame,(x1,y1),(x2,y2),GT_COLOR,2)
    cv2.putText(frame,stem.split("_")[0],(W-90,H-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1,cv2.LINE_AA)
    writer.write(frame)
writer.release()

out_comp = ROOT/"outputs/test_v4_fusion_result_compressed.mp4"
subprocess.run(["ffmpeg","-y","-i",str(out_raw),
                "-vcodec","libx264","-crf","28","-preset","fast",
                str(out_comp)], check=True, capture_output=True)
out_raw.unlink()
print(f"Video: {out_comp} ({os.path.getsize(out_comp)//1024//1024} MB)")
PYEOF

echo ""
echo "=== ALL DONE ==="
