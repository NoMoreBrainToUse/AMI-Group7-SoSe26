#!/usr/bin/env bash
set -e
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
source .venv/bin/activate

echo "=== Downloading sequences 16-25 ==="
declare -A FILE_IDS=(
  [16]="12C11ksHrYCDhk31UV4CUQ9jT0tppF5eY"
  [17]="1eWhcvAQ3_kxhfyS_kW7NKH-Wa2nWD_Nv"
  [18]="1sbnJP6W4dM-kJbEnbPtv6dJRIbbnxjfi"
  [19]="1RdtAveuQzy2rj4d3YNFFvt3yIZcAVLmo"
  [20]="1MixbIZOcoFloUHPwgS16zdHHruS7uLvs"
  [21]="1JL-SryXd0RIBHlficEpcCXOlD9EgKba5"
  [22]="1fj1LH-K-Js5NZYdginzoMc_p3QPcYEU_"
  [23]="1m1st8ZT4AXFk7K9pdfa0_tLOFPWnz4tG"
  [24]="1rOMxPCBSiMT_HpFwWsR1znmbgwO0JBiz"
  [25]="1avoNPRqodkZ0CBHnfXYY32pAKuL2dz8Q"
)

for seq in 16 17 18 19 20 21 22 23 24 25; do
  if [ -d "dataset/${seq}" ]; then
    echo "seq${seq} already exists, skipping download"
    continue
  fi
  echo "Downloading seq${seq}..."
  gdown "${FILE_IDS[$seq]}" -O "dataset/downloads/${seq}.zip" --quiet
  echo "Extracting seq${seq}..."
  unzip -n -q "dataset/downloads/${seq}.zip" -d "dataset/${seq}"
  rm "dataset/downloads/${seq}.zip"
  echo "seq${seq} done."
done

echo ""
echo "=== Preprocessing v3 dataset ==="
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root processed/fred_subset_v3 \
  --train-seqs 0,1,2,3,4,5,6,7,8,11,12,13,101,102,103,21,22,23,24,25 \
  --val-seqs   9,10,15,19,20 \
  --test-seqs  14,16,17,18,100 \
  --overwrite

echo ""
echo "=== Training Event YOLO on v3 ==="
python src/training/train_event_yolo.py \
  --data processed/fred_subset_v3/event_yolo/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name fred_subset_v3_event_yolo11m \
  --exist-ok

echo ""
echo "=== Training RGB YOLO on v3 ==="
python src/training/train_rgb_yolo.py \
  --data processed/fred_subset_v3/rgb_yolo/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name fred_subset_v3_rgb_yolo11m \
  --exist-ok

echo ""
echo "=== ALL DONE ==="
