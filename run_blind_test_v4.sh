#!/usr/bin/env bash
set -e
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------
MAKE_VIDEO=false
OVERWRITE=false
NO_DOWNLOAD=false
for arg in "$@"; do
  case $arg in
    --make-video)  MAKE_VIDEO=true    ;;
    --overwrite)   OVERWRITE=true     ;;
    --no-download) NO_DOWNLOAD=true   ;;
    --help|-h)
      echo "Usage: $0 [--make-video] [--overwrite] [--no-download]"
      echo ""
      echo "  --make-video   Also render the fusion result video (slow; optional)"
      echo "  --overwrite    Recompute all outputs even if they already exist"
      echo "  --no-download  Skip Google Drive downloads; raise error if data missing"
      exit 0
      ;;
  esac
done

OVERWRITE_FLAG=""
$OVERWRITE && OVERWRITE_FLAG="--overwrite"

# ---------------------------------------------------------------------------
# Optional venv activation (Docker-compatible: skipped if .venv absent)
# ---------------------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# ---------------------------------------------------------------------------
# Phase timing helpers
# ---------------------------------------------------------------------------
_PHASE_START=0
phase_start() {
  echo ""
  echo "=== $* ===  ($(date '+%H:%M:%S'))"
  _PHASE_START=$(date +%s)
}
phase_end() {
  local dur=$(( $(date +%s) - _PHASE_START ))
  echo "  Done in ${dur}s"
}

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
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
WEB_OUT="outputs/web"
VIDEO_OUT="outputs/videos"

# ---------------------------------------------------------------------------
# PHASE 1: Download / verify blind sequences
# ---------------------------------------------------------------------------
phase_start "PHASE 1: Download / verify blind sequences (seq40,43,46,49)"

if $NO_DOWNLOAD; then
  for seq in 40 43 46 49; do
    if [ ! -d "dataset/${seq}" ]; then
      echo "ERROR: dataset/${seq} not found." >&2
      echo "       Mount the dataset or remove --no-download to enable downloads." >&2
      exit 1
    fi
  done
  echo "  --no-download: all sequences verified present"
else
  mkdir -p dataset/downloads
  for seq in 40 43 46 49; do
    if [ -d "dataset/${seq}" ]; then
      echo "  seq${seq} already exists, skipping"
      continue
    fi
    command -v gdown >/dev/null 2>&1 || {
      echo "ERROR: gdown not installed. Run: pip install gdown" >&2; exit 1
    }
    echo "  Downloading seq${seq}..."
    gdown "${FILE_IDS[$seq]}" -O "dataset/downloads/${seq}.zip" --quiet
    unzip -n -q "dataset/downloads/${seq}.zip" -d "dataset/${seq}"
    rm "dataset/downloads/${seq}.zip"
    echo "  seq${seq} done."
  done
fi
phase_end

# ---------------------------------------------------------------------------
# PHASE 2: Preprocess (always runs — fast, creates deterministic outputs)
# ---------------------------------------------------------------------------
phase_start "PHASE 2: Preprocess (all seqs as test)"
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root "$OUTROOT" \
  --train-seqs 40 \
  --val-seqs   43 \
  --test-seqs  46,49 \
  --overwrite
phase_end

# ---------------------------------------------------------------------------
# PHASE 3: Export Event-YOLO proposals
# ---------------------------------------------------------------------------
phase_start "PHASE 3: Export proposals (all splits)"

# Pre-flight checks
if [ ! -f "$EVENT_MODEL" ]; then
  echo "ERROR: event model not found: $EVENT_MODEL" >&2
  echo "       Train it first or check the path." >&2
  exit 1
fi
if [ ! -d "$OUTROOT/event_yolo/images" ]; then
  echo "ERROR: event image directory not found: $OUTROOT/event_yolo/images" >&2
  echo "       Did Phase 2 (preprocessing) succeed?" >&2
  exit 1
fi
if [ ! -d "$OUTROOT/event_yolo/labels" ]; then
  echo "ERROR: event label directory not found: $OUTROOT/event_yolo/labels" >&2
  echo "       Did Phase 2 (preprocessing) succeed?" >&2
  exit 1
fi
mkdir -p "$PROPOSALS"

for SPLIT in train val test; do
  PFILE="$PROPOSALS/detections_conf${CONF}_${SPLIT}.jsonl"
  if [ -f "$PFILE" ] && ! $OVERWRITE; then
    echo "  $SPLIT: $PFILE exists, skipping (--overwrite to recompute)"
    continue
  fi
  echo "  --- proposals $SPLIT ---"
  python src/verifier/export_proposals.py \
    --model         "$EVENT_MODEL" \
    --event-images  "$OUTROOT/event_yolo/images" \
    --event-labels  "$OUTROOT/event_yolo/labels" \
    --split         "$SPLIT" \
    --output        "$PROPOSALS"
done
phase_end

# ---------------------------------------------------------------------------
# PHASE 4: Extract RGB and event crops
# ---------------------------------------------------------------------------
phase_start "PHASE 4: Extract crops"
for SPLIT in train val test; do
  DETS="$PROPOSALS/detections_conf${CONF}_${SPLIT}.jsonl"
  for MOD in rgb event; do
    MANIFEST="$CROPS/${MOD}/crop_manifest_${MOD}_${SPLIT}_conf${CONF}.jsonl"
    if [ -f "$MANIFEST" ] && ! $OVERWRITE; then
      echo "  $MOD $SPLIT: manifest exists, skipping"
      continue
    fi
    python src/verifier/extract_crops.py \
      --detections  "$DETS" \
      --modality    "$MOD" \
      --images-dir  "$OUTROOT/${MOD}_yolo/images" \
      --output-dir  "$CROPS" \
      --split       "$SPLIT"
  done
done
phase_end

# ---------------------------------------------------------------------------
# PHASE 5: Score all crops with v4 verifiers
# ---------------------------------------------------------------------------
phase_start "PHASE 5: Score all crops with v4 verifiers"
mkdir -p "runs/verifier/rgb_v4/blind_v4"

for SPLIT in train val test; do
  for MOD_PAIR in "rgb:$RGB_VER" "event:$EVT_VER"; do
    MOD="${MOD_PAIR%%:*}"
    MODEL="${MOD_PAIR##*:}"
    SCORED="runs/verifier/rgb_v4/blind_v4/scored_${MOD}_${SPLIT}_conf${CONF}.jsonl"
    if [ -f "$SCORED" ] && ! $OVERWRITE; then
      echo "  $MOD $SPLIT: scored file exists, skipping"
      continue
    fi
    MANIFEST="$CROPS/${MOD}/crop_manifest_${MOD}_${SPLIT}_conf${CONF}.jsonl"
    python src/verifier/eval_verifier.py \
      --model    "$MODEL" \
      --manifest "$MANIFEST" \
      --output   "runs/verifier/rgb_v4/blind_v4"
  done
done
phase_end

# ---------------------------------------------------------------------------
# PHASE 6: Fusion metrics + web output
# ---------------------------------------------------------------------------
phase_start "PHASE 6: Fusion metrics + web output"
mkdir -p "$WEB_OUT"
python src/verifier/compute_fusion_metrics.py \
  --scored-dir     "runs/verifier/rgb_v4/blind_v4" \
  --fusion-config  "$FUSION_JSON" \
  --processed-root "$OUTROOT" \
  --splits         train,val,test \
  --conf           "$CONF" \
  --output-dir     "$WEB_OUT" \
  --confusion-plot "outputs/confusion_matrices_blind_test_v4.png" \
  --name           "blind_test_v4" \
  $OVERWRITE_FLAG
phase_end

echo ""
echo "=== WEB OUTPUT (GUI team: use these files) ==="
echo "  Manifest:   $WEB_OUT/fusion_manifest_blind_test_v4.json"
echo "  Detections: $WEB_OUT/fusion_detections_blind_test_v4.jsonl"
echo "  Plot:       outputs/confusion_matrices_blind_test_v4.png"

# ---------------------------------------------------------------------------
# PHASE 7 (optional): Render video
# ---------------------------------------------------------------------------
if $MAKE_VIDEO; then
  phase_start "PHASE 7 (optional): Render video"
  mkdir -p "$VIDEO_OUT"
  python src/visualization/render_fusion_video.py \
    --detections "$WEB_OUT/fusion_detections_blind_test_v4.jsonl" \
    --output     "$VIDEO_OUT/blind_test_v4_fusion_result_compressed.mp4"
  phase_end
  echo "  Video: $VIDEO_OUT/blind_test_v4_fusion_result_compressed.mp4"
else
  echo ""
  echo "(Video not rendered — pass --make-video to render)"
fi

echo ""
echo "=== ALL DONE ==="
