#!/usr/bin/env bash
set -e
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------
SEQS="49"
DATASET_DIR="dataset"
MAKE_VIDEO=false
NO_DOWNLOAD=false
PLOTS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --seqs)        SEQS="$2";        shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --make-video)  MAKE_VIDEO=true;  shift   ;;
    --no-download) NO_DOWNLOAD=true; shift   ;;
    --plots)       PLOTS=true;       shift   ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "  --seqs <seq1,seq2,...>   Sequences to evaluate (default: 49)"
      echo "  --dataset-dir <path>     Dataset root directory (default: dataset)"
      echo "  --make-video             Also render the fusion result video (slow; optional)"
      echo "  --no-download            Skip Google Drive downloads; raise error if data missing"
      echo "  --plots                  Generate 6-panel curve plot (F1/Precision/Recall/PR/CM)"
      echo ""
      echo "Examples:"
      echo "  $0                                    # run on seq 49 (default)"
      echo "  $0 --seqs 46,49                       # run on seq 46 and 49"
      echo "  $0 --seqs 40,43,46,49                 # reproduce old all-sequence output"
      echo "  $0 --seqs 49 --no-download            # skip download, data already on disk"
      echo "  $0 --seqs 49 --dataset-dir /mnt/data --no-download"
      echo "  $0 --seqs 46,49 --plots               # include 6-panel evaluation plot"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Always overwrite — output paths are fixed, so cached results from a
# different sequence selection must never be reused silently.
OVERWRITE_FLAG="--overwrite"

# Parse SEQS into array
IFS=',' read -ra SEQ_ARRAY <<< "$SEQS"

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

CURVE_PLOT_FLAG=""
$PLOTS && CURVE_PLOT_FLAG="--curve-plot outputs/curve_plots_blind_test_v4.png"

# ---------------------------------------------------------------------------
# PHASE 1: Download / verify sequences
# ---------------------------------------------------------------------------
phase_start "PHASE 1: Download / verify sequences (seq: $SEQS)"

if $NO_DOWNLOAD; then
  for seq in "${SEQ_ARRAY[@]}"; do
    if [ ! -d "$DATASET_DIR/${seq}" ]; then
      echo "ERROR: $DATASET_DIR/${seq} not found." >&2
      echo "       Mount the dataset or remove --no-download to enable downloads." >&2
      exit 1
    fi
  done
  echo "  --no-download: all sequences verified present"
else
  mkdir -p "$DATASET_DIR/downloads"
  for seq in "${SEQ_ARRAY[@]}"; do
    if [ -d "$DATASET_DIR/${seq}" ]; then
      echo "  seq${seq} already exists, skipping"
      continue
    fi
    if [ -z "${FILE_IDS[$seq]+x}" ]; then
      echo "ERROR: No download FILE_ID configured for seq${seq}." >&2
      echo "       Place the data in $DATASET_DIR/${seq} and use --no-download." >&2
      exit 1
    fi
    command -v gdown >/dev/null 2>&1 || {
      echo "ERROR: gdown not installed. Run: pip install gdown" >&2; exit 1
    }
    echo "  Downloading seq${seq}..."
    gdown "${FILE_IDS[$seq]}" -O "$DATASET_DIR/downloads/${seq}.zip" --quiet
    unzip -n -q "$DATASET_DIR/downloads/${seq}.zip" -d "$DATASET_DIR/${seq}"
    rm "$DATASET_DIR/downloads/${seq}.zip"
    echo "  seq${seq} done."
  done
fi
phase_end

# ---------------------------------------------------------------------------
# PHASE 2: Preprocess (all seqs as test)
# ---------------------------------------------------------------------------
phase_start "PHASE 2: Preprocess (all seqs as test)"
python src/preprocessing/prepare_fred_yolo.py \
  --dataset-root "$DATASET_DIR" \
  --output-root  "$OUTROOT" \
  --train-seqs   "" \
  --val-seqs     "" \
  --test-seqs    "$SEQS" \
  --overwrite
phase_end

# ------------

# ---------------------------------------------------------------------------
# PHASE 3: Export Event-YOLO proposals
# ---------------------------------------------------------------------------
phase_start "PHASE 3: Export proposals (test)"

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

for SPLIT in test; do
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
for SPLIT in test; do
  DETS="$PROPOSALS/detections_conf${CONF}_${SPLIT}.jsonl"
  for MOD in rgb event; do
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

for SPLIT in test; do
  for MOD_PAIR in "rgb:$RGB_VER" "event:$EVT_VER"; do
    MOD="${MOD_PAIR%%:*}"
    MODEL="${MOD_PAIR##*:}"
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
  --splits         test \
  --conf           "$CONF" \
  --output-dir     "$WEB_OUT" \
  --confusion-plot "outputs/confusion_matrices_blind_test_v4.png" \
  --name           "blind_test_v4" \
  $CURVE_PLOT_FLAG \
  $OVERWRITE_FLAG
phase_end

echo ""
echo "=== WEB OUTPUT (GUI team: use these files) ==="
echo "  Manifest:   $WEB_OUT/fusion_manifest_blind_test_v4.json"
echo "  Detections: $WEB_OUT/fusion_detections_blind_test_v4.jsonl"
echo "  Plot:       outputs/confusion_matrices_blind_test_v4.png"
$PLOTS && echo "  Curves:     outputs/curve_plots_blind_test_v4.png"

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
