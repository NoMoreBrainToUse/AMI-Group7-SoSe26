# Update — Web GUI (2026-07-07): full pipeline from the GUI

The sidebar now has a **Step 3 - Run pipeline** button that runs the complete
inference pipeline on the uploaded sequence (any FRED-format zip, not just the
four blind-test sequences):

1. `src/preprocessing/prepare_fred_yolo.py` — RGB/event YOLO layout (whole
   sequence as the `test` split) → `processed/pipeline_<seq>/`
2. `src/verifier/export_proposals.py` — event-YOLO proposals with the **v5
   weights** (`runs/event_yolo/fred_subset_v5_event_yolo11m/weights/best.pt`)
3. `src/verifier/extract_crops.py` — RGB + event crops
4. `src/verifier/eval_verifier.py` — v4 verifier scores for both modalities
5. `src/verifier/compute_fusion_metrics.py` — late fusion with the v5-calibrated
   config (`runs/verifier/rgb_v4/calib_v5/fusion_results_rgb_test_conf0.20.json`),
   confusion matrices, and web outputs
   → `outputs/web/fusion_{detections,manifest}_pipeline_<seq>.{jsonl,json}`,
   `outputs/confusion_matrices_pipeline_<seq>.png`

Orchestrated by `src/pipeline/gui_pipeline.py` (also runnable standalone:
`python src/pipeline/gui_pipeline.py <seq> [--overwrite]`). Finished phases are
skipped, so an interrupted run resumes where it stopped.

Once pipeline results exist for the loaded sequence, the **Tracking**, **Fusion**
and **confusion matrices** tabs automatically switch from the committed
blind-test v4 results to that run's outputs.

Extra dependencies (CPU inference), installed on top of `requirements-gui.txt`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

---

# Update — Web GUI (2026-07-02)

Additions on the `feature/gui` branch, complementing `README_PIPELINE.md`
(which is unchanged and still documents the ML pipeline).

## What changed

| File | Change |
|---|---|
| `gui.py` | Streamlit web interface (see below) |
| `run_gui.sh` | Launch script |
| `requirements-gui.txt` | GUI-only dependencies (pinned) |
| `.streamlit/config.toml` | Raises the upload limit to 4 GB; picked up automatically |
| `src/preprocessing/preprocess_fred_dataset.py` | Copied from the project repo; generates the `matched/` per-sequence layout used by the GUI |
| `processed/preprocessed_fred_{40,43,46,49}/` | Dashboard preview datasets (not committed, regenerable — see below) |

The unused `processed/preprocessed_fred_*` datasets (sequences 10–42 except 40)
were deleted; `processed/` went from 26 GB to ~3 GB.

## Running the GUI

```bash
pip install -r requirements-gui.txt   # into the venv used by run_gui.sh
./run_gui.sh                          # opens http://localhost:8501
```

The GUI starts a small HTTP server on port **8765** that serves frames from
`processed/` to the browser (the in-browser players load images by URL). When
accessing the GUI from another machine, open port 8765 alongside 8501.

Workflow: select a dataset in the sidebar (Step 1) and press **Run** (Step 2)
to load it, then use the tabs.

| Tab | Data source | How to generate |
|---|---|---|
| Dashboard | `processed/preprocessed_fred_{40,43,46,49}/matched/` | `preprocess_fred_dataset.py` (below) |
| Side-by-Side | `outputs/web/fusion_detections_blind_test_v4.jsonl` (committed) + `processed/fred_blind_test_v4/` images | `./run_blind_test_v4.sh --no-download` |
| confusion matrices | `outputs/web/fusion_manifest_blind_test_v4.json` (committed) | nothing to do |

Both players share the same controls: `<` / `>` step one frame, `▶` plays at the
chosen fps (default 30), the progress bar seeks (also while playing). The
Dashboard viewer wipes between RGB and Event by dragging on the image; the
Side-by-Side viewer draws ground truth (green), kept detections (yellow, with
fusion score) and rejected detections (red) on both streams. The confusion
matrices tab renders the metrics from the manifest (split selector, per-model
cards); the original PNG stays available in an expander.

Note: `README_PIPELINE.md` suggests cyan for kept detections; the GUI uses
yellow.

## Generating the Dashboard preview datasets

The Dashboard needs per-sequence aligned frames in the `matched/` layout,
produced by `src/preprocessing/preprocess_fred_dataset.py` (stdlib-only; not
to be confused with `src/preprocessing/prepare_fred_yolo.py`, which emits the
`rgb_yolo/` layout used for training):

```bash
python src/preprocessing/preprocess_fred_dataset.py \
  --dataset-root dataset \
  --output-root  processed \
  --train-seqs   40,43,46,49 \
  --val-seqs     "" \
  --test-seqs    ""
# -> processed/preprocessed_fred_<seq>/{matched,labels,paired,report}/
```

Requires the raw sequences in `dataset/<seq>/` (`PADDED_RGB/`, `Event/Frames/`,
`interpolated_coordinates.txt` or `coordinates.txt`). Images are materialized
as hardlinks into `dataset/`, so the output adds almost no disk space.
Frame counts: seq40 2771, seq43 2135, seq46 3162, seq49 3318.
