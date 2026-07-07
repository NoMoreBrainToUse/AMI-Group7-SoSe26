#!/usr/bin/env python3
"""
Run the full detection pipeline for ONE sequence (GUI "Run pipeline" step).

Mirrors run_blind_test_v4.sh but parameterized on the sequence name and using
the v5 event-YOLO weights, so any FRED-format sequence uploaded through the
GUI can be evaluated end to end:

  Phase 1  prepare_fred_yolo.py        processed/pipeline_<seq>/{rgb,event}_yolo/
  Phase 2  export_proposals.py         runs/proposals/pipeline_<seq>/detections_conf0.20_test.jsonl
  Phase 3  extract_crops.py (rgb+evt)  processed/pipeline_<seq>/crops/
  Phase 4  eval_verifier.py (rgb+evt)  runs/verifier/pipeline_<seq>/scored_*_test_conf0.20.jsonl
  Phase 5  compute_fusion_metrics.py   outputs/web/fusion_{detections,manifest}_pipeline_<seq>.{jsonl,json}
                                       outputs/confusion_matrices_pipeline_<seq>.png

The whole sequence is treated as the "test" split. Each phase is skipped when
its outputs already exist (pass overwrite=True to recompute).

CLI:
  python src/pipeline/gui_pipeline.py 40 [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# v5 event detector; the v4 verifiers are the current (only) trained verifiers,
# and calib_v5 holds the fusion lambda/threshold calibrated on v5 proposals.
EVENT_MODEL = REPO_ROOT / "runs/event_yolo/fred_subset_v5_event_yolo11m/weights/best.pt"
RGB_VERIFIER = REPO_ROOT / "runs/verifier/rgb_v4/efficientnet_b0/best.pt"
EVENT_VERIFIER = REPO_ROOT / "runs/verifier/rgb_v4/event_efficientnet_b0/best.pt"
FUSION_CONFIG = REPO_ROOT / "runs/verifier/rgb_v4/calib_v5/fusion_results_rgb_test_conf0.20.json"

CONF = "0.20"
SPLIT = "test"


def pipeline_name(seq: str) -> str:
    return f"pipeline_{seq}"


def pipeline_paths(seq: str) -> dict[str, Path]:
    """All input/output locations of the pipeline run for one sequence."""
    name = pipeline_name(seq)
    return {
        "sequence_dir": REPO_ROOT / "dataset" / seq,
        "processed_root": REPO_ROOT / "processed" / name,
        "proposals_dir": REPO_ROOT / "runs" / "proposals" / name,
        "scored_dir": REPO_ROOT / "runs" / "verifier" / name,
        "web_detections": REPO_ROOT / "outputs" / "web" / f"fusion_detections_{name}.jsonl",
        "web_manifest": REPO_ROOT / "outputs" / "web" / f"fusion_manifest_{name}.json",
        "confusion_plot": REPO_ROOT / "outputs" / f"confusion_matrices_{name}.png",
    }


def pipeline_results(seq: str) -> dict | None:
    """Manifest dict of a finished pipeline run for this sequence, or None."""
    p = pipeline_paths(seq)
    if not (p["web_manifest"].is_file() and p["web_detections"].is_file()):
        return None
    return json.loads(p["web_manifest"].read_text(encoding="utf-8"))


def _run(cmd: list[str | Path], progress) -> None:
    """Run one phase script from the repo root, streaming its output."""
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            progress(line)
    if proc.wait() != 0:
        raise RuntimeError(f"Pipeline step failed (exit {proc.returncode}): {cmd[1]}")


def run_pipeline(seq: str, progress=print, overwrite: bool = False) -> dict:
    """Run all pipeline phases for one sequence; returns the fusion manifest.

    `progress(msg)` is called with one line per subprocess output line and
    with "PHASE k/5 ..." markers between phases.
    """
    p = pipeline_paths(seq)
    py = sys.executable

    for f, what in [
        (EVENT_MODEL, "event-YOLO v5 weights"),
        (RGB_VERIFIER, "RGB verifier weights"),
        (EVENT_VERIFIER, "event verifier weights"),
        (FUSION_CONFIG, "v5 fusion config"),
    ]:
        if not f.is_file():
            raise FileNotFoundError(f"Missing {what}: {f}")
    if not p["sequence_dir"].is_dir():
        raise FileNotFoundError(
            f"Sequence directory not found: {p['sequence_dir']} — "
            "upload and Process the zip first (sidebar steps 1-2)."
        )

    # --- Phase 1: preprocess into the rgb_yolo/event_yolo layout ---
    progress(f"PHASE 1/5 Preprocess (YOLO layout) — sequence {seq}")
    event_images = p["processed_root"] / "event_yolo" / "images"
    if not overwrite and (event_images / SPLIT).is_dir() and any((event_images / SPLIT).iterdir()):
        progress("  outputs exist, skipping")
    else:
        _run([
            py, "src/preprocessing/prepare_fred_yolo.py",
            "--dataset-root", "dataset",
            "--output-root", p["processed_root"],
            "--train-seqs", "", "--val-seqs", "", "--test-seqs", seq,
            "--overwrite",
        ], progress)

    # --- Phase 2: event-YOLO proposals (v5 weights) ---
    progress("PHASE 2/5 Event-YOLO proposals (v5 weights) — this is the slow part")
    detections = p["proposals_dir"] / f"detections_conf{CONF}_{SPLIT}.jsonl"
    if not overwrite and detections.is_file():
        progress("  proposals exist, skipping")
    else:
        _run([
            py, "src/verifier/export_proposals.py",
            "--model", EVENT_MODEL,
            "--event-images", p["processed_root"] / "event_yolo" / "images",
            "--event-labels", p["processed_root"] / "event_yolo" / "labels",
            "--split", SPLIT,
            "--output", p["proposals_dir"],
        ], progress)

    # --- Phase 3: RGB + event crops at the proposal boxes ---
    progress("PHASE 3/5 Extract crops")
    crops = p["processed_root"] / "crops"
    for mod in ("rgb", "event"):
        manifest = crops / mod / f"crop_manifest_{mod}_{SPLIT}_conf{CONF}.jsonl"
        if not overwrite and manifest.is_file():
            progress(f"  {mod}: manifest exists, skipping")
            continue
        _run([
            py, "src/verifier/extract_crops.py",
            "--detections", detections,
            "--modality", mod,
            "--images-dir", p["processed_root"] / f"{mod}_yolo" / "images",
            "--output-dir", crops,
            "--split", SPLIT,
        ], progress)

    # --- Phase 4: score crops with both verifiers ---
    progress("PHASE 4/5 Score crops with verifiers")
    for mod, model in (("rgb", RGB_VERIFIER), ("event", EVENT_VERIFIER)):
        scored = p["scored_dir"] / f"scored_{mod}_{SPLIT}_conf{CONF}.jsonl"
        if not overwrite and scored.is_file():
            progress(f"  {mod}: scored file exists, skipping")
            continue
        _run([
            py, "src/verifier/eval_verifier.py",
            "--model", model,
            "--manifest", crops / mod / f"crop_manifest_{mod}_{SPLIT}_conf{CONF}.jsonl",
            "--output", p["scored_dir"],
        ], progress)

    # --- Phase 5: fusion metrics, confusion matrices, web outputs ---
    progress("PHASE 5/5 Fusion metrics + confusion matrices")
    cmd = [
        py, "src/verifier/compute_fusion_metrics.py",
        "--scored-dir", p["scored_dir"],
        "--fusion-config", FUSION_CONFIG,
        "--processed-root", p["processed_root"],
        "--splits", SPLIT,
        "--conf", CONF,
        "--output-dir", REPO_ROOT / "outputs" / "web",
        "--confusion-plot", p["confusion_plot"],
        "--name", pipeline_name(seq),
    ]
    if overwrite:
        cmd.append("--overwrite")
    _run(cmd, progress)

    manifest = pipeline_results(seq)
    if manifest is None:
        raise RuntimeError("Pipeline finished but the fusion manifest was not written.")
    progress(f"DONE — {manifest['n_detections_total']} detections "
             f"on {manifest['n_frames_total']} frames")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("seq", help="Sequence name = zip stem, e.g. 40")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.seq, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
