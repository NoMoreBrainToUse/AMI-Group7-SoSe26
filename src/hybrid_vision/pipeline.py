"""End-to-end orchestration for one FRED sequence.

  1. align     FRED sequence -> paired rgb/event frames (YOLO layout)
  2. detect    event-YOLO + RGB-YOLO proposals (one low-conf pass each)
  3. merge     hybrid union with cross-modality dedup + hard RGB gate
  4. verify    RGB + event EfficientNet crop scores (in memory)
  5. fuse      logit fusion -> kept/rejected, honest metrics, web outputs

Every phase writes its artifact and is skipped when that artifact already
exists (pass overwrite=True to recompute). All phases run in-process — no
subprocesses, no shell scripts.

Artifacts for sequence <seq>:
  processed/<seq>/{rgb_yolo,event_yolo}/...      aligned frames + labels
  outputs/<seq>/proposals_{event,rgb}.jsonl      raw detections (conf>=base)
  outputs/<seq>/proposals_merged.jsonl           hybrid proposal set
  outputs/<seq>/merge_stats.json                 counts + recall ceiling
  outputs/<seq>/scored_{rgb,event}.jsonl         verifier scores
  outputs/<seq>/web/fusion_detections_<seq>.jsonl
  outputs/<seq>/web/fusion_manifest_<seq>.json
  outputs/<seq>/confusion_matrices.png
"""

from __future__ import annotations

import json
from pathlib import Path

from .align import align_sequence
from .common import read_jsonl, write_jsonl
from .config import REPO_ROOT, PipelineConfig
from .detect import generate_proposals
from .fusion import fuse_scores, write_web_outputs
from .merge import coverage_stats, merge_proposals
from .verifier import score_proposals

SPLIT = "test"  # a pipeline run treats the whole sequence as one test split


def run_sequence(
    seq_dir: str | Path,
    cfg: PipelineConfig | None = None,
    processed_root: Path | None = None,
    outputs_root: Path | None = None,
    overwrite: bool = False,
    progress=print,
) -> dict:
    """Run all phases for one FRED sequence directory; return the manifest."""
    cfg = cfg or PipelineConfig()
    cfg.validate()
    seq_dir = Path(seq_dir)
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    seq = seq_dir.name

    processed = (processed_root or REPO_ROOT / "processed") / seq
    out = (outputs_root or REPO_ROOT / "outputs") / seq
    out.mkdir(parents=True, exist_ok=True)

    # --- 1 align -----------------------------------------------------------
    progress(f"PHASE 1/5 Align RGB/event frames — sequence {seq}")
    event_images = processed / "event_yolo" / "images" / SPLIT
    if not overwrite and event_images.is_dir() and any(event_images.iterdir()):
        progress("  aligned frames exist, skipping")
    else:
        result = align_sequence(seq_dir, processed, SPLIT, cfg)
        progress(f"  {result.samples_written} aligned pairs "
                 f"({result.unmatched} unmatched, "
                 f"{result.invalid_boxes} invalid boxes)")

    # --- 2 detect ------------------------------------------------------------
    progress("PHASE 2/5 Detector proposals (event + RGB) — the slow part")
    proposals: dict[str, list[dict]] = {}
    for modality, model, imgsz in (
        ("event", cfg.event_model, cfg.event_imgsz),
        ("rgb", cfg.rgb_model, cfg.rgb_imgsz),
    ):
        path = out / f"proposals_{modality}.jsonl"
        if not overwrite and path.is_file():
            progress(f"  {modality} proposals exist, skipping")
            proposals[modality] = read_jsonl(path)
            continue
        progress(f"  running {modality}-YOLO (imgsz {imgsz})...")
        dets = generate_proposals(
            Path(model), processed / f"{modality}_yolo" / "images",
            processed / f"{modality}_yolo" / "labels",
            SPLIT, imgsz, cfg, progress=progress)
        write_jsonl(path, dets)
        proposals[modality] = dets

    # --- 3 merge -------------------------------------------------------------
    progress("PHASE 3/5 Hybrid proposal merge")
    merged_path = out / "proposals_merged.jsonl"
    stats_path = out / "merge_stats.json"
    if not overwrite and merged_path.is_file():
        progress("  merged proposals exist, skipping")
        merged = read_jsonl(merged_path)
    else:
        event_props = [d for d in proposals["event"]
                       if d["detector_score"] >= cfg.proposal_conf]
        merged, stats = merge_proposals(
            event_props, proposals["rgb"], cfg,
            rgb_images_dir=processed / "rgb_yolo" / "images" / SPLIT)
        stats.update(coverage_stats(
            merged, processed / "event_yolo" / "labels" / SPLIT, cfg))
        write_jsonl(merged_path, merged)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        progress(f"  {stats['merged_detections']} proposals "
                 f"(by source: {stats['by_source']}), "
                 f"ceiling recall {stats['proposal_ceiling_recall']}")

    # --- 4 verify --------------------------------------------------------------
    progress("PHASE 4/5 Verifier scoring (RGB + event crops)")
    scored: dict[str, list[dict]] = {}
    for modality, model, ext, key in (
        ("rgb", cfg.rgb_verifier, ".jpg", "rgb_verifier_score"),
        ("event", cfg.event_verifier, ".png", "event_verifier_score"),
    ):
        path = out / f"scored_{modality}.jsonl"
        if not overwrite and path.is_file():
            progress(f"  {modality} scores exist, skipping")
            scored[modality] = read_jsonl(path)
            continue
        recs = score_proposals(
            merged, processed / f"{modality}_yolo" / "images", SPLIT, ext,
            Path(model), key, cfg,
            measure_activity=(modality == "event"), progress=progress)
        write_jsonl(path, recs)
        scored[modality] = recs

    # --- 5 fuse ---------------------------------------------------------------
    progress("PHASE 5/5 Fusion, metrics, web outputs")
    manifest_path = out / "web" / f"fusion_manifest_{seq}.json"
    if not overwrite and manifest_path.is_file():
        progress("  web outputs exist, skipping")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    fused = fuse_scores(scored["rgb"], scored["event"], cfg)
    manifest = write_web_outputs(
        fused, processed, SPLIT, out / "web", seq, cfg,
        confusion_plot=out / "confusion_matrices.png", progress=progress)
    progress(f"DONE — {manifest['n_detections_total']} detections "
             f"on {manifest['n_frames_total']} frames")
    return manifest
