"""Hybrid proposal merge: event-YOLO ∪ hard-gated RGB-YOLO.

The event detector leads — an event camera excels on moving drones but goes
blind when one hovers (no brightness change, no events). The RGB detector
exists for exactly those frames and is gated hard (cfg.rgb_min_conf): it only
contributes proposals it is very sure about, and they still have to pass the
verifier + fusion stages downstream.

Merge rule (per frame): keep every event proposal; add each gated RGB
proposal that does not overlap a kept box with IoU >= cfg.dedup_iou. A
duplicate keeps the event box (marked source="both") with the max of both
scores. Every record carries source: "event" | "rgb" | "both".
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .common import iou_xyxy, read_yolo_labels
from .config import PipelineConfig


def merge_proposals(
    event_dets: list[dict],
    rgb_dets: list[dict],
    cfg: PipelineConfig,
) -> tuple[list[dict], dict]:
    """Return (merged detections, stats). Inputs are full detection lists."""
    rgb_gated = [d for d in rgb_dets if d["detector_score"] >= cfg.rgb_min_conf]

    by_stem_event: dict[str, list[dict]] = defaultdict(list)
    by_stem_rgb: dict[str, list[dict]] = defaultdict(list)
    for d in event_dets:
        by_stem_event[d["stem"]].append(d)
    for d in rgb_gated:
        by_stem_rgb[d["stem"]].append(d)

    merged: list[dict] = []
    for stem in sorted(set(by_stem_event) | set(by_stem_rgb)):
        frame = [{**d, "source": "event"} for d in by_stem_event[stem]]
        for r in by_stem_rgb[stem]:
            dup = next((m for m in frame
                        if iou_xyxy(r["bbox_xyxy"], m["bbox_xyxy"]) >= cfg.dedup_iou),
                       None)
            if dup is None:
                frame.append({**r, "source": "rgb"})
            else:
                dup["source"] = "both"
                dup["detector_score"] = round(
                    max(dup["detector_score"], r["detector_score"]), 6)
        merged.extend(frame)

    counts: dict[str, int] = defaultdict(int)
    for d in merged:
        counts[d["source"]] += 1
    stats = {
        "event_detections": len(event_dets),
        "rgb_detections_total": len(rgb_dets),
        "rgb_detections_gated": len(rgb_gated),
        "rgb_min_conf": cfg.rgb_min_conf,
        "dedup_iou": cfg.dedup_iou,
        "merged_detections": len(merged),
        "by_source": dict(counts),
    }
    return merged, stats


def coverage_stats(merged: list[dict], labels_dir: Path,
                   cfg: PipelineConfig) -> dict:
    """Proposal recall ceiling: which GT boxes have any proposal at all?

    No downstream stage can recover a GT box without a proposal, so this is
    the hard upper bound on recall — reported per source to show what the
    RGB safety net adds over the event detector alone.
    """
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for d in merged:
        by_stem[d["stem"]].append(d)

    n_gt = covered = covered_event = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        dets = by_stem.get(label_path.stem, [])
        for g in read_yolo_labels(label_path, cfg.img_width, cfg.img_height):
            n_gt += 1
            if any(iou_xyxy(g, d["bbox_xyxy"]) >= cfg.iou_match for d in dets):
                covered += 1
            if any(iou_xyxy(g, d["bbox_xyxy"]) >= cfg.iou_match
                   for d in dets if d["source"] in ("event", "both")):
                covered_event += 1
    return {
        "n_gt_boxes": n_gt,
        "n_covered_gt": covered,
        "n_missed_gt": n_gt - covered,
        "proposal_ceiling_recall": round(covered / n_gt, 4) if n_gt else None,
        "event_only_ceiling_recall": (round(covered_event / n_gt, 4)
                                      if n_gt else None),
    }
