#!/usr/bin/env python3
"""
Compute fusion metrics and produce web-ready outputs for the blind-test pipeline.

Loads per-detection scored JSONL files (from eval_verifier.py), applies the
late-fusion formula, computes per-split and combined confusion-matrix metrics,
saves a confusion-matrix plot, and writes two files the web interface can
consume without running any model inference:

  outputs/web/fusion_detections_<name>.jsonl   one entry per frame (all frames)
  outputs/web/fusion_manifest_<name>.json      summary + per-split metrics

RGB and event records are merged by crop_id (set in extract_crops.py) rather
than by position, so missing crops in one modality do not silently corrupt the
alignment.

Usage
-----
  python src/verifier/compute_fusion_metrics.py \\
    --scored-dir    runs/verifier/rgb_v4/blind_v4 \\
    --fusion-config runs/verifier/rgb_v4/fusion_results_rgb_val_conf0.20.json \\
    --processed-root processed/fred_blind_test_v4 \\
    --splits train,val,test \\
    --conf  0.20 \\
    --output-dir  outputs/web \\
    --confusion-plot outputs/confusion_matrices_blind_test_v4.png \\
    --name  blind_test_v4

  # force overwrite even if outputs already exist
  python src/verifier/compute_fusion_metrics.py ... --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gt_boxes(label_path: Path) -> list[list[float]]:
    """Return GT boxes as normalized YOLO [cx, cy, w, h]."""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append([round(float(v), 6) for v in parts[1:5]])
    return boxes


def compute_cm(detections: list[dict], mode: str, thresh: float) -> dict:
    """TP/FP/FN/precision/recall/f1 for a detection list."""
    TP = FP = FN = 0
    for d in detections:
        if mode == "fusion":
            pred = d["fusion_score"] >= thresh
        else:
            pred = d["detector_score"] >= thresh
        gt = d["label"] == 1
        if pred and gt:
            TP += 1
        elif pred and not gt:
            FP += 1
        elif not pred and gt:
            FN += 1
    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec  = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def metrics_for_subset(dets: list[dict], fusion_threshold: float) -> dict:
    return {
        "event_yolo_conf0.25": compute_cm(dets, "detector", 0.25),
        "event_yolo_conf0.50": compute_cm(dets, "detector", 0.50),
        "fusion_v4":           compute_cm(dets, "fusion",   fusion_threshold),
    }


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="Compute fusion metrics and write web-ready outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scored-dir",    default=str(root / "runs/verifier/rgb_v4/blind_v4"))
    p.add_argument("--fusion-config", default=str(root / "runs/verifier/rgb_v4/fusion_results_rgb_val_conf0.20.json"))
    p.add_argument("--processed-root", default=str(root / "processed/fred_blind_test_v4"))
    p.add_argument("--splits",  default="train,val,test")
    p.add_argument("--conf",    default="0.20")
    p.add_argument("--output-dir",     default=str(root / "outputs/web"))
    p.add_argument("--confusion-plot", default=str(root / "outputs/confusion_matrices_blind_test_v4.png"))
    p.add_argument("--name",    default="blind_test_v4")
    p.add_argument("--fps",     type=float, default=30.0,
                   help="Frames per second assumed for frame_index/timestamp_sec in the web JSONL.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if outputs exist.")
    return p


def main() -> int:
    root = repo_root()
    args = build_parser().parse_args()

    scored_dir  = resolve_path(args.scored_dir, root)
    fusion_cfg  = resolve_path(args.fusion_config, root)
    proc_root   = resolve_path(args.processed_root, root)
    output_dir  = resolve_path(args.output_dir, root)
    plot_path   = resolve_path(args.confusion_plot, root)
    splits      = [s.strip() for s in args.splits.split(",")]
    conf        = args.conf
    name        = args.name

    web_dets = output_dir / f"fusion_detections_{name}.jsonl"
    web_mani = output_dir / f"fusion_manifest_{name}.json"

    if web_dets.is_file() and web_mani.is_file() and not args.overwrite:
        print("Web outputs already exist — skipping (pass --overwrite to recompute)")
        print(f"  {web_dets}")
        print(f"  {web_mani}")
        return 0

    # Load fusion config
    if not fusion_cfg.is_file():
        print(f"ERROR: fusion config not found: {fusion_cfg}", file=sys.stderr)
        return 1
    cfg = json.loads(fusion_cfg.read_text(encoding="utf-8"))
    LAMBDA = float(cfg["best_lambda"])
    THRESH = float(cfg["best_threshold"])
    print(f"Fusion config: λ={LAMBDA}, threshold={THRESH:.6f}")

    # Load scored files — collect as lists (preserving order) and as dicts (keyed by crop_id)
    rgb_recs_all: list[dict] = []
    evt_recs_all: list[dict] = []
    loaded_splits: list[str] = []

    for split in splits:
        rgb_f = scored_dir / f"scored_rgb_{split}_conf{conf}.jsonl"
        evt_f = scored_dir / f"scored_event_{split}_conf{conf}.jsonl"
        if not rgb_f.is_file():
            print(f"WARNING: missing {rgb_f} — skipping split {split}", file=sys.stderr)
            continue
        if not evt_f.is_file():
            print(f"WARNING: missing {evt_f} — skipping split {split}", file=sys.stderr)
            continue
        rgb_recs_all.extend(load_jsonl(rgb_f))
        evt_recs_all.extend(load_jsonl(evt_f))
        loaded_splits.append(split)

    print(f"Loaded splits: {loaded_splits}")
    print(f"RGB records: {len(rgb_recs_all)}  |  Event records: {len(evt_recs_all)}")

    # Detect whether crop_id is present (requires re-running phases 4+5 after the update)
    has_crop_id = bool(rgb_recs_all) and "crop_id" in rgb_recs_all[0]

    if has_crop_id:
        print("Merging by crop_id (stable key)")
        rgb_by_id = {r["crop_id"]: r for r in rgb_recs_all}
        evt_by_id = {r["crop_id"]: r for r in evt_recs_all}

        rgb_only = set(rgb_by_id) - set(evt_by_id)
        evt_only = set(evt_by_id) - set(rgb_by_id)
        if rgb_only:
            print(f"WARNING: {len(rgb_only)} crop_ids in RGB but not event — excluded", file=sys.stderr)
        if evt_only:
            print(f"WARNING: {len(evt_only)} crop_ids in event but not RGB — excluded", file=sys.stderr)
        pairs = [
            (rgb_by_id[cid], evt_by_id[cid])
            for cid in sorted(set(rgb_by_id) & set(evt_by_id), key=natural_key)
        ]
    else:
        print(
            "WARNING: crop_id not found in scored files — falling back to positional merge.\n"
            "         Re-run phases 4 and 5 (--overwrite) after updating extract_crops.py\n"
            "         to get robust crop_id-based alignment.",
            file=sys.stderr,
        )
        if len(rgb_recs_all) != len(evt_recs_all):
            print(
                f"ERROR: record count mismatch (rgb={len(rgb_recs_all)} event={len(evt_recs_all)}) "
                "and crop_id is missing — cannot merge safely.",
                file=sys.stderr,
            )
            return 1
        pairs = list(zip(rgb_recs_all, evt_recs_all))

    # Validate alignment and compute fusion scores
    flat: list[dict] = []
    for r, e in pairs:
        crop_id = r.get("crop_id", f"{r['split']}_{r['stem']}_pos{len(flat):07d}")

        if r["stem"] != e["stem"]:
            raise ValueError(
                f"stem mismatch for crop_id={crop_id!r}: "
                f"rgb={r['stem']!r} vs event={e['stem']!r}\n"
                "Both scored files must come from the same detections JSONL."
            )
        if r["bbox_xyxy"] != e["bbox_xyxy"]:
            raise ValueError(
                f"bbox_xyxy mismatch for crop_id={crop_id!r}: "
                f"rgb={r['bbox_xyxy']} vs event={e['bbox_xyxy']}\n"
                "Both scored files must come from the same detections JSONL."
            )

        fs = sigmoid(logit(r["verifier_score"]) + LAMBDA * logit(e["verifier_score"]))
        flat.append({
            "crop_id":              crop_id,
            "stem":                 r["stem"],
            "split":                r["split"],
            "sequence":             r["sequence"],
            "bbox_xyxy":            r["bbox_xyxy"],
            "detector_score":       round(r["detector_score"], 6),
            "rgb_verifier_score":   round(r["verifier_score"], 6),
            "event_verifier_score": round(e["verifier_score"], 6),
            "fusion_score":         round(fs, 6),
            "kept":                 fs >= THRESH,
            "iou_with_gt":          r.get("iou_with_gt", 0.0),
            "label":                r["label"],
        })

    print(f"Merged detections: {len(flat)}")

    # Per-split and combined metrics
    metrics: dict[str, dict] = {}
    split_dets: dict[str, list[dict]] = defaultdict(list)
    for d in flat:
        split_dets[d["split"]].append(d)
    for sp in loaded_splits:
        metrics[sp] = metrics_for_subset(split_dets[sp], THRESH)
    metrics["all"] = metrics_for_subset(flat, THRESH)

    print("\n=== Metrics ===")
    for subset, ms in metrics.items():
        print(f"  [{subset}]")
        for sname, m in ms.items():
            print(f"    {sname:<30} TP={m['TP']:6d} FP={m['FP']:5d} FN={m['FN']:5d}"
                  f"  Prec={m['precision']:.1%} Rec={m['recall']:.1%} F1={m['f1']:.1%}")

    # Confusion matrix plot
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fusion_label = f"Fusion v4\n(λ={LAMBDA}, t={THRESH:.3f})"
        display = {
            "Event YOLO\n(conf≥0.25)": metrics["all"]["event_yolo_conf0.25"],
            "Event YOLO\n(conf≥0.50)": metrics["all"]["event_yolo_conf0.50"],
            fusion_label:              metrics["all"]["fusion_v4"],
        }
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"Blind Test {name} — NEVER seen during training",
            fontsize=13, fontweight="bold",
        )
        for ax, (sname, s) in zip(axes, display.items()):
            mat = np.array([[s["TP"], s["FN"]], [s["FP"], 0]])
            ax.imshow(mat, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Pos", "Pred Neg"])
            ax.set_yticklabels(["GT Pos", "GT Neg"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                            fontsize=13,
                            color="white" if mat[i, j] > mat.max() * 0.5 else "black")
            ax.set_title(
                f"{sname}\nPrec={s['precision']:.1%} Rec={s['recall']:.1%} F1={s['f1']:.1%}",
                fontsize=9,
            )
        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nConfusion plot: {plot_path}")
    except ImportError as exc:
        print(f"WARNING: skipping confusion plot ({exc})", file=sys.stderr)
        plot_path = None

    # Group detections by stem for the web JSONL
    stem_dets: dict[str, list[dict]] = defaultdict(list)
    stem_meta: dict[str, tuple[str, str]] = {}  # stem → (split, sequence)
    for d in flat:
        stem_dets[d["stem"]].append(d)
        stem_meta[d["stem"]] = (d["split"], d["sequence"])

    # Determine relative path prefix for image paths
    try:
        proc_rel = proc_root.relative_to(root)
    except ValueError:
        proc_rel = proc_root  # absolute fallback if outside repo

    rgb_img_root = proc_root / "rgb_yolo" / "images"
    evt_img_root = proc_root / "event_yolo" / "images"
    label_root   = proc_root / "event_yolo" / "labels"

    # Write web JSONL — one line per frame (all frames, including those with no detections)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_frames_total = 0
    n_frames_with_dets = 0
    sequences: set[str] = set()
    fps = args.fps

    with web_dets.open("w", encoding="utf-8") as fh:
        for split in splits:
            split_img_dir = rgb_img_root / split
            if not split_img_dir.is_dir():
                print(f"WARNING: image dir not found: {split_img_dir}", file=sys.stderr)
                continue
            img_paths = sorted(split_img_dir.glob("*.jpg"), key=lambda p: natural_key(p.name))
            for img_path in img_paths:
                stem = img_path.stem
                # derive sequence name: everything before the last underscore-separated number
                sequence = stem.rsplit("_", 1)[0] if "_" in stem else stem
                # use stem_meta if available for consistency
                if stem in stem_meta:
                    _split, sequence = stem_meta[stem]
                sequences.add(sequence)

                rgb_rel = str(proc_rel / "rgb_yolo" / "images" / split / f"{stem}.jpg")
                evt_rel = str(proc_rel / "event_yolo" / "images" / split / f"{stem}.png")
                gt_boxes = read_gt_boxes(label_root / split / f"{stem}.txt")

                dets = stem_dets.get(stem, [])
                det_records = [
                    {
                        "crop_id":              d["crop_id"],
                        "bbox_xyxy":            d["bbox_xyxy"],
                        "detector_score":       d["detector_score"],
                        "rgb_verifier_score":   d["rgb_verifier_score"],
                        "event_verifier_score": d["event_verifier_score"],
                        "fusion_score":         d["fusion_score"],
                        "kept":                 d["kept"],
                        "iou_with_gt":          d["iou_with_gt"],
                        "label":                d["label"],
                    }
                    for d in dets
                ]

                record = {
                    "stem":          stem,
                    "split":         split,
                    "sequence":      sequence,
                    "frame_index":   n_frames_total,
                    "timestamp_sec": round(n_frames_total / fps, 6),
                    "rgb_image":     rgb_rel,
                    "event_image":   evt_rel,
                    "gt_boxes_norm": gt_boxes,
                    "detections":    det_records,
                }
                fh.write(json.dumps(record) + "\n")
                n_frames_total += 1
                if dets:
                    n_frames_with_dets += 1

    print(f"\nWeb JSONL: {web_dets}")
    print(f"  Total frames:             {n_frames_total}")
    print(f"  Frames with detections:   {n_frames_with_dets}")
    print(f"  Frames with no detections:{n_frames_total - n_frames_with_dets}")

    # Web manifest
    try:
        plot_rel = str(plot_path.relative_to(root)) if plot_path else None
    except ValueError:
        plot_rel = str(plot_path) if plot_path else None
    try:
        web_dets_rel = str(web_dets.relative_to(root))
    except ValueError:
        web_dets_rel = str(web_dets)

    manifest = {
        "name":    name,
        "created": datetime.now(timezone.utc).isoformat(),
        "sequences":              sorted(sequences, key=natural_key),
        "splits":                 loaded_splits,
        "n_frames_total":         n_frames_total,
        "n_frames_with_detections": n_frames_with_dets,
        "n_detections_total":     len(flat),
        "fps":                    fps,
        "fusion_lambda":          LAMBDA,
        "fusion_threshold":       THRESH,
        "conf":                   conf,
        "metrics":                metrics,
        "paths": {
            "web_detections": web_dets_rel,
            "confusion_plot": plot_rel,
            "processed_root": str(proc_rel),
        },
    }
    web_mani.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Web manifest: {web_mani}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
