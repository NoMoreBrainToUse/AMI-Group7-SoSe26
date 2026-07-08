"""Late fusion, honest metrics, and web-ready outputs.

Fusion combines the two verifier probabilities in log-odds space —
independent evidence adds in logits (naive-Bayes fusion):

    fusion = sigmoid( logit(s_rgb) + lambda * logit(s_event) )

A detection is kept when fusion >= tau. Lambda and tau come from
weights/fusion_config.json and were calibrated on a diverse validation pool.

Metrics are GT-box level over ALL frames: a ground-truth drone that no kept
detection matches (IoU >= cfg.iou_match) counts as a false negative — in
particular when the detectors proposed nothing at all. The older
proposal-conditional view (which silently ignored such misses and once
reported 97% F1 at a true recall of 82%) is kept in the manifest under
"metrics_conditional" for reference only.

Outputs (schema identical to the previous pipeline, so the web GUI can
consume them unchanged):
  fusion_detections_<name>.jsonl   one line per frame
  fusion_manifest_<name>.json      metrics + operating point + paths
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import (iou_xyxy, logit, natural_key, read_yolo_labels_norm,
                     sigmoid, yolo_to_xyxy)
from .config import PipelineConfig

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def fuse_scores(rgb_scored: list[dict], event_scored: list[dict],
                cfg: PipelineConfig) -> list[dict]:
    """Pair the two scored lists (same detections, same order) and fuse."""
    if len(rgb_scored) != len(event_scored):
        raise ValueError(
            f"Scored list length mismatch: rgb={len(rgb_scored)} "
            f"event={len(event_scored)}")
    fused = []
    for r, e in zip(rgb_scored, event_scored):
        if r["crop_id"] != e["crop_id"]:
            raise ValueError(f"crop_id mismatch: {r['crop_id']} vs {e['crop_id']}")
        fs = sigmoid(logit(r["rgb_verifier_score"])
                     + cfg.fusion_lambda * logit(e["event_verifier_score"]))
        fused.append({
            "crop_id": r["crop_id"],
            "stem": r["stem"],
            "split": r["split"],
            "sequence": r["sequence"],
            "source": r.get("source", "event"),
            "class": "drone",  # single-class detector; explicit per spec
            "bbox_xyxy": r["bbox_xyxy"],
            "detector_score": r["detector_score"],
            "rgb_verifier_score": r["rgb_verifier_score"],
            "event_verifier_score": e["event_verifier_score"],
            "fusion_score": round(fs, 6),
            "kept": fs >= cfg.fusion_threshold,
            "iou_with_gt": r.get("iou_with_gt", 0.0),
            "label": r["label"],
        })
    return fused


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _cm(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def true_cm(frames: list[dict], keep_fn, iou_match: float) -> dict:
    """GT-box-level confusion counts; missed drones are FN."""
    tp = fp = fn = 0
    for fr in frames:
        gts = fr["gt_boxes_px"]
        kept = [d for d in fr["detections"] if keep_fn(d)]
        for g in gts:
            if any(iou_xyxy(g, d["bbox_xyxy"]) >= iou_match for d in kept):
                tp += 1
            else:
                fn += 1
        for d in kept:
            if not gts or all(iou_xyxy(g, d["bbox_xyxy"]) < iou_match for g in gts):
                fp += 1
    return _cm(tp, fp, fn)


def conditional_cm(dets: list[dict], keep_fn) -> dict:
    """Legacy proposal-only view: a GT box without a proposal is invisible."""
    tp = fp = fn = 0
    for d in dets:
        pred, gt = keep_fn(d), d["label"] == 1
        tp += pred and gt
        fp += pred and not gt
        fn += (not pred) and gt
    return _cm(tp, fp, fn)


def _systems(cfg: PipelineConfig, hybrid: bool) -> dict:
    def evt(conf):
        return lambda d: (d["detector_score"] >= conf
                          and d.get("source", "event") in ("event", "both"))
    systems = {
        "event_yolo_conf0.25": evt(0.25),
        "event_yolo_conf0.50": evt(0.50),
    }
    if hybrid:
        systems["hybrid_yolo_conf0.25"] = lambda d: d["detector_score"] >= 0.25
    systems["fusion_v4"] = lambda d: d["kept"]
    return systems


# ---------------------------------------------------------------------------
# Web outputs
# ---------------------------------------------------------------------------

def write_web_outputs(
    fused: list[dict],
    processed_root: Path,
    split: str,
    output_dir: Path,
    name: str,
    cfg: PipelineConfig,
    confusion_plot: Path | None = None,
    progress=print,
) -> dict:
    """Write fusion_detections/manifest for one split; return the manifest."""
    rgb_images = processed_root / "rgb_yolo" / "images" / split
    labels_dir = processed_root / "event_yolo" / "labels" / split

    by_stem: dict[str, list[dict]] = defaultdict(list)
    for d in fused:
        by_stem[d["stem"]].append(d)

    det_fields = ("crop_id", "source", "class", "bbox_xyxy", "detector_score",
                  "rgb_verifier_score", "event_verifier_score",
                  "fusion_score", "kept", "iou_with_gt", "label")

    frames: list[dict] = []
    records: list[str] = []
    sequences: set[str] = set()
    for i, img in enumerate(sorted(rgb_images.glob("*"),
                                   key=lambda p: natural_key(p.name))):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img.stem
        dets = by_stem.get(stem, [])
        seq = dets[0]["sequence"] if dets else (
            stem.rsplit("_", 1)[0] if "_" in stem else stem)
        sequences.add(seq)
        gt_norm = read_yolo_labels_norm(labels_dir / f"{stem}.txt")
        det_records = [{k: d[k] for k in det_fields} for d in dets]
        records.append(json.dumps({
            "stem": stem,
            "split": split,
            "sequence": seq,
            "frame_index": len(frames),
            "timestamp_sec": round(len(frames) / cfg.fps, 6),
            "rgb_image": str((rgb_images / img.name).as_posix()),
            "event_image": str((processed_root / "event_yolo" / "images"
                                / split / f"{stem}.png").as_posix()),
            "gt_boxes_norm": gt_norm,
            "detections": det_records,
        }))
        frames.append({
            "gt_boxes_px": [yolo_to_xyxy(*b, cfg.img_width, cfg.img_height)
                            for b in gt_norm],
            "detections": det_records,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    web_dets = output_dir / f"fusion_detections_{name}.jsonl"
    web_dets.write_text("\n".join(records) + "\n", encoding="utf-8")

    hybrid = any(d["source"] == "rgb" for d in fused)
    metrics = {sp: {} for sp in (split, "all")}
    for sys_name, keep_fn in _systems(cfg, hybrid).items():
        m = true_cm(frames, keep_fn, cfg.iou_match)
        metrics[split][sys_name] = m
        metrics["all"][sys_name] = m
    metrics_cond = {sp: {} for sp in (split, "all")}
    for sys_name, keep_fn in _systems(cfg, hybrid).items():
        m = conditional_cm(fused, keep_fn)
        metrics_cond[split][sys_name] = m
        metrics_cond["all"][sys_name] = m

    progress("TRUE detection metrics (GT-box level, misses counted as FN):")
    for sys_name, m in metrics["all"].items():
        progress(f"  {sys_name:<24} P={m['precision']:.1%} "
                 f"R={m['recall']:.1%} F1={m['f1']:.1%}")

    plot_rel = None
    if confusion_plot is not None:
        if _confusion_plot(metrics["all"], confusion_plot, name, cfg):
            plot_rel = str(confusion_plot)

    manifest = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "sequences": sorted(sequences, key=natural_key),
        "splits": [split],
        "n_frames_total": len(frames),
        "n_frames_with_detections": sum(1 for f in frames if f["detections"]),
        "n_detections_total": len(fused),
        "fps": cfg.fps,
        "fusion_lambda": cfg.fusion_lambda,
        "fusion_threshold": cfg.fusion_threshold,
        "conf": f"{cfg.proposal_conf:.2f}",
        "hybrid_proposals": hybrid,
        "metrics": metrics,
        "metrics_conditional": metrics_cond,
        "paths": {
            "web_detections": str(web_dets),
            "confusion_plot": plot_rel,
            "processed_root": str(processed_root),
        },
    }
    (output_dir / f"fusion_manifest_{name}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _confusion_plot(metrics_all: dict, plot_path: Path, name: str,
                    cfg: PipelineConfig) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    labels = {
        "event_yolo_conf0.25": "Event YOLO\n(conf≥0.25)",
        "event_yolo_conf0.50": "Event YOLO\n(conf≥0.50)",
        "hybrid_yolo_conf0.25": "Hybrid Event+RGB YOLO\n(conf≥0.25)",
        "fusion_v4": (f"Fusion\n(λ={cfg.fusion_lambda}, "
                      f"t={cfg.fusion_threshold:.3f})"),
    }
    fig, axes = plt.subplots(1, len(metrics_all),
                             figsize=(4.7 * len(metrics_all), 5))
    fig.suptitle(f"{name} — GT-box-level metrics (missed drones counted)",
                 fontsize=13, fontweight="bold")
    for ax, (key, m) in zip(axes, metrics_all.items()):
        mat = np.array([[m["TP"], m["FN"]], [m["FP"], 0]])
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Pos", "Pred Neg"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["GT Pos", "GT Neg"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        fontsize=13,
                        color="white" if mat[i, j] > mat.max() * 0.5 else "black")
        ax.set_title(f"{labels.get(key, key)}\nPrec={m['precision']:.1%} "
                     f"Rec={m['recall']:.1%} F1={m['f1']:.1%}", fontsize=9)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    return True
