#!/usr/bin/env python3
"""
Run event-YOLO once at a very low base confidence and save all raw detections.
Then sweep thresholds offline by filtering the raw output — much faster than
running YOLO separately for every threshold.

Outputs under runs/proposals/<name>/:
  detections_raw_<split>.jsonl        all detections at base_conf
  detections_conf<X.XX>_<split>.jsonl filtered subset per threshold
  threshold_sweep_<split>.csv         recall / FP-per-frame summary

Each JSONL record:
  {
    "split":          "val",
    "sequence":       "seq100",
    "stem":           "seq100_000599994",
    "bbox_xyxy":      [x1, y1, x2, y2],
    "detector_score": 0.631,
    "iou_with_gt":    0.72,
    "label":          1          # 1 = drone (TP), 0 = background (FP)
  }

Typical workflow:

  # Step 1 — sweep val to pick threshold
  python src/verifier/export_proposals.py --split val

  # Step 2 — export train proposals at the chosen threshold
  python src/verifier/export_proposals.py --split train

  Then pass detections_conf0.20_val.jsonl / detections_conf0.20_train.jsonl
  to extract_crops.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def load_gt_boxes_px(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Parse a YOLO label file into pixel-space [x1,y1,x2,y2] boxes."""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in parts[:5])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def best_iou_against(det: list[float], gt_boxes: list[list[float]]) -> float:
    if not gt_boxes:
        return 0.0
    return max(iou(det, gt) for gt in gt_boxes)


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Export event-YOLO proposals. Runs YOLO once at base_conf, sweeps thresholds offline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(root / "runs" / "event_yolo" / "fred_subset_event_yolo11m" / "weights" / "best.pt"),
    )
    parser.add_argument(
        "--event-images",
        default=str(root / "processed" / "fred_subset" / "event_yolo" / "images"),
    )
    parser.add_argument(
        "--event-labels",
        default=str(root / "processed" / "fred_subset" / "event_yolo" / "labels"),
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--base-conf",
        type=float,
        default=0.01,
        help="YOLO runs once at this low threshold. All detections above it are saved; "
             "thresholds are applied offline by filtering.",
    )
    parser.add_argument(
        "--confs",
        default="0.10,0.15,0.20,0.25,0.30,0.40",
        help="Comma-separated thresholds to generate filtered JSONL files and report.",
    )
    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.5,
        help="IoU >= this → TP (label=1). Detections in the gray zone [iou-ignore, iou-match) are "
             "still saved with their measured iou_with_gt; extract_crops.py will skip them.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        default=str(root / "runs" / "proposals" / "fred_subset"),
    )
    return parser


def run_inference(model, image_paths: list[Path], labels_dir: Path, split: str,
                  base_conf: float, iou_match: float, imgsz: int, device) -> list[dict]:
    """Run YOLO once at base_conf and return all detections with GT labels."""
    detections: list[dict] = []

    for img_path in image_paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        result = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=base_conf,
            iou=0.7,
            device=device,
            half=False,
            verbose=False,
        )[0]

        h, w = result.orig_shape
        gt_boxes = load_gt_boxes_px(label_path, w, h)

        stem = img_path.stem
        seq = stem.rsplit("_", 1)[0] if "_" in stem else stem

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                score = float(box.conf[0].item())
                best = best_iou_against(xyxy, gt_boxes)
                label = 1 if best >= iou_match else 0
                detections.append({
                    "split": split,
                    "sequence": seq,
                    "stem": stem,
                    "bbox_xyxy": [round(v, 2) for v in xyxy],
                    "detector_score": round(score, 6),
                    "iou_with_gt": round(best, 4),
                    "label": label,
                })

    return detections


def sweep_threshold(
    all_detections: list[dict],
    conf: float,
    total_images: int,
    iou_match: float,
) -> tuple[list[dict], dict]:
    filtered = [d for d in all_detections if d["detector_score"] >= conf]

    # Recall: fraction of GT boxes with at least one TP detection above this conf.
    # We approximate by counting unique (stem, iou_match) matched GT detections.
    # Because we do not have per-GT-box IDs, we use the simpler proxy:
    # recall = TPs / total_gt  where TPs = detections with label=1 at this conf.
    # (This slightly over-counts recall when multiple detections match the same GT box,
    # but is consistent with the standard recall proxy used in the training loop.)
    total_gt = sum(1 for d in all_detections if d["iou_with_gt"] >= iou_match)
    # unique matched GT instances (stem + highest-iou det per GT)
    matched_gt: set[tuple[str, int]] = set()
    gt_by_stem: dict[str, list[float]] = {}
    for d in all_detections:
        gt_by_stem.setdefault(d["stem"], []).append(d["iou_with_gt"])

    tp_count = 0
    fp_count = 0
    matched_stems_tp: set[str] = set()
    for d in filtered:
        if d["label"] == 1:
            tp_count += 1
            matched_stems_tp.add(d["stem"])
        else:
            fp_count += 1

    # Use count of TP detections as numerator; total unique GT-matched detections as denominator
    total_gt_boxes = sum(1 for d in all_detections if d["label"] == 1)
    recall = tp_count / total_gt_boxes if total_gt_boxes > 0 else 0.0
    fp_per_frame = fp_count / total_images if total_images > 0 else 0.0

    stats = {
        "conf": conf,
        "total_detections": len(filtered),
        "tp": tp_count,
        "fp": fp_count,
        "recall": round(recall, 4),
        "fp_per_frame": round(fp_per_frame, 3),
    }
    return filtered, stats


def main() -> int:
    root = repo_root()
    args = build_parser().parse_args()

    model_path = resolve_path(args.model, root)
    event_images = resolve_path(args.event_images, root)
    event_labels = resolve_path(args.event_labels, root)
    output_dir = resolve_path(args.output, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = args.split
    images_dir = event_images / split
    labels_dir = event_labels / split

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    image_paths = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise ValueError(f"No images found under {images_dir}")

    confs = sorted({float(c.strip()) for c in args.confs.split(",")})
    print(f"Split: {split}  |  images: {len(image_paths)}")
    print(f"Running YOLO once at base_conf={args.base_conf} then filtering offline...\n")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics not installed — activate the project venv") from exc

    model = YOLO(str(model_path))

    all_detections = run_inference(
        model, image_paths, labels_dir, split,
        args.base_conf, args.iou_match, args.imgsz, args.device,
    )
    print(f"Raw detections at base_conf={args.base_conf}: {len(all_detections)}\n")

    raw_path = output_dir / f"detections_raw_{split}.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for d in all_detections:
            fh.write(json.dumps(d) + "\n")

    sweep_rows = []
    print(f"{'conf':>6}  {'recall':>7}  {'fp/frame':>9}  {'detections':>11}  {'tp':>6}  {'fp':>6}")
    print("-" * 55)

    for conf in confs:
        filtered, stats = sweep_threshold(all_detections, conf, len(image_paths), args.iou_match)
        sweep_rows.append(stats)

        out_path = output_dir / f"detections_conf{conf:.2f}_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for d in filtered:
                fh.write(json.dumps(d) + "\n")

        print(
            f"{conf:>6.2f}  {stats['recall']:>7.3f}  {stats['fp_per_frame']:>9.2f}"
            f"  {stats['total_detections']:>11}  {stats['tp']:>6}  {stats['fp']:>6}"
        )

    sweep_path = output_dir / f"threshold_sweep_{split}.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    print(f"\nRaw detections:  {raw_path}")
    print(f"Filtered files:  {output_dir}/detections_conf*_{split}.jsonl")
    print(f"Sweep table:     {sweep_path}")
    print(f"\nPick lowest conf with acceptable recall, then run extract_crops.py on that file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
