#!/usr/bin/env python3
"""
Extract fixed-size crops from RGB or event images at event-YOLO detection boxes.

Reads a detections JSONL (from export_proposals.py), looks up the paired
image for each detection, expands the box by --box-scale, and saves a crop.

Use --modality rgb  (default) to crop from rgb_yolo/images/
Use --modality event          to crop from event_yolo/images/  (ext .png)

IoU labeling rules:
  iou_with_gt >= 0.5          → drone      (positive)
  iou_with_gt < 0.3           → background (negative)
  0.3 <= iou < 0.5            → IGNORED    (gray zone)

Output directory structure:
  <output-dir>/<modality>/<split>/drone/
  <output-dir>/<modality>/<split>/background/

Manifest:
  <output-dir>/<modality>/crop_manifest_<modality>_<split>_conf<X.XX>.jsonl

Example:
  # RGB crops
  python src/verifier/extract_crops.py \\
    --detections runs/proposals/fred_subset/detections_conf0.20_val.jsonl

  # Event crops (same detections, different image source)
  python src/verifier/extract_crops.py \\
    --detections runs/proposals/fred_subset/detections_conf0.20_val.jsonl \\
    --modality event
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def parse_conf_from_stem(stem: str) -> str:
    """Extract conf value string from filename stem like detections_conf0.20_val."""
    m = re.search(r"conf([\d.]+)", stem)
    return m.group(1) if m else "unknown"


def parse_split_from_stem(stem: str) -> str | None:
    """Extract split name from filename stem like detections_conf0.20_val."""
    for s in ("train", "val", "test"):
        if stem.endswith(f"_{s}"):
            return s
    return None


def expand_box(box: list[float], scale: float, img_w: int, img_h: int) -> list[int]:
    """Expand a box around its centre by scale, clamped to image bounds."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    ex1 = max(0, int(cx - bw / 2))
    ey1 = max(0, int(cy - bh / 2))
    ex2 = min(img_w, int(cx + bw / 2 + 0.5))
    ey2 = min(img_h, int(cy + bh / 2 + 0.5))
    return [ex1, ey1, ex2, ey2]


MODALITY_DEFAULTS = {
    "rgb":   {"images_subdir": "rgb_yolo/images",   "ext": ".jpg"},
    "event": {"images_subdir": "event_yolo/images", "ext": ".png"},
}


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Crop RGB or event images at event-YOLO detection boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detections",
        required=True,
        help="Detections JSONL from export_proposals.py.",
    )
    parser.add_argument(
        "--modality",
        default="rgb",
        choices=["rgb", "event"],
        help="Which image source to crop. Controls default --images-dir and --img-ext.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Override the image root directory (default auto-set from --modality).",
    )
    parser.add_argument(
        "--img-ext",
        default=None,
        help="Override the image file extension (default auto-set from --modality).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "processed" / "fred_subset" / "crops"),
        help="Root output directory for crops and manifest.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="Side length (pixels) of the square output crop.",
    )
    parser.add_argument(
        "--box-scale",
        type=float,
        default=1.5,
        help="Expand each detection box by this factor before cropping.",
    )
    parser.add_argument(
        "--iou-positive",
        type=float,
        default=0.5,
        help="iou_with_gt >= this → drone (positive).",
    )
    parser.add_argument(
        "--iou-ignore",
        type=float,
        default=0.3,
        help="iou_with_gt in [iou-ignore, iou-positive) → gray zone, skipped.",
    )
    parser.add_argument(
        "--min-box-px",
        type=int,
        default=4,
        help="Skip boxes with expanded side < this many pixels after clamping.",
    )
    parser.add_argument(
        "--conf",
        default=None,
        help="Confidence threshold string for manifest filename (auto-parsed from input if omitted).",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "val", "test"],
        help="Split override (auto-parsed from input filename if omitted).",
    )
    return parser


def main() -> int:
    root = repo_root()
    args = build_parser().parse_args()

    det_path = resolve_path(args.detections, root)
    output_dir = resolve_path(args.output_dir, root)

    # Resolve image source from modality or explicit override
    modality = args.modality
    defaults = MODALITY_DEFAULTS[modality]
    images_root = resolve_path(
        args.images_dir if args.images_dir else root / "processed" / "fred_subset" / defaults["images_subdir"],
        root,
    )
    img_ext = args.img_ext if args.img_ext else defaults["ext"]

    if not det_path.is_file():
        raise FileNotFoundError(f"Detections file not found: {det_path}")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python not installed") from exc

    records = [
        json.loads(line)
        for line in det_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        print("No detections found in file.")
        return 0

    # Determine split and conf for manifest naming
    stem = det_path.stem
    conf_str = args.conf if args.conf is not None else parse_conf_from_stem(stem)
    split = args.split if args.split is not None else parse_split_from_stem(stem)
    if split is None:
        # Fall back to split stored in records
        split = records[0].get("split", "unknown")
    print(f"Modality: {modality}  |  Split: {split}  |  Conf: {conf_str}  |  Detections: {len(records)}")

    label_names = {1: "drone", 0: "background"}
    counts: dict[str, int] = {"drone": 0, "background": 0, "gray_zone": 0, "missing": 0, "bad_box": 0}
    manifest: list[dict] = []

    for idx, rec in enumerate(records):
        iou_gt = rec.get("iou_with_gt", 0.0)

        if args.iou_ignore <= iou_gt < args.iou_positive:
            counts["gray_zone"] += 1
            continue

        label = rec["label"]
        box = rec["bbox_xyxy"]
        rec_split = rec.get("split", split)

        img_path = images_root / rec_split / f"{rec['stem']}{img_ext}"
        if not img_path.is_file():
            counts["missing"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            counts["missing"] += 1
            continue

        h, w = img.shape[:2]
        ex1, ey1, ex2, ey2 = expand_box(box, args.box_scale, w, h)

        if (ex2 - ex1) < args.min_box_px or (ey2 - ey1) < args.min_box_px:
            counts["bad_box"] += 1
            continue

        crop = img[ey1:ey2, ex1:ex2]
        crop_resized = cv2.resize(
            crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR
        )

        label_name = label_names[label]
        out_path = (
            output_dir / modality / rec_split / label_name
            / f"{rec['stem']}_box{idx:07d}.jpg"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        counts[label_name] += 1
        manifest.append({
            **rec,
            "crop_path": str(out_path),
            "crop_box_expanded": [ex1, ey1, ex2, ey2],
        })

    # Manifest: crop_manifest_<modality>_<split>_conf<X.XX>.jsonl
    manifest_path = output_dir / modality / f"crop_manifest_{modality}_{split}_conf{conf_str}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")

    print(f"\nCrops saved:")
    print(f"  drone (positive):      {counts['drone']}")
    print(f"  background (negative): {counts['background']}")
    print(f"  gray zone skipped:     {counts['gray_zone']}")
    print(f"  missing image skipped: {counts['missing']}")
    print(f"  bad box skipped:       {counts['bad_box']}")
    print(f"\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
