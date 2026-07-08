#!/usr/bin/env python3
"""Export a labeled crop dataset for verifier training.

The inference pipeline scores crops in memory; verifier TRAINING needs them
on disk. This dumps drone/background crops (gray-zone IoU skipped) from a
proposals JSONL produced by the pipeline's detect phase.

Output:
  <out>/<modality>/<split>/{drone,background}/<stem>_boxNNNNNNN.jpg
  <out>/<modality>/crop_manifest_<modality>_<split>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_vision.common import expand_box, read_jsonl  # noqa: E402
from hybrid_vision.config import PipelineConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump labeled verifier-training crops from proposals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("proposals", type=Path,
                        help="proposals_*.jsonl from the pipeline detect phase")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="processed/<seq>/<modality>_yolo/images")
    parser.add_argument("--modality", choices=["rgb", "event"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-conf", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = PipelineConfig()
    ext = ".jpg" if args.modality == "rgb" else ".png"
    counts = {"drone": 0, "background": 0, "gray_zone": 0, "skipped": 0}
    manifest: list[dict] = []

    for idx, rec in enumerate(read_jsonl(args.proposals)):
        if rec["detector_score"] < args.min_conf:
            continue
        iou = rec.get("iou_with_gt", 0.0)
        if cfg.iou_ignore <= iou < cfg.iou_match:
            counts["gray_zone"] += 1
            continue
        img = cv2.imread(
            str(args.images_dir / args.split / f"{rec['stem']}{ext}"))
        if img is None:
            counts["skipped"] += 1
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = expand_box(rec["bbox_xyxy"], cfg.box_scale, w, h)
        if (x2 - x1) < cfg.min_box_px or (y2 - y1) < cfg.min_box_px:
            counts["skipped"] += 1
            continue
        crop = cv2.resize(img[y1:y2, x1:x2], (cfg.crop_size, cfg.crop_size),
                          interpolation=cv2.INTER_LINEAR)
        label_name = "drone" if rec["label"] == 1 else "background"
        out = (args.output_dir / args.modality / args.split / label_name
               / f"{rec['stem']}_box{idx:07d}.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        counts[label_name] += 1
        manifest.append({**rec, "crop_path": str(out)})

    manifest_path = (args.output_dir / args.modality
                     / f"crop_manifest_{args.modality}_{args.split}.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")

    print(f"crops: {counts}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
