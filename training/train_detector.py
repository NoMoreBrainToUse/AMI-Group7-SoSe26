#!/usr/bin/env python3
"""Train a YOLO11 drone detector on aligned FRED frames (either modality).

The dataset comes from hybrid_vision.align (one sequence per call) or any
directory tree in the same rgb_yolo/event_yolo layout with a data.yaml.

Examples:
  # Event detector (trained at 640 — event frames are low-texture)
  python training/train_detector.py processed/train_set/event_yolo/data.yaml \\
      --name event_yolo11m --imgsz 640

  # RGB detector (1280: FRED drones are small; 640 halves a 57 px drone
  # to 28 px and was the main reason the first RGB model failed)
  python training/train_detector.py processed/train_set/rgb_yolo/data.yaml \\
      --name rgb_yolo11m --imgsz 1280
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train YOLO11 on aligned FRED frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_yaml", type=Path)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--name", default="detector")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 = Ultralytics autobatch")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mosaic", type=float, default=None,
                        help="Override mosaic augmentation; set 0 for a "
                             "confidence-sharpening fine-tune.")
    parser.add_argument("--lr0", type=float, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    kwargs = dict(
        data=str(args.data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, patience=args.patience, device=args.device,
        project=str(args.project), name=args.name, seed=args.seed,
        amp=True, exist_ok=True)
    if args.mosaic is not None:
        kwargs["mosaic"] = args.mosaic
    if args.lr0 is not None:
        kwargs["lr0"] = args.lr0

    YOLO(args.model).train(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
