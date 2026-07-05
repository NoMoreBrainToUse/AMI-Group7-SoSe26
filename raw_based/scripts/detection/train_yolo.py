#!/usr/bin/env python3
"""Train YOLO11m on a reconstruction-based dataset built by one of the
prepare_*_yolo_multiseq.py scripts.

Run from the repo root with the V2V venv, e.g.:

  external/V2V/.venv/bin/python scripts/detection/train_yolo.py \
      --data artifacts/yolo/e2vid_multiseq/data.yaml \
      --name e2vid_multiseq_yolo11m

Note: ultralytics prefixes its global runs_dir, so results land under
external/V2V/runs/detect/<project>/<name> unless runs_dir is configured;
consolidate into artifacts/yolo/runs afterwards if needed.
"""

import argparse

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8,
                    help="8 fits an 8GB RTX 4060; 16 OOMs")
    args = ap.parse_args()

    m = YOLO(args.model)
    m.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=0, workers=8, seed=0,
            project="artifacts/yolo/runs", name=args.name, exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
