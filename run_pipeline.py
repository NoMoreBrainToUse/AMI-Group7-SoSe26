#!/usr/bin/env python3
"""Run the hybrid drone-detection pipeline on one FRED sequence.

Examples:
  python run_pipeline.py dataset/40
  python run_pipeline.py /data/fred/31 --device cpu
  python run_pipeline.py dataset/40 --rgb-model weights/rgb_yolo11m_v6.pt \\
      --rgb-imgsz 1280 --overwrite

The sequence directory must contain PADDED_RGB/, Event/Frames/ and
interpolated_coordinates.txt (or coordinates.txt) as shipped in FRED zips.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hybrid_vision import PipelineConfig, run_sequence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sequence", type=Path,
                        help="FRED sequence directory, e.g. dataset/40")
    parser.add_argument("--rgb-model", type=Path, default=None,
                        help="Override the RGB detector weights.")
    parser.add_argument("--rgb-imgsz", type=int, default=None,
                        help="Inference size for the RGB detector "
                             "(must match how the weights were trained).")
    parser.add_argument("--rgb-min-conf", type=float, default=None,
                        help="Confidence gate for RGB proposals.")
    parser.add_argument("--device", default=None,
                        help="cuda, cpu, or 0/1/... (default: auto)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute all phases even if artifacts exist.")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.rgb_model is not None:
        cfg.rgb_model = args.rgb_model
    if args.rgb_imgsz is not None:
        cfg.rgb_imgsz = args.rgb_imgsz
    if args.rgb_min_conf is not None:
        cfg.rgb_min_conf = args.rgb_min_conf
    if args.device is not None:
        cfg.device = args.device

    run_sequence(args.sequence, cfg, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
