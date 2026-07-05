#!/usr/bin/env python3
"""Validate a trained YOLO checkpoint on a multiseq dataset, overall and per
validation sequence (19 / 31 / 34 / 43), printing one line per split.

  external/V2V/.venv/bin/python scripts/detection/val_yolo_per_sequence.py \
      --weights <path>/best.pt --dataset artifacts/yolo/e2vid_multiseq
"""

import argparse
import glob
from pathlib import Path

from ultralytics import YOLO

VAL_SEQS = [19, 31, 34, 43]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True,
                    help="dataset root containing data.yaml + images/val")
    ap.add_argument("--run-prefix", default="finalval")
    args = ap.parse_args()

    ds = Path(args.dataset).resolve()
    m = YOLO(args.weights)

    r = m.val(data=str(ds / "data.yaml"), verbose=False,
              project="artifacts/yolo/runs", name=f"{args.run_prefix}_all",
              exist_ok=True)
    print(f"ALL: mAP50={r.box.map50:.3f} mAP50-95={r.box.map:.3f} "
          f"P={r.box.mp:.3f} R={r.box.mr:.3f}")

    for s in VAL_SEQS:
        lst = ds / f"val_seq{s}.txt"
        lst.write_text("\n".join(
            sorted(glob.glob(str(ds / "images" / "val" / f"seq{s}_*.png"))))
            + "\n", encoding="utf-8")
        yaml = ds / f"data_seq{s}.yaml"
        yaml.write_text(f"path: {ds}\ntrain: images/train\n"
                        f"val: val_seq{s}.txt\nnames:\n  0: drone\n",
                        encoding="utf-8")
        r = m.val(data=str(yaml), verbose=False, project="artifacts/yolo/runs",
                  name=f"{args.run_prefix}_seq{s}", exist_ok=True)
        print(f"SEQ{s}: mAP50={r.box.map50:.3f} mAP50-95={r.box.map:.3f} "
              f"P={r.box.mp:.3f} R={r.box.mr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
