#!/usr/bin/env python3
"""Run a trained YOLO checkpoint over one validation sequence of a multiseq
dataset and write an annotated video (cyan = predictions with confidence,
green = ground-truth box). Frames are on the 12.5 fps grid, so 12.5 fps
playback is real time.

  external/V2V/.venv/bin/python scripts/detection/predict_yolo_video.py \
      --weights <path>/best.pt --dataset artifacts/yolo/e2vid_multiseq \
      --seq 31 --out artifacts/yolo/runs/pred_seq31.mp4
"""

import argparse
import glob

import cv2
from ultralytics import YOLO

W, H = 1280, 720


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", default="31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    m = YOLO(args.weights)
    imgs = sorted(glob.glob(
        f"{args.dataset}/images/val/seq{args.seq}_*.png"))
    if not imgs:
        raise SystemExit(f"no val images for seq{args.seq} in {args.dataset}")
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         12.5, (W, H))
    for i in range(0, len(imgs), 64):
        batch = imgs[i:i + 64]
        for p, r in zip(batch, m.predict(batch, conf=args.conf,
                                         verbose=False)):
            im = cv2.imread(p)
            lab = open(p.replace("/images/", "/labels/")
                       .replace(".png", ".txt")).read().split()
            if lab:
                cx, cy, bw, bh = (float(v) for v in lab[1:5])
                cv2.rectangle(im,
                              (int((cx - bw / 2) * W), int((cy - bh / 2) * H)),
                              (int((cx + bw / 2) * W), int((cy + bh / 2) * H)),
                              (0, 200, 0), 1)
            for b in r.boxes:
                x0, y0, x1, y1 = (int(v) for v in b.xyxy[0])
                cv2.rectangle(im, (x0, y0), (x1, y1), (255, 200, 0), 2)
                cv2.putText(im, f"{float(b.conf):.2f}", (x0, max(y0 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
                            cv2.LINE_AA)
            vw.write(im)
    vw.release()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
