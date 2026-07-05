#!/usr/bin/env python3
"""Assemble example_result/: one showcase sequence (default 31) through every
pipeline stage — preprocessed RGB + event videos, E2VID and E2VID++
reconstructions, the hybrid E2VID composite, and both trained YOLO11m models
run on the hybrid composite.

All videos are on the manifest ~30 fps grid (real-time playback), 1280x720.

  external/V2V/.venv/bin/python scripts/presentation/build_example_result.py
"""

import argparse
import csv
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30.0
HYBRID_OUT = Path("/home/spacezhang/Desktop/AMI_Course/reconstruction/reconstruction_hybrid/out")


def rows_for(seq):
    pre = Path(f"data/preprocessed_all_val/preprocessed_fred_{seq}")
    rows = [r for r in csv.DictReader(open(pre / "paired/manifest_val.csv"))
            if r.get("split") == "val"]
    rows.sort(key=lambda r: float(r["event_time_s"]))
    return pre, rows


def write_video(path, frame_iter, n):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                         (W, H))
    for k, im in enumerate(frame_iter):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        vw.write(im)
        if k % 500 == 0:
            print(f"  {path.name}: {k}/{n}")
    vw.release()
    print("wrote", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="31")
    ap.add_argument("--out", type=Path, default=Path("example_result"))
    ap.add_argument("--only-yolo", action="store_true",
                    help="only regenerate the YOLO prediction videos")
    args = ap.parse_args()
    seq = args.seq
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    pre, rows = rows_for(seq)
    n = len(rows)
    print(f"seq{seq}: {n} samples")

    if args.only_yolo:
        run_yolo_stage(out, seq, pre, rows, n)
        return 0

    # 1) preprocessed RGB / event frame videos
    write_video(out / f"preprocessed_rgb_seq{seq}.mp4",
                (cv2.imread(str(pre / r["rgb_image"])) for r in rows), n)
    write_video(out / f"preprocessed_event_seq{seq}.mp4",
                (cv2.imread(str(pre / r["event_image"])) for r in rows), n)

    # 2) E2VID (rpg_e2vid): frame names are event indices -> map via the
    #    time-mapping npys built by the hybrid pipeline
    names = np.load(HYBRID_OUT / f"e2vid_names_{seq}.npy")
    frame_t = np.load(HYBRID_OUT / f"e2vid_frame_t_{seq}.npy")
    e2vid_dir = Path(f"external/rpg_e2vid/output_tests/all_120fps/seq{seq}_120fps")

    def e2vid_frames():
        for r in rows:
            j = int(np.abs(frame_t - float(r["event_time_s"])).argmin())
            yield cv2.imread(str(e2vid_dir / f"frame_{names[j]:010d}.png"))
    write_video(out / f"reconstructed_e2vid_seq{seq}.mp4", e2vid_frames(), n)

    # 3) E2VID++ (V2V/EVBIRD): reconstruction i corresponds to row i+1
    v2v_dir = Path(f"external/V2V/generated_tests/seq{seq}_{n-1}frames"
                   f"/results/EVBIRD/seq{seq}_{n-1}frames")
    v2v = sorted(v2v_dir.glob("*.png"), key=lambda p: int(p.stem))
    write_video(out / f"reconstructed_e2vidpp_seq{seq}.mp4",
                (cv2.imread(str(p)) for p in v2v), len(v2v))

    # 4) hybrid E2VID composite (rendered by reconstruction_hybrid)
    shutil.copy2(HYBRID_OUT / f"composite_grid_seq{seq}_clean.mp4",
                 out / f"hybrid_e2vid_composite_seq{seq}.mp4")
    print("copied hybrid composite")

    run_yolo_stage(out, seq, pre, rows, n)
    return 0


def run_yolo_stage(out, seq, pre, rows, n):
    # 5) each trained YOLO11m run on its own reconstruction (in-domain):
    #    E2VID++-trained model on the E2VID++ video, E2VID-trained on E2VID
    from ultralytics import YOLO
    runs = Path("artifacts/yolo/runs")
    models = {
        "e2vidpp": (runs / "e2vid_multiseq_yolo11m/weights/best.pt",
                    out / f"reconstructed_e2vidpp_seq{seq}.mp4"),
        "e2vid": (runs / "rpge2vid_multiseq_yolo11m/weights/best.pt",
                  out / f"reconstructed_e2vid_seq{seq}.mp4"),
    }
    for tag, (wpath, src_video) in models.items():
        m = YOLO(str(wpath))
        cap = cv2.VideoCapture(str(src_video))
        vw = cv2.VideoWriter(
            str(out / f"yolo11m_on_{tag}_seq{seq}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for k, r in enumerate(rows):
            ok, im = cap.read()
            if not ok:
                break
            lab = open(pre / r["rgb_label"]).read().split()
            if lab:
                cx, cy, bw, bh = (float(v) for v in lab[1:5])
                cv2.rectangle(im,
                              (int((cx - bw / 2) * W), int((cy - bh / 2) * H)),
                              (int((cx + bw / 2) * W), int((cy + bh / 2) * H)),
                              (0, 200, 0), 1)
            for b in m.predict(im, conf=0.25, verbose=False)[0].boxes:
                x0, y0, x1, y1 = (int(v) for v in b.xyxy[0])
                cv2.rectangle(im, (x0, y0), (x1, y1), (255, 200, 0), 2)
                cv2.putText(im, f"{float(b.conf):.2f}",
                            (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 200, 0), 1, cv2.LINE_AA)
            vw.write(im)
            if k % 500 == 0:
                print(f"  yolo {tag}: {k}/{n}")
        cap.release()
        vw.release()
        print(f"wrote yolo11m_on_{tag}_seq{seq}.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
