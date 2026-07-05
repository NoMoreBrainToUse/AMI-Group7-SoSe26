#!/usr/bin/env python3
"""Build the rpg_e2vid (E2VID 120fps) YOLO dataset with the same
sequence-level split and 12.5 fps time grid as prepare_e2vid_yolo_multiseq.py
(V2V/EVBIRD), so the two reconstruction methods are directly comparable.

rpg_e2vid output frames are named by the index of the last event in each
1/120 s window, not by timestamp. The name->real-time mapping npy files
(e2vid_names_<seq>.npy / e2vid_frame_t_<seq>.npy) built by the reconstruction_hybrid
pipeline are reused via --mapping-dir; for each manifest row on the grid the
temporally nearest e2vid frame is taken, with the same label file as the V2V
dataset.
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

TRAIN_SEQS = [18, 26, 33, 36, 40, 46, 49]
VAL_SEQS = [19, 31, 34, 43]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--output-root", type=Path,
                   default=Path("artifacts/yolo/rpge2vid_multiseq"))
    p.add_argument("--mapping-dir", type=Path,
                   default=Path("/home/spacezhang/Desktop/AMI_Course/reconstruction/reconstruction_hybrid/out"))
    p.add_argument("--target-fps", type=float, default=12.5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("split") == "val"]
    rows.sort(key=lambda r: float(r["event_time_s"]))
    return rows


def select_grid_indices(times: list[float], fps: float) -> list[int]:
    step = 1.0 / fps
    t0 = times[0]
    sel, last, i = [], -1, 0
    for k in range(int((times[-1] - t0) / step) + 1):
        t = t0 + k * step
        while i + 1 < len(times) and abs(times[i + 1] - t) <= abs(times[i] - t):
            i += 1
        if i > last:
            sel.append(i)
            last = i
    return sel


def build_sequence(repo_root: Path, seq: int, split: str, out_root: Path,
                   fps: float, mapping_dir: Path) -> int:
    pre = repo_root / "data" / "preprocessed_all_val" / f"preprocessed_fred_{seq}"
    rows = read_manifest_rows(pre / "paired" / "manifest_val.csv")
    e2vid_dir = (repo_root / "external" / "rpg_e2vid" / "output_tests"
                 / "all_120fps" / f"seq{seq}_120fps")
    names = np.load(mapping_dir / f"e2vid_names_{seq}.npy")
    frame_t = np.load(mapping_dir / f"e2vid_frame_t_{seq}.npy")

    # keep row indexing identical to the V2V dataset (rows[1:] on the grid)
    times = [float(r["event_time_s"]) for r in rows[1:]]
    chosen = select_grid_indices(times, fps)

    img_dir = out_root / "images" / split
    lab_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for out_idx, src_idx in enumerate(chosen):
        row = rows[src_idx + 1]
        j = int(np.abs(frame_t - float(row["event_time_s"])).argmin())
        src_img = e2vid_dir / f"frame_{names[j]:010d}.png"
        stem = f"seq{seq}_{out_idx:06d}"
        shutil.copy2(src_img, img_dir / f"{stem}.png")
        label_src = pre / row["rgb_label"]
        if label_src.exists():
            shutil.copy2(label_src, lab_dir / f"{stem}.txt")
        else:
            (lab_dir / f"{stem}.txt").write_text("", encoding="utf-8")
    return len(chosen)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_root = (repo_root / args.output_root).resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split, seqs in (("train", TRAIN_SEQS), ("val", VAL_SEQS)):
        total = 0
        for seq in seqs:
            n = build_sequence(repo_root, seq, split, out_root,
                               args.target_fps, args.mapping_dir)
            print(f"{split} seq{seq}: {n} samples")
            total += n
        print(f"{split} total: {total}")

    (out_root / "data.yaml").write_text(
        "\n".join([f"path: {out_root}", "train: images/train",
                   "val: images/val", "names:", "  0: drone", ""]),
        encoding="utf-8")
    print(f"data yaml: {out_root/'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
