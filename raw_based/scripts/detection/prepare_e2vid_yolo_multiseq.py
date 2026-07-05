#!/usr/bin/env python3
"""Build a sequence-level train/val YOLO dataset from V2V (EVBIRD)
reconstructions of all fully reconstructed FRED sequences.

Unlike prepare_e2vid_yolo_seq18_31.py (train seq18 / val seq31 on a shared
clamped time grid), this samples every sequence over its full duration on a
fixed-fps grid and splits at sequence level, one held-out sequence per
scene/lighting group:

  train: 18, 26, 33, 36, 40, 46, 49
  val:   19 (wall/day), 31 (courtyard/dusk), 34 (dark), 43 (garden/bright)

Reconstructed frame i corresponds to manifest row i+1 (rows sorted by
event_time_s), matching the V2V run convention.
"""

import argparse
import csv
import shutil
from pathlib import Path

TRAIN_SEQS = [18, 26, 33, 36, 40, 46, 49]
VAL_SEQS = [19, 31, 34, 43]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--output-root", type=Path,
                   default=Path("artifacts/yolo/e2vid_multiseq"))
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
                   fps: float) -> int:
    pre = repo_root / "data" / "preprocessed_all_val" / f"preprocessed_fred_{seq}"
    rows = read_manifest_rows(pre / "paired" / "manifest_val.csv")
    recon_dir = (repo_root / "external" / "V2V" / "generated_tests"
                 / f"seq{seq}_{len(rows)-1}frames" / "results" / "EVBIRD"
                 / f"seq{seq}_{len(rows)-1}frames")
    frames = sorted(recon_dir.glob("*.png"), key=lambda p: int(p.stem))
    if len(frames) != len(rows) - 1:
        raise RuntimeError(
            f"seq{seq}: {len(frames)} recon frames, expected {len(rows)-1}")

    times = [float(r["event_time_s"]) for r in rows[1:]]
    chosen = select_grid_indices(times, fps)

    img_dir = out_root / "images" / split
    lab_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for out_idx, src_idx in enumerate(chosen):
        stem = f"seq{seq}_{out_idx:06d}"
        shutil.copy2(frames[src_idx], img_dir / f"{stem}.png")
        label_src = pre / rows[src_idx + 1]["rgb_label"]
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
            n = build_sequence(repo_root, seq, split, out_root, args.target_fps)
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
