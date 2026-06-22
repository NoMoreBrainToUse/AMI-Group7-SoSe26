"""
Re-split an existing YOLO dataset using frame-difference diversity scoring.

The 50% of frames with the highest visual difference to their temporal
neighbour go to train. The remaining frames are split evenly into val/test.

Usage:
    python src/preprocessing/resplit.py
    python src/preprocessing/resplit.py --source fred_yolo_0 --output fred_yolo_1
    python src/preprocessing/resplit.py --train 0.50 --val 0.25

Output layout  (symlinks — no data is duplicated):
    dataset/<output>/
        images/          train / val / test  -> ../../../<source>/images/<orig-split>/<id>.jpg
        images_event/    train / val / test  -> symlinks to event frames
        labels/          train / val / test  -> hard copies (small txt files)
    configs/<output>/
        data_rgb.yaml
        data_event.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
CONFIG_ROOT  = PROJECT_ROOT / "configs"


# ── helpers ──────────────────────────────────────────────────────────────────

def resolve_manifest_path(rel: str) -> Path:
    """
    Manifest was written with images_rgb/ but the dataset on disk uses images/.
    Normalise so both resolve to the actual file.
    """
    p = PROJECT_ROOT / rel
    if not p.exists():
        p = Path(str(p).replace("/images_rgb/", "/images/"))
    return p


def frame_diff(path_a: Path, path_b: Path) -> float:
    """Mean absolute pixel difference between two RGB images (0-255 scale)."""
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        b = np.asarray(
            Image.open(path_b).convert("RGB").resize((a.shape[1], a.shape[0])),
            dtype=np.float32,
        )
    return float(np.mean(np.abs(a - b)))


def yaml_content(dataset_name: str, image_dir: str) -> str:
    return (
        f"path: {DATASET_ROOT / dataset_name}\n"
        f"train: {image_dir}/train\n"
        f"val:   {image_dir}/val\n"
        f"test:  {image_dir}/test\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: object\n"
    )


# ── core logic ────────────────────────────────────────────────────────────────

def load_manifest(source_root: Path) -> list[dict]:
    with open(source_root / "manifest.csv") as f:
        return list(csv.DictReader(f))


def score_frames(rows: list[dict], source_root: Path) -> list[float]:
    """
    Compute frame-to-frame L1 difference for every sample.

    Sample 0 gets score 0 (no predecessor). The rest get the mean absolute
    pixel difference to the previous frame in temporal order.
    """
    scores: list[float] = [0.0]
    print(f"  scoring {len(rows)} frames …", flush=True)
    for i in range(1, len(rows)):
        prev_rel = rows[i - 1]["rgb_dst"]          # e.g. dataset/fred_yolo_0/images/train/000001.jpg
        curr_rel = rows[i]["rgb_dst"]
        prev_path = resolve_manifest_path(prev_rel)
        curr_path = resolve_manifest_path(curr_rel)
        if prev_path.exists() and curr_path.exists():
            scores.append(frame_diff(prev_path, curr_path))
        else:
            scores.append(0.0)
        if i % 200 == 0:
            print(f"    {i}/{len(rows)}", flush=True)
    return scores


def assign_splits(
    rows: list[dict],
    scores: list[float],
    train_ratio: float,
    val_ratio: float,
    start_index: int = 80,
) -> list[str]:
    """
    Sort indices by score descending, assign top train_ratio → train,
    next val_ratio → val, rest → test.
    Keep val/test in temporal order so evaluation stays realistic.
    """
    n = len(rows)
    n_train = round(n * train_ratio)
    n_val   = round(n * val_ratio)

    candidate_pics = list(range(start_index, n)) #dont use first 80 bc of phone
    if n_train > len(range(start_index, n)):
        raise ValueError(
            f"Not enough candidate frames after index{start_index}:"
            f"need {n_train}, have {len(candidate_pics)}"
        )
                        

    order = sorted(candidate_pics, key=lambda i: scores[i], reverse=True)
    train_idx = set(order[:n_train])
    remaining = [i for i in range(n) if i not in train_idx]  # temporal order
    val_idx   = set(remaining[:n_val])

    splits = []
    for i in range(n):
        if i in train_idx:
            splits.append("train")
        elif i in val_idx:
            splits.append("val")
        else:
            splits.append("test")
    return splits


def build_dataset(
    rows: list[dict],
    new_splits: list[str],
    source_root: Path,
    output_root: Path,
) -> dict[str, int]:
    for modality in ("images", "event/images"):
        for split in ("train", "val", "test"):
            (output_root / modality / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_root / "labels"       / split).mkdir(parents=True, exist_ok=True)
        (output_root / "event/labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    for row, split in zip(rows, new_splits):
        sample_id = row["sample_id"]

        rgb_src   = resolve_manifest_path(row["rgb_dst"])
        event_src = resolve_manifest_path(row["event_dst"])
        label_src = resolve_manifest_path(row["label_dst"])

        rgb_ext   = rgb_src.suffix
        event_ext = event_src.suffix

        rgb_dst        = output_root / "images"        / split / f"{sample_id}{rgb_ext}"
        event_img_dst  = output_root / "event/images"  / split / f"{sample_id}{event_ext}"
        label_dst      = output_root / "labels"        / split / f"{sample_id}.txt"
        event_lbl_dst  = output_root / "event/labels"  / split / f"{sample_id}.txt"

        # RGB: symlink is fine — YOLO resolves to images/ which maps to labels/
        if not rgb_dst.exists():
            os.symlink(rgb_src.resolve(), rgb_dst)

        # Event: hardlink to the real file so Path.resolve() stays in this dataset tree
        if not event_img_dst.exists():
            os.link(event_src.resolve(), event_img_dst)

        # Labels: copies for both modalities (small, avoids cache conflicts)
        if label_src.exists():
            shutil.copy2(label_src, label_dst)
            shutil.copy2(label_src, event_lbl_dst)
        else:
            label_dst.write_text("")
            event_lbl_dst.write_text("")

        counts[split] += 1

    return counts


def write_configs(output_name: str) -> None:
    config_dir = CONFIG_ROOT / output_name
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "data_rgb.yaml").write_text(yaml_content(output_name, "images"))
    (config_dir / "data_event.yaml").write_text(
        yaml_content(output_name + "/event", "images")
        .replace(
            f"path: {DATASET_ROOT / output_name}/event",
            f"path: {DATASET_ROOT / output_name / 'event'}",
        )
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",  default="fred_yolo_0")
    p.add_argument("--output",  default="fred_yolo_1")
    p.add_argument("--train",   type=float, default=0.50)
    p.add_argument("--val",     type=float, default=0.25)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.train + args.val >= 1.0:
        raise SystemExit("--train + --val must be < 1.0")

    source_root = DATASET_ROOT / args.source
    output_root = DATASET_ROOT / args.output

    if output_root.exists() and not args.overwrite:
        raise SystemExit(f"{output_root} already exists. Pass --overwrite to rebuild.")
    if output_root.exists():
        shutil.rmtree(output_root)

    print(f"Source : {source_root}")
    print(f"Output : {output_root}")
    print(f"Split  : {args.train:.0%} / {args.val:.0%} / {1-args.train-args.val:.0%}  (train/val/test)")

    rows = load_manifest(source_root)
    print(f"\nStep 1/3  computing frame differences …")
    scores = score_frames(rows, source_root)

    print(f"\nStep 2/3  assigning splits …")
    new_splits = assign_splits(rows, scores, args.train, args.val, start_index=80)

    print(f"\nStep 3/3  building dataset …")
    counts = build_dataset(rows, new_splits, source_root, output_root)
    write_configs(args.output)

    print(f"\nDone.")
    for split, n in counts.items():
        print(f"  {split:6s}  {n} samples  ({n/len(rows):.1%})")
    print(f"\n  RGB   config : {CONFIG_ROOT / args.output / 'data_rgb.yaml'}")
    print(f"  Event config : {CONFIG_ROOT / args.output / 'data_event.yaml'}")


if __name__ == "__main__":
    main()
