#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a time-aligned YOLO dataset from reconstructed E2Vid seq18 and seq31 frames.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/yolo/e2vid_seq18_seq31_timealigned"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Common time grid fps for both sequences.",
    )
    parser.add_argument(
        "--train-sequence",
        type=int,
        default=18,
        help="Sequence id used for train split.",
    )
    parser.add_argument(
        "--val-sequence",
        type=int,
        default=31,
        help="Sequence id used for val split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") == "val":
                rows.append(row)
    rows.sort(key=lambda row: float(row["event_time_s"]))
    return rows


def parse_reconstructed_index(frame_path: Path) -> int:
    # V2V outputs 000000.png, 000001.png, ...
    return int(frame_path.stem)


def select_time_aligned_indices(times: list[float], target_fps: float, max_duration: float) -> list[int]:
    if not times:
        return []

    start_t = times[0]
    rel_times = [t - start_t for t in times]
    step = 1.0 / target_fps
    sample_count = int(max_duration / step) + 1

    selected: list[int] = []
    last_idx = -1
    for i in range(sample_count):
        t = i * step
        # Find nearest available frame timestamp.
        best_idx = min(range(len(rel_times)), key=lambda k: abs(rel_times[k] - t))
        if best_idx <= last_idx:
            continue
        selected.append(best_idx)
        last_idx = best_idx

    return selected


def copy_sample(
    image_src: Path,
    label_src: Path,
    image_dst: Path,
    label_dst: Path,
) -> None:
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_src, image_dst)
    if label_src.exists():
        shutil.copy2(label_src, label_dst)
    else:
        label_dst.write_text("", encoding="utf-8")


def build_split(
    repo_root: Path,
    sequence_id: int,
    split_name: str,
    output_root: Path,
    target_fps: float,
    max_duration: float,
) -> tuple[int, float]:
    preprocessed_dir = repo_root / "data" / "preprocessed_all_val" / f"preprocessed_fred_{sequence_id}"
    manifest_path = preprocessed_dir / "paired" / "manifest_val.csv"
    rows = read_manifest_rows(manifest_path)

    recon_dir = repo_root / "external" / "V2V" / "generated_tests" / f"seq{sequence_id}_{len(rows)-1}frames" / "results" / "EVBIRD" / f"seq{sequence_id}_{len(rows)-1}frames"
    recon_frames = sorted(recon_dir.glob("*.png"), key=parse_reconstructed_index)

    if len(recon_frames) != len(rows) - 1:
        raise RuntimeError(
            f"Unexpected reconstructed frame count for seq{sequence_id}: got {len(recon_frames)}, expected {len(rows)-1}"
        )

    event_times = [float(r["event_time_s"]) for r in rows[1:]]
    rel_duration = event_times[-1] - event_times[0]
    use_duration = min(rel_duration, max_duration)
    chosen = select_time_aligned_indices(event_times, target_fps, use_duration)

    for out_idx, src_idx in enumerate(chosen):
        row = rows[src_idx + 1]
        src_image = recon_frames[src_idx]
        src_label = (preprocessed_dir / row["rgb_label"]).resolve()

        stem = f"seq{sequence_id}_{out_idx:06d}"
        dst_image = output_root / "images" / split_name / f"{stem}.png"
        dst_label = output_root / "labels" / split_name / f"{stem}.txt"
        copy_sample(src_image, src_label, dst_image, dst_label)

    return len(chosen), use_duration


def write_data_yaml(dataset_root: Path) -> Path:
    yaml_path = dataset_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: drone",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = (repo_root / args.output_root).resolve()

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_manifest = repo_root / "data" / "preprocessed_all_val" / f"preprocessed_fred_{args.train_sequence}" / "paired" / "manifest_val.csv"
    val_manifest = repo_root / "data" / "preprocessed_all_val" / f"preprocessed_fred_{args.val_sequence}" / "paired" / "manifest_val.csv"
    train_rows = read_manifest_rows(train_manifest)
    val_rows = read_manifest_rows(val_manifest)

    train_duration = float(train_rows[-1]["event_time_s"]) - float(train_rows[1]["event_time_s"])
    val_duration = float(val_rows[-1]["event_time_s"]) - float(val_rows[1]["event_time_s"])
    max_duration = min(train_duration, val_duration)

    train_count, _ = build_split(
        repo_root=repo_root,
        sequence_id=args.train_sequence,
        split_name="train",
        output_root=output_root,
        target_fps=args.target_fps,
        max_duration=max_duration,
    )
    val_count, _ = build_split(
        repo_root=repo_root,
        sequence_id=args.val_sequence,
        split_name="val",
        output_root=output_root,
        target_fps=args.target_fps,
        max_duration=max_duration,
    )

    yaml_path = write_data_yaml(output_root)

    print(f"Dataset root: {output_root}")
    print(f"Train sequence: {args.train_sequence}, samples: {train_count}")
    print(f"Val sequence: {args.val_sequence}, samples: {val_count}")
    print(f"Time-aligned duration used: {max_duration:.3f}s")
    print(f"Target fps: {args.target_fps:.3f}")
    print(f"Data yaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
