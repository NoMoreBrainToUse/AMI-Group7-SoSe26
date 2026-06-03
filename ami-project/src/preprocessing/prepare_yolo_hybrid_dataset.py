"""Prepare a local YOLO hybrid dataset under the test folder.

Run from any working directory:

    python test/src/preprocessing/prepare_yolo_hybrid_dataset.py

Select the source sequence by editing SOURCE_SEQUENCE in the script config
section below.

The script reads one selected local source sequence from:

    test/dataset/<sequence>/

and writes the project-style layout:

    test/dataset/yolo_hybrid<sequence>/
    test/configs/yolo_hybrid<sequence>/data_rgb.yaml
    test/configs/yolo_hybrid<sequence>/data_event.yaml

RGB frames and Event frames are paired by timestamp and renamed to the same
sample id so later YOLO/YOLO-hybrid code can address both modalities with the
same stem.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


TEST_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = TEST_ROOT / "dataset"
CONFIG_ROOT = TEST_ROOT / "configs"
OUTPUT_PREFIX = "yolo_hybrid"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_POLICIES = {"rgb", "event", "merge"}

# Script config: edit these values, then run this file directly.
SOURCE_SEQUENCE = "17"
OUTPUT_NAME = None
RGB_DIR = "RGB"
LABEL_DIR = "RGB_YOLO"
EVENT_LABEL_DIR = "Event_YOLO"
LABEL_POLICY = "merge"
MERGE_IOU_THRESHOLD = 0.80
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
MAX_DELTA_MS = 10.0
OVERWRITE_OUTPUT = True


def rgb_yaml(dataset_name: str) -> str:
    return f"""path: dataset/{dataset_name}
train: images_rgb/train
val: images_rgb/val
test: images_rgb/test
nc: 1
names:
  0: object
"""


def event_yaml(dataset_name: str) -> str:
    return f"""path: dataset/{dataset_name}
train: images_event/train
val: images_event/val
test: images_event/test
nc: 1
names:
  0: object
"""


def hybrid_yaml(dataset_name: str) -> str:
    return f"""path: dataset/{dataset_name}
train: images_rgb/train
val: images_rgb/val
test: images_rgb/test
train_rgb: images_rgb/train
val_rgb: images_rgb/val
test_rgb: images_rgb/test
train_event: images_event/train
val_event: images_event/val
test_event: images_event/test
nc: 1
names:
  0: object
"""


@dataclass(frozen=True)
class Frame:
    path: Path
    time_s: float


@dataclass(frozen=True)
class Pair:
    sequence: str
    rgb: Frame
    event: Frame
    label_path: Path
    event_label_path: Path
    delta_ms: float


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


def parse_event_time(path: Path) -> Optional[float]:
    """Parse Event frame timestamps such as Video_230_33333.png."""
    match = re.search(r"_(\d+)$", path.stem)
    if not match:
        return None
    return int(match.group(1)) / 1_000_000.0


def parse_rgb_clock_time(path: Path) -> Optional[float]:
    """Parse RGB wall-clock filenames such as Video_230_17_16_02.213938.jpg."""
    match = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{1,2})\.(\d+)$", path.stem)
    if not match:
        return None

    hour, minute, second, fraction = match.groups()
    return (
        int(hour) * 3600
        + int(minute) * 60
        + int(second)
        + float(f"0.{fraction}")
    )


def collect_frames(directory: Path, parser) -> list[Frame]:
    if not directory.exists():
        return []

    frames: list[Frame] = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        time_s = parser(path)
        if time_s is None:
            continue
        frames.append(Frame(path=path, time_s=time_s))

    return sorted(frames, key=lambda frame: (frame.time_s, frame.path.name))


def count_objects(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_yolo_boxes(label_path: Path) -> list[YoloBox]:
    if not label_path.exists():
        return []

    boxes: list[YoloBox] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {label_path}: {raw_line!r}")
        boxes.append(
            YoloBox(
                class_id=int(parts[0]),
                center_x=float(parts[1]),
                center_y=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return boxes


def box_bounds(box: YoloBox) -> tuple[float, float, float, float]:
    half_width = box.width / 2.0
    half_height = box.height / 2.0
    return (
        box.center_x - half_width,
        box.center_y - half_height,
        box.center_x + half_width,
        box.center_y + half_height,
    )


def yolo_iou(first: YoloBox, second: YoloBox) -> float:
    first_left, first_top, first_right, first_bottom = box_bounds(first)
    second_left, second_top, second_right, second_bottom = box_bounds(second)

    intersection_width = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first_top, second_top))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0

    first_area = max(0.0, first_right - first_left) * max(0.0, first_bottom - first_top)
    second_area = max(0.0, second_right - second_left) * max(0.0, second_bottom - second_top)
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def format_yolo_box(box: YoloBox) -> str:
    return (
        f"{box.class_id} "
        f"{box.center_x:.12g} "
        f"{box.center_y:.12g} "
        f"{box.width:.12g} "
        f"{box.height:.12g}"
    )


def build_output_labels(
    rgb_label_path: Path,
    event_label_path: Path,
    label_policy: str,
    merge_iou_threshold: float,
) -> tuple[str, int, str]:
    rgb_boxes = read_yolo_boxes(rgb_label_path)
    event_boxes = read_yolo_boxes(event_label_path)

    if label_policy == "rgb":
        boxes = rgb_boxes
    elif label_policy == "event":
        boxes = event_boxes
    elif label_policy == "merge":
        boxes = list(rgb_boxes)
        for event_box in event_boxes:
            duplicate = any(
                event_box.class_id == existing_box.class_id
                and yolo_iou(event_box, existing_box) >= merge_iou_threshold
                for existing_box in boxes
            )
            if not duplicate:
                boxes.append(event_box)
    else:
        raise ValueError(f"Unsupported label policy: {label_policy}")

    if not boxes:
        label_source = "empty"
    elif boxes == rgb_boxes:
        label_source = "rgb"
    elif boxes == event_boxes:
        label_source = "event"
    else:
        label_source = "merged"

    content = "".join(f"{format_yolo_box(box)}\n" for box in boxes)
    return content, len(boxes), label_source


def discover_sequences(dataset_root: Path, rgb_dir_name: str) -> list[Path]:
    sequences: list[Path] = []
    if not dataset_root.exists():
        return sequences

    for child in sorted(dataset_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name == "yolo_hybrid":
            continue
        if (child / rgb_dir_name).exists() and (child / "Event" / "Frames").exists():
            sequences.append(child)

    return sequences


def resolve_sequence(sequence: str, dataset_root: Path) -> Path:
    candidate = Path(sequence)
    if candidate.exists():
        return candidate.resolve()
    return (dataset_root / sequence).resolve()


def default_output_name(sequence_root: Path) -> str:
    return f"{OUTPUT_PREFIX}{sequence_root.name}"


def make_pairs(
    sequence_root: Path,
    rgb_dir_name: str,
    label_dir_name: str,
    event_label_dir_name: str,
    max_delta_ms: float,
) -> tuple[list[Pair], int]:
    rgb_abs_frames = collect_frames(sequence_root / rgb_dir_name, parse_rgb_clock_time)
    event_frames = collect_frames(sequence_root / "Event" / "Frames", parse_event_time)

    if not rgb_abs_frames:
        raise ValueError(f"No RGB frames found in {sequence_root / rgb_dir_name}")
    if not event_frames:
        raise ValueError(f"No Event frames found in {sequence_root / 'Event' / 'Frames'}")

    first_rgb_time = rgb_abs_frames[0].time_s
    first_event_time = event_frames[0].time_s
    rgb_frames = [
        Frame(path=frame.path, time_s=frame.time_s - first_rgb_time + first_event_time)
        for frame in rgb_abs_frames
    ]
    event_times = [frame.time_s for frame in event_frames]

    pairs: list[Pair] = []
    skipped = 0
    cursor = 0
    max_delta_s = max_delta_ms / 1000.0
    label_dir = sequence_root / label_dir_name
    event_label_dir = sequence_root / event_label_dir_name

    for rgb_frame in rgb_frames:
        index = bisect.bisect_left(event_times, rgb_frame.time_s, lo=cursor)
        candidates = []
        for candidate in (index - 1, index, index + 1):
            if cursor <= candidate < len(event_frames):
                candidates.append(candidate)

        if not candidates:
            skipped += 1
            continue

        best_index = min(candidates, key=lambda value: abs(event_times[value] - rgb_frame.time_s))
        event_frame = event_frames[best_index]
        delta_s = abs(event_frame.time_s - rgb_frame.time_s)

        if delta_s > max_delta_s:
            skipped += 1
            continue

        pairs.append(
            Pair(
                sequence=sequence_root.name,
                rgb=rgb_frame,
                event=event_frame,
                label_path=label_dir / f"{rgb_frame.path.stem}.txt",
                event_label_path=event_label_dir / f"{event_frame.path.stem}.txt",
                delta_ms=delta_s * 1000.0,
            )
        )
        cursor = best_index + 1

    return pairs, skipped


def split_for_index(index: int, train_end: int, val_end: int) -> str:
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def relative_to_root(path: Path) -> str:
    return path.relative_to(TEST_ROOT).as_posix()


def relative_to_sequence(path: Path, sequence_root: Path) -> str:
    try:
        relative = path.relative_to(sequence_root)
    except ValueError:
        relative = path
    return ".\\" + str(relative).replace("/", "\\")


def ensure_output_ready(output_root: Path, overwrite: bool) -> None:
    generated_dirs = [output_root / "images_rgb", output_root / "images_event", output_root / "labels"]
    generated_files = [output_root / "manifest.csv", output_root / "data.yaml"]

    if output_root.exists() and not overwrite:
        existing = [path for path in generated_dirs + generated_files if path.exists()]
        if existing:
            existing_text = ", ".join(relative_to_root(path) for path in existing)
            raise SystemExit(
                f"Output already exists ({existing_text}). "
                "Set OVERWRITE_OUTPUT = True in the script config to rebuild it."
            )

    if overwrite:
        for directory in generated_dirs:
            if directory.exists():
                shutil.rmtree(directory)
        for file_path in generated_files:
            if file_path.exists():
                file_path.unlink()

    for split in SPLITS:
        (output_root / "images_rgb" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "images_event" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_metadata_files(sequence_root: Path, output_root: Path) -> list[Path]:
    copied: list[Path] = []
    for filename in ("tracks.txt", "coordinates.txt"):
        source = sequence_root / filename
        destination = output_root / filename
        if source.exists():
            shutil.copy2(source, destination)
            copied.append(destination)
    return copied


def write_configs(config_dir: Path, output_root: Path, dataset_name: str) -> None:
    write_text(config_dir / "data_rgb.yaml", rgb_yaml(dataset_name))
    write_text(config_dir / "data_event.yaml", event_yaml(dataset_name))
    write_text(output_root / "data.yaml", hybrid_yaml(dataset_name))


def write_dataset(
    pairs: list[Pair],
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    label_policy: str,
    merge_iou_threshold: float,
) -> dict[str, int]:
    train_end = int(len(pairs) * train_ratio)
    val_end = train_end + int(len(pairs) * val_ratio)
    split_counts = {split: 0 for split in SPLITS}
    manifest_path = output_root / "manifest.csv"

    with manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "sample_id",
            "sequence",
            "split",
            "rgb_time_s",
            "event_time_s",
            "delta_ms",
            "rgb_src",
            "event_src",
            "label_src",
            "event_label_src",
            "objects",
            "event_label_objects",
            "output_label_objects",
            "label_policy",
            "label_source",
            "rgb_dst",
            "event_dst",
            "label_dst",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, pair in enumerate(pairs):
            sample_id = f"{index + 1:06d}"
            split = split_for_index(index, train_end, val_end)
            split_counts[split] += 1

            rgb_dst = output_root / "images_rgb" / split / f"{sample_id}{pair.rgb.path.suffix.lower()}"
            event_dst = output_root / "images_event" / split / f"{sample_id}{pair.event.path.suffix.lower()}"
            label_dst = output_root / "labels" / split / f"{sample_id}.txt"

            shutil.copy2(pair.rgb.path, rgb_dst)
            shutil.copy2(pair.event.path, event_dst)
            label_content, output_label_objects, label_source = build_output_labels(
                pair.label_path,
                pair.event_label_path,
                label_policy=label_policy,
                merge_iou_threshold=merge_iou_threshold,
            )
            write_text(label_dst, label_content)

            sequence_root = pair.rgb.path.parents[1]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "sequence": pair.sequence,
                    "split": split,
                    "rgb_time_s": f"{pair.rgb.time_s:.6f}",
                    "event_time_s": f"{pair.event.time_s:.6f}",
                    "delta_ms": f"{pair.delta_ms:.3f}",
                    "rgb_src": relative_to_sequence(pair.rgb.path, sequence_root),
                    "event_src": relative_to_sequence(pair.event.path, sequence_root),
                    "label_src": relative_to_sequence(pair.label_path, sequence_root),
                    "event_label_src": relative_to_sequence(pair.event_label_path, sequence_root),
                    "objects": count_objects(pair.label_path),
                    "event_label_objects": count_objects(pair.event_label_path),
                    "output_label_objects": output_label_objects,
                    "label_policy": label_policy,
                    "label_source": label_source,
                    "rgb_dst": relative_to_root(rgb_dst),
                    "event_dst": relative_to_root(event_dst),
                    "label_dst": relative_to_root(label_dst),
                }
            )

    return split_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        nargs="?",
        default=SOURCE_SEQUENCE,
        help="Source sequence folder name under test/dataset, or a path such as test/dataset/230.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    parser.add_argument(
        "--output-name",
        default=OUTPUT_NAME,
        help="Output dataset/config folder name. Default: yolo_hybrid<sequence folder name>.",
    )
    parser.add_argument("--rgb-dir", default=RGB_DIR, help="Source RGB directory inside each sequence.")
    parser.add_argument("--label-dir", default=LABEL_DIR, help="Source YOLO labels for RGB frames.")
    parser.add_argument("--event-label-dir", default=EVENT_LABEL_DIR)
    parser.add_argument(
        "--label-policy",
        choices=sorted(LABEL_POLICIES),
        default=LABEL_POLICY,
        help="Which source labels to write: rgb, event, or merged RGB+Event labels.",
    )
    parser.add_argument(
        "--merge-iou-threshold",
        type=float,
        default=MERGE_IOU_THRESHOLD,
        help="IoU threshold used to deduplicate boxes when --label-policy merge is active.",
    )
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--max-delta-ms", type=float, default=MAX_DELTA_MS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_OUTPUT,
        help="Rebuild generated output dataset files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    config_root = args.config_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train-ratio and val-ratio must be non-negative and sum to less than 1.")
    if args.merge_iou_threshold < 0.0 or args.merge_iou_threshold > 1.0:
        raise SystemExit("merge-iou-threshold must be in [0, 1].")

    sequence_root = resolve_sequence(args.sequence, dataset_root)
    if not sequence_root.exists() or not sequence_root.is_dir():
        available = ", ".join(sequence.name for sequence in discover_sequences(dataset_root, args.rgb_dir))
        raise SystemExit(
            f"Source sequence not found: {sequence_root}"
            + (f"\nAvailable sequences: {available}" if available else "")
        )
    if not (sequence_root / args.rgb_dir).exists() or not (sequence_root / "Event" / "Frames").exists():
        raise SystemExit(
            f"Selected sequence must contain {args.rgb_dir} and Event/Frames: {sequence_root}"
        )

    output_name = args.output_name or default_output_name(sequence_root)
    output_root = dataset_root / output_name
    config_dir = config_root / output_name

    ensure_output_ready(output_root, args.overwrite)

    all_pairs: list[Pair] = []
    pairs, total_skipped = make_pairs(
        sequence_root,
        rgb_dir_name=args.rgb_dir,
        label_dir_name=args.label_dir,
        event_label_dir_name=args.event_label_dir,
        max_delta_ms=args.max_delta_ms,
    )
    all_pairs.extend(pairs)

    if not all_pairs:
        raise SystemExit("No RGB/Event frame pairs were produced.")

    split_counts = write_dataset(
        all_pairs,
        output_root=output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        label_policy=args.label_policy,
        merge_iou_threshold=args.merge_iou_threshold,
    )
    write_configs(config_dir, output_root, output_name)
    copied_metadata = copy_metadata_files(sequence_root, output_root)

    print("Prepared local YOLO hybrid dataset")
    print(f"- source sequence: {sequence_root.name}")
    print(f"- paired samples: {len(all_pairs)}")
    print(f"- skipped RGB frames: {total_skipped}")
    print(f"- label policy: {args.label_policy}")
    if args.label_policy == "merge":
        print(f"- merge IoU threshold: {args.merge_iou_threshold}")
    for split in SPLITS:
        print(f"- {split}: {split_counts[split]}")
    print(f"- output dataset: {relative_to_root(output_root)}")
    print(f"- manifest: {relative_to_root(output_root / 'manifest.csv')}")
    print(f"- RGB config: {relative_to_root(config_dir / 'data_rgb.yaml')}")
    print(f"- Event config: {relative_to_root(config_dir / 'data_event.yaml')}")
    for path in copied_metadata:
        print(f"- copied metadata: {relative_to_root(path)}")


if __name__ == "__main__":
    main()
