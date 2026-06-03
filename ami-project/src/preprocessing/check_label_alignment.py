"""Check YOLO label alignment and format in the local test dataset.

Run from any working directory:

    python test/src/preprocessing/check_label_alignment.py

Select the dataset by editing DATASET_NAME in the script config section below.

The report is written to:

    test/outputs/metrics/<sequence>/label_alignment_report.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional


TEST_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET_ROOT = TEST_ROOT / "dataset"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EPSILON = 1e-9

# Script config: edit these values, then run this file directly.
DATASET_NAME = "yolo_hybrid17"
DATASET_ROOT_OVERRIDE = None
NO_STRICT = False


def relative(path: Path) -> str:
    return path.relative_to(TEST_ROOT).as_posix()


def report_dir_for_dataset(dataset_root: Path) -> Path:
    dataset_name = dataset_root.name
    if dataset_name.startswith("yolo_hybrid") and dataset_name != "yolo_hybrid":
        dataset_name = dataset_name.removeprefix("yolo_hybrid")
    return TEST_ROOT / "outputs" / "metrics" / dataset_name


def list_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ],
        key=lambda path: path.name,
    )


def label_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def label_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_label_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []

    boxes: list[tuple[int, float, float, float, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def labels_equal(first_path: Path, second_path: Path) -> bool:
    first_boxes = read_label_boxes(first_path)
    second_boxes = read_label_boxes(second_path)
    if len(first_boxes) != len(second_boxes):
        return False

    for first_box, second_box in zip(first_boxes, second_boxes):
        if first_box[0] != second_box[0]:
            return False
        if any(abs(first_box[index] - second_box[index]) > LABEL_EPSILON for index in range(1, 5)):
            return False
    return True


def validate_label_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Missing label file: {relative(path)}"]

    if path.stat().st_size == 0:
        return errors

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{relative(path)} line {line_number}: expected 5 columns, got {len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{relative(path)} line {line_number}: non-numeric YOLO value")
            continue

        if class_id != 0:
            errors.append(f"{relative(path)} line {line_number}: class_id should be 0, got {class_id}")

        values = (center_x, center_y, width, height)
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{relative(path)} line {line_number}: bbox values must be in [0, 1]")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{relative(path)} line {line_number}: width and height must be positive")

    return errors


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def source_path(row: dict[str, str], key: str) -> Optional[Path]:
    sequence = row.get("sequence", "")
    value = row.get(key, "")
    if not sequence or not value:
        return None

    normalized = value
    if normalized.startswith(".\\") or normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.replace("\\", "/")
    return SOURCE_DATASET_ROOT / sequence / normalized


def check_manifest_labels(rows: list[dict[str, str]], dataset_root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "")
        split = row.get("split", "")
        if split not in SPLITS or not sample_id:
            errors.append(f"manifest row {row_number}: missing sample_id or invalid split")
            continue

        destination = dataset_root / "labels" / split / f"{sample_id}.txt"
        rgb_source = source_path(row, "label_src")
        event_source = source_path(row, "event_label_src")
        label_policy = row.get("label_policy", "rgb")
        label_source = row.get("label_source", "")

        rgb_exists = rgb_source is not None and rgb_source.exists()
        event_exists = event_source is not None and event_source.exists()
        destination_text = label_text(destination)

        if label_policy == "rgb" and not rgb_exists:
            errors.append(f"manifest row {row_number}: RGB label source missing for sample {sample_id}")
        elif label_policy == "event" and not event_exists:
            errors.append(f"manifest row {row_number}: Event label source missing for sample {sample_id}")
        elif label_policy == "merge" and not rgb_exists and not event_exists:
            errors.append(f"manifest row {row_number}: no label source exists for sample {sample_id}")

        if label_policy == "rgb" and rgb_exists and not labels_equal(destination, rgb_source):
            errors.append(
                f"manifest row {row_number}: prepared label differs from RGB source for sample {sample_id}"
            )
        elif label_policy == "event" and event_exists and not labels_equal(destination, event_source):
            errors.append(
                f"manifest row {row_number}: prepared label differs from Event source for sample {sample_id}"
            )
        elif label_policy == "merge":
            output_objects = row.get("output_label_objects", "")
            if output_objects:
                try:
                    expected_count = int(output_objects)
                except ValueError:
                    errors.append(f"manifest row {row_number}: non-numeric output_label_objects")
                else:
                    actual_count = label_line_count(destination)
                    if actual_count != expected_count:
                        errors.append(
                            f"manifest row {row_number}: output label count {actual_count} "
                            f"differs from manifest count {expected_count} for sample {sample_id}"
                        )
            if label_source == "rgb" and rgb_exists and not labels_equal(destination, rgb_source):
                errors.append(
                    f"manifest row {row_number}: RGB-sourced prepared label differs for sample {sample_id}"
                )
            elif label_source == "event" and event_exists and not labels_equal(destination, event_source):
                errors.append(
                    f"manifest row {row_number}: Event-sourced prepared label differs for sample {sample_id}"
                )
            elif label_source == "empty" and destination_text:
                errors.append(f"manifest row {row_number}: empty-sourced label is not empty for sample {sample_id}")

        if not event_exists:
            warnings.append(f"manifest row {row_number}: Event label source missing for sample {sample_id}")
        elif label_policy != "merge" and rgb_exists:
            if not labels_equal(rgb_source, event_source):
                warnings.append(
                    f"manifest row {row_number}: RGB/Event source labels differ for sample {sample_id}"
                )

    return errors, warnings


def build_report(dataset_root: Path) -> tuple[str, int, int]:
    report: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    total_labels = 0
    empty_labels = 0
    non_empty_labels = 0

    report.append("# Label Alignment Report")
    report.append("")
    report.append(f"- dataset root: {relative(dataset_root)}")
    report.append("")
    report.append("| Split | Images RGB | Images Event | Labels | Empty Labels | Non-empty Labels |")
    report.append("|---|---:|---:|---:|---:|---:|")

    for split in SPLITS:
        rgb_files = list_files(dataset_root / "images_rgb" / split, IMAGE_EXTENSIONS)
        event_files = list_files(dataset_root / "images_event" / split, IMAGE_EXTENSIONS)
        label_files = list_files(dataset_root / "labels" / split, {".txt"})

        rgb_stems = {path.stem for path in rgb_files}
        event_stems = {path.stem for path in event_files}
        label_stems = {path.stem for path in label_files}
        all_image_stems = rgb_stems | event_stems

        for stem in sorted(all_image_stems - label_stems)[:50]:
            errors.append(f"{split}: missing label for image sample {stem}")
        for stem in sorted(label_stems - all_image_stems)[:50]:
            errors.append(f"{split}: label has no matching image sample {stem}")

        split_empty = sum(1 for path in label_files if path.stat().st_size == 0)
        split_non_empty = len(label_files) - split_empty
        total_labels += len(label_files)
        empty_labels += split_empty
        non_empty_labels += split_non_empty

        for label_path in label_files:
            errors.extend(validate_label_file(label_path))

        report.append(
            f"| {split} | {len(rgb_files)} | {len(event_files)} | {len(label_files)} | "
            f"{split_empty} | {split_non_empty} |"
        )

    manifest_rows = read_manifest(dataset_root / "manifest.csv")
    if not manifest_rows:
        warnings.append(f"Missing manifest, source-label comparison skipped: {relative(dataset_root / 'manifest.csv')}")
    else:
        manifest_errors, manifest_warnings = check_manifest_labels(manifest_rows, dataset_root=dataset_root)
        errors.extend(manifest_errors)
        warnings.extend(manifest_warnings)

    report.append("")
    report.append("## Totals")
    report.append("")
    report.append(f"- Label files: {total_labels}")
    report.append(f"- Empty label files: {empty_labels}")
    report.append(f"- Non-empty label files: {non_empty_labels}")
    report.append(f"- Errors: {len(errors)}")
    report.append(f"- Warnings: {len(warnings)}")
    report.append("")
    report.append("## Errors")
    report.append("")
    if errors:
        for error in errors[:100]:
            report.append(f"- {error}")
        if len(errors) > 100:
            report.append(f"- ... {len(errors) - 100} more errors")
    else:
        report.append("- None")

    report.append("")
    report.append("## Warnings")
    report.append("")
    if warnings:
        for warning in warnings[:100]:
            report.append(f"- {warning}")
        if len(warnings) > 100:
            report.append(f"- ... {len(warnings) - 100} more warnings")
    else:
        report.append("- None")

    return "\n".join(report) + "\n", len(errors), len(warnings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help="Folder name under test/dataset, for example yolo_hybrid230.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT_OVERRIDE,
        help="Explicit dataset output path. Overrides --dataset-name.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        default=NO_STRICT,
        help="Always exit with status 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = (args.dataset_root or (TEST_ROOT / "dataset" / args.dataset_name)).resolve()
    report_path = report_dir_for_dataset(dataset_root) / "label_alignment_report.txt"
    report, error_count, warning_count = build_report(dataset_root=dataset_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    print(f"Wrote {relative(report_path)}")
    print(f"Label alignment errors: {error_count}")
    print(f"Label alignment warnings: {warning_count}")
    if error_count and not args.no_strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
