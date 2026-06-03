"""Check RGB/Event/label alignment in the local test YOLO hybrid dataset.

Run from any working directory:

    python test/src/preprocessing/check_dataset_alignment.py

Select the dataset by editing DATASET_NAME in the script config section below.

The report is written to:

    test/outputs/metrics/<sequence>/dataset_alignment_report.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional


TEST_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Script config: edit these values, then run this file directly.
DATASET_NAME = "yolo_hybrid17"
DATASET_ROOT_OVERRIDE = None
MAX_DELTA_MS = 20.0
NO_STRICT = False


def file_stems(directory: Path, extensions: Optional[set[str]] = None) -> set[str]:
    if not directory.exists():
        return set()
    stems: set[str] = set()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if extensions is not None and path.suffix.lower() not in extensions:
            continue
        stems.add(path.stem)
    return stems


def list_files(directory: Path, extensions: Optional[set[str]] = None) -> list[Path]:
    if not directory.exists():
        return []
    files = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if extensions is not None and path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.name)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def relative(path: Path) -> str:
    return path.relative_to(TEST_ROOT).as_posix()


def report_dir_for_dataset(dataset_root: Path) -> Path:
    dataset_name = dataset_root.name
    if dataset_name.startswith("yolo_hybrid") and dataset_name != "yolo_hybrid":
        dataset_name = dataset_name.removeprefix("yolo_hybrid")
    return TEST_ROOT / "outputs" / "metrics" / dataset_name


def check_manifest(rows: list[dict[str, str]], dataset_root: Path, max_delta_ms: float) -> list[str]:
    issues: list[str] = []
    seen_samples: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "")
        split = row.get("split", "")
        if sample_id in seen_samples:
            issues.append(f"manifest row {row_number}: duplicated sample_id {sample_id}")
        seen_samples.add(sample_id)

        if split not in SPLITS:
            issues.append(f"manifest row {row_number}: invalid split {split!r}")
            continue

        for key in ("rgb_dst", "event_dst", "label_dst"):
            value = row.get(key, "")
            if not value:
                issues.append(f"manifest row {row_number}: missing {key}")
                continue
            candidate = TEST_ROOT / value
            if not candidate.exists():
                issues.append(f"manifest row {row_number}: {key} does not exist: {value}")

        try:
            delta_ms = float(row.get("delta_ms", "nan"))
        except ValueError:
            issues.append(f"manifest row {row_number}: non-numeric delta_ms")
            continue
        if delta_ms > max_delta_ms:
            issues.append(
                f"manifest row {row_number}: delta_ms {delta_ms:.3f} exceeds {max_delta_ms:.3f}"
            )

    return issues


def build_report(dataset_root: Path, max_delta_ms: float) -> tuple[str, int]:
    report: list[str] = []
    issues: list[str] = []

    report.append("# Dataset Alignment Report")
    report.append("")
    report.append(f"- dataset root: {relative(dataset_root)}")
    report.append("")
    report.append("| Split | RGB | Event | Labels | Missing RGB | Missing Event | Missing Label |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")

    total_rgb = 0
    total_event = 0
    total_labels = 0
    split_membership: dict[str, str] = {}

    for split in SPLITS:
        rgb_dir = dataset_root / "images_rgb" / split
        event_dir = dataset_root / "images_event" / split
        label_dir = dataset_root / "labels" / split

        for directory in (rgb_dir, event_dir, label_dir):
            if not directory.exists():
                issues.append(f"Missing directory: {relative(directory)}")

        rgb_stems = file_stems(rgb_dir, IMAGE_EXTENSIONS)
        event_stems = file_stems(event_dir, IMAGE_EXTENSIONS)
        label_stems = file_stems(label_dir, {".txt"})
        all_stems = rgb_stems | event_stems | label_stems

        missing_rgb = sorted(all_stems - rgb_stems)
        missing_event = sorted(all_stems - event_stems)
        missing_label = sorted(all_stems - label_stems)

        for stem in all_stems:
            previous_split = split_membership.get(stem)
            if previous_split is not None:
                issues.append(f"Sample {stem} appears in both {previous_split} and {split}")
            split_membership[stem] = split

        for stem in missing_rgb[:20]:
            issues.append(f"{split}: missing RGB image for sample {stem}")
        for stem in missing_event[:20]:
            issues.append(f"{split}: missing Event image for sample {stem}")
        for stem in missing_label[:20]:
            issues.append(f"{split}: missing label for sample {stem}")

        total_rgb += len(rgb_stems)
        total_event += len(event_stems)
        total_labels += len(label_stems)
        report.append(
            f"| {split} | {len(rgb_stems)} | {len(event_stems)} | {len(label_stems)} | "
            f"{len(missing_rgb)} | {len(missing_event)} | {len(missing_label)} |"
        )

    manifest_path = dataset_root / "manifest.csv"
    manifest_rows = read_manifest(manifest_path)
    if not manifest_rows:
        issues.append(f"Missing or empty manifest: {relative(manifest_path)}")
    else:
        manifest_issues = check_manifest(manifest_rows, dataset_root=dataset_root, max_delta_ms=max_delta_ms)
        issues.extend(manifest_issues)

        manifest_ids = {row.get("sample_id", "") for row in manifest_rows}
        dataset_ids = set(split_membership)
        for sample_id in sorted(dataset_ids - manifest_ids)[:20]:
            issues.append(f"Sample {sample_id} exists in dataset but not manifest")
        for sample_id in sorted(manifest_ids - dataset_ids)[:20]:
            issues.append(f"Sample {sample_id} exists in manifest but not dataset")

    report.append("")
    report.append("## Totals")
    report.append("")
    report.append(f"- RGB images: {total_rgb}")
    report.append(f"- Event images: {total_event}")
    report.append(f"- Label files: {total_labels}")
    report.append(f"- Manifest rows: {len(manifest_rows)}")
    report.append(f"- Issues: {len(issues)}")
    report.append("")
    report.append("## Issues")
    report.append("")
    if issues:
        for issue in issues[:100]:
            report.append(f"- {issue}")
        if len(issues) > 100:
            report.append(f"- ... {len(issues) - 100} more issues")
    else:
        report.append("- None")

    return "\n".join(report) + "\n", len(issues)


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
    parser.add_argument("--max-delta-ms", type=float, default=MAX_DELTA_MS)
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
    report_path = report_dir_for_dataset(dataset_root) / "dataset_alignment_report.txt"
    report, issue_count = build_report(dataset_root=dataset_root, max_delta_ms=args.max_delta_ms)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    print(f"Wrote {relative(report_path)}")
    print(f"Dataset alignment issues: {issue_count}")
    if issue_count and not args.no_strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
