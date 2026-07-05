#!/usr/bin/env python3
"""
Prepare a small FRED subset for RGB-YOLO, Event-YOLO, and late/paired fusion.

The script is intentionally dependency-free. It:
  - parses FRED annotations,
  - aligns labels, padded RGB frames, and event frames by relative time,
  - exports YOLO labels for RGB and Event images,
  - writes paired manifests for fusion experiments.

Example:
  python scripts/preprocessing/prepare_fred_yolo.py \
    --dataset-root dataset \
    --output-root processed/fred10 \
    --train-seqs 0,1,11,101,102,103 \
    --val-seqs 10,21 \
    --test-seqs 34,110
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import random
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class TimedPath:
    time_s: float
    path: Path


@dataclass
class SequenceResult:
    sequence: str
    split: str
    annotation_file: Optional[str]
    labels_seen: int = 0
    samples_written: int = 0
    invalid_boxes: int = 0
    unmatched_rgb: int = 0
    unmatched_event: int = 0
    negative_candidates: int = 0
    negative_samples_written: int = 0
    missing_dirs: Optional[list[str]] = None
    rgb_t0_offset: float = 0.0


def parse_seq_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [part.strip().strip("/\\") for part in value.split(",") if part.strip()]


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def find_sequence_dir(dataset_root: Path, seq: str) -> Optional[Path]:
    candidates = [
        dataset_root / seq,
        dataset_root / "train" / seq,
        dataset_root / "test" / seq,
        dataset_root / "val" / seq,
        dataset_root / "val_challenging" / seq,
        dataset_root / "test_demo" / seq,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )


def parse_annotations(path: Path) -> dict[float, list[Box]]:
    annotations: dict[float, list[Box]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            timestamp_raw, rest = line.split(":", 1)
            try:
                timestamp = float(timestamp_raw.strip())
            except ValueError:
                continue
            parts = [part.strip() for part in rest.split(",")]
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = (float(parts[i]) for i in range(4))
            except ValueError:
                continue
            annotations.setdefault(timestamp, []).append(Box(x1, y1, x2, y2))
    return annotations


def choose_annotation_file(seq_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for name in candidates:
        path = seq_dir / name
        if path.is_file():
            return path
    return None


def parse_event_time(path: Path) -> Optional[float]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if not match:
        return None
    return int(match.group(1)) / 1_000_000.0


def parse_rgb_wallclock(path: Path) -> Optional[float]:
    """Return wall-clock seconds since midnight encoded in a PADDED_RGB/RGB filename.

    Expected format: ``Video_N_HH_MM_SS.ffffff.ext``
    (e.g. ``Video_1_16_11_01.592169.jpg`` → 58261.592169 s since midnight).
    Returns *None* if the filename does not match.
    """
    match = re.search(r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)\.", path.name)
    if not match:
        return None
    h, m, s, frac_str = int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
    frac = int(frac_str) / (10 ** len(frac_str))
    return h * 3600 + m * 60 + s + frac


def build_event_timeline(event_dir: Path) -> list[TimedPath]:
    items: list[TimedPath] = []
    for path in list_images(event_dir):
        time_s = parse_event_time(path)
        if time_s is not None:
            items.append(TimedPath(time_s, path))
    return sorted(items, key=lambda item: item.time_s)


def build_rgb_elapsed_timeline(
    seq_dir: Path,
    rgb_dir_name: str,
    frame_period: float,
    include_removed: bool,
) -> list[TimedPath]:
    """Build PADDED_RGB frame timeline with elapsed times starting at 0 for the first frame.

    When filenames encode wall-clock timestamps (``Video_N_HH_MM_SS.ffffff.ext``),
    these are parsed directly so that inter-frame timing is exact — no drift from
    an approximate ``frame_period``.  Falls back to ``index × frame_period`` (0-based)
    when the filenames do not carry timestamps.

    The returned times are *elapsed* times, not sensor times.  Add the
    per-sequence sensor-start offset (see :func:`estimate_rgb_t0_offset`) before
    performing nearest-neighbour matching against annotation or event timestamps.
    """
    rgb_dir = seq_dir / rgb_dir_name
    rgb_paths = list_images(rgb_dir)
    if not rgb_paths:
        return []

    wallclock_times = [parse_rgb_wallclock(p) for p in rgb_paths]
    if all(t is not None for t in wallclock_times):
        # Use actual wall-clock elapsed times: eliminates drift from approximate frame_period.
        t0 = wallclock_times[0]
        timeline = [TimedPath(t - t0, p) for t, p in zip(wallclock_times, rgb_paths)]
    else:
        # Fallback: 0-based index × frame_period.  The sensor-start offset is added
        # externally, so index 0 → 0 s here (offset will add ~frame_period).
        # FRED annotations normally index the usable PADDED_RGB sequence, not
        # the earlier Removed_frames folder; keep include_removed as a diagnostic opt-in.
        if include_removed:
            removed_paths = list_images(seq_dir / "Removed_frames")
            combined_names = sorted(
                {p.name for p in rgb_paths} | {p.name for p in removed_paths},
                key=lambda n: natural_key(Path(n)),
            )
        else:
            combined_names = [p.name for p in rgb_paths]
        index_by_name = {name: idx for idx, name in enumerate(combined_names)}
        timeline = [
            TimedPath(index_by_name[p.name] * frame_period, p)
            for p in rgb_paths
            if p.name in index_by_name
        ]

    return sorted(timeline, key=lambda item: item.time_s)


def nearest(timeline: list[TimedPath], target_s: float, max_delta_s: float) -> Optional[TimedPath]:
    if not timeline:
        return None
    times = [item.time_s for item in timeline]
    pos = bisect.bisect_left(times, target_s)
    candidates = []
    if pos < len(timeline):
        candidates.append(timeline[pos])
    if pos > 0:
        candidates.append(timeline[pos - 1])
    best = min(candidates, key=lambda item: abs(item.time_s - target_s))
    if abs(best.time_s - target_s) <= max_delta_s:
        return best
    return None


def has_near_time(times: list[float], target_s: float, exclusion_s: float) -> bool:
    if not times:
        return False
    pos = bisect.bisect_left(times, target_s)
    if pos < len(times) and abs(times[pos] - target_s) <= exclusion_s:
        return True
    if pos > 0 and abs(times[pos - 1] - target_s) <= exclusion_s:
        return True
    return False


def read_image_size(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                width, height = struct.unpack(">II", header[16:24])
                return int(width), int(height)
            if header.startswith(b"\xff\xd8"):
                handle.seek(2)
                while True:
                    marker_start = handle.read(1)
                    if not marker_start:
                        break
                    if marker_start != b"\xff":
                        continue
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if marker in {b"\xd8", b"\xd9"}:
                        continue
                    length_raw = handle.read(2)
                    if len(length_raw) != 2:
                        break
                    length = struct.unpack(">H", length_raw)[0]
                    if marker in {
                        b"\xc0",
                        b"\xc1",
                        b"\xc2",
                        b"\xc3",
                        b"\xc5",
                        b"\xc6",
                        b"\xc7",
                        b"\xc9",
                        b"\xca",
                        b"\xcb",
                        b"\xcd",
                        b"\xce",
                        b"\xcf",
                    }:
                        data = handle.read(5)
                        if len(data) != 5:
                            break
                        height, width = struct.unpack(">HH", data[1:5])
                        return int(width), int(height)
                    handle.seek(length - 2, os.SEEK_CUR)
    except OSError:
        pass
    return DEFAULT_WIDTH, DEFAULT_HEIGHT


def clamp_box(box: Box, width: int, height: int) -> tuple[Optional[Box], bool]:
    x1 = min(max(box.x1, 0.0), float(width - 1))
    y1 = min(max(box.y1, 0.0), float(height - 1))
    x2 = min(max(box.x2, 0.0), float(width - 1))
    y2 = min(max(box.y2, 0.0), float(height - 1))
    changed = (x1, y1, x2, y2) != (box.x1, box.y1, box.x2, box.y2)
    if x2 <= x1 or y2 <= y1:
        return None, changed
    return Box(x1, y1, x2, y2), changed


def to_yolo_line(box: Box, width: int, height: int, class_id: int = 0) -> str:
    x_center = ((box.x1 + box.x2) / 2.0) / width
    y_center = ((box.y1 + box.y2) / 2.0) / height
    box_w = (box.x2 - box.x1) / width
    box_h = (box.y2 - box.y1) / height
    return f"{class_id} {x_center:.8f} {y_center:.8f} {box_w:.8f} {box_h:.8f}"


def materialize_image(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    if dst.exists() and overwrite:
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def estimate_rgb_t0_offset(
    rgb_elapsed_timeline: list[TimedPath],
    sorted_annotation_times: list[float],
    initial_guess: float,
    search_radius_s: float = 1.0,
    max_anchors: int = 60,
) -> float:
    """Estimate the sensor time that corresponds to elapsed=0 in the PADDED_RGB timeline.

    Samples up to *max_anchors* evenly-spaced annotation times and, for each, finds
    the PADDED_RGB frame whose elapsed time is closest to ``annotation_time - initial_guess``.
    The per-anchor offset is ``annotation_time - elapsed_of_matched_frame``.  The
    function returns the median of all accepted offset samples.

    Args:
        rgb_elapsed_timeline: PADDED_RGB frames sorted by elapsed time (offset not yet applied).
        sorted_annotation_times: annotation sensor times, sorted ascending.
        initial_guess: seed estimate for the offset (e.g. ``frame_period``).
        search_radius_s: maximum elapsed-time distance from the seeded guess to accept a
            candidate frame.  Default 1.0 s is generous enough for any realistic frame-rate.
        max_anchors: maximum number of evenly-spaced annotation anchors to sample.
    """
    if not sorted_annotation_times or not rgb_elapsed_timeline:
        return initial_guess

    elapsed_times = [item.time_s for item in rgb_elapsed_timeline]
    offsets: list[float] = []

    step = max(1, len(sorted_annotation_times) // max_anchors)
    for ann_t in sorted_annotation_times[::step]:
        target_elapsed = ann_t - initial_guess
        pos = bisect.bisect_left(elapsed_times, target_elapsed)
        candidates: list[TimedPath] = []
        if pos < len(rgb_elapsed_timeline):
            candidates.append(rgb_elapsed_timeline[pos])
        if pos > 0:
            candidates.append(rgb_elapsed_timeline[pos - 1])
        if not candidates:
            continue
        best = min(candidates, key=lambda x: abs(x.time_s - target_elapsed))
        if abs(best.time_s - target_elapsed) <= search_radius_s:
            offsets.append(ann_t - best.time_s)

    if not offsets:
        return initial_guess
    offsets.sort()
    return offsets[len(offsets) // 2]


def process_sequence(
    seq: str,
    split: str,
    seq_dir: Path,
    output_root: Path,
    annotation_candidates: list[str],
    rgb_dir_name: str,
    frame_period: float,
    max_delta_s: float,
    materialize: str,
    overwrite: bool,
    include_removed_in_rgb_timeline: bool,
    negative_ratio: float,
    negative_exclusion_s: float,
    negative_seed: int,
) -> tuple[SequenceResult, list[dict[str, str]]]:
    result = SequenceResult(sequence=seq, split=split, annotation_file=None, missing_dirs=[])
    rgb_dir = seq_dir / rgb_dir_name
    event_dir = seq_dir / "Event" / "Frames"

    if not rgb_dir.is_dir():
        result.missing_dirs.append(str(rgb_dir))
    if not event_dir.is_dir():
        result.missing_dirs.append(str(event_dir))

    annotation_file = choose_annotation_file(seq_dir, annotation_candidates)
    if annotation_file is None:
        result.missing_dirs.append("annotation: " + ",".join(annotation_candidates))

    if result.missing_dirs:
        return result, []

    assert annotation_file is not None
    result.annotation_file = annotation_file.name
    annotations = parse_annotations(annotation_file)
    rgb_elapsed = build_rgb_elapsed_timeline(seq_dir, rgb_dir_name, frame_period, include_removed_in_rgb_timeline)
    # Estimate the sensor time of the first PADDED_RGB frame by fitting annotation anchors.
    # This is more accurate than assuming a fixed frame_period offset, especially when the
    # RGB and event cameras run at slightly different rates (accumulated drift otherwise
    # reaches 20–30 ms over 500–2000 frames, consuming most of the max_delta budget).
    sorted_label_times = sorted(annotations.keys())
    rgb_t0_offset = estimate_rgb_t0_offset(rgb_elapsed, sorted_label_times, initial_guess=frame_period)
    rgb_timeline = sorted(
        [TimedPath(item.time_s + rgb_t0_offset, item.path) for item in rgb_elapsed],
        key=lambda x: x.time_s,
    )
    result.rgb_t0_offset = rgb_t0_offset
    event_timeline = build_event_timeline(event_dir)
    result.labels_seen = len(annotations)

    manifest_rows: list[dict[str, str]] = []
    size_cache: dict[Path, tuple[int, int]] = {}
    used_rgb_paths: set[Path] = set()
    used_event_paths: set[Path] = set()

    for label_time in sorted(annotations):
        rgb_item = nearest(rgb_timeline, label_time, max_delta_s)
        event_item = nearest(event_timeline, label_time, max_delta_s)

        if rgb_item is None:
            result.unmatched_rgb += 1
            continue
        if event_item is None:
            result.unmatched_event += 1
            continue

        rgb_size = size_cache.setdefault(rgb_item.path, read_image_size(rgb_item.path))
        event_size = size_cache.setdefault(event_item.path, read_image_size(event_item.path))

        rgb_lines: list[str] = []
        event_lines: list[str] = []
        for box in annotations[label_time]:
            rgb_box, _ = clamp_box(box, rgb_size[0], rgb_size[1])
            event_box, _ = clamp_box(box, event_size[0], event_size[1])
            if rgb_box is None or event_box is None:
                result.invalid_boxes += 1
                continue
            rgb_lines.append(to_yolo_line(rgb_box, rgb_size[0], rgb_size[1]))
            event_lines.append(to_yolo_line(event_box, event_size[0], event_size[1]))

        if not rgb_lines:
            continue

        time_us = int(round(label_time * 1_000_000))
        stem = f"seq{seq}_{time_us:09d}"
        rgb_dst = output_root / "matched" / "rgb" / "images" / split / f"{stem}{rgb_item.path.suffix.lower()}"
        event_dst = output_root / "matched" / "event" / "images" / split / f"{stem}{event_item.path.suffix.lower()}"
        rgb_label_dst = output_root / "labels" / "rgb" / split / f"{stem}.txt"
        event_label_dst = output_root / "labels" / "event" / split / f"{stem}.txt"

        materialize_image(rgb_item.path, rgb_dst, materialize, overwrite)
        materialize_image(event_item.path, event_dst, materialize, overwrite)
        write_label(rgb_label_dst, rgb_lines)
        write_label(event_label_dst, event_lines)
        used_rgb_paths.add(rgb_item.path)
        used_event_paths.add(event_item.path)

        manifest_rows.append(
            {
                "sequence": seq,
                "split": split,
                "label_time_s": f"{label_time:.6f}",
                "rgb_time_s": f"{rgb_item.time_s:.6f}",
                "event_time_s": f"{event_item.time_s:.6f}",
                "rgb_delta_s": f"{abs(rgb_item.time_s - label_time):.6f}",
                "event_delta_s": f"{abs(event_item.time_s - label_time):.6f}",
                "rgb_image": rel(rgb_dst, output_root),
                "event_image": rel(event_dst, output_root),
                "rgb_label": rel(rgb_label_dst, output_root),
                "event_label": rel(event_label_dst, output_root),
                "source_rgb": str(rgb_item.path),
                "source_event": str(event_item.path),
                "annotation_file": str(annotation_file),
                "num_boxes": str(len(rgb_lines)),
            }
        )
        result.samples_written += 1

    if negative_ratio > 0 and result.samples_written > 0:
        annotation_times = sorted(annotations)
        negative_candidates: list[tuple[TimedPath, TimedPath]] = []
        for rgb_item in rgb_timeline:
            if rgb_item.path in used_rgb_paths:
                continue
            event_item = nearest(event_timeline, rgb_item.time_s, max_delta_s)
            if event_item is None or event_item.path in used_event_paths:
                continue
            if has_near_time(annotation_times, rgb_item.time_s, negative_exclusion_s):
                continue
            if has_near_time(annotation_times, event_item.time_s, negative_exclusion_s):
                continue
            negative_candidates.append((rgb_item, event_item))

        result.negative_candidates = len(negative_candidates)
        negative_limit = min(len(negative_candidates), int(round(result.samples_written * negative_ratio)))
        rng = random.Random(f"{negative_seed}:{split}:{seq}")
        rng.shuffle(negative_candidates)
        selected_negatives = sorted(negative_candidates[:negative_limit], key=lambda pair: pair[0].time_s)

        for rgb_item, event_item in selected_negatives:
            time_us = int(round(rgb_item.time_s * 1_000_000))
            stem = f"seq{seq}_neg_{time_us:09d}"
            rgb_dst = output_root / "matched" / "rgb" / "images" / split / f"{stem}{rgb_item.path.suffix.lower()}"
            event_dst = output_root / "matched" / "event" / "images" / split / f"{stem}{event_item.path.suffix.lower()}"
            rgb_label_dst = output_root / "labels" / "rgb" / split / f"{stem}.txt"
            event_label_dst = output_root / "labels" / "event" / split / f"{stem}.txt"

            materialize_image(rgb_item.path, rgb_dst, materialize, overwrite)
            materialize_image(event_item.path, event_dst, materialize, overwrite)
            write_label(rgb_label_dst, [])
            write_label(event_label_dst, [])

            manifest_rows.append(
                {
                    "sequence": seq,
                    "split": split,
                    "label_time_s": f"{rgb_item.time_s:.6f}",
                    "rgb_time_s": f"{rgb_item.time_s:.6f}",
                    "event_time_s": f"{event_item.time_s:.6f}",
                    "rgb_delta_s": "0.000000",
                    "event_delta_s": f"{abs(event_item.time_s - rgb_item.time_s):.6f}",
                    "rgb_image": rel(rgb_dst, output_root),
                    "event_image": rel(event_dst, output_root),
                    "rgb_label": rel(rgb_label_dst, output_root),
                    "event_label": rel(event_label_dst, output_root),
                    "source_rgb": str(rgb_item.path),
                    "source_event": str(event_item.path),
                    "annotation_file": str(annotation_file),
                    "num_boxes": "0",
                }
            )
            result.negative_samples_written += 1

    return result, manifest_rows


def write_manifest(output_root: Path, split: str, rows: list[dict[str, str]]) -> None:
    manifest_dir = output_root / "paired"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"manifest_{split}.csv"
    fieldnames = [
        "sequence",
        "split",
        "label_time_s",
        "rgb_time_s",
        "event_time_s",
        "rgb_delta_s",
        "event_delta_s",
        "rgb_image",
        "event_image",
        "rgb_label",
        "event_label",
        "source_rgb",
        "source_event",
        "annotation_file",
        "num_boxes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sequence_report(output_root: Path, report: dict) -> None:
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FRED sequences for RGB/Event YOLO and paired fusion.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("processed/fred_subset"))
    parser.add_argument("--train-seqs", default="0,1,11,101,102,103")
    parser.add_argument("--val-seqs", default="10,21")
    parser.add_argument("--test-seqs", default="34,110")
    parser.add_argument("--rgb-dir", default="PADDED_RGB", help="Use PADDED_RGB for shared RGB/Event coordinates.")
    parser.add_argument(
        "--annotation-files",
        default="interpolated_coordinates.txt,coordinates.txt",
        help="Comma-separated fallback list. Use coordinates.txt first for strict official extended boxes.",
    )
    parser.add_argument("--frame-period", type=float, default=0.033333)
    parser.add_argument("--max-delta", type=float, default=0.04, help="Maximum allowed label/image time delta in seconds.")
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.0,
        help="Add synchronized empty-label samples per sequence as ratio of positive samples. 0 keeps legacy behavior.",
    )
    parser.add_argument(
        "--negative-exclusion",
        type=float,
        default=0.066,
        help="Minimum time distance in seconds from any annotated box when sampling negative frames.",
    )
    parser.add_argument("--negative-seed", type=int, default=42, help="Seed for deterministic negative-frame sampling.")
    parser.add_argument("--materialize", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-removed-in-rgb-timeline",
        action="store_true",
        help="Diagnostic option. FRED labels usually align to PADDED_RGB index, so this is off by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.negative_ratio < 0:
        raise ValueError("--negative-ratio must be non-negative.")
    if args.negative_exclusion < 0:
        raise ValueError("--negative-exclusion must be non-negative.")

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    annotation_candidates = parse_seq_list(args.annotation_files)

    splits = {
        "train": parse_seq_list(args.train_seqs),
        "val": parse_seq_list(args.val_seqs),
        "test": parse_seq_list(args.test_seqs),
    }

    all_results: list[SequenceResult] = []
    sequence_reports: list[dict[str, object]] = []

    for split, seqs in splits.items():
        for seq in seqs:
            seq_dir = find_sequence_dir(dataset_root, seq)
            if seq_dir is None:
                all_results.append(
                    SequenceResult(
                        sequence=seq,
                        split=split,
                        annotation_file=None,
                        missing_dirs=[f"sequence {seq} not found under {dataset_root}"],
                    )
                )
                continue

            sequence_output_root = output_root / f"preprocessed_fred_{seq}"
            result, rows = process_sequence(
                seq=seq,
                split=split,
                seq_dir=seq_dir,
                output_root=sequence_output_root,
                annotation_candidates=annotation_candidates,
                rgb_dir_name=args.rgb_dir,
                frame_period=args.frame_period,
                max_delta_s=args.max_delta,
                materialize=args.materialize,
                overwrite=args.overwrite,
                include_removed_in_rgb_timeline=args.include_removed_in_rgb_timeline,
                negative_ratio=args.negative_ratio,
                negative_exclusion_s=args.negative_exclusion,
                negative_seed=args.negative_seed,
            )
            all_results.append(result)

            write_manifest(sequence_output_root, split, rows)

            sequence_report = {
                "dataset_root": str(dataset_root),
                "sequence": seq,
                "split": split,
                "output_root": str(sequence_output_root),
                "rgb_dir": args.rgb_dir,
                "annotation_files": annotation_candidates,
                "frame_period": args.frame_period,
                "max_delta": args.max_delta,
                "negative_ratio": args.negative_ratio,
                "negative_exclusion": args.negative_exclusion,
                "negative_seed": args.negative_seed,
                "materialize": args.materialize,
                "include_removed_in_rgb_timeline": args.include_removed_in_rgb_timeline,
                "result": result.__dict__,
                "totals": {
                    split: len(rows)
                },
            }
            write_sequence_report(sequence_output_root, sequence_report)
            sequence_reports.append(sequence_report)

    summary_report = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "rgb_dir": args.rgb_dir,
        "annotation_files": annotation_candidates,
        "frame_period": args.frame_period,
        "max_delta": args.max_delta,
        "negative_ratio": args.negative_ratio,
        "negative_exclusion": args.negative_exclusion,
        "negative_seed": args.negative_seed,
        "materialize": args.materialize,
        "include_removed_in_rgb_timeline": args.include_removed_in_rgb_timeline,
        "splits": splits,
        "results": [result.__dict__ for result in all_results],
        "sequences": sequence_reports,
        "totals": {
            "sequences_processed": len(sequence_reports),
            "samples_written": sum(r.samples_written for r in all_results),
            "negative_samples_written": sum(r.negative_samples_written for r in all_results),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preprocessing_report.json").write_text(json.dumps(summary_report, indent=2), encoding="utf-8")

    print(json.dumps(summary_report["totals"], indent=2))
    missing = [r for r in all_results if r.missing_dirs]
    if missing:
        print("Warnings: some requested sequences or required files were missing. See preprocessing_report.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
