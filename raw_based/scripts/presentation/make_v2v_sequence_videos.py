#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MP4 videos from V2V generated sequence PNG frames.",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("external/V2V/generated_tests"),
        help="Root directory containing V2V generated sequence folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for MP4 files (default: <generated-root>/videos).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="Fallback FPS when implied FPS cannot be computed.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/preprocessed_all_val"),
        help="Root directory containing preprocessed_fred_*/paired/manifest_val.csv files.",
    )
    parser.add_argument(
        "--implied-fps",
        action="store_true",
        help="Use per-sequence FPS implied by manifest_val timestamps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    return parser.parse_args()


def collect_sequence_dirs(generated_root: Path) -> list[Path]:
    dirs = [
        path
        for path in generated_root.glob("*/results/EVBIRD/*")
        if path.is_dir() and list(path.glob("*.png"))
    ]
    return sorted(dirs)


def parse_sequence_descriptor(sequence_name: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"seq(\d+)_(\d+)frames", sequence_name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def implied_fps_from_manifest(manifest_root: Path, sequence_id: int, frame_count: int) -> float | None:
    manifest_path = manifest_root / f"preprocessed_fred_{sequence_id}" / "paired" / "manifest_val.csv"
    if not manifest_path.exists():
        return None

    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") == "val":
                rows.append(row)

    rows.sort(key=lambda row: float(row["event_time_s"]))
    needed = frame_count + 1
    if len(rows) < needed:
        return None

    used_rows = rows[:needed]
    t0 = float(used_rows[0]["event_time_s"])
    t1 = float(used_rows[-1]["event_time_s"])
    duration = t1 - t0
    if duration <= 0:
        return None

    return frame_count / duration


def build_video_from_frames(frame_dir: Path, output_video: Path, fps: int) -> int:
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        return 0
    # Validate first frame early to fail fast on corrupted sequences.
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame in {frame_dir}")

    fps_str = f"{fps:.6f}"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        fps_str,
        "-i",
        str(frame_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {frame_dir}: {result.stderr.strip()}")

    return len(frame_paths)


def main() -> int:
    args = parse_args()
    generated_root = args.generated_root.resolve()
    manifest_root = args.manifest_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else (generated_root / "videos").resolve())
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = collect_sequence_dirs(generated_root)
    if not sequence_dirs:
        print(f"No sequence frame directories found under {generated_root}")
        return 1

    print(f"Found {len(sequence_dirs)} sequences under {generated_root}")
    if args.implied_fps:
        print(f"Writing videos to {output_dir} with per-sequence implied FPS from {manifest_root}")
    else:
        print(f"Writing videos to {output_dir} at {args.fps:.3f} FPS")

    created = 0
    skipped = 0
    failed = 0

    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        output_video = output_dir / f"{sequence_name}.mp4"

        fps = args.fps
        if args.implied_fps:
            parsed = parse_sequence_descriptor(sequence_name)
            if parsed is not None:
                sequence_id, frame_count = parsed
                implied = implied_fps_from_manifest(manifest_root, sequence_id, frame_count)
                if implied is not None:
                    fps = implied

        if output_video.exists() and not args.overwrite:
            print(f"[skip] {sequence_name}: {output_video.name} already exists")
            skipped += 1
            continue

        try:
            written = build_video_from_frames(sequence_dir, output_video, fps)
            if written == 0:
                print(f"[skip] {sequence_name}: no valid frames")
                skipped += 1
                continue
            print(f"[ok]   {sequence_name}: wrote {written} frames at {fps:.3f} FPS -> {output_video.name}")
            created += 1
        except Exception as exc:
            print(f"[fail] {sequence_name}: {exc}")
            failed += 1

    print(f"Done. created={created}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
