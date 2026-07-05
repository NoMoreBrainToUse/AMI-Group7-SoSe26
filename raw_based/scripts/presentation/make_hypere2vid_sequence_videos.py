#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MP4 videos from HyperE2VID generated frame PNGs.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("external/HyperE2VID/output_tests/all_120fps"),
        help="Root containing seq*_120fps frame folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for videos (default: <frames-root>/videos).",
    )
    parser.add_argument("--fps", type=int, default=120, help="Video FPS.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mp4 files.")
    return parser.parse_args()


def collect_sequence_dirs(frames_root: Path) -> list[Path]:
    dirs = [
        path
        for path in sorted(frames_root.glob("seq*_120fps"))
        if path.is_dir() and list(path.glob("frame_*.png"))
    ]
    return dirs


def write_video(frame_dir: Path, output_video: Path, fps: int) -> int:
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        return 0

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame in {frame_dir}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_video}")

    written = 0
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        writer.write(frame)
        written += 1

    writer.release()
    return written


def main() -> int:
    args = parse_args()
    frames_root = args.frames_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else (frames_root / "videos").resolve())
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = collect_sequence_dirs(frames_root)
    if not sequence_dirs:
        print(f"No HyperE2VID frame folders found under {frames_root}")
        return 1

    print(f"Found {len(sequence_dirs)} HyperE2VID sequences")
    print(f"Writing videos to {output_dir} at {args.fps} FPS")

    created = 0
    skipped = 0
    failed = 0

    for sequence_dir in sequence_dirs:
        output_video = output_dir / f"{sequence_dir.name}.mp4"
        if output_video.exists() and not args.overwrite:
            print(f"[skip] {sequence_dir.name}: {output_video.name} already exists")
            skipped += 1
            continue

        try:
            written = write_video(sequence_dir, output_video, args.fps)
            if written == 0:
                print(f"[skip] {sequence_dir.name}: no readable frames")
                skipped += 1
                continue
            print(f"[ok]   {sequence_dir.name}: wrote {written} frames -> {output_video.name}")
            created += 1
        except Exception as exc:
            print(f"[fail] {sequence_dir.name}: {exc}")
            failed += 1

    print(f"Done. created={created}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
