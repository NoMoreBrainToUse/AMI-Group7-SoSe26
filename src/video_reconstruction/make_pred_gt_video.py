#!/usr/bin/env python3
"""
Create a reconstruction video by combining prediction frames with ground truth boxes.

The input prediction frames are expected to be images already rendered with
prediction boxes and confidence percentages, for example the output from:

  python src/training/predict_event_yolo_val.py

This script overlays matching YOLO ground-truth boxes in green and writes a video.

Example:
  python src/video_reconstruction/make_pred_gt_video.py \
    --pred-source outputs/fred_subset_event_yolo_val_pred \
    --label-dir processed/fred_subset/event_yolo/labels/val \
    --output-video outputs/fred_subset_event_yolo_val_pred_gt.mp4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def list_images(source: Path, pattern: str) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        (
            path
            for path in source.rglob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ),
        key=natural_key,
    )


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    if not label_path.is_file():
        return []

    boxes: list[tuple[int, int, int, int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x_center, y_center, box_w, box_h = (float(value) for value in parts[1:5])
        except ValueError:
            continue

        x1 = int(round((x_center - box_w / 2.0) * width))
        y1 = int(round((y_center - box_h / 2.0) * height))
        x2 = int(round((x_center + box_w / 2.0) * width))
        y2 = int(round((y_center + box_h / 2.0) * height))
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, 0), width - 1)
        y2 = min(max(y2, 0), height - 1)
        if x2 > x1 and y2 > y1:
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def draw_text_box(image, text: str, x: int, y: int, color: tuple[int, int, int], line_width: int) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = max(1, line_width)
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = max(0, min(x, image.shape[1] - text_w - 6))
    text_y = max(text_h + baseline + 2, min(y, image.shape[0] - 4))
    cv2.rectangle(
        image,
        (text_x, text_y - text_h - baseline - 4),
        (text_x + text_w + 4, text_y + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (text_x + 2, text_y - baseline - 1),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_legend(image, pred_color: tuple[int, int, int], gt_color: tuple[int, int, int], line_width: int) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = max(1, line_width)
    entries = [("prediction", pred_color), ("ground truth", gt_color)]
    cv2.rectangle(image, (6, 6), (190, 62), (28, 28, 28), -1)
    for index, (text, color) in enumerate(entries):
        y = 17 + index * 24
        cv2.rectangle(image, (13, y), (31, y + 12), color, -1)
        cv2.putText(
            image,
            text,
            (40, y + 12),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def overlay_ground_truth(
    image,
    label_path: Path,
    class_names: dict[int, str],
    gt_color: tuple[int, int, int],
    line_width: int,
) -> int:
    import cv2

    height, width = image.shape[:2]
    boxes = read_yolo_boxes(label_path, width, height)
    for cls_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), gt_color, line_width)
        label = f"GT {class_names.get(cls_id, str(cls_id))}"
        draw_text_box(image, label, x1, y2 + 18, gt_color, line_width)
    return len(boxes)


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Build a prediction-vs-ground-truth reconstruction video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred-source",
        default=str(root / "outputs" / "fred_subset_event_yolo_val_pred"),
        help="Directory or image file from the prediction output.",
    )
    parser.add_argument("--pattern", default="*.png", help="Image glob pattern when --pred-source is a directory.")
    parser.add_argument(
        "--label-dir",
        default=str(root / "processed" / "fred_subset" / "event_yolo" / "labels" / "val"),
        help="YOLO ground-truth label directory with matching filename stems.",
    )
    parser.add_argument(
        "--output-video",
        default=str(root / "outputs" / "fred_subset_event_yolo_val_pred_gt.mp4"),
        help="Output video path.",
    )
    parser.add_argument("--fps", type=positive_float, default=30.0, help="Output video FPS.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec, e.g. mp4v or XVID.")
    parser.add_argument("--line-width", type=positive_int, default=2, help="Ground-truth box line width.")
    parser.add_argument("--max-frames", type=positive_int, default=None, help="Optional limit for quick previews.")
    parser.add_argument("--no-legend", action="store_true", help="Do not draw the color legend.")
    return parser


def main() -> int:
    import cv2

    root = repo_root()
    args = build_parser().parse_args()
    pred_source = resolve_path(args.pred_source, root)
    label_dir = resolve_path(args.label_dir, root)
    output_video = resolve_path(args.output_video, root)

    if not pred_source.exists():
        raise FileNotFoundError(f"Prediction source not found: {pred_source}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth label directory not found: {label_dir}")

    frames = list_images(pred_source, args.pattern)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError(f"No prediction frames found under: {pred_source}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"Could not read first frame: {frames[0]}")

    height, width = first.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_video), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_video}")

    class_names = {0: "drone"}
    pred_color = (255, 80, 20)
    gt_color = (40, 220, 40)
    missing_labels = 0
    written = 0

    try:
        for index, frame_path in enumerate(frames, start=1):
            image = cv2.imread(str(frame_path))
            if image is None:
                print(f"Warning: could not read frame, skipped: {frame_path}")
                continue
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            label_path = label_dir / f"{frame_path.stem}.txt"
            if not label_path.exists():
                missing_labels += 1
            overlay_ground_truth(image, label_path, class_names, gt_color, args.line_width)
            if not args.no_legend:
                draw_legend(image, pred_color, gt_color, args.line_width)

            writer.write(image)
            written += 1
            if index % 100 == 0 or index == len(frames):
                print(f"Written {index}/{len(frames)} frames")
    finally:
        writer.release()

    print(f"Video: {output_video}")
    print(f"Frames written: {written}")
    print(f"Missing labels: {missing_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
