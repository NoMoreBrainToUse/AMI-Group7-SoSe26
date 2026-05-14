#!/usr/bin/env python3
"""
Render Event-YOLO validation predictions with bounding boxes and confidence percentages.

Example:
  python scripts/training/predict_event_yolo_val.py \
    --model runs/event_yolo/event_yolo11m_neg50_e5/weights/best.pt \
    --source processed/fred10/event_yolo/images/val \
    --output outputs/event_yolo11m_neg50_e5_val_pred \
    --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def draw_predictions(result, output_path: Path, line_width: int) -> None:
    import cv2

    image = result.orig_img.copy()
    boxes = result.boxes
    names = result.names

    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            cls_id = int(box.cls[0].detach().cpu().item())
            conf = float(box.conf[0].detach().cpu().item())
            x1, y1, x2, y2 = (int(round(value)) for value in xyxy)

            label = f"{names.get(cls_id, str(cls_id))} {conf * 100:.1f}%"
            color = (255, 80, 20)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = max(1, line_width)
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = max(0, x1)
            text_y = max(text_h + baseline + 2, y1)
            cv2.rectangle(
                image,
                (text_x, text_y - text_h - baseline - 4),
                (text_x + text_w + 4, text_y + 2),
                color,
                -1,
            )
            cv2.putText(
                image,
                label,
                (text_x + 2, text_y - baseline - 1),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Draw Event-YOLO predictions on validation images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(root / "runs" / "event_yolo" / "event_yolo11m_neg50_e5" / "weights" / "best.pt"),
        help="Path to YOLO weights.",
    )
    parser.add_argument(
        "--source",
        default=str(root / "processed" / "fred10" / "event_yolo" / "images" / "val"),
        help="Image file or directory to predict.",
    )
    parser.add_argument(
        "--output",
        default=str(root / "outputs" / "event_yolo11m_neg50_e5_val_pred"),
        help="Directory for rendered prediction images.",
    )
    parser.add_argument("--imgsz", type=positive_int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="Use FP16 inference on CUDA.")
    parser.add_argument("--device", default=None, help="CUDA device like 0, cpu, or mps.")
    parser.add_argument("--line-width", type=positive_int, default=2, help="Box line width.")
    parser.add_argument("--empty-cache-every", type=positive_int, default=25, help="Clear CUDA cache every N images.")
    parser.add_argument("--max-images", type=positive_int, default=None, help="Optional limit for quick previews.")
    return parser


def main() -> int:
    root = repo_root()
    args = build_parser().parse_args()
    model_path = resolve_path(args.model, root)
    source = resolve_path(args.source, root)
    output_dir = resolve_path(args.output, root)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    images = list_images(source)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise ValueError(f"No images found under: {source}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Could not import ultralytics. Activate the project environment first.") from exc

    model = YOLO(str(model_path))
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(images)}")

    torch = None
    if args.device not in {None, "cpu"}:
        try:
            import torch as torch_module

            torch = torch_module
        except ImportError:
            torch = None

    for index, image_path in enumerate(images, start=1):
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            verbose=False,
        )[0]
        image_path = images[index - 1]
        relative = image_path.relative_to(source) if source.is_dir() else Path(image_path.name)
        output_path = output_dir / relative
        draw_predictions(result, output_path, args.line_width)
        del result
        if torch is not None and index % args.empty_cache_every == 0:
            torch.cuda.empty_cache()
        if index % 100 == 0 or index == len(images):
            print(f"Rendered {index}/{len(images)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
