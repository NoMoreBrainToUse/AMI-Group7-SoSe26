#!/usr/bin/env python3
"""
Train a YOLO11 detector on the preprocessed FRED Event-only dataset.

Default input:
  processed/fred10/event_yolo/data.yaml

Example:
  python scripts/training/train_event_yolo.py --epochs 100 --imgsz 640 --batch 16

Use a larger YOLO11 checkpoint if the GPU has enough memory:
  python scripts/training/train_event_yolo.py --model yolo11s.pt --epochs 150
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def normalize_model_name(model: str) -> str:
    """Accept the common YOLOv11 spelling while using Ultralytics filenames."""
    name = model.strip()
    lower = name.lower()
    if lower.startswith("yolov11"):
        return "yolo11" + name[len("yolov11") :]
    return name


def count_split(dataset_dir: Path, split: str) -> tuple[int, int, int]:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    images = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    labels = [path for path in label_dir.rglob("*.txt") if path.is_file()]

    label_stems = {path.stem for path in labels}
    missing_labels = sum(1 for image in images if image.stem not in label_stems)
    return len(images), len(labels), missing_labels


def validate_event_dataset(data_yaml: Path) -> None:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    dataset_dir = data_yaml.parent
    required_dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    ]
    missing = [str(path) for path in required_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing Event YOLO directories:\n  " + "\n  ".join(missing))

    print(f"Event dataset: {dataset_dir}")
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not image_dir.is_dir() and not label_dir.is_dir():
            continue
        num_images, num_labels, missing_labels = count_split(dataset_dir, split)
        print(
            f"  {split}: {num_images} images, {num_labels} labels, "
            f"{missing_labels} images without labels"
        )
        if split in {"train", "val"} and num_images == 0:
            raise ValueError(f"No {split} images found under {image_dir}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    default_data = root / "processed" / "fred10" / "event_yolo" / "data.yaml"
    default_project = root / "runs" / "event_yolo"
    default_workers = 0 if os.name == "nt" else 8

    parser = argparse.ArgumentParser(
        description="Train Ultralytics YOLO11 on processed FRED Event frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default=str(default_data), help="Path to event_yolo/data.yaml.")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO11 checkpoint or YAML. 'yolov11*.pt' aliases are normalized to 'yolo11*.pt'.",
    )
    parser.add_argument("--epochs", type=positive_int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=positive_int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Use -1 for Ultralytics autobatch.")
    parser.add_argument("--device", default=None, help="CUDA device like 0, 0,1, cpu, or mps.")
    parser.add_argument("--workers", type=int, default=default_workers, help="DataLoader workers.")
    parser.add_argument("--project", default=str(default_project), help="Output directory for training runs.")
    parser.add_argument("--name", default="fred10_event_yolo11n", help="Run name under --project.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience.")
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer setting.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate; leave unset for default.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay; leave unset for default.")
    parser.add_argument("--cache", choices=["none", "ram", "disk"], default="none", help="Dataset cache mode.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--resume", default=None, help="Resume from a previous last.pt checkpoint.")
    parser.add_argument("--val-only", action="store_true", help="Run validation only, no training.")
    parser.add_argument("--test-after-train", action="store_true", help="Evaluate the best checkpoint on test split.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print the planned command only.")
    return parser


def train_or_validate(args: argparse.Namespace) -> None:
    root = repo_root()
    data_yaml = resolve_path(args.data, root)
    project_dir = resolve_path(args.project, root)
    model_name = normalize_model_name(args.model)

    validate_event_dataset(data_yaml)

    train_kwargs = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(project_dir),
        "name": args.name,
        "seed": args.seed,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "amp": args.amp,
        "exist_ok": args.exist_ok,
    }
    if args.cache != "none":
        train_kwargs["cache"] = args.cache
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay

    print(f"Model: {model_name}")
    print(f"Data yaml: {data_yaml}")
    print(f"Project: {project_dir}")

    if args.dry_run:
        print("Dry run only. Training arguments:")
        for key, value in train_kwargs.items():
            print(f"  {key}: {value}")
        if args.resume:
            print(f"  resume: {args.resume}")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Could not import ultralytics. Activate the conda environment where YOLO11 is installed, "
            "then run this script again."
        ) from exc

    if args.resume:
        checkpoint = resolve_path(args.resume, root)
        model = YOLO(str(checkpoint))
        print(f"Resuming from: {checkpoint}")
        model.train(resume=True)
        return

    model = YOLO(model_name)

    if args.val_only:
        model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, split="val")
        return

    train_kwargs["epochs"] = args.epochs
    results = model.train(**train_kwargs)
    print(results)

    if args.test_after_train:
        best = project_dir / args.name / "weights" / "best.pt"
        eval_model = YOLO(str(best)) if best.is_file() else model
        eval_model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, split="test")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    train_or_validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
