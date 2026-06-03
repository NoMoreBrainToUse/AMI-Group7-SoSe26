"""
Train YOLO on RGB and event data, then print a comparison.

Usage:
    python train.py                              # both modalities, fred_yolo_1 split
    python train.py --mode rgb                   # only RGB
    python train.py --mode event                 # only event
    python train.py --model yolo11m              # different base model
    python train.py --model runs/detect/rgb_yolo11s/weights/best  # fine-tune from checkpoint
    python train.py --dataset fred_yolo_0        # use original split instead
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

CONFIGS = "/home/simonwesenick/Projects/ami-project/configs"

TRAIN_ARGS = dict(
    epochs=5,
    imgsz=640,
    batch=16,
    patience=20,
    device=0,
    pretrained=True,
)


def best_map50(run_dir: Path) -> float:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return float("nan")
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    values = [float(r["metrics/mAP50(B)"]) for r in rows if r.get("metrics/mAP50(B)")]
    return max(values) if values else float("nan")


def train(mode: str, model_arg: str, dataset: str) -> Path:
    data = f"{CONFIGS}/{dataset}/data_{mode}.yaml"

    # model_arg is either a short name like "yolo11s" or a path to a .pt file
    model_path = Path(model_arg)
    if model_path.exists() or model_path.with_suffix(".pt").exists():
        model_file = str(model_path.with_suffix(".pt")) if not model_arg.endswith(".pt") else model_arg
    else:
        model_file = f"{model_arg}.pt"

    model = YOLO(model_file)
    results = model.train(
        data=data,
        name=f"{mode}_{Path(model_arg).stem}_{dataset}",
        **TRAIN_ARGS,
    )
    return Path(results.save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default="both", choices=["both", "rgb", "event"])
    parser.add_argument("--model",   default="yolo11s", help="Model name or path to .pt checkpoint")
    parser.add_argument("--dataset", default="fred_yolo_1", help="Dataset config folder name")
    args = parser.parse_args()

    modes = ["rgb", "event"] if args.mode == "both" else [args.mode]
    save_dirs: dict[str, Path] = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"  Training : {mode.upper()}")
        print(f"  Model    : {args.model}")
        print(f"  Dataset  : {args.dataset}")
        print(f"{'='*50}\n")
        save_dirs[mode] = train(mode, args.model, args.dataset)

    print("\n" + "="*50)
    print("  Results")
    print("="*50)
    for mode, d in save_dirs.items():
        map50 = best_map50(d)
        print(f"  {mode:<8}  mAP50={map50:.4f}  weights: {d / 'weights' / 'best.pt'}")
    print()


if __name__ == "__main__":
    main()
