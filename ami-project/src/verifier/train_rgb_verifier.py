#!/usr/bin/env python3
"""
Train a binary RGB crop verifier (drone / background).

Takes the crop manifests produced by extract_crops.py directly — no directory
scanning, no risk of mixing train and val crops.

Output under runs/verifier/rgb/<name>/:
  best.pt        best checkpoint (highest val AUC)
  last.pt        final epoch checkpoint
  metrics.csv    per-epoch train/val loss, accuracy, AUC
  train_config.json

Architecture: MobileNetV3 Small pretrained on ImageNet, final layer replaced
with a single-logit head. Sigmoid of the logit = P(drone).

Class imbalance is handled by WeightedRandomSampler so each batch is ~50/50.

Example:
  python src/verifier/train_rgb_verifier.py \\
    --train-manifest processed/fred_subset/crops/rgb/crop_manifest_train_conf0.20.jsonl \\
    --val-manifest   processed/fred_subset/crops/rgb/crop_manifest_val_conf0.20.jsonl

  # Larger backbone:
  python src/verifier/train_rgb_verifier.py \\
    --train-manifest ... --val-manifest ... \\
    --arch efficientnet_b0 --epochs 30

  # Quick smoke-test:
  python src/verifier/train_rgb_verifier.py \\
    --train-manifest ... --val-manifest ... \\
    --epochs 2 --batch 32
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def positive_int(v: str) -> int:
    n = int(v)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


ARCH_CHOICES = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Train a binary RGB crop verifier from crop manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-manifest",
        required=True,
        help="crop_manifest_train_conf*.jsonl produced by extract_crops.py",
    )
    parser.add_argument(
        "--val-manifest",
        required=True,
        help="crop_manifest_val_conf*.jsonl produced by extract_crops.py",
    )
    parser.add_argument(
        "--output",
        default=str(root / "runs" / "verifier" / "rgb"),
        help="Parent directory for training run output.",
    )
    parser.add_argument("--name", default="mobilenet_v3_small", help="Run subdirectory name.")
    parser.add_argument(
        "--arch",
        default="mobilenet_v3_small",
        choices=ARCH_CHOICES,
    )
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch", type=positive_int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--crop-size",
        type=positive_int,
        default=96,
        help="Must match the size used in extract_crops.py.",
    )
    parser.add_argument("--workers", type=int, default=4 if os.name != "nt" else 0)
    parser.add_argument("--device", default=None, help="cuda:0, cpu, mps, or None for auto.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze all layers except the final classifier head.",
    )
    return parser


# ---------------------------------------------------------------------------
# Dataset — manifest-based, no directory scanning
# ---------------------------------------------------------------------------

class CropDataset:
    """Loads crops from a JSONL manifest produced by extract_crops.py."""

    def __init__(self, manifest_path: Path, transform=None):
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Only keep records with valid label (0 or 1); gray-zone records were
        # already excluded by extract_crops.py, but be defensive.
        self.records = [r for r in records if r.get("label") in (0, 1)]
        self.transform = transform

        n_drone = sum(1 for r in self.records if r["label"] == 1)
        n_bg = len(self.records) - n_drone
        print(f"  {manifest_path.name}: {len(self.records)} crops  "
              f"(drone={n_drone}, background={n_bg})")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        from PIL import Image
        import torch

        rec = self.records[idx]
        try:
            img = Image.open(rec["crop_path"]).convert("RGB")
        except Exception:
            # Fallback for missing crops: black image
            img = Image.new("RGB", (96, 96))

        if self.transform is not None:
            img = self.transform(img)

        label = rec["label"]
        return img, label

    @property
    def labels(self) -> list[int]:
        return [r["label"] for r in self.records]


def make_transforms(crop_size: int, is_train: bool):
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if is_train:
        return transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize,
    ])


def make_weighted_sampler(dataset: CropDataset):
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = dataset.labels
    class_counts = [labels.count(0), labels.count(1)]
    weights_per_class = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = torch.tensor([weights_per_class[l] for l in labels])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(arch: str, pretrained: bool, freeze_backbone: bool):
    import torch.nn as nn
    from torchvision import models

    weights_arg = "IMAGENET1K_V1" if pretrained else None

    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
    elif arch == "resnet18":
        model = models.resnet18(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_auc(labels: list[int], scores: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    auc = 0.0
    tp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            auc += tp / n_pos / n_neg
    return auc


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    import torch

    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_scores: list[float] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels_f = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels_f)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scores = torch.sigmoid(logits).squeeze(1)
            preds = (scores >= 0.5).long()
            correct += (preds == labels.to(device)).sum().item()
            total += len(labels)
            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.tolist())
            all_scores.extend(scores.detach().cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    auc = compute_auc(all_labels, all_scores)
    return avg_loss, accuracy, auc


def main() -> int:
    root = repo_root()
    args = build_parser().parse_args()

    import torch
    torch.manual_seed(args.seed)

    train_manifest = resolve_path(args.train_manifest, root)
    val_manifest = resolve_path(args.val_manifest, root)
    output_dir = resolve_path(args.output, root) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for p, name in [(train_manifest, "--train-manifest"), (val_manifest, "--val-manifest")]:
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}\n")

    print("Loading datasets:")
    train_ds = CropDataset(train_manifest, transform=make_transforms(args.crop_size, is_train=True))
    val_ds = CropDataset(val_manifest, transform=make_transforms(args.crop_size, is_train=False))

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=make_weighted_sampler(train_ds),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.arch, pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = -1.0
    metrics_rows = []

    print(f"\n{'Ep':>4}  {'t_loss':>7}  {'t_acc':>6}  {'t_auc':>6}  {'v_loss':>7}  {'v_acc':>6}  {'v_auc':>6}  {'lr':>9}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        t_loss, t_acc, t_auc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        v_loss, v_acc, v_auc = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"{epoch:>4}  {t_loss:>7.4f}  {t_acc:>6.3f}  {t_auc:>6.3f}"
            f"  {v_loss:>7.4f}  {v_acc:>6.3f}  {v_auc:>6.3f}  {lr:>9.2e}  ({elapsed:.0f}s)"
        )

        metrics_rows.append({
            "epoch": epoch,
            "train_loss": round(t_loss, 6), "train_acc": round(t_acc, 4), "train_auc": round(t_auc, 4),
            "val_loss": round(v_loss, 6), "val_acc": round(v_acc, 4), "val_auc": round(v_auc, 4),
            "lr": round(lr, 8),
        })

        is_best = not (v_auc != v_auc) and v_auc > best_val_auc  # nan-safe
        if is_best:
            best_val_auc = v_auc
            torch.save({
                "epoch": epoch, "arch": args.arch,
                "crop_size": args.crop_size, "val_auc": v_auc,
                "model_state": model.state_dict(),
            }, output_dir / "best.pt")

    torch.save({
        "epoch": args.epochs, "arch": args.arch,
        "crop_size": args.crop_size, "val_auc": metrics_rows[-1]["val_auc"] if metrics_rows else None,
        "model_state": model.state_dict(),
    }, output_dir / "last.pt")

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (output_dir / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"\nBest val AUC: {best_val_auc:.4f}")
    print(f"Weights:  {output_dir}/best.pt")
    print(f"Metrics:  {metrics_path}")
    print(f"\nNext: python src/verifier/eval_verifier.py \\")
    print(f"        --model {output_dir}/best.pt \\")
    print(f"        --manifest {val_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
