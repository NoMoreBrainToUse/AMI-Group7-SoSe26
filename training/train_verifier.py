#!/usr/bin/env python3
"""Train an EfficientNet-B0 drone/background crop verifier.

Input: crop manifests from training/export_crops.py (one train, one val).
Saves checkpoints in the format hybrid_vision.verifier loads:
  {"arch", "crop_size", "val_auc", "model_state"}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_vision.common import read_jsonl  # noqa: E402
from hybrid_vision.verifier import build_model  # noqa: E402


def auc(labels: list[int], scores: list[float]) -> float:
    """ROC-AUC via rank statistic (no sklearn dependency)."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = sum(i + 1 for i, (_, l) in enumerate(pairs) if l == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet-B0 crop verifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for best.pt / last.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                scale=(0.9, 1.1)),
        transforms.ToTensor(), norm])
    val_tf = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(), norm])

    class CropDataset(Dataset):
        def __init__(self, manifest: Path, tf):
            self.records = read_jsonl(manifest)
            self.tf = tf

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            rec = self.records[i]
            img = Image.open(rec["crop_path"]).convert("RGB")
            return self.tf(img), float(rec["label"])

    train_dl = DataLoader(CropDataset(args.train_manifest, train_tf),
                          batch_size=args.batch, shuffle=True, num_workers=4)
    val_dl = DataLoader(CropDataset(args.val_manifest, val_tf),
                        batch_size=args.batch, num_workers=4)

    model = build_model(args.arch).to(device)
    # imagenet init: torchvision weights arg lives in build_model's caller;
    # start from pretrained backbone for sample efficiency
    from torchvision import models
    pretrained = models.efficientnet_b0(weights="IMAGENET1K_V1").state_dict()
    missing = model.load_state_dict(pretrained, strict=False)
    del missing

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)

        model.eval()
        labels, scores = [], []
        with torch.no_grad():
            for x, y in val_dl:
                s = torch.sigmoid(model(x.to(device)).squeeze(1))
                scores.extend(s.cpu().tolist())
                labels.extend(int(v) for v in y.tolist())
        v_auc = auc(labels, scores)
        print(f"epoch {epoch:3d}  train_loss {total / len(train_dl.dataset):.4f}"
              f"  val_auc {v_auc:.4f}")

        ckpt = {"epoch": epoch, "arch": args.arch,
                "crop_size": args.crop_size, "val_auc": v_auc,
                "model_state": model.state_dict()}
        torch.save(ckpt, args.output / "last.pt")
        if v_auc > best_auc:
            best_auc = v_auc
            torch.save(ckpt, args.output / "best.pt")

    print(f"best val AUC: {best_auc:.4f} -> {args.output / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
