#!/usr/bin/env python3
"""
Event-guided RGB contrast pilot (label-free).

Key ideas:
- Learn a small event->RGB alignment flow.
- Predict a gated residual enhancement (apply mainly where event activity exists).
- Optimize label-free losses focused on local contrast in event-active regions.

This script is intended for fast pilot runs before full training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor as tv_pil_to_tensor
from torchvision.utils import save_image


@dataclass(frozen=True)
class PairRecord:
    seq_root: Path
    rgb_path: Path
    event_path: Path
    split: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return tv_pil_to_tensor(image).float() / 255.0


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = torch.clamp(t, 0.0, 1.0)
    t = (t * 255.0).byte().permute(1, 2, 0).contiguous().numpy()
    if t.shape[2] == 1:
        return Image.fromarray(t[:, :, 0], mode="L")
    return Image.fromarray(t, mode="RGB")


def parse_dataset_list(value: str) -> set[str]:
    if not value.strip():
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def matches_dataset(seq_dir_name: str, selected: set[str]) -> bool:
    if not selected:
        return True
    if seq_dir_name in selected:
        return True
    prefix = "preprocessed_fred_"
    if seq_dir_name.startswith(prefix) and seq_dir_name[len(prefix) :] in selected:
        return True
    return False


def discover_records(data_root: Path, split: str, datasets: set[str] | None = None) -> list[PairRecord]:
    records: list[PairRecord] = []
    selected = datasets or set()
    seq_dirs = sorted(p for p in data_root.glob("preprocessed_fred_*") if p.is_dir())

    for seq_dir in seq_dirs:
        if not matches_dataset(seq_dir.name, selected):
            continue
        manifest_path = seq_dir / "paired" / f"manifest_{split}.csv"
        if not manifest_path.is_file():
            continue

        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rgb_rel = row.get("rgb_image", "").strip()
                event_rel = row.get("event_image", "").strip()
                if not rgb_rel or not event_rel:
                    continue

                rgb_path = seq_dir / rgb_rel
                event_path = seq_dir / event_rel
                if not rgb_path.is_file() or not event_path.is_file():
                    continue

                records.append(
                    PairRecord(
                        seq_root=seq_dir,
                        rgb_path=rgb_path,
                        event_path=event_path,
                        split=split,
                    )
                )

    return records


class PairManifestDataset(Dataset):
    def __init__(self, records: list[PairRecord], image_size: int = 256) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.records[index]

        rgb = Image.open(item.rgb_path).convert("RGB")
        event = Image.open(item.event_path).convert("L")

        rgb = rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        event = event.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        return {
            "rgb": pil_to_tensor(rgb),
            "event": pil_to_tensor(event),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _base_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height, device=device),
        torch.linspace(-1.0, 1.0, steps=width, device=device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1)
    return grid.unsqueeze(0).repeat(batch, 1, 1, 1)


def warp_with_flow(x: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
    # flow_px: Bx2xHxW in pixel units (dx, dy).
    b, _, h, w = x.shape
    grid = _base_grid(b, h, w, x.device)

    flow_x = flow_px[:, 0:1, :, :] * (2.0 / max(1.0, float(w - 1)))
    flow_y = flow_px[:, 1:2, :, :] * (2.0 / max(1.0, float(h - 1)))
    flow = torch.cat([flow_x, flow_y], dim=1).permute(0, 2, 3, 1)

    return F.grid_sample(x, grid + flow, mode="bilinear", padding_mode="border", align_corners=True)


class EventGuidedContrastEnhancer(nn.Module):
    def __init__(
        self,
        base_channels: int = 24,
        max_flow_px: float = 4.0,
        residual_scale: float = 0.3,
        gate_prior_mix: float = 0.75,
        gate_prior_temperature: float = 2.4,
        align_mix: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_flow_px = max_flow_px
        self.residual_scale = residual_scale
        self.gate_prior_mix = gate_prior_mix
        self.gate_prior_temperature = gate_prior_temperature
        self.align_mix = align_mix

        self.rgb_enc = ConvBlock(3, base_channels)
        self.event_enc = ConvBlock(1, base_channels)

        self.flow_head = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, 2, kernel_size=3, padding=1),
        )

        self.fuse = ConvBlock(base_channels * 2, base_channels)
        self.gate_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=1),
        )

    def _event_prior(self, event_aligned: torch.Tensor) -> torch.Tensor:
        # Build a sparse hotspot prior from event-aligned intensity and local high-pass response.
        smooth = F.avg_pool2d(event_aligned, kernel_size=7, stride=1, padding=3)
        highpass = torch.clamp(event_aligned - smooth, min=0.0)
        score = torch.clamp(0.6 * event_aligned + 0.4 * highpass, 0.0, 1.0)
        peak = score.amax(dim=(2, 3), keepdim=True)
        norm = score / (peak + 1e-6)
        prior = torch.clamp(norm, 0.0, 1.0) ** self.gate_prior_temperature
        return F.max_pool2d(prior, kernel_size=5, stride=1, padding=2)

    def forward(self, rgb: torch.Tensor, event: torch.Tensor) -> dict[str, torch.Tensor]:
        f_rgb = self.rgb_enc(rgb)
        f_event = self.event_enc(event)

        flow = torch.tanh(self.flow_head(torch.cat([f_rgb, f_event], dim=1))) * self.max_flow_px
        event_warped = warp_with_flow(event, flow)
        f_event_warped = warp_with_flow(f_event, flow)

        # Keep learned alignment as a small correction when raw event/RGB are already well aligned.
        align_mix = torch.clamp(torch.tensor(self.align_mix, device=rgb.device), 0.0, 1.0)
        event_aligned = (1.0 - align_mix) * event + align_mix * event_warped
        f_event_aligned = (1.0 - align_mix) * f_event + align_mix * f_event_warped

        fused = self.fuse(torch.cat([f_rgb, f_event_aligned], dim=1))

        # Keep gate event-driven: logits from aligned event features + explicit event hotspot prior.
        gate_learned = torch.sigmoid(self.gate_head(f_event_aligned))
        gate_prior = self._event_prior(event_aligned)
        mix = torch.clamp(torch.tensor(self.gate_prior_mix, device=rgb.device), 0.0, 1.0)
        gate = torch.clamp((1.0 - mix) * gate_learned + mix * gate_prior, 0.0, 1.0)

        residual = torch.tanh(self.residual_head(fused)) * self.residual_scale
        gated_residual = gate * residual
        enhanced = torch.clamp(rgb + gated_residual, 0.0, 1.0)

        return {
            "enhanced": enhanced,
            "residual": residual,
            "gated_residual": gated_residual,
            "gate": gate,
            "gate_prior": gate_prior,
            "flow": flow,
            "event_warped": event_warped,
            "event_aligned": event_aligned,
        }


class ContrastPilotLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        gy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("gx", gx.view(1, 1, 3, 3))
        self.register_buffer("gy", gy.view(1, 1, 3, 3))

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        dx = F.conv2d(gray, self.gx, padding=1)
        dy = F.conv2d(gray, self.gy, padding=1)
        return torch.sqrt(dx * dx + dy * dy + self.eps)

    def _tv(self, x: torch.Tensor) -> torch.Tensor:
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    def _event_mask(self, event_aligned: torch.Tensor, threshold: float, blur_kernel: int) -> torch.Tensor:
        mask = torch.sigmoid((event_aligned - threshold) * 12.0)
        if blur_kernel > 1:
            blur = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            mask = F.avg_pool2d(mask, kernel_size=blur, stride=1, padding=blur // 2)
        return torch.clamp(mask, 0.0, 1.0)

    def _peak_mask(self, event_aligned: torch.Tensor, temperature: float) -> torch.Tensor:
        peak = event_aligned.amax(dim=(2, 3), keepdim=True)
        normalized = event_aligned / (peak + self.eps)
        return torch.clamp(normalized, 0.0, 1.0) ** temperature

    def _hotspot_mask(self, event_aligned: torch.Tensor, temperature: float) -> torch.Tensor:
        peak_mask = self._peak_mask(event_aligned, temperature)
        hotspot = F.max_pool2d(peak_mask, kernel_size=7, stride=1, padding=3)
        return torch.clamp(hotspot, 0.0, 1.0)

    def forward(
        self,
        rgb: torch.Tensor,
        enhanced: torch.Tensor,
        gated_residual: torch.Tensor,
        gate: torch.Tensor,
        event_aligned: torch.Tensor,
        flow: torch.Tensor,
        event_threshold: float,
        event_blur: int,
        peak_temperature: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask = self._event_mask(event_aligned, event_threshold, event_blur)
        peak_mask = self._peak_mask(event_aligned, peak_temperature)
        hotspot_mask = self._hotspot_mask(event_aligned, peak_temperature)
        focus_mask = torch.clamp(mask + hotspot_mask, 0.0, 1.0)
        inv_mask = 1.0 - mask
        inv_hotspot = 1.0 - hotspot_mask

        l_recon = F.l1_loss(enhanced, rgb)

        grad_rgb = self._grad_mag(rgb)
        grad_enh = self._grad_mag(enhanced)
        edge_gain = F.relu(0.03 - (grad_enh - grad_rgb))
        l_edge_event = (focus_mask * edge_gain).mean()

        gray_enh = enhanced.mean(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(gray_enh, kernel_size=9, stride=1, padding=4)
        local_sq = F.avg_pool2d(gray_enh * gray_enh, kernel_size=9, stride=1, padding=4)
        local_std = torch.sqrt(torch.clamp(local_sq - local_mean * local_mean, min=0.0) + self.eps)

        foreground = gray_enh
        background = local_mean
        contrast_score = (foreground - background).abs() / (local_std + self.eps)
        contrast_deficit = F.relu(0.75 - contrast_score)
        l_contrast_event = (focus_mask * contrast_deficit).mean()

        l_bg_tv = (inv_hotspot * (enhanced - rgb)).abs().mean() + inv_hotspot.mean() * self._tv(enhanced)
        l_gate_event = (focus_mask * F.relu(0.3 - gate)).mean()
        l_gate_bg = (inv_hotspot * gate).mean()
        l_gate_sparse = gate.mean()

        flow_mag = torch.sqrt(flow[:, 0:1] * flow[:, 0:1] + flow[:, 1:2] * flow[:, 1:2] + self.eps)
        l_flow_reg = flow_mag.mean()

        l_residual_event = (focus_mask * F.relu(0.02 - gated_residual.abs().mean(dim=1, keepdim=True))).mean()

        stats = {
            "loss_recon": float(l_recon.detach().cpu()),
            "loss_edge_event": float(l_edge_event.detach().cpu()),
            "loss_contrast_event": float(l_contrast_event.detach().cpu()),
            "loss_bg_tv": float(l_bg_tv.detach().cpu()),
            "loss_gate_event": float(l_gate_event.detach().cpu()),
            "loss_gate_bg": float(l_gate_bg.detach().cpu()),
            "loss_gate_sparse": float(l_gate_sparse.detach().cpu()),
            "loss_flow_reg": float(l_flow_reg.detach().cpu()),
            "loss_residual_event": float(l_residual_event.detach().cpu()),
            "mask_mean": float(mask.mean().detach().cpu()),
            "peak_mean": float(peak_mask.mean().detach().cpu()),
            "hotspot_mean": float(hotspot_mask.mean().detach().cpu()),
        }
        return (
            l_recon,
            l_edge_event,
            l_contrast_event,
            l_bg_tv,
            l_gate_event,
            l_gate_bg,
            l_gate_sparse,
            l_flow_reg,
            l_residual_event,
            stats,
        )


def run_epoch(
    model: EventGuidedContrastEnhancer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    loss_fn: ContrastPilotLoss,
    args: argparse.Namespace,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_edge_event": 0.0,
        "loss_contrast_event": 0.0,
        "loss_bg_tv": 0.0,
        "loss_gate_event": 0.0,
        "loss_gate_bg": 0.0,
        "loss_gate_sparse": 0.0,
        "loss_flow_reg": 0.0,
        "loss_residual_event": 0.0,
        "mask_mean": 0.0,
        "peak_mean": 0.0,
        "hotspot_mean": 0.0,
    }
    count = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        rgb = batch["rgb"].to(device, non_blocking=True)
        event = batch["event"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(rgb, event)
                (
                    l_recon,
                    l_edge_event,
                    l_contrast_event,
                    l_bg_tv,
                    l_gate_event,
                    l_gate_bg,
                    l_gate_sparse,
                    l_flow_reg,
                    l_residual_event,
                    stats,
                ) = loss_fn(
                    rgb=rgb,
                    enhanced=out["enhanced"],
                    gated_residual=out["gated_residual"],
                    gate=out["gate"],
                    event_aligned=out["event_aligned"],
                    flow=out["flow"],
                    event_threshold=args.event_threshold,
                    event_blur=args.event_blur,
                    peak_temperature=args.peak_temperature,
                )

                loss = (
                    args.w_recon * l_recon
                    + args.w_edge_event * l_edge_event
                    + args.w_contrast_event * l_contrast_event
                    + args.w_bg_tv * l_bg_tv
                    + args.w_gate_event * l_gate_event
                    + args.w_gate_bg * l_gate_bg
                    + args.w_gate_sparse * l_gate_sparse
                    + args.w_flow_reg * l_flow_reg
                    + args.w_residual_event * l_residual_event
                )

            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        totals["loss_total"] += float(loss.detach().cpu())
        for key in stats:
            totals[key] += stats[key]
        count += 1

    if count == 0:
        return totals

    for key in totals:
        totals[key] /= count
    return totals


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_safe = {}
    for key, value in cfg.items():
        cfg_safe[key] = str(value) if isinstance(value, Path) else value
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg_safe,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = parse_dataset_list(args.datasets)
    train_records = discover_records(data_root, "train", selected_datasets)
    val_records = discover_records(data_root, "val", selected_datasets)

    if args.limit_train_records is not None:
        train_records = train_records[: max(1, args.limit_train_records)]
    if args.limit_val_records is not None and val_records:
        val_records = val_records[: max(1, args.limit_val_records)]

    if not train_records:
        raise RuntimeError(f"No training records found under {data_root}")

    train_ds = PairManifestDataset(train_records, image_size=args.image_size)
    val_ds = PairManifestDataset(val_records, image_size=args.image_size) if val_records else None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    model = EventGuidedContrastEnhancer(
        base_channels=args.base_channels,
        max_flow_px=args.max_flow_px,
        residual_scale=args.residual_scale,
        gate_prior_mix=args.gate_prior_mix,
        gate_prior_temperature=args.gate_prior_temperature,
        align_mix=args.align_mix,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = ContrastPilotLoss().to(device)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    history: list[dict] = []
    cfg = vars(args).copy()

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            loss_fn=loss_fn,
            args=args,
            max_batches=args.max_train_batches,
        )

        row = {"epoch": epoch, "train": train_stats}

        if val_loader is not None:
            with torch.no_grad():
                val_stats = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    scaler=None,
                    use_amp=use_amp,
                    loss_fn=loss_fn,
                    args=args,
                    max_batches=args.max_val_batches,
                )
            row["val"] = val_stats
            print(
                f"Epoch {epoch:03d} | train total={train_stats['loss_total']:.4f} "
                f"val total={val_stats['loss_total']:.4f} train mask={train_stats['mask_mean']:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train total={train_stats['loss_total']:.4f} train mask={train_stats['mask_mean']:.4f}")

        history.append(row)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(out_dir / "checkpoints" / f"pilot_epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg)

    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


@torch.no_grad()
def infer(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = EventGuidedContrastEnhancer(
        base_channels=int(cfg.get("base_channels", args.base_channels)),
        max_flow_px=float(cfg.get("max_flow_px", args.max_flow_px)),
        residual_scale=float(cfg.get("residual_scale", args.residual_scale)),
        gate_prior_mix=float(cfg.get("gate_prior_mix", args.gate_prior_mix)),
        gate_prior_temperature=float(cfg.get("gate_prior_temperature", args.gate_prior_temperature)),
        align_mix=float(cfg.get("align_mix", args.align_mix)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    selected_datasets = parse_dataset_list(args.datasets)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    records: list[PairRecord] = []
    for split in splits:
        records.extend(discover_records(args.data_root.resolve(), split, selected_datasets))

    if not records:
        raise RuntimeError(f"No records found for split={args.split}")
    if args.max_records is not None:
        records = records[: max(1, args.max_records)]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = out_dir / "enhanced"
    triplet_dir = out_dir / "triplets"
    debug_dir = out_dir / "debug"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    if args.save_triplets:
        triplet_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(records):
        rgb = Image.open(rec.rgb_path).convert("RGB")
        event = Image.open(rec.event_path).convert("L")

        orig_size = rgb.size
        rgb_r = rgb.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
        event_r = event.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)

        rgb_t = pil_to_tensor(rgb_r).unsqueeze(0).to(device)
        event_t = pil_to_tensor(event_r).unsqueeze(0).to(device)

        out = model(rgb_t, event_t)
        peak = out["event_aligned"].amax(dim=(2, 3), keepdim=True)
        peak_mask = torch.clamp(out["event_aligned"] / (peak + 1e-6), 0.0, 1.0) ** 1.8
        focus_residual = out["gated_residual"] * (0.25 + 0.75 * peak_mask) * args.focus_gain
        enhanced_t = torch.clamp(rgb_t + focus_residual * args.residual_gain, 0.0, 1.0)

        enhanced = enhanced_t.squeeze(0).cpu()
        gate = out["gate"].repeat(1, 3, 1, 1).squeeze(0).cpu()
        gate_prior = out["gate_prior"].repeat(1, 3, 1, 1).squeeze(0).cpu()
        event_aligned = out["event_aligned"].repeat(1, 3, 1, 1).squeeze(0).cpu()

        enh_pil = tensor_to_pil(enhanced).resize(orig_size, Image.Resampling.BILINEAR)
        stem = f"{rec.seq_root.name}_{rec.rgb_path.stem}"
        enh_pil.save(enhanced_dir / f"{stem}_enhanced.png")

        if args.save_triplets:
            triplet = torch.cat(
                [
                    rgb_t.squeeze(0).cpu(),
                    enhanced,
                    event_t.repeat(1, 3, 1, 1).squeeze(0).cpu(),
                ],
                dim=2,
            )
            save_image(triplet, triplet_dir / f"{stem}_triplet.png")

            debug = torch.cat(
                [
                    event_t.repeat(1, 3, 1, 1).squeeze(0).cpu(),
                    event_aligned,
                    gate_prior,
                    gate,
                ],
                dim=2,
            )
            save_image(debug, debug_dir / f"{stem}_event_align_gate.png")
            save_image(peak_mask.repeat(1, 3, 1, 1).squeeze(0).cpu(), debug_dir / f"{stem}_peak_mask.png")

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(records)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-guided RGB contrast pilot (label-free).")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train pilot enhancer.")
    train_p.add_argument("--data-root", type=Path, default=Path("data/preprocessed"))
    train_p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset folders or IDs, e.g. preprocessed_fred_1,preprocessed_fred_10 or 1,10.",
    )
    train_p.add_argument("--out-dir", type=Path, default=Path("artifacts/enhancer_pilot_contrast"))
    train_p.add_argument("--epochs", type=int, default=6)
    train_p.add_argument("--batch-size", type=int, default=1)
    train_p.add_argument("--num-workers", type=int, default=4)
    train_p.add_argument("--image-size", type=int, default=256)
    train_p.add_argument("--lr", type=float, default=8e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--base-channels", type=int, default=24)
    train_p.add_argument("--max-flow-px", type=float, default=4.0)
    train_p.add_argument("--residual-scale", type=float, default=0.3)
    train_p.add_argument("--gate-prior-mix", type=float, default=0.75)
    train_p.add_argument("--gate-prior-temperature", type=float, default=2.4)
    train_p.add_argument("--align-mix", type=float, default=0.2)

    train_p.add_argument("--w-recon", type=float, default=1.0)
    train_p.add_argument("--w-edge-event", type=float, default=1.2)
    train_p.add_argument("--w-contrast-event", type=float, default=2.0)
    train_p.add_argument("--w-bg-tv", type=float, default=0.35)
    train_p.add_argument("--w-gate-event", type=float, default=0.5)
    train_p.add_argument("--w-gate-bg", type=float, default=0.6)
    train_p.add_argument("--w-gate-sparse", type=float, default=0.2)
    train_p.add_argument("--w-flow-reg", type=float, default=0.02)
    train_p.add_argument("--w-residual-event", type=float, default=0.8)

    train_p.add_argument("--event-threshold", type=float, default=0.08)
    train_p.add_argument("--event-blur", type=int, default=5)
    train_p.add_argument("--peak-temperature", type=float, default=1.8)

    train_p.add_argument("--max-train-batches", type=int, default=220)
    train_p.add_argument("--max-val-batches", type=int, default=40)
    train_p.add_argument("--limit-train-records", type=int, default=800)
    train_p.add_argument("--limit-val-records", type=int, default=200)

    train_p.add_argument("--save-every", type=int, default=1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--no-amp", action="store_true")
    train_p.add_argument("--cpu", action="store_true")

    infer_p = sub.add_parser("infer", help="Run pilot inference.")
    infer_p.add_argument("--data-root", type=Path, default=Path("data/preprocessed"))
    infer_p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset folders or IDs, e.g. preprocessed_fred_1,preprocessed_fred_10 or 1,10.",
    )
    infer_p.add_argument("--checkpoint", type=Path, required=True)
    infer_p.add_argument("--out-dir", type=Path, default=Path("artifacts/enhancer_pilot_contrast_samples"))
    infer_p.add_argument("--split", choices=["all", "train", "val", "test"], default="val")
    infer_p.add_argument("--image-size", type=int, default=256)
    infer_p.add_argument("--max-records", type=int, default=50)
    infer_p.add_argument("--residual-gain", type=float, default=1.5)
    infer_p.add_argument("--focus-gain", type=float, default=1.0)
    infer_p.add_argument("--base-channels", type=int, default=24)
    infer_p.add_argument("--max-flow-px", type=float, default=4.0)
    infer_p.add_argument("--residual-scale", type=float, default=0.3)
    infer_p.add_argument("--gate-prior-mix", type=float, default=0.75)
    infer_p.add_argument("--gate-prior-temperature", type=float, default=2.4)
    infer_p.add_argument("--align-mix", type=float, default=0.2)
    infer_p.add_argument("--save-triplets", action="store_true")
    infer_p.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
