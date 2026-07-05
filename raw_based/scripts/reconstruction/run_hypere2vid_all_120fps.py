#!/usr/bin/env python3
"""Run HyperE2VID reconstruction for all preprocessed datasets at 120 FPS."""

from __future__ import annotations

import argparse
import csv
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HyperE2VID on all preprocessed sequences at 120 FPS")
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data/preprocessed_all_val"),
        help="Root folder containing preprocessed_fred_* directories",
    )
    parser.add_argument(
        "--hypere2vid-root",
        type=Path,
        default=Path("external/HyperE2VID"),
        help="HyperE2VID repository root",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external/HyperE2VID/pretrained/model.pth"),
        help="Checkpoint path",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("external/HyperE2VID/output_tests/all_120fps"),
        help="Output root for all sequence reconstructions",
    )
    parser.add_argument("--fps", type=float, default=120.0, help="Target FPS")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output folders")
    parser.add_argument("--max-seqs", type=int, default=None, help="Optional sequence limit")
    return parser.parse_args()


def pick_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: Path):
    parse_config_stub = types.ModuleType("parse_config")

    class ConfigParser:
        pass

    parse_config_stub.ConfigParser = ConfigParser
    sys.modules["parse_config"] = parse_config_stub

    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def build_model(hypere2vid_root: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(hypere2vid_root))
    from model.model import E2VIDRecurrent

    ckpt = load_checkpoint(checkpoint_path)
    cfg_obj = ckpt.get("config")
    if cfg_obj is None or not hasattr(cfg_obj, "_config"):
        raise RuntimeError("Checkpoint does not contain expected config object")

    unet_kwargs = cfg_obj._config["arch"]["args"]["unet_kwargs"]
    if unet_kwargs.get("norm") == "none":
        unet_kwargs = dict(unet_kwargs)
        unet_kwargs["norm"] = None

    model = E2VIDRecurrent(unet_kwargs=unet_kwargs)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.reset_states()
    return model


def events_to_voxel_grid(events: np.ndarray, num_bins: int, width: int, height: int) -> np.ndarray:
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if events.size == 0:
        return voxel

    t = events[:, 0].astype(np.float64)
    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    p = events[:, 3].astype(np.float32)
    p = np.where(p > 0, 1.0, -1.0)

    t0 = t[0]
    t1 = t[-1]
    if t1 <= t0:
        t_norm = np.zeros_like(t)
    else:
        t_norm = (num_bins - 1) * (t - t0) / (t1 - t0)

    tis = np.floor(t_norm).astype(np.int64)
    dts = t_norm - tis

    valid_xy = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    for b in range(num_bins):
        m0 = valid_xy & (tis == b)
        if np.any(m0):
            np.add.at(voxel[b], (y[m0], x[m0]), p[m0] * (1.0 - dts[m0]))

        m1 = valid_xy & (tis + 1 == b)
        if np.any(m1):
            np.add.at(voxel[b], (y[m1], x[m1]), p[m1] * dts[m1])

    nonzero = voxel != 0
    if np.any(nonzero):
        vals = voxel[nonzero]
        mean = vals.mean()
        std = vals.std()
        if std > 0:
            voxel[nonzero] = (vals - mean) / std

    return voxel


def iter_fixed_duration_windows(csv_path: Path, window_us: float):
    current_events: list[tuple[int, int, int, int]] = []
    t0_us: int | None = None
    current_end_us: float | None = None

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            x = int(row[0])
            y = int(row[1])
            pol = int(row[2])
            t_us = int(row[3])

            if t0_us is None:
                t0_us = t_us
                current_end_us = t0_us + window_us

            assert current_end_us is not None

            while t_us > current_end_us:
                if current_events:
                    yield np.array(current_events, dtype=np.int64)
                else:
                    yield np.empty((0, 4), dtype=np.int64)
                current_events = []
                current_end_us += window_us

            current_events.append((t_us, x, y, pol))

    if current_events:
        yield np.array(current_events, dtype=np.int64)


def clear_output_folder(folder: Path) -> None:
    if not folder.exists():
        return
    for p in folder.glob("*"):
        if p.is_file():
            p.unlink()


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    preprocessed_root = (project_root / args.preprocessed_root).resolve()
    hypere2vid_root = (project_root / args.hypere2vid_root).resolve()
    checkpoint = (project_root / args.checkpoint).resolve()
    output_root = (project_root / args.output_root).resolve()

    if not preprocessed_root.exists():
        print(f"ERROR: preprocessed root not found: {preprocessed_root}")
        return 1
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    seq_dirs = sorted(
        [p for p in preprocessed_root.glob("preprocessed_fred_*") if p.is_dir()],
        key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else p.name,
    )
    if args.max_seqs is not None:
        seq_dirs = seq_dirs[: args.max_seqs]

    if not seq_dirs:
        print("ERROR: no preprocessed_fred_* folders found")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model = build_model(hypere2vid_root, checkpoint, device)

    window_us = 1_000_000.0 / args.fps

    print("=" * 70)
    print("HYPERE2VID ALL-SEQUENCE RUNNER")
    print("=" * 70)
    print(f"Preprocessed root: {preprocessed_root}")
    print(f"Output root:       {output_root}")
    print(f"Checkpoint:        {checkpoint}")
    print(f"Device:            {device}")
    print(f"FPS:               {args.fps:.3f} (window {window_us:.6f} us)")
    print(f"Sequences:         {len(seq_dirs)}")

    failed: list[str] = []

    for seq_dir in seq_dirs:
        seq_id = seq_dir.name.rsplit("_", 1)[-1]
        input_csv = seq_dir / "eventRaw" / "events.txt"
        if not input_csv.exists():
            print(f"[SKIP] seq {seq_id}: missing {input_csv}")
            failed.append(f"{seq_id} (missing events)")
            continue

        seq_out = output_root / f"seq{seq_id}_{int(args.fps)}fps"
        seq_out.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            clear_output_folder(seq_out)

        print("-" * 70)
        print(f"Sequence {seq_id}")

        model.reset_states()
        frame_count = 0

        timestamps_path = seq_out / "timestamps.txt"
        with timestamps_path.open("w", encoding="utf-8") as tsf:
            for ev in iter_fixed_duration_windows(input_csv, window_us):
                voxel = events_to_voxel_grid(ev, args.num_bins, args.width, args.height)
                event_tensor = torch.from_numpy(voxel).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(event_tensor)["image"]

                img = out[0, 0].detach().cpu().numpy()
                img = np.clip(img, 0.0, 1.0)
                img_u8 = (img * 255.0).astype(np.uint8)

                frame_count += 1
                frame_path = seq_out / f"frame_{frame_count:010d}.png"
                cv2.imwrite(str(frame_path), img_u8)

                if ev.size > 0:
                    tsf.write(f"{ev[-1,0] / 1_000_000.0:.6f}\n")
                else:
                    tsf.write("0.000000\n")

        print(f"[OK] seq {seq_id}: {frame_count} frames -> {seq_out}")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total sequences attempted: {len(seq_dirs)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for item in failed:
            print(f"  - {item}")
        return 1

    print("All reconstructions completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
