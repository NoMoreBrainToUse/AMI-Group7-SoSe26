#!/usr/bin/env python3
"""Run HyperE2VID reconstruction on preprocessed event CSV files.

This script reads events from `eventRaw/events.txt` produced by preprocessing
(x,y,pol,t_us), creates fixed-duration event windows, builds voxel grids, and
runs HyperE2VID to save reconstructed frames.
"""

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
    parser = argparse.ArgumentParser(description="Run HyperE2VID reconstruction")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/preprocessed_all_val/preprocessed_fred_18/eventRaw/events.txt"),
        help="Input CSV with columns x,y,pol,t_us",
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
        "--output-dir",
        type=Path,
        default=Path("external/HyperE2VID/output_tests/seq18_100fps_100img"),
        help="Directory for reconstructed frames",
    )
    parser.add_argument("--fps", type=float, default=100.0, help="Target output fps")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to generate")
    parser.add_argument("--width", type=int, default=1280, help="Sensor width")
    parser.add_argument("--height", type=int, default=720, help="Sensor height")
    parser.add_argument("--num-bins", type=int, default=5, help="Number of voxel grid bins")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device",
    )
    parser.add_argument("--overwrite", action="store_true", help="Clear output directory first")
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
    # The official checkpoint stores a custom ConfigParser object. Stub it so
    # torch can unpickle safely from this trusted source.
    parse_config_stub = types.ModuleType("parse_config")

    class ConfigParser:  # noqa: D401 - simple pickle stub
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
    # Repository code expects None for no norm in practice.
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
    """Convert events [N,4]=[t_us,x,y,pol] into a voxel grid [B,H,W]."""
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


def collect_windows(csv_path: Path, target_frames: int, window_us: float) -> list[np.ndarray]:
    windows: list[np.ndarray] = []

    current_events: list[tuple[int, int, int, int]] = []
    t0_us: int | None = None
    current_end_us: float | None = None

    target_collect = max(target_frames + 20, int(target_frames * 1.2))

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

            while t_us > current_end_us and len(windows) < target_collect:
                if current_events:
                    windows.append(np.array(current_events, dtype=np.int64))
                else:
                    windows.append(np.empty((0, 4), dtype=np.int64))
                current_events = []
                current_end_us += window_us

            current_events.append((t_us, x, y, pol))

            if len(windows) >= target_collect:
                break

    if len(windows) < target_collect:
        if current_events:
            windows.append(np.array(current_events, dtype=np.int64))
        while len(windows) < target_collect:
            windows.append(np.empty((0, 4), dtype=np.int64))

    return windows[:target_collect]


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_csv = (project_root / args.input_csv).resolve()
    hypere2vid_root = (project_root / args.hypere2vid_root).resolve()
    checkpoint = (project_root / args.checkpoint).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not input_csv.exists():
        print(f"ERROR: input file not found: {input_csv}")
        return 1
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    if args.overwrite and output_dir.exists():
        for p in output_dir.glob("*"):
            if p.is_file():
                p.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"Device: {device}")

    model = build_model(hypere2vid_root, checkpoint, device)

    window_us = 1_000_000.0 / args.fps
    windows = collect_windows(input_csv, args.num_frames, window_us)
    print(f"Collected windows: {len(windows)}")

    timestamps_path = output_dir / "timestamps.txt"
    with timestamps_path.open("w", encoding="utf-8") as tsf:
        written = 0
        for ev in windows:
            if written >= args.num_frames:
                break
            voxel = events_to_voxel_grid(ev, args.num_bins, args.width, args.height)
            event_tensor = torch.from_numpy(voxel).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(event_tensor)["image"]

            img = out[0, 0].detach().cpu().numpy()
            img = np.clip(img, 0.0, 1.0)
            img_u8 = (img * 255.0).astype(np.uint8)

            written += 1
            frame_name = output_dir / f"frame_{written:010d}.png"
            cv2.imwrite(str(frame_name), img_u8)

            if ev.size > 0:
                tsf.write(f"{ev[-1,0] / 1_000_000.0:.6f}\n")
            else:
                tsf.write("0.000000\n")

    print(f"Saved {written} frames to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
