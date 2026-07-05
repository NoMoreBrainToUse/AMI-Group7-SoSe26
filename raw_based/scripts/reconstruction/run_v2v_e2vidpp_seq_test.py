#!/usr/bin/env python3

import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


CHECKPOINT_FILE_ID = "1Tf2CxR4gR3xYCTRvzIpNjFOaKw0H1u2r"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small seq test set and run V2V e2vid++ inference."
    )
    parser.add_argument("--sequence", type=int, default=18)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest_rows(sequence_dir: Path, frame_count: int) -> list[dict[str, str]]:
    manifest_path = sequence_dir / "paired" / "manifest_val.csv"
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] == "val":
                rows.append(row)

    rows.sort(key=lambda row: float(row["event_time_s"]))
    required_rows = frame_count + 1
    if len(rows) < required_rows:
        raise ValueError(
            f"Need at least {required_rows} validation frames, found {len(rows)} in {manifest_path}."
        )
    return rows[:required_rows]


def find_event_indices(events_path: Path, target_times_us: np.ndarray, chunk_size: int = 2_000_000) -> np.ndarray:
    indices = np.empty(len(target_times_us), dtype=np.int64)
    target_idx = 0
    offset = 0

    for chunk in pd.read_csv(
        events_path,
        header=None,
        usecols=[3],
        names=["t"],
        dtype=np.int64,
        chunksize=chunk_size,
    ):
        ts = chunk["t"].to_numpy(copy=False)
        if ts.size == 0:
            continue

        while target_idx < len(target_times_us) and target_times_us[target_idx] <= ts[-1]:
            local_idx = int(np.searchsorted(ts, target_times_us[target_idx], side="left"))
            indices[target_idx] = offset + local_idx
            target_idx += 1

        offset += ts.size
        if target_idx == len(target_times_us):
            break

    if target_idx != len(target_times_us):
        raise ValueError(
            f"Only matched {target_idx} of {len(target_times_us)} frame timestamps in {events_path}."
        )

    return indices


def build_h5(sequence_dir: Path, output_h5: Path, frame_count: int) -> tuple[Path, str]:
    rows = load_manifest_rows(sequence_dir, frame_count)
    frame_times_us = np.array([round(float(row["event_time_s"]) * 1_000_000) for row in rows], dtype=np.int64)

    events_path = sequence_dir / "eventRaw" / "events.txt"
    event_indices = find_event_indices(events_path, frame_times_us)
    start_idx = int(event_indices[0])
    end_idx = int(event_indices[-1])

    event_frame = pd.read_csv(
        events_path,
        header=None,
        names=["x", "y", "p", "t"],
        dtype={"x": np.int32, "y": np.int32, "p": np.int8, "t": np.int64},
        nrows=end_idx,
    )
    event_frame = event_frame.iloc[start_idx:end_idx].reset_index(drop=True)
    relative_event_indices = event_indices - start_idx

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.exists():
        output_h5.unlink()

    with h5py.File(output_h5, "w") as handle:
        handle.create_dataset("events/ts", data=event_frame["t"].to_numpy(dtype=np.float64) / 1_000_000.0, dtype=np.float64)
        handle.create_dataset("events/xs", data=event_frame["x"].to_numpy(dtype=np.uint16), dtype=np.uint16)
        handle.create_dataset("events/ys", data=event_frame["y"].to_numpy(dtype=np.uint16), dtype=np.uint16)
        handle.create_dataset("events/ps", data=event_frame["p"].to_numpy(dtype=np.uint8), dtype=np.uint8)

        for idx, row in enumerate(rows):
            image_path = Path(row["source_rgb"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to read image {image_path}")
            dataset = handle.create_dataset(f"images/{idx:06d}", data=image, dtype=np.uint8)
            dataset.attrs["event_idx"] = int(relative_event_indices[idx])

    return output_h5, output_h5.stem


def ensure_checkpoint(v2v_root: Path) -> Path:
    checkpoint_path = v2v_root / "checkpoints" / "e2vid++_original" / "reconstruction_model_state_dict.pth"
    if checkpoint_path.exists():
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    import gdown

    url = f"https://drive.google.com/uc?id={CHECKPOINT_FILE_ID}"
    gdown.download(url=url, output=str(checkpoint_path), quiet=False)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint download failed: {checkpoint_path}")
    return checkpoint_path


def write_test_inputs(v2v_root: Path, h5_path: Path, sequence_name: str, frame_count: int) -> Path:
    generated_dir = v2v_root / "generated_tests" / sequence_name
    generated_dir.mkdir(parents=True, exist_ok=True)

    data_file = generated_dir / "data.txt"
    data_file.write_text(f"{h5_path}\n", encoding="utf-8")

    with (v2v_root / "config" / "test_e2vid++_original.yaml").open() as handle:
        config = yaml.load(handle, Loader=yaml.Loader)

    chunk_length = min(frame_count, 50)
    max_samples = math.ceil(frame_count / chunk_length)

    config["module"]["loss"]["lpips_weight"] = 0
    config["module"]["loss"]["l1_weight"] = 0
    config["module"]["loss"]["l2_weight"] = 0
    config["module"]["loss"]["ssim_weight"] = 0
    config["module"]["loss"]["temporal_consistency_weight"] = 0

    config["experiment_name"] = f"e2vidpp_{sequence_name}"
    config["test_output_dir"] = str(generated_dir / "results")
    config["test_stage"] = {
        "test_num_workers": 0,
        "need_multi_255": True,
        "test": [
            {
                "data_file": str(data_file),
                "class_name": "data.testh5.TestH5Dataset",
                "dataset_name": "evbird",
                "num_bins": 5,
                "sequence_length": chunk_length,
                "interpolate_bins": True,
                "max_samples": max_samples,
            }
        ],
    }

    config_path = generated_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def run_inference_only(model: torch.nn.Module, dataloader, device: torch.device, config: dict, use_fp16: bool) -> None:
    output_dir = Path(config["test_output_dir"])
    model.eval()
    previous_sequence_name = None
    output_img_idx = 0
    seq_output_dir: Path | None = None
    normalize_voxels = config["module"].get("normalize_voxels", False)
    pad_size = 16

    with torch.inference_mode():
        for batch in dataloader:
            sequence_name = batch["sequence_name"][0][0]
            if previous_sequence_name != sequence_name:
                model.reset_states()
                output_img_idx = 0
                seq_output_dir = output_dir / "EVBIRD" / sequence_name
                seq_output_dir.mkdir(parents=True, exist_ok=True)

            events = batch["events"]
            if normalize_voxels:
                from model.train_utils import normalize_batch_voxel

                events = normalize_batch_voxel(events)

            _, time_steps, channels, height, width = events.shape
            padded_h = int(np.ceil(height / pad_size) * pad_size)
            padded_w = int(np.ceil(width / pad_size) * pad_size)

            assert seq_output_dir is not None
            for t in range(time_steps):
                event_dtype = torch.float16 if (device.type == "cuda" and use_fp16) else torch.float32
                event_tensor = events[:, t].to(device=device, dtype=event_dtype)
                if padded_h != height or padded_w != width:
                    event_tensor = F.pad(event_tensor, (0, padded_w - width, 0, padded_h - height))

                if device.type == "cuda" and use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_dict = model(event_tensor)
                else:
                    pred_dict = model(event_tensor)

                pred_frame = pred_dict["image"][:, :, :height, :width]
                if config["test_stage"].get("need_multi_255", True):
                    pred_frame = pred_frame * 255
                pred_frame = torch.clamp(pred_frame, 0, 255)

                img = pred_frame[0].detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0)).squeeze()
                img = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(str(seq_output_dir / f"{output_img_idx:06d}.png"), img)
                output_img_idx += 1

            previous_sequence_name = sequence_name


def run_test(v2v_root: Path, checkpoint_path: Path, config_path: Path, device_name: str, use_fp16: bool) -> Path:
    if device_name == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sys.path.insert(0, str(v2v_root))
    from utils.util import instantiate_from_config
    from test_e2vid import create_test_dataloader
    from train import convert_to_compiled

    with config_path.open() as handle:
        config = yaml.load(handle, Loader=yaml.Loader)

    device = torch.device(device_name)
    model = instantiate_from_config(config["module"]["model"]).to(device)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = saved["state_dict"]
    new_state_dict = convert_to_compiled(state_dict=state_dict, local_rank=None, use_compile=False)
    model.load_state_dict(new_state_dict, strict=False)
    if device.type == "cuda" and use_fp16:
        model = model.half()
    print(f"Loaded checkpoint on {device_name} ({'fp16' if use_fp16 and device.type == 'cuda' else 'fp32'}): {checkpoint_path}")
    model.to(device)
    test_dataloader = create_test_dataloader(config["test_stage"])

    original_cwd = Path.cwd()
    try:
        os.chdir(v2v_root)
        run_inference_only(model, test_dataloader, device, config, use_fp16)
    finally:
        os.chdir(original_cwd)

    output_dir = Path(config["test_output_dir"]) / "EVBIRD" / Path(config_path).parent.name
    return output_dir


def resolve_fp16_mode(precision_arg: str, device_name: str) -> bool:
    if device_name != "cuda":
        return False
    if precision_arg == "fp16":
        return True
    if precision_arg == "fp32":
        return False
    return True


def resolve_devices(device_arg: str) -> list[str]:
    if device_arg == "cpu":
        return ["cpu"]
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return ["cuda"]
    if torch.cuda.is_available():
        return ["cuda", "cpu"]
    return ["cpu"]


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root()
    v2v_root = repo_root / "external" / "V2V"
    sequence_dir = repo_root / "data" / "preprocessed_all_val" / f"preprocessed_fred_{args.sequence}"
    sequence_name = f"seq{args.sequence}_{args.frames}frames"
    generated_dir = v2v_root / "generated_tests" / sequence_name

    if generated_dir.exists():
        shutil.rmtree(generated_dir)

    h5_path, h5_stem = build_h5(sequence_dir, generated_dir / f"{sequence_name}.h5", args.frames)
    checkpoint_path = ensure_checkpoint(v2v_root)
    config_path = write_test_inputs(v2v_root, h5_path, h5_stem, args.frames)
    last_error: Exception | None = None
    output_dir: Path | None = None
    used_device: str | None = None
    for device_name in resolve_devices(args.device):
        use_fp16 = resolve_fp16_mode(args.precision, device_name)
        try:
            output_dir = run_test(v2v_root, checkpoint_path, config_path, device_name, use_fp16)
            used_device = device_name
            break
        except torch.OutOfMemoryError as exc:
            last_error = exc
            print(f"Device {device_name} failed with CUDA OOM; trying next option if available.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if output_dir is None or used_device is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Test run did not produce an output directory.")

    output_count = len(list(output_dir.glob("*.png")))
    print(f"Device used: {used_device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"H5: {h5_path}")
    print(f"Config: {config_path}")
    print(f"Output dir: {output_dir}")
    print(f"Output frames: {output_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())