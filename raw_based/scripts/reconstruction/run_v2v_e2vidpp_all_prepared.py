#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import run_v2v_e2vidpp_seq_test as single_run


DEFAULT_SECONDS_PER_100_FRAMES = {
    "cpu": 102.19,
    "cuda": 10.29,
    "auto": 10.29,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V2V e2vid++ on all prepared validation sequences."
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequences", nargs="*", type=int, help="Optional subset of sequence ids to run.")
    parser.add_argument(
        "--isolate-process",
        action="store_true",
        default=True,
        help="Run each sequence in a fresh Python subprocess (recommended for CUDA memory stability).",
    )
    parser.add_argument(
        "--seconds-per-100-frames",
        type=float,
        default=None,
        help="Measured wall time for 100 frames, used only for ETA reporting.",
    )
    return parser.parse_args()


def count_usable_frames(manifest_path: Path) -> int:
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        count = sum(1 for row in reader if row.get("split") == "val")
    return max(count - 1, 0)


def run_sequence_subprocess(repo_root: Path, sequence_id: int, frame_count: int, device: str) -> None:
    seq_runner = repo_root / "scripts" / "tools" / "run_v2v_e2vidpp_seq_test.py"
    command = [
        sys.executable,
        str(seq_runner),
        "--sequence",
        str(sequence_id),
        "--frames",
        str(frame_count),
        "--device",
        device,
    ]

    env = os.environ.copy()
    if device == "cuda":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    subprocess.run(command, cwd=repo_root, check=True, env=env)


def discover_sequences(repo_root: Path, requested_sequences: list[int] | None) -> list[tuple[int, Path, int]]:
    root = repo_root / "data" / "preprocessed_all_val"
    discovered: list[tuple[int, Path, int]] = []

    for seq_dir in sorted(root.glob("preprocessed_fred_*"), key=lambda path: int(path.name.rsplit("_", 1)[-1])):
        sequence_id = int(seq_dir.name.rsplit("_", 1)[-1])
        if requested_sequences and sequence_id not in requested_sequences:
            continue

        manifest_path = seq_dir / "paired" / "manifest_val.csv"
        usable_frames = count_usable_frames(manifest_path)
        if usable_frames <= 0:
            continue
        discovered.append((sequence_id, seq_dir, usable_frames))

    return discovered


def format_seconds(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def main() -> int:
    args = parse_args()
    repo_root = single_run.find_repo_root()
    v2v_root = repo_root / "external" / "V2V"
    checkpoint_path = single_run.ensure_checkpoint(v2v_root)
    seconds_per_100_frames = args.seconds_per_100_frames
    if seconds_per_100_frames is None:
        seconds_per_100_frames = DEFAULT_SECONDS_PER_100_FRAMES[args.device]

    sequences = discover_sequences(repo_root, args.sequences)
    if not sequences:
        raise SystemExit("No prepared sequences matched the requested selection.")

    total_frames = sum(frame_count for _, _, frame_count in sequences)
    estimated_seconds = total_frames * seconds_per_100_frames / 100.0

    print(f"Prepared sequences: {len(sequences)}")
    print(f"Total frames: {total_frames}")
    print(f"Estimated runtime on {args.device}: {format_seconds(estimated_seconds)}")
    print(f"Measured baseline: {seconds_per_100_frames:.2f}s per 100 frames")

    results: list[dict[str, object]] = []
    failures = 0
    started_at = time.time()

    for index, (sequence_id, sequence_dir, frame_count) in enumerate(sequences, start=1):
        sequence_name = f"seq{sequence_id}_{frame_count}frames"
        generated_dir = v2v_root / "generated_tests" / sequence_name
        output_dir = generated_dir / "results" / "EVBIRD" / sequence_name

        if generated_dir.exists() and args.overwrite:
            shutil.rmtree(generated_dir)

        if output_dir.exists() and not args.overwrite:
            output_count = len(list(output_dir.glob("*.png")))
            if output_count == frame_count:
                print(f"[{index}/{len(sequences)}] Skipping seq{sequence_id}: found {output_count} existing frames")
                results.append(
                    {
                        "sequence": sequence_id,
                        "frames": frame_count,
                        "status": "skipped",
                        "output_dir": str(output_dir),
                        "output_frames": output_count,
                    }
                )
                continue

        print(f"[{index}/{len(sequences)}] Running seq{sequence_id} for {frame_count} frames")
        seq_started_at = time.time()
        try:
            if args.isolate_process:
                run_sequence_subprocess(repo_root, sequence_id, frame_count, args.device)
            else:
                h5_path, h5_stem = single_run.build_h5(sequence_dir, generated_dir / f"{sequence_name}.h5", frame_count)
                config_path = single_run.write_test_inputs(v2v_root, h5_path, h5_stem, frame_count)
                single_run.run_test(v2v_root, checkpoint_path, config_path, args.device)

            output_dir = generated_dir / "results" / "EVBIRD" / sequence_name
            output_count = len(list(output_dir.glob("*.png")))
            elapsed_seconds = time.time() - seq_started_at
            print(
                f"[{index}/{len(sequences)}] Finished seq{sequence_id}: {output_count} frames in {format_seconds(elapsed_seconds)}"
            )
            results.append(
                {
                    "sequence": sequence_id,
                    "frames": frame_count,
                    "status": "completed",
                    "output_dir": str(output_dir),
                    "output_frames": output_count,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
        except Exception as exc:
            failures += 1
            elapsed_seconds = time.time() - seq_started_at
            print(f"[{index}/{len(sequences)}] Failed seq{sequence_id} after {format_seconds(elapsed_seconds)}: {exc}")
            results.append(
                {
                    "sequence": sequence_id,
                    "frames": frame_count,
                    "status": "failed",
                    "elapsed_seconds": elapsed_seconds,
                    "error": str(exc),
                }
            )

    summary = {
        "device": args.device,
        "total_sequences": len(sequences),
        "total_frames": total_frames,
        "estimated_seconds": estimated_seconds,
        "elapsed_seconds": time.time() - started_at,
        "failures": failures,
        "results": results,
    }
    summary_path = v2v_root / "generated_tests" / "all_prepared_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Summary: {summary_path}")
    print(f"Completed with {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
