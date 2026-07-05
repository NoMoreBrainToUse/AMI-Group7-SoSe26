#!/usr/bin/env python3
"""Run E2VID reconstruction for all preprocessed FRED sequences at 120 FPS.

This script expects preprocessed folders such as:
  data/preprocessed_all_val/preprocessed_fred_<seq>/eventRaw/events.txt

For each sequence, it converts the CSV-style event file (x,y,pol,t_us) into the
text format expected by rpg_e2vid (header + t x y pol), then runs
`run_reconstruction.py` with fixed-duration windows at 120 FPS.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2VID for all preprocessed sequences at 120 FPS.")
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data/preprocessed_all_val"),
        help="Root containing preprocessed_fred_<seq> folders.",
    )
    parser.add_argument(
        "--rpg-e2vid-root",
        type=Path,
        default=Path("external/rpg_e2vid"),
        help="Path to rpg_e2vid repository root.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("pretrained/E2VID_lightweight.pth.tar"),
        help="Model path relative to rpg_e2vid root (or absolute path).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="Target reconstruction FPS (default: 120).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Sensor width for E2VID event header (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Sensor height for E2VID event header (default: 720).",
    )
    parser.add_argument(
        "--converted-root",
        type=Path,
        default=Path("data/e2vid_inputs_120fps"),
        help="Folder (relative to rpg_e2vid root) to store converted event files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_tests/all_120fps"),
        help="Folder (relative to rpg_e2vid root) for reconstructions.",
    )
    parser.add_argument(
        "--compute-voxel-grid-on-cpu",
        action="store_true",
        help="Pass --compute_voxel_grid_on_cpu to E2VID.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing output folder for each sequence before running.",
    )
    parser.add_argument(
        "--force-reconvert",
        action="store_true",
        help="Recreate converted E2VID input files even if they already exist.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Optional limit for number of sequences to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing reconstruction.",
    )
    return parser.parse_args()


def discover_sequence_dirs(preprocessed_root: Path) -> list[Path]:
    seq_dirs = sorted(
        [p for p in preprocessed_root.glob("preprocessed_fred_*") if p.is_dir()],
        key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else p.name,
    )
    return seq_dirs


def convert_events_csv_to_e2vid_txt(input_csv: Path, output_txt: Path, width: int, height: int) -> int:
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    event_count = 0
    t0_us: int | None = None

    with input_csv.open("r", encoding="utf-8", errors="ignore") as src, output_txt.open("w", encoding="utf-8") as dst:
        dst.write(f"{width} {height}\n")
        for line in src:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue

            x = int(parts[0])
            y = int(parts[1])
            pol = int(parts[2])
            t_us = int(parts[3])

            if t0_us is None:
                t0_us = t_us
            t_rel_s = (t_us - t0_us) / 1_000_000.0
            dst.write(f"{t_rel_s:.6f} {x} {y} {pol}\n")
            event_count += 1

    return event_count


def run_command(cmd: list[str], cwd: Path) -> int:
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    preprocessed_root = (project_root / args.preprocessed_root).resolve()
    rpg_root = (project_root / args.rpg_e2vid_root).resolve()

    if not preprocessed_root.is_dir():
        print(f"ERROR: preprocessed root not found: {preprocessed_root}")
        return 1
    if not rpg_root.is_dir():
        print(f"ERROR: rpg_e2vid root not found: {rpg_root}")
        return 1

    python_exe = rpg_root / ".venv" / "bin" / "python"
    if not python_exe.exists():
        print(f"ERROR: E2VID venv python not found: {python_exe}")
        return 1

    model_path = args.model_path if args.model_path.is_absolute() else (rpg_root / args.model_path)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        return 1

    converted_root = args.converted_root if args.converted_root.is_absolute() else (rpg_root / args.converted_root)
    output_root = args.output_root if args.output_root.is_absolute() else (rpg_root / args.output_root)
    converted_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = discover_sequence_dirs(preprocessed_root)
    if args.max_seqs is not None:
        seq_dirs = seq_dirs[: args.max_seqs]

    if not seq_dirs:
        print(f"ERROR: no preprocessed_fred_* folders found under {preprocessed_root}")
        return 1

    window_ms = 1000.0 / args.fps

    print("=" * 70)
    print("E2VID ALL-SEQUENCE RUNNER")
    print("=" * 70)
    print(f"Preprocessed root: {preprocessed_root}")
    print(f"rpg_e2vid root:    {rpg_root}")
    print(f"Model:             {model_path}")
    print(f"FPS:               {args.fps:.3f} (window {window_ms:.6f} ms)")
    print(f"Sequences:         {len(seq_dirs)}")

    failures: list[str] = []

    for seq_dir in seq_dirs:
        seq_id = seq_dir.name.rsplit("_", 1)[-1]
        input_csv = seq_dir / "eventRaw" / "events.txt"
        if not input_csv.exists():
            print(f"[SKIP] seq {seq_id}: missing {input_csv}")
            failures.append(f"{seq_id} (missing events.txt)")
            continue

        converted_input = converted_root / f"seq{seq_id}_e2vid.txt"
        dataset_name = f"seq{seq_id}_120fps"
        dataset_out = output_root / dataset_name

        print("-" * 70)
        print(f"Sequence {seq_id}")

        if args.force_reconvert or not converted_input.exists():
            event_count = convert_events_csv_to_e2vid_txt(
                input_csv=input_csv,
                output_txt=converted_input,
                width=args.width,
                height=args.height,
            )
            print(f"Converted events: {event_count} -> {converted_input}")
        else:
            print(f"Using existing converted input: {converted_input}")

        if args.overwrite_output and dataset_out.exists():
            shutil.rmtree(dataset_out)

        cmd = [
            str(python_exe),
            "run_reconstruction.py",
            "-c",
            str(model_path),
            "-i",
            str(converted_input),
            "--fixed_duration",
            "-T",
            f"{window_ms:.6f}",
            "--output_folder",
            str(output_root),
            "--dataset_name",
            dataset_name,
        ]

        if args.compute_voxel_grid_on_cpu:
            cmd.append("--compute_voxel_grid_on_cpu")

        if args.dry_run:
            print("[DRY RUN]", " ".join(cmd))
            continue

        rc = run_command(cmd, cwd=rpg_root)
        if rc != 0:
            failures.append(f"{seq_id} (exit {rc})")
            print(f"[FAIL] seq {seq_id}")
            continue

        frame_count = len(list(dataset_out.glob("frame_*.png")))
        print(f"[OK] seq {seq_id}: {frame_count} frames -> {dataset_out}")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total sequences attempted: {len(seq_dirs)}")
    print(f"Failed: {len(failures)}")

    if failures:
        for fail in failures:
            print(f"  - {fail}")
        return 1

    print("All reconstructions completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
