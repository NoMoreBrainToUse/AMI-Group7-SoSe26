#!/usr/bin/env python3
"""
Runner script for all motion analysis variants on raw RGB sequences.
Generates: L1 differences, L2 differences, optical flow (magnitude), optical flow (HSV).
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        return result.returncode
    return 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run all motion analysis variants on dataset"
    )
    parser.add_argument(
        "--dataset",
        type=int,
        default=10,
        help="Dataset ID to process (default: 10)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate videos for all variants",
    )
    
    args = parser.parse_args()
    dataset = args.dataset
    video_flag = ["--video"] if args.video else []
    
    venv_python = Path("./.venv/bin/python").resolve()
    
    if not venv_python.exists():
        print(f"ERROR: Virtual environment not found at {venv_python}")
        sys.exit(1)
    
    print(f"\n{'*'*70}")
    print(f"  MOTION ANALYSIS VARIANTS - DATASET {dataset}")
    print(f"{'*'*70}")
    
    experiments = [
        {
            "name": "Frame Difference (L1 - Absolute)",
            "script": "scripts/frame_difference_analysis.py",
            "args": [
                f"--dataset", str(dataset),
                "--method", "l1",
                "--variant", "l1_gray_norm",
            ] + video_flag,
        },
        {
            "name": "Frame Difference (L2 - Euclidean)",
            "script": "scripts/frame_difference_analysis.py",
            "args": [
                f"--dataset", str(dataset),
                "--method", "l2",
                "--variant", "l2_gray_norm",
            ] + video_flag,
        },
        {
            "name": "Frame Difference with JET Colormap",
            "script": "scripts/frame_difference_analysis.py",
            "args": [
                f"--dataset", str(dataset),
                "--method", "l1",
                "--colormap",
                "--variant", "l1_jet_norm",
            ] + video_flag,
        },
        {
            "name": "Optical Flow (Magnitude - Grayscale)",
            "script": "scripts/optical_flow_analysis.py",
            "args": [
                f"--dataset", str(dataset),
                "--magnitude-only",
                "--variant", "farneback_mag",
            ] + video_flag,
        },
        {
            "name": "Optical Flow (HSV Color Encoding)",
            "script": "scripts/optical_flow_analysis.py",
            "args": [
                f"--dataset", str(dataset),
                "--variant", "farneback_hsv",
            ] + video_flag,
        },
    ]
    
    failed_experiments = []
    
    for exp in experiments:
        cmd = [str(venv_python), exp["script"]] + exp["args"]
        result = run_command(cmd, exp["name"])
        if result != 0:
            failed_experiments.append(exp["name"])
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(experiments) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
        return 1
    
    print("\nAll experiments completed successfully!")
    print("\nOutput locations:")
    print(f"  Frame differences: artifacts/frame_differences/dataset_{dataset}_l1_gray_norm/")
    print(f"  Frame differences: artifacts/frame_differences/dataset_{dataset}_l2_gray_norm/")
    print(f"  Frame differences: artifacts/frame_differences/dataset_{dataset}_l1_jet_norm/")
    print(f"  Optical flow:      artifacts/optical_flow/dataset_{dataset}_farneback_mag/")
    print(f"  Optical flow:      artifacts/optical_flow/dataset_{dataset}_farneback_hsv/")
    
    if args.video:
        print("\nVideos generated:")
        print(f"  Frame differences: artifacts/frame_differences/frame_differences_dataset_{dataset}_l1_gray_norm.mp4")
        print(f"  Frame differences: artifacts/frame_differences/frame_differences_dataset_{dataset}_l2_gray_norm.mp4")
        print(f"  Frame differences: artifacts/frame_differences/frame_differences_dataset_{dataset}_l1_jet_norm.mp4")
        print(f"  Optical flow:      artifacts/optical_flow/optical_flow_dataset_{dataset}_farneback_mag.mp4")
        print(f"  Optical flow:      artifacts/optical_flow/optical_flow_dataset_{dataset}_farneback_hsv.mp4")
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
