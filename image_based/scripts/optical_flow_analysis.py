#!/usr/bin/env python3
"""
Optical flow analysis on raw RGB sequences.
Computes dense optical flow using Farneback algorithm for motion visualization.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def compute_optical_flow(
    rgb_dir: Path,
    output_dir: Path,
    flow_type: str = "farneback",
    magnitude_only: bool = False,
) -> dict:
    """
    Compute optical flow between consecutive frames.
    
    Args:
        rgb_dir: Directory containing raw RGB frames
        output_dir: Where to save optical flow visualizations
        flow_type: 'farneback' (dense) or 'lk' (Lucas-Kanade sparse)
        magnitude_only: If True, save only magnitude; if False, save magnitude and angle
    
    Returns:
        Dictionary with processing statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load frame sequence
    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        raise ValueError(f"No RGB frames found in {rgb_dir}")
    
    print(f"Processing {len(rgb_files)} frames from {rgb_dir.name}")
    
    stats = {
        "total_frames": len(rgb_files),
        "flow_images_saved": 0,
        "mean_flow_magnitudes": [],
        "max_flow_magnitudes": [],
    }
    
    prev_gray = None
    
    for i, frame_path in enumerate(tqdm(rgb_files, desc="Computing optical flow")):
        # Load current frame
        current_frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        
        if current_frame is None:
            print(f"  Warning: Could not load {frame_path}")
            continue
        
        # Convert to grayscale for flow computation
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # Compute dense optical flow
            if flow_type == "farneback":
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    current_gray,
                    None,    # flow output
                    0.5,     # pyr_scale
                    3,       # levels
                    15,      # winsize
                    3,       # iterations
                    5,       # poly_n
                    1.2,     # poly_sigma
                    cv2.OPTFLOW_FARNEBACK_GAUSSIAN,  # flags
                )
            else:
                raise ValueError(f"Unknown flow type: {flow_type}")
            
            # Compute magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Normalize magnitude
            magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            # Store statistics
            stats["mean_flow_magnitudes"].append(float(np.mean(magnitude)))
            stats["max_flow_magnitudes"].append(float(np.max(magnitude)))
            
            # Create visualization
            if magnitude_only:
                # Just magnitude as grayscale
                flow_vis = magnitude_normalized.astype(np.uint8)
                flow_vis = cv2.cvtColor(flow_vis, cv2.COLOR_GRAY2BGR)
            else:
                # HSV visualization: Hue=angle, Saturation=1, Value=magnitude
                hsv = np.zeros_like(current_frame, dtype=np.uint8)
                hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue: angle
                hsv[..., 1] = 255  # Saturation: full
                hsv[..., 2] = magnitude_normalized.astype(np.uint8)  # Value: magnitude
                flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Save optical flow image
            output_filename = f"optical_flow_{i:06d}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), flow_vis)
            stats["flow_images_saved"] += 1
        
        # Current becomes previous for next iteration
        prev_gray = current_gray.copy()
    
    # Compute aggregate statistics
    if stats["mean_flow_magnitudes"]:
        stats["avg_mean_flow_magnitude"] = float(np.mean(stats["mean_flow_magnitudes"]))
        stats["avg_max_flow_magnitude"] = float(np.mean(stats["max_flow_magnitudes"]))
    else:
        stats["avg_mean_flow_magnitude"] = 0.0
        stats["avg_max_flow_magnitude"] = 0.0
    
    return stats


def generate_video_from_flow(
    flow_dir: Path,
    output_video: Path,
    fps: int = 20,
) -> None:
    """Generate MP4 video from optical flow images."""
    flow_images = sorted(flow_dir.glob("optical_flow_*.jpg"))
    
    if not flow_images:
        print(f"  No optical flow images found in {flow_dir}")
        return
    
    # Get frame dimensions
    first_frame = cv2.imread(str(flow_images[0]))
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    print(f"Generating video: {output_video.name} ({len(flow_images)} frames at {fps} FPS)")
    
    for flow_image_path in tqdm(flow_images, desc="Writing video frames"):
        frame = cv2.imread(str(flow_image_path))
        if frame is not None:
            writer.write(frame)
    
    writer.release()
    print(f"  Video saved to {output_video}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute optical flow from raw RGB sequences",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing raw data (default: data/raw)",
    )
    parser.add_argument(
        "--dataset",
        type=int,
        default=10,
        help="Dataset ID to process (default: 10)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/optical_flow"),
        help="Output directory for flow images (default: artifacts/optical_flow)",
    )
    parser.add_argument(
        "--flow-type",
        choices=["farneback"],
        default="farneback",
        help="Optical flow algorithm (farneback = dense flow)",
    )
    parser.add_argument(
        "--magnitude-only",
        action="store_true",
        help="Save only flow magnitude (grayscale); default is HSV color encoding",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate MP4 video from flow images",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
        help="FPS for output video (default: 20)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional variant name to isolate outputs (e.g. farneback_hsv)",
    )
    
    args = parser.parse_args()
    
    # Construct paths
    rgb_dir = args.data_root / str(args.dataset) / "RGB"
    if args.variant:
        variant_name = args.variant
    else:
        mode_tag = "mag" if args.magnitude_only else "hsv"
        variant_name = f"{args.flow_type}_{mode_tag}"
    dataset_out_dir = args.out_dir / f"dataset_{args.dataset}_{variant_name}"
    
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    print(f"Configuration:")
    print(f"  Data root: {args.data_root}")
    print(f"  Dataset: {args.dataset}")
    print(f"  RGB input: {rgb_dir}")
    print(f"  Output: {dataset_out_dir}")
    print(f"  Variant: {variant_name}")
    print(f"  Flow type: {args.flow_type}")
    print(f"  Magnitude only: {args.magnitude_only}")
    print()
    
    # Compute optical flow
    stats = compute_optical_flow(
        rgb_dir=rgb_dir,
        output_dir=dataset_out_dir,
        flow_type=args.flow_type,
        magnitude_only=args.magnitude_only,
    )
    
    # Print statistics
    print()
    print("Statistics:")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Optical flow images saved: {stats['flow_images_saved']}")
    if stats["avg_mean_flow_magnitude"] > 0:
        print(f"  Average mean flow magnitude: {stats['avg_mean_flow_magnitude']:.2f}")
        print(f"  Average max flow magnitude: {stats['avg_max_flow_magnitude']:.2f}")
    
    # Save statistics
    stats_path = dataset_out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_path.name}")
    
    # Generate video if requested
    if args.video:
        video_path = args.out_dir / f"optical_flow_dataset_{args.dataset}_{variant_name}.mp4"
        generate_video_from_flow(
            flow_dir=dataset_out_dir,
            output_video=video_path,
            fps=args.video_fps,
        )
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
