#!/usr/bin/env python3
"""
Frame difference analysis on raw RGB sequences.
Computes temporal differences between consecutive frames to visualize motion.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_frame_sequence(rgb_dir: Path) -> list:
    """
    Load and sort RGB frames from dataset directory.
    Frames are sorted by their timestamp embedded in the filename.
    """
    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        raise ValueError(f"No RGB frames found in {rgb_dir}")
    return rgb_files


def compute_frame_differences(
    rgb_dir: Path,
    output_dir: Path,
    method: str = "l1",
    normalize: bool = True,
    apply_colormap: bool = False,
) -> dict:
    """
    Compute frame-by-frame differences and save as images.
    
    Args:
        rgb_dir: Directory containing raw RGB frames
        output_dir: Where to save difference images
        method: 'l1' (absolute difference) or 'l2' (squared difference)
        normalize: Whether to normalize to 0-255 range
        apply_colormap: Whether to apply a colormap (jet) for visualization
    
    Returns:
        Dictionary with processing statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load frame sequence
    frame_paths = load_frame_sequence(rgb_dir)
    
    print(f"Processing {len(frame_paths)} frames from {rgb_dir.name}")
    
    stats = {
        "total_frames": len(frame_paths),
        "difference_images_saved": 0,
        "mean_differences": [],
        "max_differences": [],
    }
    
    prev_frame = None
    prev_path = None
    
    for i, frame_path in enumerate(tqdm(frame_paths, desc="Computing differences")):
        # Load current frame
        current_frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        
        if current_frame is None:
            print(f"  Warning: Could not load {frame_path}")
            continue
        
        # Convert to float for computation
        current_frame_f = current_frame.astype(np.float32)
        
        if prev_frame is not None:
            prev_frame_f = prev_frame.astype(np.float32)
            
            # Compute difference
            if method == "l1":
                diff = np.abs(current_frame_f - prev_frame_f)
            elif method == "l2":
                diff = np.sqrt((current_frame_f - prev_frame_f) ** 2)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Compute statistics
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            stats["mean_differences"].append(float(mean_diff))
            stats["max_differences"].append(float(max_diff))
            
            # Normalize for visualization
            if normalize:
                # Normalize to 0-255 range based on max value in the difference image
                if max_diff > 0:
                    diff_normalized = (diff / max_diff * 255).astype(np.uint8)
                else:
                    diff_normalized = np.zeros_like(diff, dtype=np.uint8)
            else:
                diff_normalized = np.clip(diff, 0, 255).astype(np.uint8)
            
            # Average across channels to create grayscale difference map
            diff_gray = np.mean(diff_normalized, axis=2).astype(np.uint8)
            
            # Apply colormap if requested
            if apply_colormap:
                diff_vis = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            else:
                # Create 3-channel BGR image from grayscale
                diff_vis = cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR)
            
            # Save difference image with frame index
            output_filename = f"frame_diff_{i:06d}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), diff_vis)
            stats["difference_images_saved"] += 1
        
        # Current becomes previous for next iteration
        prev_frame = current_frame.copy()
        prev_path = frame_path
    
    # Compute aggregate statistics
    if stats["mean_differences"]:
        stats["avg_mean_difference"] = float(np.mean(stats["mean_differences"]))
        stats["avg_max_difference"] = float(np.mean(stats["max_differences"]))
    else:
        stats["avg_mean_difference"] = 0.0
        stats["avg_max_difference"] = 0.0
    
    return stats


def generate_video_from_differences(
    diff_dir: Path,
    output_video: Path,
    fps: int = 20,
) -> None:
    """
    Generate MP4 video from frame difference images.
    """
    diff_images = sorted(diff_dir.glob("frame_diff_*.jpg"))
    
    if not diff_images:
        print(f"  No difference images found in {diff_dir}")
        return
    
    # Get frame dimensions
    first_frame = cv2.imread(str(diff_images[0]))
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    print(f"Generating video: {output_video.name} ({len(diff_images)} frames at {fps} FPS)")
    
    for diff_image_path in tqdm(diff_images, desc="Writing video frames"):
        frame = cv2.imread(str(diff_image_path))
        if frame is not None:
            writer.write(frame)
    
    writer.release()
    print(f"  Video saved to {output_video}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute frame differences from raw RGB sequences",
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
        default=Path("artifacts/frame_differences"),
        help="Output directory for difference images (default: artifacts/frame_differences)",
    )
    parser.add_argument(
        "--method",
        choices=["l1", "l2"],
        default="l1",
        help="Difference computation method: l1 (absolute) or l2 (euclidean)",
    )
    parser.add_argument(
        "--colormap",
        action="store_true",
        help="Apply JET colormap to difference images",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable normalization (keep raw difference values)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate MP4 video from difference images",
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
        help="Optional variant name to isolate outputs (e.g. l1_gray)",
    )
    
    args = parser.parse_args()
    
    # Construct paths
    rgb_dir = args.data_root / str(args.dataset) / "RGB"
    if args.variant:
        variant_name = args.variant
    else:
        color_tag = "jet" if args.colormap else "gray"
        norm_tag = "raw" if args.no_normalize else "norm"
        variant_name = f"{args.method}_{color_tag}_{norm_tag}"
    dataset_out_dir = args.out_dir / f"dataset_{args.dataset}_{variant_name}"
    
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    print(f"Configuration:")
    print(f"  Data root: {args.data_root}")
    print(f"  Dataset: {args.dataset}")
    print(f"  RGB input: {rgb_dir}")
    print(f"  Output: {dataset_out_dir}")
    print(f"  Variant: {variant_name}")
    print(f"  Method: {args.method}")
    print(f"  Colormap: {args.colormap}")
    print(f"  Normalize: {not args.no_normalize}")
    print()
    
    # Compute frame differences
    stats = compute_frame_differences(
        rgb_dir=rgb_dir,
        output_dir=dataset_out_dir,
        method=args.method,
        normalize=not args.no_normalize,
        apply_colormap=args.colormap,
    )
    
    # Print statistics
    print()
    print("Statistics:")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Difference images saved: {stats['difference_images_saved']}")
    if stats["avg_mean_difference"] > 0:
        print(f"  Average mean difference: {stats['avg_mean_difference']:.2f}")
        print(f"  Average max difference: {stats['avg_max_difference']:.2f}")
    
    # Save statistics
    stats_path = dataset_out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_path.name}")
    
    # Generate video if requested
    if args.video:
        video_path = args.out_dir / f"frame_differences_dataset_{args.dataset}_{variant_name}.mp4"
        generate_video_from_differences(
            diff_dir=dataset_out_dir,
            output_video=video_path,
            fps=args.video_fps,
        )
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
