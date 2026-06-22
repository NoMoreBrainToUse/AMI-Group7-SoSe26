#!/usr/bin/env python3
"""
Render a fusion result video by piping raw frames directly into ffmpeg.

Reads the web JSONL produced by compute_fusion_metrics.py (one entry per frame),
draws GT boxes (green), kept detections (cyan), and rejected detections (red),
and writes the result directly to an mp4 via an ffmpeg stdin pipe — no
intermediate video file is created.

Box colours
-----------
  Green  (40, 220,  40) — ground-truth box
  Cyan   (20, 200, 255) — kept detection (fusion_score >= threshold)
  Red    ( 0,   0, 220) — rejected detection

Codec options
-------------
  --codec libx264      CPU encoding, default, Docker-compatible
  --codec h264_nvenc   NVIDIA hardware encoding; falls back to libx264 if
                       unavailable (checked automatically before the pipe opens)

Example
-------
  # CPU encoding (default)
  python src/visualization/render_fusion_video.py \\
    --detections outputs/web/fusion_detections_blind_test_v4.jsonl \\
    --output outputs/videos/blind_test_v4_fusion_result_compressed.mp4

  # NVIDIA hardware encoding with automatic CPU fallback
  python src/visualization/render_fusion_video.py \\
    --detections outputs/web/fusion_detections_blind_test_v4.jsonl \\
    --output outputs/videos/blind_test_v4_fusion_result_compressed.mp4 \\
    --codec h264_nvenc
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


GT_COLOR   = (40, 220,  40)
KEPT_COLOR = (20, 200, 255)
REJ_COLOR  = ( 0,   0, 220)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_ffmpeg() -> str:
    """Return the ffmpeg binary path or raise RuntimeError."""
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError(
            "ffmpeg not found on PATH.\n"
            "  Linux:  apt-get install ffmpeg\n"
            "  macOS:  brew install ffmpeg\n"
            "  Docker: add 'RUN apt-get install -y ffmpeg' to your Dockerfile"
        )
    return binary


def codec_available(ffmpeg: str, codec: str) -> bool:
    """Return True if ffmpeg can encode with this codec."""
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner",
             "-f", "lavfi", "-i", "nullsrc=s=64x64",
             "-t", "0.1", "-c:v", codec, "-f", "null", "-"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def norm_to_px(
    cx: float, cy: float, bw: float, bh: float, W: int, H: int
) -> tuple[int, int, int, int]:
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def draw_boxes(frame, detections: list[dict], gt_boxes_norm: list[list[float]]) -> None:
    import cv2

    H, W = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        color = KEPT_COLOR if det["kept"] else REJ_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if det["kept"]:
            cv2.putText(
                frame,
                f"DRONE {det['fusion_score']:.2f}",
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEPT_COLOR, 1, cv2.LINE_AA,
            )

    for box in gt_boxes_norm:
        cx, cy, bw, bh = box
        x1, y1, x2, y2 = norm_to_px(cx, cy, bw, bh, W, H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GT_COLOR, 2)


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="Render fusion result video via ffmpeg pipe (no intermediate file).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--detections",
        default=str(root / "outputs/web/fusion_detections_blind_test_v4.jsonl"),
        help="Web JSONL from compute_fusion_metrics.py (one entry per frame).",
    )
    p.add_argument(
        "--output",
        default=str(root / "outputs/videos/blind_test_v4_fusion_result_compressed.mp4"),
        help="Output video path.",
    )
    p.add_argument(
        "--codec",
        default="libx264",
        help="ffmpeg video codec. Use 'libx264' (CPU, default) or 'h264_nvenc' (NVIDIA GPU).",
    )
    p.add_argument("--fps",    type=float, default=30.0)
    p.add_argument("--crf",    type=int,   default=28, help="Quality (libx264 only; lower = better).")
    p.add_argument("--preset", default="veryfast",     help="Encoding preset (libx264 only).")
    p.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split order for the video.",
    )
    return p


def main() -> int:
    try:
        import cv2
    except ImportError as exc:
        print(f"ERROR: opencv-python not installed: {exc}", file=sys.stderr)
        return 1

    root = repo_root()
    args = build_parser().parse_args()

    det_path = resolve_path(args.detections, root)
    out_path = resolve_path(args.output, root)
    splits   = [s.strip() for s in args.splits.split(",")]

    if not det_path.is_file():
        print(f"ERROR: detections file not found: {det_path}", file=sys.stderr)
        print("  Run compute_fusion_metrics.py first.", file=sys.stderr)
        return 1

    ffmpeg = check_ffmpeg()

    # Codec selection with fallback
    codec = args.codec
    if codec == "h264_nvenc":
        if codec_available(ffmpeg, "h264_nvenc"):
            print("Using NVIDIA hardware encoding (h264_nvenc)")
        else:
            print("WARNING: h264_nvenc not available — falling back to libx264", file=sys.stderr)
            codec = "libx264"
    else:
        print(f"Using codec: {codec}")

    # Load JSONL and build stem index
    print(f"Loading detections from {det_path}")
    records = load_jsonl(det_path)
    stem_index: dict[str, dict] = {r["stem"]: r for r in records}
    print(f"  Frames in JSONL: {len(records)}")

    # Collect frames in split order from rgb_image paths in the JSONL
    SPLIT_ORDER = {s: i for i, s in enumerate(splits)}
    frames_by_split: dict[str, list[dict]] = {s: [] for s in splits}
    for rec in records:
        sp = rec.get("split", "")
        if sp in frames_by_split:
            frames_by_split[sp].append(rec)

    ordered_records: list[dict] = []
    for sp in splits:
        ordered_records.extend(
            sorted(frames_by_split[sp], key=lambda r: natural_key(r["stem"]))
        )

    if not ordered_records:
        print("ERROR: no frames to render", file=sys.stderr)
        return 1

    # Get W×H from first available frame
    first_rgb = root / ordered_records[0]["rgb_image"]
    first_frame = cv2.imread(str(first_rgb))
    if first_frame is None:
        print(f"ERROR: cannot read first frame: {first_rgb}", file=sys.stderr)
        return 1
    H, W = first_frame.shape[:2]
    print(f"Frame size: {W}x{H}  |  Total frames: {len(ordered_records)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command — write raw BGR24 to stdin
    ffmpeg_cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "bgr24",
        "-r", str(args.fps), "-i", "pipe:0",
        "-vcodec", codec,
        "-pix_fmt", "yuv420p",
    ]
    if codec == "libx264":
        ffmpeg_cmd += ["-crf", str(args.crf), "-preset", args.preset]
    ffmpeg_cmd.append(str(out_path))

    print(f"ffmpeg: {' '.join(ffmpeg_cmd)}")

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    written = 0
    try:
        for rec in ordered_records:
            rgb_abs = root / rec["rgb_image"]
            frame = cv2.imread(str(rgb_abs))
            if frame is None:
                print(f"  WARNING: cannot read {rgb_abs} — substituting blank frame")
                frame = __import__("numpy").zeros((H, W, 3), dtype=__import__("numpy").uint8)
            elif frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))

            draw_boxes(frame, rec.get("detections", []), rec.get("gt_boxes_norm", []))

            sequence = rec.get("sequence", rec["stem"].rsplit("_", 1)[0])
            cv2.putText(
                frame, sequence,
                (W - 90, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

            ffmpeg_proc.stdin.write(frame.tobytes())
            written += 1
            if written % 500 == 0 or written == len(ordered_records):
                print(f"  {written}/{len(ordered_records)} frames")

    except BrokenPipeError:
        stderr = ffmpeg_proc.stderr.read().decode(errors="replace")
        print(f"ERROR: ffmpeg pipe broke:\n{stderr}", file=sys.stderr)
        return 1
    finally:
        if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:
            ffmpeg_proc.stdin.close()

    ret = ffmpeg_proc.wait()
    if ret != 0:
        stderr = ffmpeg_proc.stderr.read().decode(errors="replace")
        print(f"ERROR: ffmpeg exited with code {ret}:\n{stderr}", file=sys.stderr)
        return 1

    size_mb = out_path.stat().st_size // (1024 * 1024)
    print(f"\nVideo: {out_path} ({size_mb} MB)")
    print(f"Frames written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
