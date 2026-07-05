#!/usr/bin/env python3
"""Quick OpenCV denoise + deblur pass for triplet PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply OpenCV denoise/deblur to triplet images.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="One or more triplet directories containing *_triplet.png files.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_cv2_quick",
        help="Suffix appended to each input directory name for output folder.",
    )
    parser.add_argument("--h", type=float, default=8.0, help="Denoise strength for luminance.")
    parser.add_argument("--h-color", type=float, default=8.0, help="Denoise strength for color channels.")
    parser.add_argument("--template-window", type=int, default=7, help="Template window size for NLM denoise.")
    parser.add_argument("--search-window", type=int, default=21, help="Search window size for NLM denoise.")
    parser.add_argument("--blur-sigma", type=float, default=1.2, help="Gaussian sigma used for unsharp deblur.")
    parser.add_argument("--sharpen-amount", type=float, default=1.1, help="Unsharp mask amount.")
    return parser.parse_args()


def process_image(
    image,
    h: float,
    h_color: float,
    template_window: int,
    search_window: int,
    blur_sigma: float,
    sharpen_amount: float,
):
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=h,
        hColor=h_color,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )

    ksize = max(3, int(round(blur_sigma * 4)) | 1)
    blurred = cv2.GaussianBlur(denoised, (ksize, ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)
    deblurred = cv2.addWeighted(denoised, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
    return deblurred


def process_dir(input_dir: Path, output_suffix: str, args: argparse.Namespace) -> tuple[int, Path]:
    files = sorted(input_dir.glob("*_triplet.png"))
    out_dir = input_dir.parent / f"{input_dir.name}{output_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        out = process_image(
            image=image,
            h=args.h,
            h_color=args.h_color,
            template_window=args.template_window,
            search_window=args.search_window,
            blur_sigma=args.blur_sigma,
            sharpen_amount=args.sharpen_amount,
        )

        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), out)
        count += 1

    return count, out_dir


def main() -> int:
    args = parse_args()

    total = 0
    for raw in args.input_dirs:
        in_dir = Path(raw).resolve()
        if not in_dir.is_dir():
            print(f"Skip (not found): {in_dir}")
            continue

        count, out_dir = process_dir(in_dir, args.output_suffix, args)
        total += count
        print(f"Processed {count} images -> {out_dir}")

    print(f"Total processed images: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
