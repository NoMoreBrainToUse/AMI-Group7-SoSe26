#!/usr/bin/env python3
"""
Visualize raw event windows stored under eventMatched for a preprocessed FRED sequence.

For each sample txt in:
  data/preprocessed/preprocessed_fred_<seq>/eventMatched/<split>/*.txt
this viewer renders polarity events (cyan=ON, orange=OFF) and shows them.
If the corresponding matched event PNG exists, it is displayed side-by-side.

Usage:
  python3 scripts/presentation/view_event_matched_raw.py --seq 1
  python3 scripts/presentation/view_event_matched_raw.py --seq 1 --split train --start 25

Controls:
  Right / D / N / Space  - next sample
  Left  / A / P          - previous sample
  Home                   - first sample
  End                    - last sample
  Q / Esc                - quit
"""

from __future__ import annotations

import argparse
import re
import tkinter as tk
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Missing dependency: Pillow. Install with: pip install Pillow")


COLOR_POS = (0, 200, 255)
COLOR_NEG = (255, 80, 0)
COLOR_BG = (30, 30, 30)


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View raw event windows from eventMatched txt files.")
    parser.add_argument("--data-root", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--seq", required=True, help="Sequence id, e.g. 1, 10, 110")
    parser.add_argument("--split", default="train", help="Split under eventMatched (default: train)")
    parser.add_argument("--start", type=int, default=0, help="Start sample index")
    parser.add_argument("--width", type=int, default=1280, help="Event sensor width")
    parser.add_argument("--height", type=int, default=720, help="Event sensor height")
    return parser.parse_args()


def render_events_txt(path: Path, width: int, height: int) -> tuple[Image.Image, int]:
    img = Image.new("RGB", (width, height), COLOR_BG)
    pixels = img.load()
    count = 0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue
            try:
                x, y, pol = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = COLOR_POS if pol > 0 else COLOR_NEG
                count += 1

    return img, count


def find_matched_png(preproc_dir: Path, split: str, stem: str) -> Path | None:
    candidate = preproc_dir / "matched" / "event" / "images" / split / f"{stem}.png"
    if candidate.is_file():
        return candidate
    return None


def fit_height(img: Image.Image, target_height: int) -> Image.Image:
    if img.height == target_height:
        return img
    ratio = target_height / img.height
    target_width = max(1, int(round(img.width * ratio)))
    return img.resize((target_width, target_height), Image.Resampling.BILINEAR)


class Viewer:
    def __init__(
        self,
        root: tk.Tk,
        txt_files: list[Path],
        preproc_dir: Path,
        split: str,
        width: int,
        height: int,
        start: int,
    ) -> None:
        self.root = root
        self.txt_files = txt_files
        self.preproc_dir = preproc_dir
        self.split = split
        self.width = width
        self.height = height
        self.index = max(0, min(start, len(txt_files) - 1))
        self.tk_image: Any = None

        root.title("EventMatched Raw Viewer")
        root.geometry("1800x840")
        root.configure(bg="#111")

        self.canvas_label = tk.Label(root, bg="#111")
        self.canvas_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        info_frame = tk.Frame(root, bg="#1c1c1c")
        info_frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.info_label = tk.Label(
            info_frame,
            text="",
            anchor="w",
            justify=tk.LEFT,
            bg="#1c1c1c",
            fg="#DDD",
            padx=10,
            pady=6,
        )
        self.info_label.pack(fill=tk.X)

        nav_frame = tk.Frame(root, bg="#111")
        nav_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Button(nav_frame, text="<- Previous", width=14, command=self.prev).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Next ->", width=14, command=self.next).pack(side=tk.LEFT, padx=6)

        self.pos_label = tk.Label(nav_frame, text="", anchor="e", bg="#111", fg="#AAA")
        self.pos_label.pack(side=tk.RIGHT, padx=8)

        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self.refresh()

    def _bind_keys(self) -> None:
        self.root.bind("<Right>", lambda _e: self.next())
        self.root.bind("<d>", lambda _e: self.next())
        self.root.bind("<n>", lambda _e: self.next())
        self.root.bind("<space>", lambda _e: self.next())
        self.root.bind("<Left>", lambda _e: self.prev())
        self.root.bind("<a>", lambda _e: self.prev())
        self.root.bind("<p>", lambda _e: self.prev())
        self.root.bind("<Home>", lambda _e: self.first())
        self.root.bind("<End>", lambda _e: self.last())
        self.root.bind("<q>", lambda _e: self.root.destroy())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def first(self) -> None:
        self.index = 0
        self.refresh()

    def last(self) -> None:
        self.index = len(self.txt_files) - 1
        self.refresh()

    def next(self) -> None:
        if self.index < len(self.txt_files) - 1:
            self.index += 1
            self.refresh()

    def prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.refresh()

    def refresh(self) -> None:
        txt_path = self.txt_files[self.index]
        stem = txt_path.stem

        raw_img, count = render_events_txt(txt_path, self.width, self.height)

        matched_png_path = find_matched_png(self.preproc_dir, self.split, stem)
        if matched_png_path is not None:
            matched_img = Image.open(matched_png_path).convert("RGB")
            matched_img = fit_height(matched_img, raw_img.height)

            gap = 10
            canvas = Image.new("RGB", (raw_img.width + gap + matched_img.width, raw_img.height), (20, 20, 20))
            canvas.paste(raw_img, (0, 0))
            canvas.paste(matched_img, (raw_img.width + gap, 0))
            shown = canvas
            matched_note = str(matched_png_path)
        else:
            shown = raw_img
            matched_note = "(no matched PNG found)"

        max_w, max_h = 1750, 700
        scale = min(max_w / shown.width, max_h / shown.height, 1.0)
        if scale < 1.0:
            shown = shown.resize((int(shown.width * scale), int(shown.height * scale)), Image.Resampling.BILINEAR)

        self.tk_image = ImageTk.PhotoImage(shown)
        self.canvas_label.configure(image=self.tk_image)

        self.info_label.configure(
            text=(
                f"sample: {stem} | rendered events: {count}\n"
                f"txt: {txt_path}\n"
                f"matched png: {matched_note}"
            )
        )
        self.pos_label.configure(text=f"{self.index + 1}/{len(self.txt_files)}")


def main() -> int:
    args = parse_args()

    preproc_dir = args.data_root / f"preprocessed_fred_{args.seq}"
    txt_dir = preproc_dir / "eventMatched" / args.split
    if not txt_dir.is_dir():
        raise SystemExit(f"eventMatched split folder not found: {txt_dir}")

    txt_files = sorted([p for p in txt_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"], key=natural_key)
    if not txt_files:
        raise SystemExit(f"No txt files found in: {txt_dir}")

    root = tk.Tk()
    Viewer(
        root=root,
        txt_files=txt_files,
        preproc_dir=preproc_dir,
        split=args.split,
        width=args.width,
        height=args.height,
        start=args.start,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
