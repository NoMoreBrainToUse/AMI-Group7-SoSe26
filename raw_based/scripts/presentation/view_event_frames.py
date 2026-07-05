#!/usr/bin/env python3
"""
Visualize preprocessed event frames alongside reconstructed event frames from raw.

Three-panel view per sample:
  Left   – Preprocessed event frame PNG from matched/event/images/
  Center – Event frame reconstructed on-the-fly from raw events.txt window
  Right  – Original raw event camera frame from source_event (Event/Frames/)

Sources:
  - data/preprocessed/preprocessed_fred_X/matched/event/images/<split>/<sample>.png
  - data/preprocessed/preprocessed_fred_X/eventRaw/events.txt
  - data/preprocessed/preprocessed_fred_X/eventMatched/matched_windows.csv

Usage:
  python3 scripts/presentation/view_event_frames.py --seq 1
  python3 scripts/presentation/view_event_frames.py --seq 1 --split train --start 0

Controls:
  Right / D / N / Space  – next sample
  Left  / A / P          – previous sample
  Home                   – first
  End                    – last
  Q / Esc                – quit
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import sys
import tkinter as tk
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError:
    raise SystemExit("Missing dependency: Pillow. Install with: pip install Pillow")


# ---------------------------------------------------------------------------
# Event reconstruction helpers
# ---------------------------------------------------------------------------

# Fixed sensor resolution; will be overridden from the data if possible.
SENSOR_W = 1280
SENSOR_H = 720

# Colors used for polarity rendering (RGB tuples).
COLOR_POS = (0, 200, 255)   # cyan – positive polarity
COLOR_NEG = (255, 80, 0)    # orange-red – negative polarity
COLOR_BG = (30, 30, 30)     # dark background


def render_events_from_txt(
    events_txt: Path,
    start_idx: int,
    end_idx: int,
    width: int = SENSOR_W,
    height: int = SENSOR_H,
) -> Image.Image:
    """Read events[start_idx:end_idx] from txt and render a polarity frame."""
    img = Image.new("RGB", (width, height), COLOR_BG)
    pixels = img.load()

    if start_idx < 0 or end_idx < 0 or start_idx > end_idx:
        return img

    current_idx = -1
    with events_txt.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue
            current_idx += 1
            if current_idx < start_idx:
                continue
            if current_idx > end_idx:
                break
            try:
                x, y, pol = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = COLOR_POS if pol > 0 else COLOR_NEG

    return img


def render_events_from_txt_fast(
    events_txt: Path,
    start_us: int,
    end_us: int,
    width: int = SENSOR_W,
    height: int = SENSOR_H,
) -> Image.Image:
    """Read events in [start_us, end_us] from txt using timestamps (fast path)."""
    img = Image.new("RGB", (width, height), COLOR_BG)
    pixels = img.load()

    with events_txt.open("r", encoding="utf-8", errors="ignore") as fh:
        found_any = False
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue
            try:
                t = int(parts[3])
            except ValueError:
                continue
            if t < start_us:
                continue
            if t > end_us:
                if found_any:
                    break
                continue
            found_any = True
            try:
                x, y, pol = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = COLOR_POS if pol > 0 else COLOR_NEG

    return img


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(path: Path, split_filter: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if split_filter and row.get("split") != split_filter:
                continue
            rows.append(row)
    return rows


def find_preproc_event_image(preproc_dir: Path, row: dict[str, str]) -> Path | None:
    """Resolve preprocessed event frame PNG from matched/event/images/<split>/<sample_id>.png"""
    sample_id = row["sample_id"]
    split = row.get("split", "train")
    candidates = [
        preproc_dir / "matched" / "event" / "images" / split / f"{sample_id}.png",
        preproc_dir / "matched" / "event" / "images" / f"{sample_id}.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def find_source_event_image(row: dict[str, str]) -> Path | None:
    src = row.get("source_event", "")
    if src:
        p = Path(src)
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Viewer application
# ---------------------------------------------------------------------------

PANEL_LABELS = [
    "Preprocessed event frame\n(matched/event/images)",
    "Reconstructed from raw events.txt\n(polarity: cyan=ON, orange=OFF)",
    "Original raw camera event frame\n(Event/Frames)",
]

PREPROC_TS_RE = re.compile(r"_(\d+)$")
SOURCE_TS_RE = re.compile(r"_frame_(\d+)\.[^.]+$")


def parse_preproc_time_us(path: Path) -> int | None:
    m = PREPROC_TS_RE.search(path.stem)
    if not m:
        return None
    return int(m.group(1))


def parse_source_frame_time_us(path: Path) -> int | None:
    m = SOURCE_TS_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def nearest_path_by_time(index: list[tuple[int, Path]], target_us: int) -> Path | None:
    if not index:
        return None
    times = [t for t, _ in index]
    pos = bisect.bisect_left(times, target_us)
    candidates: list[tuple[int, Path]] = []
    if pos < len(index):
        candidates.append(index[pos])
    if pos > 0:
        candidates.append(index[pos - 1])
    return min(candidates, key=lambda item: abs(item[0] - target_us))[1] if candidates else None


class ViewerApp:
    def __init__(
        self,
        root: tk.Tk,
        rows: list[dict[str, str]],
        preproc_dir: Path,
        events_txt: Path,
        source_frames_dir: Path,
        start_index: int,
    ) -> None:
        self.root = root
        self.rows = rows
        self.preproc_dir = preproc_dir
        self.events_txt = events_txt
        self.source_frames_dir = source_frames_dir
        self.index = start_index
        self.photo: Any = None
        self.prev_index: int = -1
        self.preproc_index_by_split = self._build_preproc_time_index()
        self.source_index = self._build_source_time_index()

        root.title("Event Frame Viewer")
        root.geometry("1500x620")
        root.configure(bg="#111")

        self.image_label = tk.Label(root, bg="#111")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        info_frame = tk.Frame(root, bg="#1c1c1c")
        info_frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.info_label = tk.Label(
            info_frame,
            text="",
            anchor="w",
            justify=tk.LEFT,
            font=("TkDefaultFont", 10),
            bg="#1c1c1c",
            fg="#DDD",
            padx=10,
            pady=6,
        )
        self.info_label.pack(fill=tk.X)

        nav_frame = tk.Frame(root, bg="#111")
        nav_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Button(nav_frame, text="← Previous", width=14, command=self.show_previous).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Next →", width=14, command=self.show_next, padx=8).pack(side=tk.LEFT, padx=6)

        self.pos_label = tk.Label(nav_frame, text="", anchor="e", bg="#111", fg="#AAA")
        self.pos_label.pack(side=tk.RIGHT, padx=8)

        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self.refresh()

    def _build_preproc_time_index(self) -> dict[str, list[tuple[int, Path]]]:
        root = self.preproc_dir / "matched" / "event" / "images"
        out: dict[str, list[tuple[int, Path]]] = {}
        if not root.is_dir():
            return out

        # Standard layout: images/<split>/*.png
        split_dirs = [p for p in root.iterdir() if p.is_dir()]
        if split_dirs:
            for split_dir in split_dirs:
                items: list[tuple[int, Path]] = []
                for img in split_dir.glob("*.png"):
                    t_us = parse_preproc_time_us(img)
                    if t_us is not None:
                        items.append((t_us, img))
                items.sort(key=lambda x: x[0])
                out[split_dir.name] = items
            return out

        # Fallback layout: images/*.png
        items = []
        for img in root.glob("*.png"):
            t_us = parse_preproc_time_us(img)
            if t_us is not None:
                items.append((t_us, img))
        items.sort(key=lambda x: x[0])
        out["all"] = items
        return out

    def _build_source_time_index(self) -> list[tuple[int, Path]]:
        items: list[tuple[int, Path]] = []
        if not self.source_frames_dir.is_dir():
            return items
        for img in self.source_frames_dir.glob("*.png"):
            t_us = parse_source_frame_time_us(img)
            if t_us is not None:
                items.append((t_us, img))
        items.sort(key=lambda x: x[0])
        return items

    def _nearest_preproc_image(self, split: str, anchor_us: int) -> Path | None:
        items = self.preproc_index_by_split.get(split)
        if not items:
            items = self.preproc_index_by_split.get("all", [])
        return nearest_path_by_time(items, anchor_us)

    def _nearest_source_image(self, anchor_us: int, row: dict[str, str]) -> Path | None:
        nearest_local = nearest_path_by_time(self.source_index, anchor_us)
        if nearest_local is not None:
            return nearest_local
        # Fallback to the manifest path if local Event/Frames is unavailable.
        return find_source_event_image(row)

    def _bind_keys(self) -> None:
        for key in ("<Right>", "<d>", "<n>", "<space>"):
            self.root.bind(key, lambda _e: self.show_next())
        for key in ("<Left>", "<a>", "<p>"):
            self.root.bind(key, lambda _e: self.show_previous())
        self.root.bind("<Home>", lambda _e: self._go(0))
        self.root.bind("<End>", lambda _e: self._go(len(self.rows) - 1))
        self.root.bind("<q>", lambda _e: self.root.destroy())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def _go(self, idx: int) -> None:
        self.index = max(0, min(len(self.rows) - 1, idx))
        self.refresh()

    def show_next(self) -> None:
        self._go(self.index + 1)

    def show_previous(self) -> None:
        self._go(self.index - 1)

    def _load_or_placeholder(self, path: Path | None, label: str, w: int, h: int) -> Image.Image:
        if path is not None and path.is_file():
            return Image.open(path).convert("RGB")
        img = Image.new("RGB", (w, h), (50, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.text((10, h // 2 - 10), f"[{label}]\nnot found", fill=(200, 100, 100))
        return img

    @staticmethod
    def _resize_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        scale = min(target_w / img.width, target_h / img.height)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _build_canvas(self, row: dict[str, str]) -> Image.Image:
        vp_w = max(self.root.winfo_width() - 20, 800)
        vp_h = max(self.root.winfo_height() - 140, 300)
        panel_w = vp_w // 3
        panel_h = vp_h
        anchor_us = int(row.get("frame_time_us", "0"))

        # Panel 1 – preprocessed event frame
        preproc_img_path = self._nearest_preproc_image(row.get("split", "train"), anchor_us)
        if preproc_img_path is None:
            preproc_img_path = find_preproc_event_image(self.preproc_dir, row)
        p1 = self._load_or_placeholder(preproc_img_path, "preprocessed", panel_w, panel_h)
        p1 = self._resize_fit(p1, panel_w, panel_h)

        # Panel 2 – reconstruct from raw events
        start_us = int(row.get("start_us", "-1"))
        end_us = int(row.get("end_us", "-1"))
        start_idx = int(row.get("start_idx", "-1"))
        end_idx = int(row.get("end_idx", "-1"))

        if self.events_txt.is_file() and start_us >= 0 and end_us >= 0:
            try:
                if start_idx >= 0 and end_idx >= 0:
                    p2 = render_events_from_txt(self.events_txt, start_idx, end_idx)
                else:
                    p2 = render_events_from_txt_fast(self.events_txt, start_us, end_us)
            except Exception:
                p2 = Image.new("RGB", (SENSOR_W, SENSOR_H), (60, 0, 0))
        else:
            p2 = Image.new("RGB", (SENSOR_W, SENSOR_H), (50, 50, 50))
        p2 = self._resize_fit(p2, panel_w, panel_h)

        # Panel 3 – original raw camera event frame
        src_path = self._nearest_source_image(anchor_us, row)
        p3 = self._load_or_placeholder(src_path, "source_event", panel_w, panel_h)
        p3 = self._resize_fit(p3, panel_w, panel_h)

        # Compose panels side by side with labels
        canvas = Image.new("RGB", (vp_w, vp_h + 36), (17, 17, 17))
        label_img = Image.new("RGB", (vp_w, 36), (25, 25, 25))
        draw = ImageDraw.Draw(label_img)
        for i, (panel, lbl) in enumerate(zip([p1, p2, p3], PANEL_LABELS)):
            x = i * panel_w
            ph = min(panel.height, panel_h)
            y_off = (panel_h - ph) // 2
            canvas.paste(panel, (x, y_off))
            draw.text((x + 6, 8), lbl.split("\n")[0], fill=(180, 180, 180))
        canvas.paste(label_img, (0, vp_h))

        return canvas

    def refresh(self) -> None:
        if not self.rows:
            return
        row = self.rows[self.index]

        canvas = self._build_canvas(row)
        self.photo = ImageTk.PhotoImage(canvas)
        self.image_label.configure(image=self.photo)

        n_events = row.get("event_count", "?")
        t_us = int(row.get("frame_time_us", 0))
        self.info_label.configure(
            text=(
                f"[{self.index + 1}/{len(self.rows)}]  "
                f"sample: {row['sample_id']}  |  "
                f"split: {row.get('split', '?')}  |  "
                f"frame_time: {t_us / 1e6:.6f} s  |  "
                f"window: [{int(row.get('start_us',0))/1e6:.6f} – {int(row.get('end_us',0))/1e6:.6f}] s  |  "
                f"events in window: {n_events}"
            )
        )
        self.pos_label.configure(text=f"{self.index + 1} / {len(self.rows)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize preprocessed and reconstructed event frames side by side."
    )
    p.add_argument("--seq", required=True, help="Sequence id X (for example: 1, 10, 34).")
    p.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    p.add_argument("--split", default=None, help="Filter by split (train/val/test). Default: all.")
    p.add_argument("--start", type=int, default=0, help="Start at this sample index (0-based).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = args.project_root.resolve()
    seq = str(args.seq).strip()

    preproc_dir = root_dir / "data" / "preprocessed" / f"preprocessed_fred_{seq}"
    events_txt = preproc_dir / "eventRaw" / "events.txt"
    manifest = preproc_dir / "eventMatched" / "matched_windows.csv"
    source_frames_dir = root_dir / "data" / "raw" / seq / "Event" / "Frames"

    if not preproc_dir.is_dir():
        print(f"[ERROR] Preprocessed directory not found: {preproc_dir}", file=sys.stderr)
        return 1
    if not manifest.is_file():
        print(f"[ERROR] Matched windows manifest not found: {manifest}", file=sys.stderr)
        print("  Run: python3 scripts/match_event_txt_to_frames.py --seq", seq, "--manifest-only", file=sys.stderr)
        return 1
    if not events_txt.is_file():
        print(f"[WARNING] events.txt not found: {events_txt}  (panel 2 will be blank)")

    rows = load_manifest(manifest, split_filter=args.split)
    if not rows:
        print(f"[ERROR] No entries in manifest (split filter: {args.split!r})", file=sys.stderr)
        return 1

    start_index = max(0, min(args.start, len(rows) - 1))

    tk_root = tk.Tk()
    ViewerApp(
        tk_root,
        rows=rows,
        preproc_dir=preproc_dir,
        events_txt=events_txt,
        source_frames_dir=source_frames_dir,
        start_index=start_index,
    )
    tk_root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
