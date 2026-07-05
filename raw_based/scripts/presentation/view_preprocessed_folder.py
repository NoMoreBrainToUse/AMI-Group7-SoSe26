#!/usr/bin/env python3
"""
Simple viewer for one preprocessed FRED folder (preprocessed_fred_X).

Features:
- Loads paired samples from paired/manifest_*.csv
- Shows event-over-RGB overlay above the pair view
- Shows RGB and Event matched images side by side
- Displays matched timestamps below the images
- Simple GUI with Previous/Next, Play/Pause, and keyboard navigation

Usage:
  python3 scripts/presentation/view_preprocessed_folder.py --folder data/preprocessed/preprocessed_fred_1
  python3 scripts/presentation/view_preprocessed_folder.py --folder data/preprocessed/preprocessed_fred_1 --split train

Controls:
    Enter / Space         : play/pause
    Right / D / N         : next sample
  Left / A / P          : previous sample
  Home                  : first sample
  End                   : last sample
  Q / Esc               : quit
"""

from __future__ import annotations

import argparse
import csv
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import tkinter as tk
from tkinter import messagebox

try:
    PIL_Image = importlib.import_module("PIL.Image")
    PIL_ImageTk = importlib.import_module("PIL.ImageTk")
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Pillow. Install it with: pip install Pillow"
    ) from exc


@dataclass(frozen=True)
class Sample:
    split: str
    rgb_image: Path
    event_image: Path
    label_time_s: float
    rgb_time_s: float
    event_time_s: float
    rgb_delta_s: float
    event_delta_s: float


class ViewerApp:
    def __init__(self, root: tk.Tk, samples: list[Sample], start_index: int, max_display_height: int) -> None:
        self.root = root
        self.samples = samples
        self.index = start_index
        self.max_display_height = max_display_height
        self.photo: Any = None
        self.overlay_enabled = tk.BooleanVar(value=True)
        self.overlay_alpha = tk.DoubleVar(value=0.45)
        self.playing = False
        self.play_job: str | None = None
        self.play_fps = tk.DoubleVar(value=6.0)

        self.root.title("FRED Preprocessed Pair Viewer")
        self.root.geometry("1400x900")

        self.image_label = tk.Label(self.root, bg="#101010")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))

        self.info_label = tk.Label(
            self.root,
            text="",
            anchor="w",
            justify=tk.LEFT,
            font=("TkDefaultFont", 11),
            bg="#202020",
            fg="#F0F0F0",
            padx=12,
            pady=10,
        )
        self.info_label.pack(fill=tk.X, padx=10, pady=(0, 8))

        overlay_frame = tk.Frame(self.root)
        overlay_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Checkbutton(
            overlay_frame,
            text="Show Overlay",
            variable=self.overlay_enabled,
            command=self.refresh,
        ).pack(side=tk.LEFT)

        tk.Label(overlay_frame, text="Overlay Alpha").pack(side=tk.LEFT, padx=(12, 6))
        self.alpha_scale = tk.Scale(
            overlay_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=tk.DoubleVar(value=self.overlay_alpha.get() * 100.0),
            showvalue=True,
            length=260,
            command=self._on_alpha_changed,
        )
        self.alpha_scale.pack(side=tk.LEFT)

        nav_frame = tk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.prev_button = tk.Button(nav_frame, text="Previous", width=12, command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(nav_frame, text="Next", width=12, command=self.show_next)
        self.next_button.pack(side=tk.LEFT, padx=(8, 0))

        self.play_button = tk.Button(nav_frame, text="Play", width=12, command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(nav_frame, text="FPS").pack(side=tk.LEFT, padx=(12, 6))
        self.fps_scale = tk.Scale(
            nav_frame,
            from_=1,
            to=30,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.play_fps,
            showvalue=True,
            length=200,
        )
        self.fps_scale.pack(side=tk.LEFT)

        self.pos_label = tk.Label(nav_frame, text="", anchor="e")
        self.pos_label.pack(side=tk.RIGHT)

        self.root.bind("<Left>", lambda _event: self.show_previous())
        self.root.bind("<Right>", lambda _event: self.show_next())
        self.root.bind("<a>", lambda _event: self.show_previous())
        self.root.bind("<d>", lambda _event: self.show_next())
        self.root.bind("<p>", lambda _event: self.show_previous())
        self.root.bind("<n>", lambda _event: self.show_next())
        self.root.bind("<Return>", lambda _event: self.toggle_play())
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Home>", lambda _event: self.show_first())
        self.root.bind("<End>", lambda _event: self.show_last())
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("<q>", lambda _event: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.refresh()

    def _on_alpha_changed(self, value: str) -> None:
        try:
            self.overlay_alpha.set(max(0.0, min(1.0, float(value) / 100.0)))
        except ValueError:
            return
        self.refresh()

    def _load_prepared_pair(self, sample: Sample) -> tuple[Any, Any]:
        if not sample.rgb_image.is_file():
            raise FileNotFoundError(f"Could not read RGB image: {sample.rgb_image}")
        if not sample.event_image.is_file():
            raise FileNotFoundError(f"Could not read Event image: {sample.event_image}")

        rgb = PIL_Image.open(sample.rgb_image).convert("RGB")
        event = PIL_Image.open(sample.event_image).convert("RGB")

        target_h = min(rgb.height, event.height, self.max_display_height)
        target_h = max(240, target_h)

        rgb = self._resize_to_height(rgb, target_h)
        event = self._resize_to_height(event, target_h)

        panel_h = max(rgb.height, event.height)
        rgb = self._pad_to_height(rgb, panel_h)
        event = self._pad_to_height(event, panel_h)
        return rgb, event

    def _load_side_by_side(self, sample: Sample) -> Any:
        rgb, event = self._load_prepared_pair(sample)
        panel_h = max(rgb.height, event.height)

        merged = PIL_Image.new("RGB", (rgb.width + event.width, panel_h), (18, 18, 18))
        merged.paste(rgb, (0, 0))
        merged.paste(event, (rgb.width, 0))
        return merged

    def _build_overlay(self, rgb: Any, event: Any) -> Any:
        alpha = float(self.overlay_alpha.get())
        alpha = max(0.0, min(1.0, alpha))
        return PIL_Image.blend(rgb, event, alpha)

    def _compose_view(self, sample: Sample) -> Any:
        rgb, event = self._load_prepared_pair(sample)

        side_rgb = self._scale_image(rgb, 0.62)
        side_event = self._scale_image(event, 0.62)
        side_panel_h = max(side_rgb.height, side_event.height)
        side_rgb = self._pad_to_height(side_rgb, side_panel_h)
        side_event = self._pad_to_height(side_event, side_panel_h)

        side_by_side = PIL_Image.new("RGB", (side_rgb.width + side_event.width, side_panel_h), (18, 18, 18))
        side_by_side.paste(side_rgb, (0, 0))
        side_by_side.paste(side_event, (side_rgb.width, 0))

        if not self.overlay_enabled.get():
            return side_by_side

        overlay = self._build_overlay(rgb, event)
        overlay = self._scale_image(overlay, 1.18)
        top_pad = 36
        gap = 10
        canvas_w = max(side_by_side.width, overlay.width)
        canvas_h = top_pad + overlay.height + gap + side_by_side.height
        canvas = PIL_Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))

        overlay_x = (canvas_w - overlay.width) // 2
        side_x = (canvas_w - side_by_side.width) // 2
        canvas.paste(overlay, (overlay_x, top_pad))
        canvas.paste(side_by_side, (side_x, top_pad + overlay.height + gap))
        return canvas

    @staticmethod
    def _resize_to_height(img: Any, target_h: int) -> Any:
        if img.height == target_h:
            return img
        scale = target_h / float(img.height)
        target_w = max(1, int(round(img.width * scale)))
        return img.resize((target_w, target_h), PIL_Image.Resampling.LANCZOS)

    @staticmethod
    def _pad_to_height(img: Any, target_h: int) -> Any:
        if img.height >= target_h:
            return img
        canvas = PIL_Image.new("RGB", (img.width, target_h), (18, 18, 18))
        y = (target_h - img.height) // 2
        canvas.paste(img, (0, y))
        return canvas

    @staticmethod
    def _scale_image(img: Any, scale: float) -> Any:
        scale = max(0.1, scale)
        target_w = max(1, int(round(img.width * scale)))
        target_h = max(1, int(round(img.height * scale)))
        return img.resize((target_w, target_h), PIL_Image.Resampling.LANCZOS)

    def refresh(self) -> None:
        sample = self.samples[self.index]
        try:
            merged = self._compose_view(sample)
        except Exception as exc:
            messagebox.showerror("Image Load Error", str(exc))
            return

        viewport_w = max(self.root.winfo_width() - 40, 300)
        viewport_h = max(self.root.winfo_height() - 220, 220)
        scale = min(viewport_w / merged.width, viewport_h / merged.height, 1.0)
        if scale < 1.0:
            resized = merged.resize(
                (max(1, int(merged.width * scale)), max(1, int(merged.height * scale))),
                PIL_Image.Resampling.LANCZOS,
            )
        else:
            resized = merged

        self.photo = PIL_ImageTk.PhotoImage(resized)
        self.image_label.configure(image=self.photo)

        info = (
            f"split={sample.split}\n"
            f"label={sample.label_time_s:.6f}s   rgb={sample.rgb_time_s:.6f}s   event={sample.event_time_s:.6f}s\n"
            f"delta_rgb={sample.rgb_delta_s:.6f}s   delta_event={sample.event_delta_s:.6f}s   "
            f"overlay_alpha={self.overlay_alpha.get():.2f}"
        )
        self.info_label.configure(text=info)
        self.pos_label.configure(text=f"{self.index + 1} / {len(self.samples)}")

        self.prev_button.configure(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next_button.configure(state=tk.NORMAL if self.index < len(self.samples) - 1 else tk.DISABLED)
        self.play_button.configure(text="Pause" if self.playing else "Play")

    def show_previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.refresh()

    def show_next(self) -> None:
        if self.index < len(self.samples) - 1:
            self.index += 1
            self.refresh()

    def toggle_play(self) -> None:
        if self.playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self) -> None:
        if self.playing:
            return
        self.playing = True
        self.refresh()
        self._schedule_next_tick()

    def stop_play(self) -> None:
        self.playing = False
        if self.play_job is not None:
            try:
                self.root.after_cancel(self.play_job)
            except Exception:
                pass
            self.play_job = None
        self.refresh()

    def _schedule_next_tick(self) -> None:
        fps = max(1.0, float(self.play_fps.get()))
        delay_ms = max(1, int(round(1000.0 / fps)))
        self.play_job = self.root.after(delay_ms, self._play_tick)

    def _play_tick(self) -> None:
        self.play_job = None
        if not self.playing:
            return

        if self.index >= len(self.samples) - 1:
            self.stop_play()
            return

        self.index += 1
        self.refresh()
        self._schedule_next_tick()

    def show_first(self) -> None:
        self.index = 0
        self.refresh()

    def show_last(self) -> None:
        self.index = len(self.samples) - 1
        self.refresh()

    def close(self) -> None:
        self.stop_play()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View matched RGB/Event image pairs for a preprocessed_fred_X folder.")
    parser.add_argument("--folder", type=Path, required=True, help="Path to one preprocessed_fred_X folder.")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Optional split filter. Default: all.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Initial sample index (0-based).")
    parser.add_argument(
        "--max-display-height",
        type=int,
        default=720,
        help="Maximum display height for each image panel.",
    )
    return parser.parse_args()


def iter_manifest_files(paired_dir: Path) -> Iterable[Path]:
    for split in ("train", "val", "test"):
        path = paired_dir / f"manifest_{split}.csv"
        if path.is_file():
            yield path


def load_samples(folder: Path, split_filter: str) -> list[Sample]:
    paired_dir = folder / "paired"
    manifests = list(iter_manifest_files(paired_dir))
    samples: list[Sample] = []

    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                split = row.get("split", "")
                if split_filter != "all" and split != split_filter:
                    continue

                rgb_rel = row.get("rgb_image", "")
                event_rel = row.get("event_image", "")
                if not rgb_rel or not event_rel:
                    continue

                samples.append(
                    Sample(
                        split=split,
                        rgb_image=folder / rgb_rel,
                        event_image=folder / event_rel,
                        label_time_s=float(row.get("label_time_s", 0.0)),
                        rgb_time_s=float(row.get("rgb_time_s", 0.0)),
                        event_time_s=float(row.get("event_time_s", 0.0)),
                        rgb_delta_s=float(row.get("rgb_delta_s", 0.0)),
                        event_delta_s=float(row.get("event_delta_s", 0.0)),
                    )
                )

    return samples


def main() -> int:
    args = parse_args()
    folder = args.folder.resolve()

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    samples = load_samples(folder, args.split)
    if not samples:
        print("No samples found. Check folder path and paired/manifest_*.csv files.")
        return 1

    index = min(max(args.start_index, 0), len(samples) - 1)

    root = tk.Tk()
    app = ViewerApp(root, samples, index, args.max_display_height)
    root.bind("<Configure>", lambda _event: app.refresh())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
