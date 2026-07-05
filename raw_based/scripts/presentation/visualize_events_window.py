#!/usr/bin/env python3
"""
Render a concatenated event visualization from an events.txt timestamp window.

The script reads events whose timestamps are in [start_us, end_us] from:
  data/preprocessed/preprocessed_fred_<seq>/eventRaw/events.txt
and renders a single polarity image (cyan=ON, orange=OFF) that can be shown
in a window and/or saved as PNG.

Usage examples:
  python3 scripts/presentation/visualize_events_window.py --seq 1 --start-us 2118954 --end-us 2141869
  python3 scripts/presentation/visualize_events_window.py --seq 1

If --start-us / --end-us are omitted, the script asks for them interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Missing dependency: Pillow. Install with: pip install Pillow")


COLOR_POS = (0, 200, 255)
COLOR_NEG = (255, 80, 0)
COLOR_BG = (30, 30, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an events.txt timestamp window as one image.")
    parser.add_argument("--data-root", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--seq", required=True, help="Sequence ID, e.g. 1, 10, 110")
    parser.add_argument("--start-us", type=int, default=None, help="Start timestamp in microseconds (inclusive).")
    parser.add_argument("--end-us", type=int, default=None, help="End timestamp in microseconds (inclusive).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save", type=Path, default=None, help="Optional output PNG path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open preview window.")
    return parser.parse_args()


def read_events_summary(events_txt: Path) -> Tuple[int, int, int]:
    total = 0
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None

    with events_txt.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue
            try:
                ts = int(parts[3])
            except ValueError:
                continue

            if first_ts is None:
                first_ts = ts
            last_ts = ts
            total += 1

    if total == 0 or first_ts is None or last_ts is None:
        raise SystemExit(f"No valid events found in: {events_txt}")

    return total, first_ts, last_ts


def prompt_us(name: str, minimum: int, maximum: int) -> int:
    while True:
        raw = input(f"Enter {name} [{minimum}..{maximum}]: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Out of range. Must be between {minimum} and {maximum}.")


def render_events_window(
    events_txt: Path,
    start_us: int,
    end_us: int,
    width: int,
    height: int,
) -> Tuple[Image.Image, int]:
    img = Image.new("RGB", (width, height), COLOR_BG)
    pixels = img.load()
    matched_events = 0

    if start_us < 0 or end_us < 0 or start_us > end_us:
        return img, matched_events

    with events_txt.open("r", encoding="utf-8", errors="ignore") as fh:
        found_any = False
        for line in fh:
            if not line or line.startswith("#"):
                continue

            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue

            try:
                x = int(parts[0])
                y = int(parts[1])
                pol = int(parts[2])
                ts = int(parts[3])
            except ValueError:
                continue

            if ts < start_us:
                continue
            if ts > end_us:
                if found_any:
                    break
                continue

            found_any = True
            matched_events += 1

            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = COLOR_POS if pol > 0 else COLOR_NEG

    return img, matched_events


def main() -> int:
    args = parse_args()

    preproc_dir = args.data_root / f"preprocessed_fred_{args.seq}"
    events_txt = preproc_dir / "eventRaw" / "events.txt"
    if not events_txt.is_file():
        raise SystemExit(f"events.txt not found: {events_txt}")

    total_events, min_ts, max_ts = read_events_summary(events_txt)

    start_us = args.start_us if args.start_us is not None else prompt_us("start us", min_ts, max_ts)
    end_us = args.end_us if args.end_us is not None else prompt_us("end us", start_us, max_ts)

    if start_us < min_ts or end_us > max_ts or start_us > end_us:
        raise SystemExit(f"Invalid range [{start_us}, {end_us}] for bounds [{min_ts}, {max_ts}] us.")

    img, matched_events = render_events_window(events_txt, start_us, end_us, args.width, args.height)

    save_path = args.save
    if save_path is None:
        save_path = preproc_dir / "eventRaw" / f"window_us_{start_us}_{end_us}.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)

    print(f"Sequence: {args.seq}")
    print(f"Events file: {events_txt}")
    print(f"Total events: {total_events}")
    print(f"Timestamp bounds: [{min_ts}, {max_ts}] us")
    print(f"Rendered window (us): [{start_us}, {end_us}] ({matched_events} events)")
    print(f"Saved image: {save_path}")

    if not args.no_show:
        img.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
