"""FRED sequence -> time-aligned RGB/event frame pairs in YOLO layout.

A FRED sequence directory contains:
  PADDED_RGB/Video_<seq>_<H>_<M>_<S.frac>.jpg   RGB frames, padded to the
                                                event sensor's field of view
  Event/Frames/*_<time_us>.png                  event frames (µs timestamps)
  interpolated_coordinates.txt                  '<t>: x1, y1, x2, y2[, id]'

Annotation timestamps are relative to sequence start. RGB frame times are
reconstructed from the frame index (index * frame_period); event frame times
come from the µs suffix in the filename. For every annotation timestamp we
pick the nearest RGB and event frame within max_delta_s and emit one aligned
sample: shared stem, both images (hardlinked), identical YOLO labels — the
PADDED_RGB and event sensors share one coordinate space, which is what makes
cross-modality crops and proposal merging exact.

Output layout (consumed by every later stage and by YOLO training):
  <out>/rgb_yolo/images/<split>/seq<seq>_<time_us>.jpg
  <out>/rgb_yolo/labels/<split>/seq<seq>_<time_us>.txt
  <out>/event_yolo/{images,labels}/<split>/...
  <out>/{rgb_yolo,event_yolo}/data.yaml
"""

from __future__ import annotations

import bisect
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import PipelineConfig

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class TimedPath:
    time_s: float
    path: Path


@dataclass
class AlignResult:
    sequence: str
    samples_written: int = 0
    labels_seen: int = 0
    unmatched: int = 0
    invalid_boxes: int = 0


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _parse_annotations(path: Path) -> dict[float, list[list[float]]]:
    """'<t>: x1, y1, x2, y2[, track_id]' lines -> {t: [xyxy, ...]}."""
    out: dict[float, list[list[float]]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        t_raw, rest = line.split(":", 1)
        try:
            t = float(t_raw)
            vals = [float(v) for v in rest.split(",")[:4]]
        except ValueError:
            continue
        if len(vals) == 4:
            out.setdefault(t, []).append(vals)
    return out


def _event_timeline(event_dir: Path) -> list[TimedPath]:
    items = []
    for p in _list_images(event_dir):
        m = re.search(r"(\d+)(?=\.[^.]+$)", p.name)
        if m:
            items.append(TimedPath(int(m.group(1)) / 1_000_000.0, p))
    return sorted(items, key=lambda i: i.time_s)


def _rgb_timeline(rgb_dir: Path, frame_period: float) -> list[TimedPath]:
    # FRED annotations index the PADDED_RGB sequence: frame i sits at
    # (i + 1) * frame_period on the annotation clock.
    return [TimedPath((i + 1) * frame_period, p)
            for i, p in enumerate(_list_images(rgb_dir))]


def _nearest(timeline: list[TimedPath], target_s: float,
             max_delta_s: float) -> TimedPath | None:
    if not timeline:
        return None
    times = [i.time_s for i in timeline]
    pos = bisect.bisect_left(times, target_s)
    candidates = [timeline[i] for i in (pos - 1, pos) if 0 <= i < len(timeline)]
    best = min(candidates, key=lambda i: abs(i.time_s - target_s))
    return best if abs(best.time_s - target_s) <= max_delta_s else None


def _hardlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _image_size(path: Path, cache: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    if path not in cache:
        img = cv2.imread(str(path))
        cache[path] = (img.shape[1], img.shape[0]) if img is not None else (1280, 720)
    return cache[path]


def _yolo_line(box: list[float], w: int, h: int) -> str | None:
    x1 = min(max(box[0], 0.0), w - 1.0)
    y1 = min(max(box[1], 0.0), h - 1.0)
    x2 = min(max(box[2], 0.0), w - 1.0)
    y2 = min(max(box[3], 0.0), h - 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return (f"0 {(x1 + x2) / 2 / w:.8f} {(y1 + y2) / 2 / h:.8f} "
            f"{(x2 - x1) / w:.8f} {(y2 - y1) / h:.8f}")


def _write_data_yaml(modality_root: Path) -> None:
    modality_root.mkdir(parents=True, exist_ok=True)
    (modality_root / "data.yaml").write_text(
        f"path: {modality_root.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: drone\n",
        encoding="utf-8")


def align_sequence(seq_dir: Path, output_root: Path, split: str = "test",
                   cfg: PipelineConfig | None = None) -> AlignResult:
    """Align one FRED sequence directory into <output_root> under `split`."""
    cfg = cfg or PipelineConfig()
    seq = seq_dir.name
    result = AlignResult(sequence=seq)

    rgb_dir = seq_dir / cfg.rgb_dir_name
    event_dir = seq_dir / "Event" / "Frames"
    ann_path = next((seq_dir / n for n in cfg.annotation_files
                     if (seq_dir / n).is_file()), None)
    missing = [str(d) for d in (rgb_dir, event_dir) if not d.is_dir()]
    if ann_path is None:
        missing.append(f"annotation file ({' or '.join(cfg.annotation_files)})")
    if missing:
        raise FileNotFoundError(f"Sequence {seq} is missing: {', '.join(missing)}")

    annotations = _parse_annotations(ann_path)
    rgb_tl = _rgb_timeline(rgb_dir, cfg.frame_period)
    event_tl = _event_timeline(event_dir)
    result.labels_seen = len(annotations)

    size_cache: dict[Path, tuple[int, int]] = {}
    for t in sorted(annotations):
        rgb_item = _nearest(rgb_tl, t, cfg.max_delta_s)
        event_item = _nearest(event_tl, t, cfg.max_delta_s)
        if rgb_item is None or event_item is None:
            result.unmatched += 1
            continue

        rgb_w, rgb_h = _image_size(rgb_item.path, size_cache)
        evt_w, evt_h = _image_size(event_item.path, size_cache)
        rgb_lines, evt_lines = [], []
        for box in annotations[t]:
            rl = _yolo_line(box, rgb_w, rgb_h)
            el = _yolo_line(box, evt_w, evt_h)
            if rl is None or el is None:
                result.invalid_boxes += 1
                continue
            rgb_lines.append(rl)
            evt_lines.append(el)
        if not rgb_lines:
            continue

        stem = f"seq{seq}_{int(round(t * 1_000_000)):09d}"
        for modality, item, lines in (("rgb", rgb_item, rgb_lines),
                                      ("event", event_item, evt_lines)):
            img_dst = (output_root / f"{modality}_yolo" / "images" / split
                       / f"{stem}{item.path.suffix.lower()}")
            lbl_dst = (output_root / f"{modality}_yolo" / "labels" / split
                       / f"{stem}.txt")
            _hardlink(item.path, img_dst)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result.samples_written += 1

    for modality in ("rgb", "event"):
        _write_data_yaml(output_root / f"{modality}_yolo")
    return result
