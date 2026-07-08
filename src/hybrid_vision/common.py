"""Shared helpers: JSONL IO, sorting, box geometry, score math.

Every previous pipeline script carried its own copy of these; this is the
single home for them now.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def natural_key(s: str) -> list:
    """Sort key that orders seq2 before seq10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ---------------------------------------------------------------------------
# Boxes (pixel-space xyxy unless stated otherwise)
# ---------------------------------------------------------------------------

def iou_xyxy(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def best_iou(box: list[float], others: list[list[float]]) -> float:
    return max((iou_xyxy(box, o) for o in others), default=0.0)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: float, img_h: float) -> list[float]:
    return [(cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h]


def read_yolo_labels(path: Path, img_w: float, img_h: float) -> list[list[float]]:
    """YOLO label file -> pixel xyxy boxes. Missing file -> no boxes."""
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cx, cy, w, h = (float(v) for v in parts[1:5])
            boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
    return boxes


def read_yolo_labels_norm(path: Path) -> list[list[float]]:
    """YOLO label file -> normalized [cx, cy, w, h] rows (web-output format)."""
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append([round(float(v), 6) for v in parts[1:5]])
    return boxes


def expand_box(box: list[float], scale: float,
               img_w: int, img_h: int) -> list[int]:
    """Expand around the centre by `scale`, clamped to image bounds."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    return [max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)),
            min(img_w, int(cx + bw / 2 + 0.5)), min(img_h, int(cy + bh / 2 + 0.5))]


# ---------------------------------------------------------------------------
# Score math
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))
