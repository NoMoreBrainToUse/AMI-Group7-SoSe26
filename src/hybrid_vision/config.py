"""Pipeline configuration — one place for every path, weight and threshold.

Defaults reproduce the calibrated hybrid pipeline; override per run via
PipelineConfig(...) or the run_pipeline.py CLI flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "weights"


@dataclass
class PipelineConfig:
    # --- model weights -----------------------------------------------------
    event_model: Path = WEIGHTS_DIR / "event_yolo11m.pt"
    rgb_model: Path = WEIGHTS_DIR / "rgb_yolo11m.pt"
    rgb_verifier: Path = WEIGHTS_DIR / "verifier_rgb_effb0.pt"
    event_verifier: Path = WEIGHTS_DIR / "verifier_event_effb0.pt"
    fusion_config: Path = WEIGHTS_DIR / "fusion_config.json"

    # --- detection ---------------------------------------------------------
    event_imgsz: int = 640     # event-YOLO v5 was trained at 640
    rgb_imgsz: int = 640       # set 1280 for the v6 retrained weights
    base_conf: float = 0.01    # single low-conf YOLO pass; filter offline
    proposal_conf: float = 0.20  # proposals entering the verifier stage
    nms_iou: float = 0.7
    # RGB backs up the event detector for drones it cannot see (a hovering
    # drone produces almost no events). It is gated hard so it only
    # contributes when very sure; its proposals still pass the verifiers.
    rgb_min_conf: float = 0.60
    dedup_iou: float = 0.5     # cross-modality proposal dedup

    # --- ground-truth matching --------------------------------------------
    iou_match: float = 0.5     # proposal/GT IoU for a positive label
    iou_ignore: float = 0.3    # [ignore, match) = gray zone, not verified

    # --- verifier crops ----------------------------------------------------
    crop_size: int = 96
    box_scale: float = 1.5
    min_box_px: int = 4
    verifier_batch: int = 256

    # --- alignment (FRED sequence -> paired frames) ------------------------
    rgb_dir_name: str = "PADDED_RGB"
    frame_period: float = 0.033333
    max_delta_s: float = 0.04
    annotation_files: tuple[str, ...] = (
        "interpolated_coordinates.txt", "coordinates.txt")

    # --- output ------------------------------------------------------------
    img_width: float = 1280.0
    img_height: float = 720.0
    fps: float = 30.0
    device: str | None = None  # None = auto (cuda if available)

    # fusion operating point, read from fusion_config json
    fusion_lambda: float = field(init=False, default=1.0)
    fusion_threshold: float = field(init=False, default=0.034)

    def __post_init__(self) -> None:
        cfg = json.loads(Path(self.fusion_config).read_text(encoding="utf-8"))
        self.fusion_lambda = float(cfg["best_lambda"])
        self.fusion_threshold = float(cfg["best_threshold"])

    def validate(self) -> None:
        for name in ("event_model", "rgb_model", "rgb_verifier",
                     "event_verifier", "fusion_config"):
            p = Path(getattr(self, name))
            if not p.is_file():
                raise FileNotFoundError(f"Missing {name}: {p}")
