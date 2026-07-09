"""EfficientNet-B0 crop verifiers — in-memory crop extraction and scoring.

For every merged proposal we take one RGB crop and one event crop at the same
coordinates (the aligned frames share stems and one coordinate space), expand
the box by cfg.box_scale, resize to cfg.crop_size, and score each crop with
its modality's verifier. Detections in the IoU gray zone
[cfg.iou_ignore, cfg.iou_match) are skipped, exactly as in verifier training.

The old pipeline wrote every crop to disk as a JPEG and re-read it for
scoring; this module keeps crops in memory. (Scores can differ from the old
pipeline in the 3rd decimal because the JPEG re-compression step is gone.)

Checkpoint format (from training/train_verifier.py):
  {"arch": "efficientnet_b0", "crop_size": 96, "model_state": ...}
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import expand_box
from .config import PipelineConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(arch: str):
    import torch.nn as nn
    from torchvision import models

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    else:
        raise ValueError(f"Unsupported verifier arch: {arch}")
    return model


def load_verifier(model_path: Path, device):
    import torch

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("arch", "efficientnet_b0"))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), int(ckpt.get("crop_size", 96))


def _crop_tensor(img: np.ndarray, bbox: list[float], cfg: PipelineConfig,
                 crop_size: int) -> np.ndarray | None:
    """BGR frame + xyxy box -> normalized CHW float32 array, or None."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = expand_box(bbox, cfg.box_scale, w, h)
    if (x2 - x1) < cfg.min_box_px or (y2 - y1) < cfg.min_box_px:
        return None
    crop = cv2.resize(img[y1:y2, x1:x2], (crop_size, crop_size),
                      interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def score_proposals(
    detections: list[dict],
    images_root: Path,
    split: str,
    img_ext: str,
    model_path: Path,
    score_key: str,
    cfg: PipelineConfig,
    measure_activity: bool = False,
    progress=print,
) -> list[dict]:
    """Score verifiable detections with one verifier; adds `score_key`.

    Returns only the detections outside the IoU gray zone, each augmented
    with a crop_id (stable across modalities — both verifiers see the same
    detection order) and the verifier score. With measure_activity=True
    (event modality) each record also gets `event_activity`: the fraction
    of crop pixels brighter than cfg.activity_pixel_thresh — the fusion
    stage uses it to weight the event verifier by how much its crop can
    actually know.
    """
    import torch

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, crop_size = load_verifier(model_path, device)

    keep: list[dict] = []
    for idx, det in enumerate(detections):
        iou = det.get("iou_with_gt", 0.0)
        if cfg.iou_ignore <= iou < cfg.iou_match:
            continue  # gray zone — never verified, never counted
        keep.append({**det, "crop_id": f"{det['split']}_{det['stem']}_det{idx:07d}"})

    img_cache: dict[str, np.ndarray | None] = {}

    def frame(stem: str) -> np.ndarray | None:
        if stem not in img_cache:
            img_cache.clear()  # frames are grouped by stem; cache one at a time
            img_cache[stem] = cv2.imread(
                str(images_root / split / f"{stem}{img_ext}"))
        return img_cache[stem]

    scores: list[float] = []
    activities: list[float] = []
    with torch.no_grad():
        for start in range(0, len(keep), cfg.verifier_batch):
            batch = keep[start:start + cfg.verifier_batch]
            tensors = []
            for det in batch:
                img = frame(det["stem"])
                arr = None if img is None else _crop_tensor(
                    img, det["bbox_xyxy"], cfg, crop_size)
                if arr is None:  # missing frame / degenerate box -> neutral
                    arr = np.zeros((3, crop_size, crop_size), dtype=np.float32)
                tensors.append(arr)
                if measure_activity:
                    activities.append(_crop_activity(
                        img, det["bbox_xyxy"], cfg))
            x = torch.from_numpy(np.stack(tensors)).to(device)
            scores.extend(torch.sigmoid(model(x).squeeze(1)).cpu().tolist())
            if (start // cfg.verifier_batch) % 10 == 0:
                progress(f"  scored {min(start + cfg.verifier_batch, len(keep))}"
                         f"/{len(keep)} crops ({score_key})")

    out = [{**det, score_key: round(s, 6)} for det, s in zip(keep, scores)]
    if measure_activity:
        for rec, act in zip(out, activities):
            rec["event_activity"] = round(act, 5)
    return out


def _crop_activity(img: np.ndarray | None, bbox: list[float],
                   cfg: PipelineConfig) -> float:
    """Fraction of event-crop pixels brighter than the activity threshold."""
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    x1, y1, x2, y2 = expand_box(bbox, cfg.box_scale, w, h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((gray > cfg.activity_pixel_thresh).mean())
