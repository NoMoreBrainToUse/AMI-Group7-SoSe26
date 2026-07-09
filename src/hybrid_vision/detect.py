"""YOLO proposal generation for one modality.

Runs the detector once at a very low confidence over all aligned frames and
labels every detection against ground truth. Downstream stages filter by
confidence offline — one inference pass serves any threshold.

Detection record (schema shared with the whole pipeline):
  {split, sequence, stem, bbox_xyxy, detector_score, iou_with_gt, label}
"""

from __future__ import annotations

from pathlib import Path

from .common import best_iou, read_yolo_labels
from .config import PipelineConfig

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def generate_proposals(
    model_path: Path,
    images_dir: Path,
    labels_dir: Path,
    split: str,
    imgsz: int,
    cfg: PipelineConfig,
    batch_size: int = 16,
    progress=print,
) -> list[dict]:
    """Run one YOLO detector over <images_dir>/<split> and label detections."""
    from ultralytics import YOLO

    image_paths = sorted(
        p for p in (images_dir / split).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images under {images_dir / split}")

    model = YOLO(str(model_path))
    detections: list[dict] = []

    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start:start + batch_size]
        results = model.predict(
            source=[str(p) for p in batch],
            imgsz=imgsz, conf=cfg.base_conf, iou=cfg.nms_iou,
            device=cfg.device, verbose=False)
        for path, result in zip(batch, results):
            h, w = result.orig_shape
            gt = read_yolo_labels(
                labels_dir / split / f"{path.stem}.txt", w, h)
            stem = path.stem
            seq = stem.rsplit("_", 1)[0] if "_" in stem else stem
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                iou_gt = best_iou(xyxy, gt)
                detections.append({
                    "split": split,
                    "sequence": seq,
                    "stem": stem,
                    "bbox_xyxy": [round(v, 2) for v in xyxy],
                    "detector_score": round(float(box.conf[0].item()), 6),
                    "iou_with_gt": round(iou_gt, 4),
                    "label": 1 if iou_gt >= cfg.iou_match else 0,
                })
        if (start // batch_size) % 20 == 0:
            progress(f"  {min(start + batch_size, len(image_paths))}"
                     f"/{len(image_paths)} frames")

    return detections
