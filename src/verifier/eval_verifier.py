#!/usr/bin/env python3
"""
Evaluate the RGB verifier and measure false-positive reduction at fixed recall.

This script answers the core question:
  "Does the RGB verifier reduce FPs while keeping recall close to the event YOLO?"

Pipeline:
  1. Load a crop manifest JSONL (produced by extract_crops.py).
     Each record has: label (1=drone/0=background), detector_score, crop_path.
  2. Score every crop with the trained verifier (sigmoid output = drone probability).
  3. For each possible verifier threshold, compute:
       - recall   = TPs kept / total GT (among proposals in this split)
       - FPs kept = FP proposals that still pass the threshold
  4. Print a table of key operating points and save results to CSV.

The "fixed recall" evaluation: at 95% and 90% of the detector's recall,
how many FPs does the verifier eliminate?

Example:
  python src/verifier/eval_verifier.py \\
    --model runs/verifier/rgb/mobilenet_v3_small/best.pt \\
    --manifest processed/fred_subset/crops/rgb/crop_manifest_detections_conf0.20.jsonl

  # Compare detector-only vs. verifier-filtered:
  python src/verifier/eval_verifier.py \\
    --model runs/verifier/rgb/mobilenet_v3_small/best.pt \\
    --manifest processed/fred_subset/crops/rgb/crop_manifest_detections_conf0.20.jsonl \\
    --recall-targets 0.95,0.90,0.85
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Evaluate RGB verifier: FP reduction at fixed recall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to verifier checkpoint (best.pt from train_rgb_verifier.py).",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Crop manifest JSONL from extract_crops.py.",
    )
    parser.add_argument(
        "--recall-targets",
        default="0.95,0.90,0.85",
        help="Comma-separated recall fractions to report (relative to detector recall).",
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for output CSV. Defaults to same dir as --model.",
    )
    return parser


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def score_crops(model_path: Path, manifest_records: list[dict], batch_size: int, device) -> list[float]:
    """Return verifier sigmoid scores in the same order as manifest_records."""
    import torch
    from torchvision import transforms

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "mobilenet_v3_small")
    crop_size = ckpt.get("crop_size", 96)

    # Re-build the same model architecture
    import importlib, sys
    # Use the same build_model from train_rgb_verifier
    train_module_path = Path(__file__).parent / "train_rgb_verifier.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_rgb_verifier", str(train_module_path))
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    model = train_mod.build_model(arch, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow not installed") from exc

    scores = []
    import torch

    for i in range(0, len(manifest_records), batch_size):
        batch_recs = manifest_records[i: i + batch_size]
        tensors = []
        for rec in batch_recs:
            try:
                img = Image.open(rec["crop_path"]).convert("RGB")
                tensors.append(transform(img))
            except Exception:
                # Missing crop → neutral score 0.5
                import torch as t
                tensors.append(t.zeros(3, crop_size, crop_size))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch).squeeze(1)
            batch_scores = torch.sigmoid(logits).cpu().tolist()
        scores.extend(batch_scores)

    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_operating_points(
    labels: list[int],
    detector_scores: list[float],
    verifier_scores: list[float],
    recall_targets: list[float],
) -> tuple[list[dict], dict]:
    """
    Sweep verifier thresholds and compute recall / FP count.

    Returns:
      full_curve: list of dicts, one per threshold
      operating_points: dict keyed by recall target
    """
    n_tp_total = sum(labels)
    n_fp_total = len(labels) - n_tp_total

    # Detector-only baseline
    baseline = {
        "recall": n_tp_total / n_tp_total if n_tp_total > 0 else 0.0,  # = 1.0 by definition
        "tp": n_tp_total,
        "fp": n_fp_total,
        "fp_reduction_pct": 0.0,
        "verifier_threshold": None,
    }

    # Sweep verifier thresholds from 0 → 1
    thresholds = sorted({round(s, 3) for s in verifier_scores})
    curve = []
    for thresh in thresholds:
        kept = [(l, vs) for l, vs in zip(labels, verifier_scores) if vs >= thresh]
        tp = sum(l for l, _ in kept)
        fp = len(kept) - tp
        recall = tp / n_tp_total if n_tp_total > 0 else 0.0
        fp_reduction = (n_fp_total - fp) / n_fp_total if n_fp_total > 0 else 0.0
        curve.append({
            "verifier_threshold": thresh,
            "tp": tp,
            "fp": fp,
            "recall": round(recall, 4),
            "fp_reduction_pct": round(fp_reduction * 100, 2),
            "detections_kept": len(kept),
        })

    # Find operating points
    ops: dict[float, dict] = {}
    for target in sorted(recall_targets, reverse=True):
        # Highest threshold that keeps recall >= target
        for row in sorted(curve, key=lambda r: r["verifier_threshold"], reverse=True):
            if row["recall"] >= target:
                ops[target] = {**row, "recall_target": target}
                break
        else:
            ops[target] = {"recall_target": target, "note": "no threshold achieves this recall"}

    return curve, baseline, ops


def main() -> int:
    root = repo_root()
    args = build_parser().parse_args()

    model_path = resolve_path(args.model, root)
    manifest_path = resolve_path(args.manifest, root)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    recall_targets = [float(t.strip()) for t in args.recall_targets.split(",")]

    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        print("Empty manifest.")
        return 0

    import torch
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}  |  Crops to score: {len(records)}")

    labels = [rec["label"] for rec in records]
    detector_scores = [rec["detector_score"] for rec in records]

    print("Running verifier inference...")
    verifier_scores = score_crops(model_path, records, args.batch, device)

    curve, baseline, ops = compute_operating_points(
        labels, detector_scores, verifier_scores, recall_targets
    )

    n_tp = sum(labels)
    n_fp = len(labels) - n_tp
    print(f"\nDetector baseline:  TP={n_tp}  FP={n_fp}  recall=1.000 (by definition, within proposals)")
    print(f"\nVerifier operating points:")
    print(f"{'target':>7}  {'thresh':>7}  {'recall':>7}  {'tp':>5}  {'fp':>5}  {'fp_reduc%':>10}")
    print("-" * 55)
    for tgt in sorted(recall_targets, reverse=True):
        op = ops.get(tgt, {})
        if "note" in op:
            print(f"{tgt:>7.2f}  {'—':>7}  {'—':>7}  {'—':>5}  {'—':>5}  {op['note']}")
        else:
            print(
                f"{op['recall_target']:>7.2f}"
                f"  {op['verifier_threshold']:>7.3f}"
                f"  {op['recall']:>7.4f}"
                f"  {op['tp']:>5}"
                f"  {op['fp']:>5}"
                f"  {op['fp_reduction_pct']:>10.1f}%"
            )

    # Save results
    out_dir = Path(args.output) if args.output else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_stem = manifest_path.stem.replace("crop_manifest_", "")

    curve_path = out_dir / f"eval_curve_{manifest_stem}.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    ops_path = out_dir / f"eval_ops_{manifest_stem}.json"
    ops_path.write_text(
        json.dumps({"baseline": baseline, "operating_points": {str(k): v for k, v in ops.items()}}, indent=2),
        encoding="utf-8",
    )

    # Also save per-detection scores for further analysis
    scored_path = out_dir / f"scored_{manifest_stem}.jsonl"
    with scored_path.open("w", encoding="utf-8") as fh:
        for rec, vs in zip(records, verifier_scores):
            fh.write(json.dumps({**rec, "verifier_score": round(vs, 6)}) + "\n")

    print(f"\nCurve:   {curve_path}")
    print(f"Ops:     {ops_path}")
    print(f"Scored:  {scored_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
