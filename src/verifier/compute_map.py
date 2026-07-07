#!/usr/bin/env python3
"""
Compute mAP50 and mAP50-95 from fusion_detections JSONL.

Computes three systems side-by-side:
  - Event-YOLO only  (confidence = detector_score)
  - Fusion v4        (confidence = fusion_score)

Reads the per-frame JSONL produced by compute_fusion_metrics.py.
Each detection record must contain:
  fusion_score, detector_score, iou_with_gt, label

Approximation note: gray-zone crops are excluded from the JSONL, so
mAP will be slightly optimistic. For single-drone sequences the duplicate
GT-match issue is negligible (one GT box per frame in most cases).

Usage:
  python src/verifier/compute_map.py \\
    --detections outputs/web/fusion_detections_blind_test_v4.jsonl \\
    [--sequences 40 43 46 49]     # filter; omit for all sequences
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def compute_ap(scores: list[float], is_tp: list[int], n_gt: int) -> float:
    """Area under the precision-recall curve (trapezoidal)."""
    if n_gt == 0 or not scores:
        return 0.0

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    cum_tp, cum_fp = 0, 0
    precs, recs = [], []
    for i in order:
        if is_tp[i]:
            cum_tp += 1
        else:
            cum_fp += 1
        precs.append(cum_tp / (cum_tp + cum_fp))
        recs.append(cum_tp / n_gt)

    # Monotone envelope
    max_p = 0.0
    precs_env = []
    for p in reversed(precs):
        max_p = max(max_p, p)
        precs_env.append(max_p)
    precs_env = list(reversed(precs_env))

    recs_arr = np.array([0.0] + recs + [recs[-1]])
    precs_arr = np.array([1.0] + precs_env + [0.0])
    return float(np.sum((recs_arr[1:] - recs_arr[:-1]) * precs_arr[1:]))


def compute_ap_series(dets: list[dict], score_key: str, n_gt: int) -> dict:
    """Compute mAP50 and mAP50-95 for a given score field."""
    scores = [d[score_key] for d in dets]
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    aps = []
    for iou_t in iou_thresholds:
        is_tp = [1 if d["iou_with_gt"] >= iou_t and d["label"] == 1 else 0
                 for d in dets]
        aps.append(compute_ap(scores, is_tp, n_gt))
    return {
        "AP50":     round(aps[0], 4),
        "mAP50-95": round(float(np.mean(aps)), 4),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute mAP50/mAP50-95: Event-YOLO vs Fusion v4.")
    p.add_argument("--detections", required=True,
                   help="fusion_detections_*.jsonl from compute_fusion_metrics.py")
    p.add_argument("--sequences", nargs="*", type=int, default=None,
                   help="sequence numbers to evaluate (default: all)")
    args = p.parse_args()

    frames = load_jsonl(Path(args.detections))

    # Group by sequence
    seq_frames: dict[str, list] = {}
    for f in frames:
        seq = f.get("sequence", "unknown")
        seq_frames.setdefault(seq, []).append(f)

    wanted = ({f"seq{s}" for s in args.sequences}
              if args.sequences else set(seq_frames))

    header = (f"{'Seq':<8}  {'Frames':>7}  {'GT':>6}  "
              f"{'YOLO AP50':>10}  {'YOLO mAP50-95':>14}  "
              f"{'Fus AP50':>9}  {'Fus mAP50-95':>13}  "
              f"{'ΔAP50':>7}  {'ΔmAP50-95':>10}")
    print(f"\n{header}")
    print("-" * len(header))

    all_yolo: list[dict] = []
    all_fus:  list[dict] = []
    all_n_gt = 0

    for seq in sorted(wanted):
        if seq not in seq_frames:
            print(f"{seq:<8}  (not found in JSONL)")
            continue

        frs  = seq_frames[seq]
        n_gt = sum(len(f.get("gt_boxes_norm", [])) for f in frs)
        dets = [d for f in frs for d in f.get("detections", [])
                if "fusion_score" in d and "iou_with_gt" in d
                   and "detector_score" in d]

        yolo = compute_ap_series(dets, "detector_score", n_gt)
        fus  = compute_ap_series(dets, "fusion_score",   n_gt)

        all_yolo.extend(dets)
        all_fus.extend(dets)
        all_n_gt += n_gt

        d50   = fus["AP50"]     - yolo["AP50"]
        d5095 = fus["mAP50-95"] - yolo["mAP50-95"]

        print(f"{seq:<8}  {len(frs):>7}  {n_gt:>6}  "
              f"{yolo['AP50']:>10.4f}  {yolo['mAP50-95']:>14.4f}  "
              f"{fus['AP50']:>9.4f}  {fus['mAP50-95']:>13.4f}  "
              f"{d50:>+7.4f}  {d5095:>+10.4f}")

    if len(wanted) > 1 and all_yolo:
        oy = compute_ap_series(all_yolo, "detector_score", all_n_gt)
        of = compute_ap_series(all_fus,  "fusion_score",   all_n_gt)
        d50   = of["AP50"]     - oy["AP50"]
        d5095 = of["mAP50-95"] - oy["mAP50-95"]
        print("-" * len(header))
        print(f"{'ALL':<8}  {'':>7}  {all_n_gt:>6}  "
              f"{oy['AP50']:>10.4f}  {oy['mAP50-95']:>14.4f}  "
              f"{of['AP50']:>9.4f}  {of['mAP50-95']:>13.4f}  "
              f"{d50:>+7.4f}  {d5095:>+10.4f}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
