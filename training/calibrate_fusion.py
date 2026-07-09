#!/usr/bin/env python3
"""Calibrate the fusion operating point (lambda, tau) on validation data.

Sweeps lambda over a grid; for each lambda finds the highest threshold that
still reaches --recall-target. Recall is TRUE recall when --n-missed-gt is
given (GT boxes the detectors never proposed — take n_missed_gt from the
pipeline's merge_stats.json); without it the sweep optimizes recall
conditioned on the detector having fired, which is how a degenerate
tau=0.002 was once selected.

Writes a fusion config consumable by the pipeline (weights/fusion_config.json
format: best_lambda / best_threshold).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_vision.common import logit, read_jsonl, sigmoid  # noqa: E402


def threshold_at_recall(labels: list[int], scores: list[float],
                        target: float, n_missed: int) -> tuple[float, float, int, int]:
    """Highest threshold with recall >= target; returns (t, recall, tp, fp)."""
    n_pos = sum(labels) + n_missed
    best = (min(scores), 1.0, sum(labels), len(labels) - sum(labels))
    for t in sorted(set(scores), reverse=True):
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= t)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= t)
        recall = tp / n_pos if n_pos else 0.0
        if recall >= target:
            return t, recall, tp, fp
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep lambda/threshold for verifier fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rgb-scored", type=Path, required=True)
    parser.add_argument("--event-scored", type=Path, required=True)
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--n-missed-gt", type=int, default=0,
                        help="GT boxes with no proposal at all "
                             "(merge_stats.json: n_missed_gt). Makes the "
                             "recall target a TRUE recall target.")
    parser.add_argument("--lambdas", default="0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rgb = read_jsonl(args.rgb_scored)
    evt = read_jsonl(args.event_scored)
    if len(rgb) != len(evt) or any(
            a["crop_id"] != b["crop_id"] for a, b in zip(rgb, evt)):
        raise ValueError("Scored files do not pair up (crop_id mismatch)")

    labels = [r["label"] for r in rgb]
    n_fp = len(labels) - sum(labels)
    rows = []
    print(f"records={len(labels)} pos={sum(labels)} neg={n_fp} "
          f"missed_gt={args.n_missed_gt}")
    print(f"{'lambda':>7} {'tau':>9} {'recall':>7} {'tp':>6} {'fp':>6} {'fp_cut%':>8}")
    for lam in (float(v) for v in args.lambdas.split(",")):
        fusedscores = [
            sigmoid(logit(r["rgb_verifier_score"])
                    + lam * logit(e["event_verifier_score"]))
            for r, e in zip(rgb, evt)]
        t, recall, tp, fp = threshold_at_recall(
            labels, fusedscores, args.recall_target, args.n_missed_gt)
        cut = (n_fp - fp) / n_fp * 100 if n_fp else 0.0
        print(f"{lam:>7.2f} {t:>9.4f} {recall:>7.4f} {tp:>6} {fp:>6} {cut:>7.1f}%")
        rows.append({"lambda": lam, "threshold": t, "recall": recall,
                     "tp": tp, "fp": fp, "fp_reduction_pct": round(cut, 2)})

    best = max(rows, key=lambda r: r["fp_reduction_pct"])
    result = {
        "best_lambda": best["lambda"],
        "best_threshold": best["threshold"],
        "recall_target": args.recall_target,
        "n_missed_gt": args.n_missed_gt,
        "true_recall_denominator": args.n_missed_gt > 0,
        "sweep": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nbest: lambda={best['lambda']} tau={best['threshold']:.4f}"
          f" -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
