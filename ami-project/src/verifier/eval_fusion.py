#!/usr/bin/env python3
"""
Late fusion of RGB and event verifier scores in logit space.

Takes two scored JSONL files (one per modality, from eval_verifier.py) and
combines them as:

  logit_rgb   = log(s_rgb / (1 - s_rgb))
  logit_event = log(s_event / (1 - s_event))
  final_score = sigmoid(logit_rgb + lambda * logit_event)

Records are matched by position — both files must come from manifests generated
from the same detections JSONL (same order, same number of rows).

Sweeps lambda over a grid to find the best FP reduction at 95% recall, then
prints a comparison table:

  System                  | Recall | FP   | FP reduction
  ------------------------|--------|------|-------------
  Detector only           | 1.000  | 1881 |    —
  RGB verifier            | 0.950  |  930 |  50.6%
  Event verifier          | 0.950  |  960 |  49.0%
  RGB + Event (best lambda)| 0.950  |  820 |  56.4%

Example:
  # Score event crops first (if not done yet):
  python src/verifier/eval_verifier.py \\
    --model runs/verifier/rgb/event_mobilenet_v3_small/best.pt \\
    --manifest processed/fred_subset/crops/event/crop_manifest_event_test_conf0.20.jsonl

  # Then fuse:
  python src/verifier/eval_fusion.py \\
    --rgb-scored   runs/verifier/rgb/mobilenet_v3_small_v2/scored_test_conf0.20.jsonl \\
    --event-scored runs/verifier/rgb/event_mobilenet_v3_small/scored_event_test_conf0.20.jsonl \\
    --recall-target 0.95
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def fp_reduction_at_recall(
    labels: list[int],
    scores: list[float],
    recall_target: float,
) -> tuple[float, float, int, int]:
    """Return (best_threshold, achieved_recall, tp_kept, fp_kept)."""
    n_tp = sum(labels)
    n_fp = len(labels) - n_tp
    if n_tp == 0:
        return 0.5, 0.0, 0, n_fp

    thresholds = sorted(set(scores), reverse=True)
    for thresh in thresholds:
        tp = sum(1 for l, s in zip(labels, scores) if s >= thresh and l == 1)
        fp = sum(1 for l, s in zip(labels, scores) if s >= thresh and l == 0)
        recall = tp / n_tp
        if recall >= recall_target:
            return thresh, recall, tp, fp

    # fallback: keep everything
    return min(scores), 1.0, n_tp, n_fp


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Late fusion of RGB and event verifier scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rgb-scored", required=True, help="scored_*.jsonl from eval_verifier.py (RGB model).")
    parser.add_argument("--event-scored", required=True, help="scored_*.jsonl from eval_verifier.py (event model).")
    parser.add_argument(
        "--recall-target", type=float, default=0.95,
        help="Recall level at which to compare FP counts.",
    )
    parser.add_argument(
        "--lambdas",
        default="0.25,0.5,0.75,1.0,1.5,2.0",
        help="Comma-separated lambda values to sweep (weight on event logit).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path for results JSON. Defaults to same dir as --rgb-scored.",
    )
    return parser


def load_scored(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    root = repo_root()
    args = build_parser().parse_args()

    rgb_path = resolve_path(args.rgb_scored, root)
    event_path = resolve_path(args.event_scored, root)

    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB scored file not found: {rgb_path}")
    if not event_path.is_file():
        raise FileNotFoundError(f"Event scored file not found: {event_path}")

    rgb_recs = load_scored(rgb_path)
    event_recs = load_scored(event_path)

    if len(rgb_recs) != len(event_recs):
        raise ValueError(
            f"Record count mismatch: RGB={len(rgb_recs)} event={len(event_recs)}.\n"
            "Both files must be generated from the same detections JSONL."
        )

    # Sanity-check alignment on a sample of records
    mismatches = sum(
        1 for a, b in zip(rgb_recs, event_recs) if a["stem"] != b["stem"]
    )
    if mismatches > 0:
        raise ValueError(f"{mismatches} stem mismatches between RGB and event scored files — check input order.")

    labels = [r["label"] for r in rgb_recs]
    rgb_scores = [r["verifier_score"] for r in rgb_recs]
    event_scores = [r["verifier_score"] for r in event_recs]

    n_tp = sum(labels)
    n_fp = len(labels) - n_tp
    recall_target = args.recall_target
    lambdas = [float(v.strip()) for v in args.lambdas.split(",")]

    print(f"Records: {len(labels)}  TP: {n_tp}  FP: {n_fp}")
    print(f"Recall target: {recall_target}\n")

    # Detector-only baseline (all proposals kept)
    print(f"{'System':<32} {'λ':>5}  {'thresh':>7}  {'recall':>7}  {'tp':>5}  {'fp':>5}  {'fp_reduc%':>10}")
    print("-" * 72)
    print(f"{'Detector only (all proposals)':<32} {'—':>5}  {'—':>7}  {'1.000':>7}  {n_tp:>5}  {n_fp:>5}  {'0.0%':>10}")

    # RGB-only
    t, rec, tp, fp = fp_reduction_at_recall(labels, rgb_scores, recall_target)
    fp_red = (n_fp - fp) / n_fp * 100 if n_fp > 0 else 0.0
    print(f"{'RGB verifier only':<32} {'—':>5}  {t:>7.3f}  {rec:>7.4f}  {tp:>5}  {fp:>5}  {fp_red:>9.1f}%")

    # Event-only
    t, rec, tp, fp = fp_reduction_at_recall(labels, event_scores, recall_target)
    fp_red = (n_fp - fp) / n_fp * 100 if n_fp > 0 else 0.0
    print(f"{'Event verifier only':<32} {'—':>5}  {t:>7.3f}  {rec:>7.4f}  {tp:>5}  {fp:>5}  {fp_red:>9.1f}%")

    # Fusion sweep
    best = {"fp_red": -1.0, "lambda": None, "thresh": None, "recall": None, "tp": None, "fp": None}
    fusion_rows = []
    for lam in lambdas:
        fused = [sigmoid(logit(sr) + lam * logit(se)) for sr, se in zip(rgb_scores, event_scores)]
        t, rec, tp, fp = fp_reduction_at_recall(labels, fused, recall_target)
        fp_red = (n_fp - fp) / n_fp * 100 if n_fp > 0 else 0.0
        label = f"RGB + Event fusion"
        print(f"  {label:<30} {lam:>5.2f}  {t:>7.3f}  {rec:>7.4f}  {tp:>5}  {fp:>5}  {fp_red:>9.1f}%")
        fusion_rows.append({"lambda": lam, "threshold": t, "recall": rec, "tp": tp, "fp": fp, "fp_reduction_pct": round(fp_red, 2)})
        if fp_red > best["fp_red"]:
            best.update({"fp_red": fp_red, "lambda": lam, "thresh": t, "recall": rec, "tp": tp, "fp": fp})

    print(f"\nBest fusion: λ={best['lambda']}  FP reduction={best['fp_red']:.1f}%  "
          f"recall={best['recall']:.4f}  fp={best['fp']}")

    # Save results
    out_dir = Path(args.output) if args.output else rgb_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = rgb_path.stem.replace("scored_", "")
    results = {
        "recall_target": recall_target,
        "n_tp": n_tp, "n_fp": n_fp,
        "rgb_scored": str(rgb_path),
        "event_scored": str(event_path),
        "fusion_sweep": fusion_rows,
        "best_lambda": best["lambda"],
        "best_threshold": best["thresh"],
    }
    out_path = out_dir / f"fusion_results_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
