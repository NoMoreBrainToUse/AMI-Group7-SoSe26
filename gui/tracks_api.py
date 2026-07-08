"""Kalman tracking overlay data for the GUI.

Runs the tuned MultiDroneTracker (gui/tracker.py, Celia's constant-
acceleration filter from the previous repo) once over the pipeline's kept
detections and emits one compact JSON record per frame. The old GUI rendered
these overlays into thousands of JPEGs; the new frontend draws them on a
canvas instead, so tracking is available immediately after a pipeline run.

Per frame:
  dets:      kept detection boxes (xyxy)
  gt:        current GT centroids
  gt_future: next-second GT path per GT object (matched by frame order)
  tentative: warming-up track positions
  tracks:    active tracks {id, pos, vel, speed, future:[...]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tracker import MultiDroneTracker

FUTURE_FRAMES = 30
FPS = 30.0


def _make_tracker() -> MultiDroneTracker:
    # Tuned blind-test parameters from the previous repo's Tracking tab.
    return MultiDroneTracker(
        dt=1 / FPS,
        Q_base=np.diag([11.442, 11.442, 0.00036198, 0.00036198, 2.8337, 2.8337]),
        R_pos=np.diag([1.6883, 1.6883]),
        R_vel=np.diag([0.35511, 0.35511]),
        R_acc=np.diag([10.408, 10.408]),
        alpha=0.95221,
        beta=0.96737,
        acc_coeffs=[-0.13303, 0.52543, 4.0978],
        g_max=483.51,
        future_frames=FUTURE_FRAMES,
    )


def _centroids_norm(gt_boxes_norm: list[list[float]],
                    w: float = 1280, h: float = 720) -> list[list[float]]:
    return [[cx * w, cy * h] for cx, cy, _, _ in gt_boxes_norm]


def build_tracks(detections_jsonl: Path) -> dict:
    records = [json.loads(l) for l in
               Path(detections_jsonl).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    records.sort(key=lambda r: r["frame_index"])

    tracker = _make_tracker()
    frames = []
    for rec in records:
        kept = [d for d in rec["detections"] if d["kept"]]
        positions = [np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                     for b in (d["bbox_xyxy"] for d in kept)]
        results = tracker.step(positions)

        gt_now = _centroids_norm(rec["gt_boxes_norm"])
        # GT future path: same-order centroid across the next second
        idx = rec["frame_index"]
        futures = []
        for g_i in range(len(gt_now)):
            path = []
            for k in range(1, FUTURE_FRAMES + 1):
                if idx + k >= len(records):
                    break
                nxt = _centroids_norm(records[idx + k]["gt_boxes_norm"])
                if g_i < len(nxt):
                    path.append([round(v, 1) for v in nxt[g_i]])
            futures.append(path)

        tracks = []
        for tid, pos, vel, future in results:
            fut_pos, fut_out = future[0], future[1]
            inside = [[round(float(p[0]), 1), round(float(p[1]), 1)]
                      for p, out in zip(fut_pos, fut_out) if not out]
            tracks.append({
                "id": int(tid),
                "pos": [round(float(pos[0]), 1), round(float(pos[1]), 1)],
                "speed": round(float(np.hypot(vel[0], vel[1])), 1),
                "future": inside,
            })

        frames.append({
            "i": idx,
            "dets": [d["bbox_xyxy"] for d in kept],
            "gt": [[round(v, 1) for v in g] for g in gt_now],
            "gt_future": futures,
            "tentative": [[round(float(w["buf"][-1][0]), 1),
                           round(float(w["buf"][-1][1]), 1)]
                          for w in tracker.warmup],
            "tracks": tracks,
        })

    return {"frames": frames, "future_frames": FUTURE_FRAMES, "fps": FPS}
