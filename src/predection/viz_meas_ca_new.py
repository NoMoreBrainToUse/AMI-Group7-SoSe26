#!/usr/bin/env python3
"""
viz_meas_ca_new.py — Visualise CA_Meas_kalman on a single video (multi-drone).
Detection source: Event_YOLO_new.

For every frame:
  - Event-camera PNG as background
  - Green box       : Event_YOLO_new detection
  - Magenta dot     : Tentative (not-yet-confirmed) track
  - Coloured dot    : Active KF estimate (per track)
  - Coloured line   : KF predicted future trajectory (inside frame)
  - Hollow circle   : First point where KF prediction exits frame
  - Red line/dot    : Ground-truth future / current position (per GT track)

Output: {VIDEO_ID}/meas_ca_new_viz.mp4
"""

import os, re, sys
import numpy as np

try:
    import cv2
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)   # AMI root — where video folders live
sys.path.insert(0, BASE_DIR)

from CA_Meas_kalman_new import MultiDroneTracker

# =============================================================================
# PARAMETERS  (optimised values)
# =============================================================================

VIDEO_ID = 31

Q_base = np.diag([11.442, 11.442, 0.00036198, 0.00036198, 2.8337, 2.8337])
R_pos  = np.diag([1.6883,  1.6883])
R_vel  = np.diag([0.35511, 0.35511])
R_acc  = np.diag([10.408,  10.408])
alpha        = 0.95221
beta         = 0.96737
g_max        = 483.51
acc_coeffs   = np.array([-0.13303, 0.52543, 4.0978])

FUTURE_FRAMES = 24   # 0.8 s at 30 fps
IMG_W, IMG_H  = 1280, 720

# colours (BGR)
C_BOX        = (  0, 220,   0)   # green  — Event_YOLO_new detection box
C_TENT       = (200,  50, 200)   # magenta — tentative track dot
C_GT_CURRENT = (  0,  80, 255)   # orange-red — current GT position
# GT future colours per GT track ID (red family)
GT_COLORS = [
    (  0,  50, 220),   # red
    (  0, 100, 180),   # darker orange-red
    (  0,  30, 160),   # deep red
    ( 30,  70, 200),   # brownish red
]
# KF active-track colours per assigned track ID (bright, distinct)
TRACK_COLORS = [
    (255, 120,   0),   # blue
    (  0, 165, 255),   # orange
    (180,  20, 255),   # purple
    (  0, 220, 220),   # yellow-green
    (128, 255,   0),   # lime
]

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICK = 1

# =============================================================================
# DATA LOADING
# =============================================================================

def _parse_bbox_file(path):
    from collections import defaultdict
    result = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts_str, rest = line.split(':', 1)
            ts   = round(float(ts_str), 5)
            vals = [float(v) for v in rest.split(',')]
            x1, y1, x2, y2 = vals[0], vals[1], vals[2], vals[3]
            tid  = int(round(vals[4])) if len(vals) >= 5 else 1
            result[tid][ts] = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    return dict(result)


def _load_event_yolo(vdir):
    """Returns {ts_s: [(cx, cy, bw, bh), ...]} for all frames with detections."""
    yolo_dir = os.path.join(vdir, 'Event_YOLO_new')
    dets = {}
    for fname in os.listdir(yolo_dir):
        m = re.search(r'frame_(\d+)\.txt', fname)
        if not m:
            continue
        ts = round(int(m.group(1)) / 1e6, 5)
        boxes = []
        with open(os.path.join(yolo_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx = float(parts[1]) * IMG_W
                    cy = float(parts[2]) * IMG_H
                    bw = float(parts[3]) * IMG_W
                    bh = float(parts[4]) * IMG_H
                    boxes.append((cx, cy, bw, bh))
        if boxes:
            dets[ts] = boxes
    return dets


def _ts_from_name(name):
    m = re.search(r'frame_(\d+)', name)
    return int(m.group(1)) if m else 0

# =============================================================================
# DRAW HELPERS
# =============================================================================

def draw_trajectory(frame, pts, colour, radius=3, thickness=2):
    pts = [p for p in pts if 0 <= p[0] <= IMG_W and 0 <= p[1] <= IMG_H]
    if not pts:
        return
    for p in pts:
        cv2.circle(frame, (int(p[0]), int(p[1])), radius, colour, -1)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, thickness)


def gt_future_pts(gt_ts, gt_pos, ts_now, dt, n_frames):
    """Return list of GT positions for the next n_frames steps from ts_now."""
    pts = []
    for k in range(1, n_frames + 1):
        target = ts_now + k * dt
        fi = int(np.searchsorted(gt_ts, target))
        if fi >= len(gt_ts):
            break
        if abs(gt_ts[fi] - target) <= 1.5 * dt:
            pts.append(gt_pos[fi])
    return pts


def gt_current_pos(gt_ts, gt_pos, ts_now, dt):
    """Return GT position at ts_now (within 1.5 frames), or None."""
    fi = int(np.searchsorted(gt_ts, ts_now))
    if fi < len(gt_ts) and abs(gt_ts[fi] - ts_now) <= 1.5 * dt:
        return gt_pos[fi]
    if fi > 0 and abs(gt_ts[fi - 1] - ts_now) <= 1.5 * dt:
        return gt_pos[fi - 1]
    return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    vdir       = os.path.join(DATA_DIR, str(VIDEO_ID))
    frames_dir = os.path.join(vdir, 'Event', 'Frames')
    out_path   = os.path.join(vdir, 'meas_ca_new_viz.mp4')

    # --- Ground truth (all track IDs) ----------------------------------------
    gt_all = _parse_bbox_file(os.path.join(vdir, 'interpolated_coordinates.txt'))
    gt_arrs = {}
    for tid, gd in gt_all.items():
        ts_sorted = sorted(gd.keys())
        gt_arrs[tid] = {
            'ts':  np.array(ts_sorted),
            'pos': np.array([gd[t] for t in ts_sorted]),
        }

    # --- Event_YOLO_new detections -------------------------------------------
    ev_all = _load_event_yolo(vdir)

    # --- Frame list and timing -----------------------------------------------
    pngs = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith('.png')],
        key=_ts_from_name,
    )
    if len(pngs) < 2:
        print("Not enough frames found.")
        return

    ts_us_arr = np.array([_ts_from_name(p) for p in pngs])
    dt        = float(np.median(np.diff(ts_us_arr))) / 1e6
    fps       = round(1.0 / dt)

    # --- Tracker -------------------------------------------------------------
    tracker = MultiDroneTracker(
        dt            = dt,
        Q_base        = Q_base,
        R_pos         = R_pos,
        R_vel         = R_vel,
        R_acc         = R_acc,
        alpha         = alpha,
        beta          = beta,
        acc_coeffs    = acc_coeffs,
        g_max         = g_max,
        future_frames = FUTURE_FRAMES,
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (IMG_W, IMG_H))

    print(f"Processing {len(pngs)} frames for video {VIDEO_ID} (dt={dt*1000:.2f} ms, {fps} fps) ...")

    for i, png_name in enumerate(pngs):
        ts_us = _ts_from_name(png_name)
        ts_s  = round(ts_us / 1e6, 5)

        frame = cv2.imread(os.path.join(frames_dir, png_name))
        if frame is None:
            continue

        # --- Detections at this frame ----------------------------------------
        raw_dets = ev_all.get(ts_s, [])            # [(cx, cy, bw, bh), ...]
        det_positions = [np.array([d[0], d[1]]) for d in raw_dets]

        # --- Step tracker ----------------------------------------------------
        results = tracker.step(det_positions)      # [(tid, pos, vel, fut), ...]

        # --- Draw detection boxes (green) ------------------------------------
        for cx, cy, bw, bh in raw_dets:
            x1 = int(cx - bw / 2); y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2); y2 = int(cy + bh / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_BOX, 2)

        # --- Draw GT (current position + future trajectory) ------------------
        for gt_idx, (tid, ga) in enumerate(sorted(gt_arrs.items())):
            c_gt = GT_COLORS[gt_idx % len(GT_COLORS)]
            cur  = gt_current_pos(ga['ts'], ga['pos'], ts_s, dt)
            if cur is not None:
                cv2.circle(frame, (int(cur[0]), int(cur[1])), 5, C_GT_CURRENT, -1)
            fut_pts = gt_future_pts(ga['ts'], ga['pos'], ts_s, dt, FUTURE_FRAMES)
            draw_trajectory(frame, fut_pts, c_gt, radius=3, thickness=2)

        # --- Draw tentative tracks -------------------------------------------
        for tent in tracker.warmup:
            p = tent['buf'][-1]
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, C_TENT, -1)

        # --- Draw active KF tracks -------------------------------------------
        for tid, pos, vel, (fut_pos, fut_out, fut_edge, fut_ellipses) in results:
            c = TRACK_COLORS[tid % len(TRACK_COLORS)]

            # KF current estimate
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 7, c, -1)

            # Predicted future: inside portion as filled dots + line
            inside_pts = [p for p, o in zip(fut_pos, fut_out) if not o]
            draw_trajectory(frame, inside_pts, c, radius=3, thickness=2)

            # Mark first exit point with a hollow circle
            for p, out, edge in zip(fut_pos, fut_out, fut_edge):
                if out:
                    cv2.circle(frame, (int(edge[0]), int(edge[1])), 6, c, 1)
                    break

            # 80% certainty ellipses at 0.1 s, 0.2 s, 0.3 s, ... (every 3 steps at 30 fps)
            for k, (p, out, (a, b, angle)) in enumerate(
                    zip(fut_pos, fut_out, fut_ellipses)):
                if (k + 1) % 3 != 0:
                    continue
                if out:
                    continue
                center = (int(round(p[0])), int(round(p[1])))
                axes   = (max(1, int(round(a))), max(1, int(round(b))))
                cv2.ellipse(frame, center, axes, angle, 0, 360, c, 1)

            # Speed label next to dot
            speed = float(np.hypot(vel[0], vel[1]))
            cv2.putText(frame,
                        f"T{tid} {speed:.0f}px/s",
                        (int(pos[0]) + 8, int(pos[1]) - 8),
                        FONT, FONT_SCALE, c, FONT_THICK, cv2.LINE_AA)

        # --- Legend / timestamp overlay --------------------------------------
        n_active  = len(results)
        n_tent    = len(tracker.warmup)
        cv2.putText(frame,
                    f"t={ts_s:.3f}s  frame {i+1}/{len(pngs)}  "
                    f"active={n_active}  tentative={n_tent}",
                    (8, 20), FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)
        cv2.putText(frame,
                    "green=det(new)  magenta=tent  colour=KF pred  red=GT future  orange=GT now",
                    (8, 40), FONT, FONT_SCALE, (200, 200, 200), FONT_THICK, cv2.LINE_AA)

        writer.write(frame)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pngs)} frames ...", flush=True)

    writer.release()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nDone → {out_path}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
