#!/usr/bin/env python3
"""Stage A hybrid reconstruction: RGB-anchored compositing with event-gated
e2vid injection.

Per output frame:
  1. RGB frame is the base layer (sharp static background).
  2. Events in the matched ~33ms window -> activity mask (drone region).
  3. Nearest e2vid reconstruction, tone-matched to RGB luminance on a ring
     around the mask, replaces the RGB luminance inside the feathered mask.
     Chroma stays RGB.

Run from this folder (with data_seq<N>/e2vid_seq<N> symlinked):  python composite.py --seq 18
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

W, H = 1280, 720
DATA = "data_seq18"
E2VID = "e2vid_seq18"


def set_seq(seq):
    global DATA, E2VID
    DATA = f"data_seq{seq}"
    E2VID = f"e2vid_seq{seq}"


def load_index(seq="18"):
    """Manifest rows + e2vid frame-name -> real-time mapping.

    e2vid output frames are named by the index of the last event in each
    1/120s window, not by timestamp; map via the raw event timestamps."""
    rows = list(csv.DictReader(open(f"{DATA}/paired/manifest_val.csv")))
    names_f, t_f = f"out/e2vid_names_{seq}.npy", f"out/e2vid_frame_t_{seq}.npy"
    if not (os.path.exists(names_f) and os.path.exists(t_f)):
        print("building e2vid time mapping for seq", seq)
        ts = np.loadtxt(f"{DATA}/eventRaw/events.txt", delimiter=",",
                        usecols=3, dtype=np.int64)
        names = np.sort(np.array(
            [int(os.path.basename(p)[6:-4])
             for p in glob.glob(f"{E2VID}/frame_*.png")]))
        frame_t = ts[np.clip(names, 0, len(ts) - 1)] / 1e6
        np.save(names_f, names)
        np.save(t_f, frame_t)
    return rows, np.load(names_f), np.load(t_f)


def load_events(path):
    ev = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    return ev  # columns: x, y, polarity, t_us


def activity_mask(ev, blur_sigma=6.0, rel_thresh=0.18, min_events=120,
                  min_peak=0.12, window_us=15000):
    """Event count map -> smoothed density -> threshold -> keep dense blobs.

    Only the trailing `window_us` of the event window is used so the mask
    hugs the drone's current position instead of its 33ms motion trail.
    """
    if window_us and len(ev):
        ev = ev[ev[:, 3] >= ev[-1, 3] - window_us]
    cnt = np.zeros((H, W), np.float32)
    np.add.at(cnt, (ev[:, 1], ev[:, 0]), 1.0)
    dens = cv2.GaussianBlur(cnt, (0, 0), blur_sigma)
    peak = dens.max()
    if peak <= 0:
        return np.zeros((H, W), np.uint8)
    m = (dens > rel_thresh * peak).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # keep only components with real event mass and density: sensor noise and
    # wind-blown vegetation sit at mass<~100 / peak<~0.06 ev/px, while the
    # drone (even hovering) shows mass>~180 and peak>~0.25
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 8000:
            continue
        comp = lab == i
        if cnt[comp].sum() >= min_events and dens[comp].max() >= min_peak:
            keep[comp] = 1
    return cv2.dilate(keep, np.ones((13, 13), np.uint8))


def estimate_shift(ev, mask, t_rgb_us, t_e2v_us, max_shift=60.0):
    """Translation aligning the e2vid frame to the RGB timestamp.

    Drone velocity from the event slice (centroid of the late half minus
    centroid of the early half, restricted to the mask), extrapolated over
    the known e2vid-frame-to-RGB time gap."""
    if len(ev) < 50 or not mask.any():
        return 0.0, 0.0
    e = ev[mask[ev[:, 1], ev[:, 0]] > 0]
    if len(e) < 50:
        return 0.0, 0.0
    tm = np.median(e[:, 3])
    a, b = e[e[:, 3] <= tm], e[e[:, 3] > tm]
    if len(a) < 20 or len(b) < 20:
        return 0.0, 0.0
    dt = b[:, 3].mean() - a[:, 3].mean()
    if dt <= 0:
        return 0.0, 0.0
    d = t_rgb_us - t_e2v_us
    sx = (b[:, 0].mean() - a[:, 0].mean()) / dt * d
    sy = (b[:, 1].mean() - a[:, 1].mean()) / dt * d
    n = (sx * sx + sy * sy) ** 0.5
    if n > max_shift:
        sx, sy = sx * max_shift / n, sy * max_shift / n
    return float(sx), float(sy)


def tone_match(e2v_y, rgb_y, mask):
    """Linear gain/offset for e2vid luminance, fit on a background ring
    around the mask where both layers should show the same scene."""
    ring = (cv2.dilate(mask, np.ones((41, 41), np.uint8)) > 0) & (
        cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0
    )
    if ring.sum() < 100:
        return e2v_y
    e, r = e2v_y[ring].astype(np.float32), rgb_y[ring].astype(np.float32)
    se = e.std()
    a = (r.std() / se) if se > 1e-3 else 1.0
    a = float(np.clip(a, 0.2, 5.0))
    b = float(r.mean() - a * e.mean())
    return np.clip(a * e2v_y.astype(np.float32) + b, 0, 255).astype(np.uint8)


def composite(rgb, e2v_gray, mask, feather_sigma=5.0, bg_sigma=15.0):
    """Detail-max fusion inside the mask: keep the RGB low-frequency base
    everywhere, and per pixel take the detail (local contrast) from whichever
    modality is stronger. RGB wins where it is sharp; e2vid fills in where
    RGB is dark or blurred. Never degrades the background."""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0]
    matched = tone_match(e2v_gray, y, mask).astype(np.float32)
    yf = y.astype(np.float32)
    base_r = cv2.GaussianBlur(yf, (0, 0), bg_sigma)
    det_r = yf - base_r
    det_e = matched - cv2.GaussianBlur(matched, (0, 0), bg_sigma)
    sel = np.where(np.abs(det_e) > np.abs(det_r), det_e, det_r)
    alpha = np.clip(
        cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather_sigma), 0, 1)
    fused = base_r + alpha * sel + (1 - alpha) * det_r
    ycc[:, :, 0] = np.clip(fused, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def crop_panel(rgb, evf, e2v_gray, comp, center, cw=320, ch=180, box=None):
    """2x2 grid of drone-centered crops, each upscaled to 960x540."""
    cx = int(np.clip(center[0], cw // 2, W - cw // 2))
    cy = int(np.clip(center[1], ch // 2, H - ch // 2))
    x0, y0 = cx - cw // 2, cy - ch // 2
    e2v = cv2.cvtColor(e2v_gray, cv2.COLOR_GRAY2BGR)
    if box is not None:
        cv2.rectangle(comp, box[:2], box[2:], (0, 200, 255), 1)
    tiles = []
    for im, name in ((rgb, "RGB"), (evf, "Events"),
                     (e2v, "E2VID"), (comp, "Composite")):
        t = cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (960, 540),
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 200, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def panel(rgb, evf, e2v_gray, comp, box=None):
    e2v = cv2.cvtColor(e2v_gray, cv2.COLOR_GRAY2BGR)
    if box is not None:
        for im in (comp,):
            cv2.rectangle(im, box[:2], box[2:], (0, 200, 255), 1)
    top = np.hstack([rgb, evf])
    bot = np.hstack([e2v, comp])
    grid = np.vstack([top, bot])
    return cv2.resize(grid, (1920, 1080))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="18")
    ap.add_argument("--t0", type=float, default=8.0)
    ap.add_argument("--t1", type=float, default=20.0)
    ap.add_argument("--video", default="out/composite_grid.mp4")
    ap.add_argument("--dump-frames", type=int, default=0,
                    help="also save N evenly spaced grid stills")
    ap.add_argument("--crop", action="store_true",
                    help="drone-centered crop view instead of full frames")
    ap.add_argument("--no-box", action="store_true",
                    help="don't draw the label box on the composite tile")
    ap.add_argument("--motion-comp", action="store_true",
                    help="shift e2vid to the RGB timestamp using event "
                         "velocity (fixes the 120fps quantization offset)")
    args = ap.parse_args()

    set_seq(args.seq)
    rows, names, frame_t = load_index(args.seq)
    rows = [r for r in rows if args.t0 <= float(r["event_time_s"]) <= args.t1]
    print(f"{len(rows)} frames in [{args.t0}, {args.t1}]s")

    vw = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                         (1920, 1080))
    clean_path = args.video.replace(".mp4", "_clean.mp4")
    vw_clean = None
    if not args.crop:
        vw_clean = cv2.VideoWriter(clean_path,
                                   cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                   (W, H))
    center = None  # EMA-smoothed drone-centered crop position
    dump_at = set(np.linspace(0, len(rows) - 1, args.dump_frames).astype(int)
                  ) if args.dump_frames else set()

    for k, r in enumerate(rows):
        t = float(r["event_time_s"])
        # composite onto the RGB base layer, so target its exact timestamp:
        # the drone moves fast enough (>10 px/ms) that the ~5ms label offset
        # would otherwise ghost the injected detail next to the RGB drone
        t_rgb = float(r["rgb_time_s"])
        rgb = cv2.imread(f"{DATA}/{r['rgb_image']}")
        evf = cv2.imread(f"{DATA}/{r['event_image']}")
        j = int(np.abs(frame_t - t_rgb).argmin())
        e2v = cv2.imread(f"{E2VID}/frame_{names[j]:010d}.png",
                         cv2.IMREAD_GRAYSCALE)
        stem = os.path.splitext(os.path.basename(r["rgb_image"]))[0]
        ev = load_events(f"{DATA}/eventMatched/val/{stem}.txt")
        # slice of the 33ms window nearest the RGB timestamp (the window can
        # end before rgb_time, e.g. seq31 where rgb lags the label by ~6ms)
        rgb_us = int(t_rgb * 1e6)
        t_hi = min(rgb_us + 5000, int(ev[-1, 3])) if len(ev) else rgb_us
        ev = ev[(ev[:, 3] >= t_hi - 15000) & (ev[:, 3] <= t_hi)]
        mask = activity_mask(ev, window_us=0)
        if args.motion_comp:
            sx, sy = estimate_shift(ev, mask, rgb_us,
                                    int(frame_t[j] * 1e6))
            if sx or sy:
                M = np.float32([[1, 0, sx], [0, 1, sy]])
                e2v = cv2.warpAffine(e2v, M, (W, H),
                                     borderMode=cv2.BORDER_REPLICATE)
        comp = composite(rgb, e2v, mask)
        if vw_clean is not None:
            vw_clean.write(comp)

        lab = open(f"{DATA}/{r['rgb_label']}").read().split()
        cx, cy, bw, bh = (float(v) for v in lab[1:5])
        box = None if args.no_box else (
            int((cx - bw / 2) * W), int((cy - bh / 2) * H),
            int((cx + bw / 2) * W), int((cy + bh / 2) * H))
        if args.crop:
            if mask.any():
                ys, xs = np.nonzero(mask)
                target = (xs.mean(), ys.mean())
            else:
                target = (cx * W, cy * H)
            center = target if center is None else (
                0.7 * center[0] + 0.3 * target[0],
                0.7 * center[1] + 0.3 * target[1])
            grid = crop_panel(rgb, evf, e2v, comp, center, box=box)
        else:
            grid = panel(rgb, evf, e2v, comp, box)
        vw.write(grid)
        if k in dump_at:
            cv2.imwrite(f"out/grid_{k:04d}_t{t:.2f}.jpg", grid)
        if k % 50 == 0:
            print(f"  {k}/{len(rows)} t={t:.2f}s mask_px={int(mask.sum())}")
    vw.release()
    if vw_clean is not None:
        vw_clean.release()
        print("wrote", args.video, "and", clean_path)
    else:
        print("wrote", args.video)


if __name__ == "__main__":
    main()
