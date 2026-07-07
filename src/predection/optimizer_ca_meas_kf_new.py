#!/usr/bin/env python3
"""
optimizer_ca_meas_kf_new.py — Offline optimisation of noise parameters for
CA_Meas_kalman.py.  Detection source: Event_YOLO_new.

11 free parameters
─────────────────────────────────────────────────────────────────
  Index  Name         Space    Description
─────────────────────────────────────────────────────────────────
  0      q_pos        log      position process noise base
  1      q_vel        log      velocity process noise base
  2      q_acc        log      acceleration process noise base
  3      r_pos        log      position measurement noise
  4      r_vel        log      velocity measurement noise
  5      r_acc        log      acceleration measurement noise
  6      alpha        natural  acceleration scaling per future step
  7      r_acc_c1     natural  acc R polynomial — 1st order
  8      r_acc_c2     natural  acc R polynomial — 2nd order
  9      r_acc_c3     natural  acc R polynomial — 3rd order
  10     g_max        log      polynomial output ceiling
─────────────────────────────────────────────────────────────────

Polynomials (3rd order in acceleration magnitude only, no speed/bbox term):
    g = 1 + c1*aₙ + c2*aₙ² + c3*aₙ³,  clipped to [1, g_max]
    aₙ = |a| / A_REF

For R_acc the input is the *measured* acceleration magnitude (|acc_meas|/A_REF)
computed from consecutive detections, so g_r_acc tracks the noise level of the
noisy finite-difference measurements rather than the smoothed state estimate.

Future prediction: each step scales [ax, ay] *= alpha before propagating.
"""

import os, sys, re, time, subprocess
import multiprocessing as mp
import numpy as np
from collections import defaultdict

try:
    import cma
except ModuleNotFoundError:
    print(f"Installing cma into {sys.executable} ...", flush=True)
    for _pip_args in (
        [sys.executable, '-m', 'pip', 'install', 'cma'],
        [sys.executable, '-m', 'pip', 'install', 'cma', '--break-system-packages'],
    ):
        if subprocess.call(_pip_args) == 0:
            break
    import cma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)   # AMI root — where video folders live
sys.path.insert(0, BASE_DIR)

from CA_Meas_kalman_new import (
    MeasCADroneTrack,
    make_F,
    A_REF,
)

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

TRAIN_IDS = [14, 31, 33, 79]
TEST_IDS  = [0, 21]

WARMUP = 5

HORIZONS_S      = [0.2, 0.4, 0.8]
HORIZON_WEIGHTS = [1/3, 1/3, 1/3]

W_AVG   = 1
W_FINAL = 0

STALL_WINDOW    = 300
STALL_THRESHOLD = 0.1
MIN_EVALS       = STALL_WINDOW * 2

EVENT_IMG_W = 1280
EVENT_IMG_H = 720

ASSIGN_RADIUS = 150.0   # px — max GT↔detection distance for assignment

# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------
PARAM_NAMES = [
    "q_pos", "q_vel", "q_acc",
    "r_pos", "r_vel", "r_acc",
    "alpha",
    "beta",
    "r_acc_c1", "r_acc_c2", "r_acc_c3",
    "g_max",
]

N_PARAMS = len(PARAM_NAMES)
assert N_PARAMS == 12, f"Expected 12 params, got {N_PARAMS}"

LOG_PARAM_INDICES = frozenset([0, 1, 2, 3, 4, 5, 11])   # g_max at index 11

# ---------------------------------------------------------------------------
# Bounds (natural space)
# ---------------------------------------------------------------------------
_B_POLY = [-20.0, 100.0]

# PARAM_BOUNDS = np.array([
#     [0.001,   10.0],   # q_pos
#     [0.01,    70.0],   # q_vel
#     [0.001,   10.0],   # q_acc
#     [0.001,   50.0],   # r_pos
#     [0.01,   500.0],   # r_vel
#     [0.1,   5000.0],   # r_acc
#     [0.0,      1.5],   # alpha
#     _B_POLY,           # r_acc_c1
#     _B_POLY,           # r_acc_c2
#     _B_POLY,           # r_acc_c3
#     [2.0,   100.0],    # g_max
# ])
PARAM_BOUNDS = None  # set to the array above to re-enable bounds penalty

PENALTY_WEIGHT = 1e4


# =============================================================================
# SECTION 2 — DATA LOADING
# =============================================================================

def _parse_bbox_file(path: str) -> dict:
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
            result[tid][ts] = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    return dict(result)


def _load_event_yolo(video_id: int) -> dict:
    yolo_dir = os.path.join(DATA_DIR, str(video_id), 'Event_YOLO_new')
    dets: dict = {}
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
                    cx = float(parts[1]) * EVENT_IMG_W
                    cy = float(parts[2]) * EVENT_IMG_H
                    bw = float(parts[3]) * EVENT_IMG_W
                    bh = float(parts[4]) * EVENT_IMG_H
                    boxes.append(np.array([cx, cy, float(np.sqrt(bw * bh))]))
        if boxes:
            dets[ts] = boxes
    return dets


def load_video_tracks_event(video_id: int) -> list:
    from scipy.optimize import linear_sum_assignment

    vdir   = os.path.join(DATA_DIR, str(video_id))
    gt_all = _parse_bbox_file(os.path.join(vdir, 'interpolated_coordinates.txt'))
    ev_all = _load_event_yolo(video_id)

    # Build {ts_r: {tid: pos}} for all GT tracks
    gt_by_ts = defaultdict(dict)
    for tid, gt_dict in gt_all.items():
        for ts, pos in gt_dict.items():
            gt_by_ts[round(ts, 5)][tid] = pos

    # Per-timestamp: match each detection to at most one GT track (Hungarian, radius-gated)
    det_lookups = {tid: {} for tid in gt_all}
    for ts_r, dets in ev_all.items():
        if ts_r not in gt_by_ts:
            continue
        gt_at_t = gt_by_ts[ts_r]
        tids    = list(gt_at_t.keys())
        cost    = np.full((len(dets), len(tids)), ASSIGN_RADIUS + 1.0)
        for di, d in enumerate(dets):
            for ti, tid in enumerate(tids):
                dist = np.linalg.norm(d[:2] - gt_at_t[tid])
                cost[di, ti] = dist
        if cost.size > 0:
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= ASSIGN_RADIUS:
                    det_lookups[tids[c]][ts_r] = dets[r]

    tracks = []
    for tid, gt_dict in gt_all.items():
        if len(gt_dict) < 30:
            continue
        ts_sorted = sorted(gt_dict.keys())
        gt_arr    = np.array([[ts, *gt_dict[ts]] for ts in ts_sorted])
        dts       = np.diff(ts_sorted)
        dt        = float(np.median(dts)) if len(dts) else 1.0 / 30.0
        tracks.append((det_lookups[tid], gt_arr, dt))

    return tracks


print("Loading training data (Event_YOLO_new) ...", flush=True)
_TRAIN_DATA: list = []
for _vid in TRAIN_IDS:
    try:
        loaded = load_video_tracks_event(_vid)
        _TRAIN_DATA.extend(loaded)
        print(f"  Video {_vid}: {len(loaded)} track(s)", flush=True)
    except FileNotFoundError as exc:
        print(f"  Video {_vid}: NOT FOUND — {exc}", flush=True)
print(f"  Total: {len(_TRAIN_DATA)} training tracks\n", flush=True)


# =============================================================================
# SECTION 3 — PARAMETER ENCODING / DECODING
# =============================================================================

def decode_params(raw: np.ndarray) -> dict:
    """Mixed-encoding vector (14,) → named parameter dict."""
    p = raw.copy().astype(float)
    for i in LOG_PARAM_INDICES:
        p[i] = np.exp(raw[i])
    return dict(zip(PARAM_NAMES, p))


def params_to_kf_args(params: dict) -> tuple:
    """
    Named parameter dict →
        (Q_base, R_pos, R_vel, R_acc, acc_coeffs, alpha, beta, g_max).
    """
    qp, qv, qa = params["q_pos"], params["q_vel"], params["q_acc"]
    rp, rv, ra = params["r_pos"], params["r_vel"], params["r_acc"]
    Q_base = np.diag([qp, qp, qv, qv, qa, qa])
    R_pos  = np.diag([rp, rp])
    R_vel  = np.diag([rv, rv])
    R_acc  = np.diag([ra, ra])
    r_acc_coeffs = np.array([params["r_acc_c1"], params["r_acc_c2"], params["r_acc_c3"]])
    return Q_base, R_pos, R_vel, R_acc, r_acc_coeffs, params["alpha"], params["beta"], params["g_max"]


# =============================================================================
# SECTION 4 — SIMULATION
# =============================================================================

MAX_MISSING_SIM = 12  # frames without detection before killing a track


def _precompute_liveness(det_lookup: dict, gt_arr: np.ndarray,
                         dt: float) -> np.ndarray:
    """
    First pass: returns bool array alive[i] = True when an active
    (post-warmup) track exists at frame i.

    Handles two kill triggers, both mirrored exactly in simulate_ca_meas:
      1. MAX_MISSING_SIM consecutive gt_arr entries with no detection.
      2. A time gap between consecutive gt_arr entries larger than
         MAX_MISSING_SIM * dt (drone left + different drone re-entered;
         gt_arr entries jump in time rather than having consecutive misses).
    """
    alive = np.zeros(len(gt_arr), dtype=bool)
    buf   = []
    active = False
    miss   = 0
    for i, row in enumerate(gt_arr):
        # Large time gap → treat as a track kill immediately
        if i > 0:
            gap_frames = (row[0] - gt_arr[i - 1, 0]) / dt
            if gap_frames > MAX_MISSING_SIM:
                active = False
                buf    = []
                miss   = 0

        det = det_lookup.get(round(row[0], 5))
        if not active:
            if det is not None:
                buf.append(1)
            if len(buf) >= WARMUP:
                active = True
                miss   = 0
                buf    = []
        if active:
            if det is not None:
                miss      = 0
                alive[i]  = True
            else:
                miss     += 1
                alive[i]  = miss < MAX_MISSING_SIM
                if miss >= MAX_MISSING_SIM:
                    active = False
                    buf    = []
    return alive



def simulate_ca_meas(det_lookup: dict, gt_arr: np.ndarray,
                     dt: float, params: dict) -> np.ndarray:
    """
    Simulate CA_Meas_kalman on one GT track segment.

    Track lifecycle:
      - Warmup: accumulate WARMUP detections before creating KF.
      - Kill: reset KF after MAX_MISSING_SIM consecutive missed detections.
      - After kill, warmup restarts; this prevents comparing predictions
        from drone A against ground truth of a different drone B.

    Future errors are gated by the precomputed liveness mask so that no
    comparison crosses a track-kill boundary.

    Returns (N, H) array of per-frame, per-horizon prediction errors (px).
    """
    Q_base, R_pos, R_vel, R_acc, r_acc_coeffs, alpha, beta, g_max = params_to_kf_args(params)
    F = make_F(dt)

    h_frames = [max(1, round(h / dt)) for h in HORIZONS_S]
    max_h    = max(h_frames)
    h_set    = {k: j for j, k in enumerate(h_frames)}

    alive   = _precompute_liveness(det_lookup, gt_arr, dt)
    gt_ts   = gt_arr[:, 0]                  # sorted timestamp column for searchsorted

    warmup_buf: list = []
    trk              = None
    errors: list     = []

    for i, row in enumerate(gt_arr):
        ts  = row[0]
        det = det_lookup.get(round(ts, 5), None)

        # -- LARGE TIME GAP → kill and restart warmup -----------------------
        if i > 0:
            gap_frames = (ts - gt_arr[i - 1, 0]) / dt
            if gap_frames > MAX_MISSING_SIM:
                trk        = None
                warmup_buf = []

        # -- WARM-UP ---------------------------------------------------------
        if trk is None:
            if det is not None:
                warmup_buf.append(det)
            if len(warmup_buf) >= WARMUP:
                span = (WARMUP - 1) * dt
                if span > 0:
                    v_est    = (warmup_buf[-1][:2] - warmup_buf[0][:2]) / span
                    sigma2_v = 2.0 * R_pos[0, 0] / span**2
                else:
                    v_est    = np.zeros(2)
                    sigma2_v = R_vel[0, 0]
                P_init = np.diag([R_pos[0, 0], R_pos[0, 0],
                                  sigma2_v, sigma2_v,
                                  R_acc[0, 0], R_acc[0, 0]])
                trk = MeasCADroneTrack(
                    initial_pos = warmup_buf[-1][:2],
                    dt          = dt,
                    Q_base      = Q_base,
                    R_pos       = R_pos,
                    R_vel       = R_vel,
                    R_acc       = R_acc,
                    alpha       = alpha,
                    beta        = beta,
                    initial_P   = P_init,
                    acc_coeffs  = r_acc_coeffs,
                    g_max       = g_max,
                    frame_w     = EVENT_IMG_W,
                    frame_h     = EVENT_IMG_H,
                )
                if span > 0:
                    trk.kf.x[2:4]     = v_est
                    trk._prev_pos_meas = warmup_buf[-1][:2].copy()
            continue

        # -- STEP KF (predict + update/miss) ----------------------------------
        if det is not None:
            trk.step(det[:2])
        else:
            trk.step(None)
            if trk.is_lost:
                trk        = None
                warmup_buf = []
                continue   # don't collect errors on kill frame

        # -- ITERATIVE FUTURE ROLLOUT ----------------------------------------
        frame_h    = [np.nan] * len(HORIZONS_S)
        x_fut      = trk.kf.x.copy()

        for k in range(1, max_h + 1):
            x_fut[2] *= beta
            x_fut[3] *= beta
            x_fut[4] *= alpha
            x_fut[5] *= alpha
            x_fut = F @ x_fut

            if k not in h_set:
                continue

            j         = h_set[k]
            target_ts = ts + k * dt

            # Find nearest GT frame at this horizon (searchsorted is robust to float drift)
            fi = int(np.searchsorted(gt_ts, target_ts))
            if fi >= len(gt_arr) or abs(gt_ts[fi] - target_ts) > 1.5 * dt:
                continue   # no GT frame near this horizon

            # Gate: don't compare across a track-kill boundary
            if not np.all(alive[i + 1 : fi + 1]):
                continue

            # Use the exact GT timestamp to look up the YOLO_new detection
            future_det = det_lookup.get(round(gt_ts[fi], 5), None)
            if future_det is None:
                continue   # no YOLO_new detection at this GT frame → skip

            pred     = x_fut[:2]
            pred_out = not (0 <= pred[0] <= EVENT_IMG_W and 0 <= pred[1] <= EVENT_IMG_H)

            if pred_out:
                pred_clipped = np.clip(pred, [0.0, 0.0], [float(EVENT_IMG_W), float(EVENT_IMG_H)])
                frame_h[j] = float(np.linalg.norm(pred_clipped - future_det[:2]))
            else:
                frame_h[j] = float(np.linalg.norm(pred - future_det[:2]))

        if any(not np.isnan(v) for v in frame_h):
            errors.append(frame_h)

    return np.array(errors) if errors else np.zeros((0, len(HORIZONS_S)))


# =============================================================================
# SECTION 5 — OBJECTIVE + BOUNDS PENALTY
# =============================================================================

def _track_stats(errs: np.ndarray) -> tuple:
    H       = errs.shape[1]
    h_avg   = np.full(H, np.nan)
    h_final = np.full(H, np.nan)
    for j in range(H):
        col   = errs[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            h_avg[j]   = float(np.mean(valid))
            h_final[j] = float(valid[-1])
    w              = np.array(HORIZON_WEIGHTS)
    avg_combined   = float(np.nansum(w * h_avg)   / np.nansum(w[~np.isnan(h_avg)]))
    final_combined = float(np.nansum(w * h_final) / np.nansum(w[~np.isnan(h_final)]))
    return W_AVG * avg_combined + W_FINAL * final_combined, h_avg, h_final


def _eval_per_video(video_id: int, params: dict) -> None:
    """Load one video, run simulate_ca_meas on every track, print per-horizon errors."""
    try:
        tracks = load_video_tracks_event(video_id)
    except FileNotFoundError:
        print(f"  Video {video_id:3d}: data not found, skipping")
        return

    all_h_avg = np.zeros(len(HORIZONS_S))
    n_valid   = 0
    for det_lookup, gt_arr, dt in tracks:
        errs = simulate_ca_meas(det_lookup, gt_arr, dt, params)
        if errs.shape[0] == 0:
            continue
        _, h_avg, _ = _track_stats(errs)
        all_h_avg += np.where(np.isnan(h_avg), 0.0, h_avg)
        n_valid   += 1

    if n_valid == 0:
        print(f"  Video {video_id:3d}: no valid tracks")
        return

    avg = all_h_avg / n_valid
    horizon_str = "  ".join(f"{h:.1f}s: {avg[j]:.2f} px" for j, h in enumerate(HORIZONS_S))
    combined    = float(np.sum(np.array(HORIZON_WEIGHTS) * avg)
                        / np.sum(HORIZON_WEIGHTS))
    print(f"  Video {video_id:3d}  ({n_valid} track{'s' if n_valid>1 else ''})  |  "
          f"{horizon_str}  |  weighted: {combined:.2f} px")


def _bounds_penalty(raw: np.ndarray) -> float:
    if PARAM_BOUNDS is None:
        return 0.0
    params  = decode_params(raw)
    penalty = 0.0
    for k, name in enumerate(PARAM_NAMES):
        v      = params[name]
        lo, hi = PARAM_BOUNDS[k]
        span   = hi - lo
        if v < lo:
            penalty += ((lo - v) / span) ** 2
        elif v > hi:
            penalty += ((v - hi) / span) ** 2
    return PENALTY_WEIGHT * penalty


# =============================================================================
# SECTION 6 — PARALLEL WORKER
# =============================================================================

_WORKER_DATA: list = []

def _pool_init(train_data: list) -> None:
    global _WORKER_DATA
    _WORKER_DATA = train_data

def _simulate_worker(args: tuple) -> np.ndarray:
    idx, params = args
    det_lookup, gt_arr, dt = _WORKER_DATA[idx]
    return simulate_ca_meas(det_lookup, gt_arr, dt, params)

_POOL     = None
N_WORKERS = max(1, mp.cpu_count() - 1)


# =============================================================================
# SECTION 7 — OBJECTIVE FUNCTION WITH PROGRESS REPORTING
# =============================================================================

_eval_history: list = []
_call_count         = [0]
_last_h_avg         = np.zeros(len(HORIZONS_S))
_last_h_final       = np.zeros(len(HORIZONS_S))
_last_params        = np.zeros(N_PARAMS)


def objective(raw: np.ndarray) -> float:
    params = decode_params(raw)

    if _POOL is not None:
        job_args = [(i, params) for i in range(len(_TRAIN_DATA))]
        all_errs = _POOL.map(_simulate_worker, job_args)
    else:
        all_errs = [simulate_ca_meas(d, g, dt, params) for d, g, dt in _TRAIN_DATA]

    total_scalar = 0.0
    h_avg_sum    = np.zeros(len(HORIZONS_S))
    h_final_sum  = np.zeros(len(HORIZONS_S))
    n            = 0

    for errs in all_errs:
        if errs.shape[0] == 0:
            continue
        scalar, h_avg, h_final = _track_stats(errs)
        total_scalar += scalar
        h_avg_sum    += np.where(np.isnan(h_avg),   0.0, h_avg)
        h_final_sum  += np.where(np.isnan(h_final), 0.0, h_final)
        n            += 1

    val = (total_scalar / n) if n else 1e9
    val += _bounds_penalty(raw)

    if n > 0:
        _last_h_avg[:]   = h_avg_sum   / n
        _last_h_final[:] = h_final_sum / n
    _last_params[:] = np.array(list(params.values()))

    _eval_history.append(val)
    _call_count[0] += 1

    if _call_count[0] % 100 == 0:
        best = min(_eval_history)
        print(f"\n  eval {_call_count[0]:5d}  "
              f"current={val:.4f} px  best={best:.4f} px", flush=True)
        print(f"  {'horizon':<8}  {'avg dist error':>16}  {'final dist error':>16}")
        print(f"  {'-'*44}")
        for j, h in enumerate(HORIZONS_S):
            print(f"  {h:.1f} s      "
                  f"{_last_h_avg[j]:>12.3f} px    {_last_h_final[j]:>12.3f} px")
        print(f"  {'parameter':<14}  {'value':>10}")
        print(f"  {'-'*27}")
        for name, val_ in zip(PARAM_NAMES, _last_params):
            print(f"  {name:<14}  {val_:>10.4g}", flush=True)

    return val


# =============================================================================
# SECTION 8 — MAIN
# =============================================================================

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    _POOL = mp.Pool(N_WORKERS,
                    initializer=_pool_init,
                    initargs=(_TRAIN_DATA,))
    print(f"Parallel workers : {N_WORKERS} (of {mp.cpu_count()} cores)\n")

    # -------------------------------------------------------------------------
    # Starting point
    # -------------------------------------------------------------------------
    _nat = {
        "q_pos":  11.442,    "q_vel":  0.00036198, "q_acc":  2.8337,
        "r_pos":  1.6883,    "r_vel":  0.35511,    "r_acc":  10.408,
        "alpha":  0.95221,   "beta":   0.96737,
        "r_acc_c1": -0.13303, "r_acc_c2": 0.52543, "r_acc_c3": 4.0978,
        "g_max":  483.51,
    }
    x0 = np.array([
        np.log(_nat[n]) if i in LOG_PARAM_INDICES else _nat[n]
        for i, n in enumerate(PARAM_NAMES)
    ])

    print(f"Detection source : Event_YOLO_new  (event camera, 1280×720)")
    print(f"Motion model     : constant-acceleration 6-state [x, y, vx, vy, ax, ay]")
    print(f"Measurements     : pos direct; vel = Δpos/dt; acc = Δvel_meas/dt")
    print(f"Polynomial       : 3rd order in measured |a| / A_REF — acc R block only")
    print(f"Future rollout   : alpha scales [ax,ay] each step (no physical clip)")
    print(f"Parameters       : {N_PARAMS}")
    print(f"Encoding         : log-space for {sorted(LOG_PARAM_INDICES)}, natural for all others")
    print(f"Horizons         : {HORIZONS_S} s  weights={HORIZON_WEIGHTS}")
    print(f"Early stop       : improvement < {STALL_THRESHOLD} px over {STALL_WINDOW} evals\n")

    init_val = objective(x0)
    print(f"Initial objective : {init_val:.4f} px")
    print("Starting optimisation (CMA-ES) ...\n", flush=True)

    cma_stds = [0.5] * N_PARAMS
    cma_stds[PARAM_NAMES.index("alpha")] = 0.15
    cma_stds[PARAM_NAMES.index("beta")]  = 0.15

    es = cma.CMAEvolutionStrategy(x0, 0.5, {
        'maxiter':  5000,
        'tolx':     1e-7,
        'tolfun':   1e-7,
        'verbose':  -9,
        'seed':     42,
        'CMA_stds': cma_stds,
    })

    t0       = time.time()
    best_val = init_val
    best_x   = x0.copy()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s) for s in solutions]
        es.tell(solutions, fitnesses)

        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_val:
            best_val = fitnesses[gen_best_idx]
            best_x   = solutions[gen_best_idx].copy()

        n_evals = _call_count[0]
        if n_evals >= MIN_EVALS:
            recent_best = min(_eval_history[-STALL_WINDOW:])
            older_best  = min(_eval_history[-MIN_EVALS:-STALL_WINDOW])
            if older_best - recent_best < STALL_THRESHOLD:
                print(f"\nEarly stop at eval {n_evals}: "
                      f"improvement {older_best - recent_best:.5f} px "
                      f"< {STALL_THRESHOLD} px  (best = {recent_best:.4f} px)",
                      flush=True)
                break

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s  ({_call_count[0]} evaluations)")
    print(f"Stop reason : {es.stop()}")
    print(f"Final training objective : {best_val:.4f} px\n")

    opt = decode_params(best_x)

    print("-- Optimised parameters --------------------------------------------")
    for name in PARAM_NAMES:
        print(f"  {name:<14}: {opt[name]:.5g}")

    p = opt
    print("\n-- Copy-paste snippet for CA_Meas_kalman.py ------------------------")
    print(f"    Q_base = np.diag([{p['q_pos']:.5g}, {p['q_pos']:.5g}, "
          f"{p['q_vel']:.5g}, {p['q_vel']:.5g}, "
          f"{p['q_acc']:.5g}, {p['q_acc']:.5g}])")
    print(f"    R_pos  = np.diag([{p['r_pos']:.5g}, {p['r_pos']:.5g}])")
    print(f"    R_vel  = np.diag([{p['r_vel']:.5g}, {p['r_vel']:.5g}])")
    print(f"    R_acc  = np.diag([{p['r_acc']:.5g}, {p['r_acc']:.5g}])")
    print(f"    alpha  = {p['alpha']:.5g}")
    print(f"    g_max  = {p['g_max']:.5g}")
    print(f"    r_acc_coeffs = [{p['r_acc_c1']:.5g}, {p['r_acc_c2']:.5g}, {p['r_acc_c3']:.5g}]")

    h_header = "  ".join(f"{h:.1f}s" for h in HORIZONS_S)

    print(f"\n-- Per-video errors — training videos ------------------------------")
    print(f"  {'':>9}  {h_header}  weighted")
    for vid_id in TRAIN_IDS:
        _eval_per_video(vid_id, opt)

    print(f"\n-- Per-video errors — test videos ----------------------------------")
    print(f"  {'':>9}  {h_header}  weighted")
    for vid_id in TEST_IDS:
        _eval_per_video(vid_id, opt)

    _POOL.close()
    _POOL.join()
