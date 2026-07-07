# Live Tracker — Usage Guide

`CA_Meas_kalman_new.py` provides a ready-to-use live multi-drone tracker.
It receives YOLO detections one frame at a time and outputs all active (warmed-up)
tracks together with their future position predictions and uncertainty radii.

## Quick start

```python
import numpy as np
from CA_Meas_kalman_new import MultiDroneTracker

tracker = MultiDroneTracker(
    dt          = 1/30,   # seconds per frame (30 fps)
    Q_base      = np.diag([11.442, 11.442, 0.00036198, 0.00036198, 2.8337, 2.8337]),
    R_pos       = np.diag([1.6883,  1.6883]),
    R_vel       = np.diag([0.35511, 0.35511]),
    R_acc       = np.diag([10.408,  10.408]),
    alpha       = 0.95221,   # acceleration decay over prediction horizon
    beta        = 0.96737,   # velocity decay over prediction horizon
    acc_coeffs  = [-0.13303, 0.52543, 4.0978],  # R_acc polynomial coefficients
    g_max       = 483.51,
    future_frames = 24,      # 0.8 s at 30 fps
)
```

## Calling the tracker each frame

```python
# detections: list of np.array([cx, cy]) in pixels — one entry per YOLO detection
detections = [np.array([640.0, 360.0]), np.array([200.0, 150.0])]

results = tracker.step(detections)
```

`tracker.step()` only receives the **current frame's detections** — no future knowledge
is used. It returns only **warmed-up (active)** tracks; tentative tracks still in the
warm-up window are not included.

## Output format

```python
for tid, pos, vel, (fut_pos, fut_out, fut_edge, fut_ellipses) in results:
    ...
```

| Field | Type | Description |
|-------|------|-------------|
| `tid` | `int` | Unique track ID (monotonically increasing) |
| `pos` | `np.array (2,)` | Current KF position estimate `[x, y]` in pixels |
| `vel` | `np.array (2,)` | Current KF velocity estimate `[vx, vy]` in px/s |
| `fut_pos` | `np.array (24, 2)` | Predicted positions at each of the 24 future steps |
| `fut_out` | `np.array (24,) bool` | `True` if that step's prediction is outside the frame |
| `fut_edge` | `np.array (24, 2)` | Prediction clamped to frame boundary (useful for drawing) |
| `fut_ellipses` | `list[tuple]` | 90 % confidence ellipse at each step: `(a_px, b_px, angle_deg)` |

## Extracting predictions and uncertainty at 0.1 s intervals

At 30 fps each 0.1 s step = 3 frames.  The relevant indices are 2, 5, 8, …, 23.

```python
STEP = 3   # 0.1 s at 30 fps

for tid, pos, vel, (fut_pos, fut_out, fut_edge, fut_ellipses) in results:
    print(f"Track {tid}  current pos: {pos}")
    for i, k in enumerate(range(STEP - 1, 24, STEP)):  # k = 2,5,8,...,23
        horizon_s   = (k + 1) / 30            # 0.1, 0.2, ..., 0.8
        pred_pos    = fut_pos[k]               # [x, y] in pixels
        is_outside  = fut_out[k]
        a, b, angle = fut_ellipses[k]          # 90 % ellipse semi-axes and angle
        radius      = a                        # circles are isotropic, a ≈ b
        print(f"  t+{horizon_s:.1f}s  pos={pred_pos}  uncertainty_radius={radius:.1f}px  outside={is_outside}")
```

## Tracker internals

| Parameter | Value |
|-----------|-------|
| Warm-up window | 5 consecutive detections |
| Max missing frames (active track) | 12 |
| Association radius | 100 px |
| Instant out-of-frame kill | yes |
| State vector | `[x, y, vx, vy, ax, ay]` |
| Uncertainty model | 90 % chi-squared confidence ellipse from propagated covariance P |

## Notes

- `dt` must match the actual frame rate of the input stream.
- YOLO detections should be passed as pixel-space centre points `[cx, cy]`.
  Bounding-box size is not used by the tracker.
- The tracker keeps its internal state between `step()` calls — do **not**
  re-instantiate it on every frame.
- Tentative (warming-up) tracks are accessible via `tracker.warmup` if needed.
