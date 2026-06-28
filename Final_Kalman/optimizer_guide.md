# Kalman Filter Optimizer — Usage Guide

`optimizer_ca_meas_kf_new.py` uses CMA-ES to optimize the 12 noise parameters of
`CA_Meas_kalman_new.py` against real Event_YOLO_new detection data.

## Running the optimizer

```bash
python Final_Kalman/optimizer_ca_meas_kf_new.py
```

The script loads training data at startup, then runs CMA-ES. Progress is printed to
stdout. The best parameters found are printed at the end of each generation and again
as a final summary.

## Training configuration

| Setting | Value |
|---------|-------|
| Training videos | 14, 31, 33, 79 |
| Test videos (not used during training) | 0, 21 |
| Prediction horizons evaluated | 0.2 s, 0.4 s, 0.8 s |
| Horizon weights | 1/3, 1/3, 1/3 (equal) |
| Rolling-average weight `W_AVG` | 1 |
| Final-frame weight `W_FINAL` | 0 |
| Detection source | Event_YOLO_new |
| Detection assignment radius | 150 px |

The optimizer compares future KF predictions to **actual YOLO_new detections** (not
interpolated ground truth). Frames without a detection at a given horizon are excluded
from the error — only frames where the detector fires contribute to the loss.

## Parameters optimized (12 total)

| Index | Name | Space | Description |
|-------|------|-------|-------------|
| 0 | `q_pos` | log | Position process noise base |
| 1 | `q_vel` | log | Velocity process noise base |
| 2 | `q_acc` | log | Acceleration process noise base |
| 3 | `r_pos` | log | Position measurement noise |
| 4 | `r_vel` | log | Velocity measurement noise |
| 5 | `r_acc` | log | Acceleration measurement noise base |
| 6 | `alpha` | natural | Acceleration decay per prediction step |
| 7 | `beta` | natural | Velocity decay per prediction step |
| 8 | `r_acc_c1` | natural | R_acc polynomial — 1st-order coefficient |
| 9 | `r_acc_c2` | natural | R_acc polynomial — 2nd-order coefficient |
| 10 | `r_acc_c3` | natural | R_acc polynomial — 3rd-order coefficient |
| 11 | `g_max` | log | Polynomial output ceiling |

The R_acc polynomial adaptively scales the acceleration measurement noise with
the measured acceleration magnitude:

```
g = 1 + c1·aₙ + c2·aₙ² + c3·aₙ³,  clipped to [1, g_max],  aₙ = |a| / 100 px/s²
```

## Currently optimized values

```python
Q_base     = np.diag([11.442, 11.442, 0.00036198, 0.00036198, 2.8337, 2.8337])
R_pos      = np.diag([1.6883,  1.6883])
R_vel      = np.diag([0.35511, 0.35511])
R_acc      = np.diag([10.408,  10.408])
alpha      = 0.95221
beta       = 0.96737
acc_coeffs = [-0.13303, 0.52543, 4.0978]
g_max      = 483.51
```

These are the values to pass to `MultiDroneTracker` (see `tracker_usage.md`).

## Updating the starting point

After a successful optimization run, copy the printed best parameters into the
`_nat` dict at the bottom of `optimizer_ca_meas_kf_new.py`:

```python
_nat = {
    "q_pos":  ...,  "q_vel":  ...,  "q_acc":  ...,
    "r_pos":  ...,  "r_vel":  ...,  "r_acc":  ...,
    "alpha":  ...,  "beta":   ...,
    "r_acc_c1": ..., "r_acc_c2": ..., "r_acc_c3": ...,
    "g_max":  ...,
}
```

This warm-starts the next optimization run from the current best solution.

## Data requirements

Each training/test video folder must contain:
- `interpolated_coordinates.txt` — ground-truth bounding boxes with track IDs
- `Event_YOLO_new/` — per-frame YOLO detection `.txt` files
  (format: `class cx cy w h [confidence]`, normalized coordinates)

The optimizer resolves detection–track assignments using Hungarian matching with a
150 px radius gate, so each YOLO detection is assigned to at most one GT track per
frame.
