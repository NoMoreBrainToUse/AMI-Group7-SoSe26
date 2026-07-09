#!/usr/bin/env python3
"""
CA_Meas_kalman_new.py -- 6-state CA Kalman filter with measurement-derived
velocity and acceleration, plus a simple multi-drone tracker.

Variant for Event_YOLO_new detections:
  - MAX_MISSING = 9  (tracks survive longer gaps between detections)
  - miss_step() applies alpha/beta velocity/acceleration decay before coasting,
    matching the same dynamics used in predict_future().

State vector: [x, y, vx, vy, ax, ay]

Measurement model
-----------------
Position is measured directly from the bounding-box centre each frame.
Velocity and acceleration are derived from consecutive measurements:

    vel_meas = (pos_meas - prev_pos_meas) / dt
    acc_meas = (vel_meas  - prev_vel_meas)  / dt

Full measurement vector: z = [cx, cy, vx, vy, ax, ay]  (H = I_6).
When fewer consecutive detections are available the update falls back to
a smaller H:
    1 detection since init  -> H_pos     (2x6), z = [cx, cy]
    2 consecutive           -> H_pos_vel (4x6), z = [cx, cy, vx, vy]
    3+ consecutive          -> H_full    (6x6), z = [cx, cy, vx, vy, ax, ay]

Motion model
------------
First-order constant-acceleration matrix (no 0.5*dt^2 pos term):
    x  += vx*dt,  y  += vy*dt
    vx += ax*dt,  vy += ay*dt
    ax, ay constant

Noise modulation
----------------
Velocity and acceleration Q blocks scaled by independent 3rd-order
polynomials in acceleration magnitude:
    g = 1 + c1*a_n + c2*a_n^2 + c3*a_n^3,  clipped to [1, g_max],  a_n = |a|/A_REF

Future prediction
-----------------
predict_future() scales [ax, ay] by alpha before each step (alpha < 1
models a drone that stops accelerating over the horizon).

Multi-drone tracking
--------------------
Deliberately simple association (see MultiDroneTracker):
  - A detection matches an active track if the track's predicted position
    is within MATCH_RADIUS pixels of the detection (plain Euclidean circle).
  - Hungarian assignment resolves the case where several detections /
    tracks compete, but the gate is just the fixed circle -- no Mahalanobis,
    no uncertainty scaling.
  - A detection that matches no active track is handed to the nearest
    warming-up track (by plain distance); if there is none, it starts a new
    warming-up track. No cost metric during warmup.
  - WARMUP consecutive detections promote a warming-up track to active.
  - MAX_MISSING consecutive misses kill an active track.
"""

import numpy as np

# ===========================================================================
# MODULE CONSTANTS
# ===========================================================================

A_REF   = 100.0   # px/s^2 -- normalises |a| for the polynomial input
CHI2_90 = 4.6052  # chi2.ppf(0.90, df=2) -- scale for 90% confidence ellipse


# ===========================================================================
# MOTION MODEL
# ===========================================================================

def make_F(dt: float) -> np.ndarray:
    """First-order CA transition matrix (6x6), no 0.5*dt^2 position term."""
    F = np.eye(6)
    F[0, 2] = dt   # x  += vx*dt
    F[1, 3] = dt   # y  += vy*dt
    F[2, 4] = dt   # vx += ax*dt
    F[3, 5] = dt   # vy += ay*dt
    return F


# ===========================================================================
# NOISE MODULATION
# ===========================================================================

def modulation_factor(a_mag: float, coeffs, g_max: float = 20.0) -> float:
    """
    3rd-order polynomial in normalised acceleration magnitude.
        g = 1 + c1*a_n + c2*a_n^2 + c3*a_n^3,  clipped to [1, g_max]
        a_n = a_mag / A_REF
    coeffs : array-like of length 3 -- [c1, c2, c3].
    """
    a_n = a_mag / A_REF
    c1, c2, c3 = coeffs[0], coeffs[1], coeffs[2]
    g = 1.0 + c1 * a_n + c2 * a_n**2 + c3 * a_n**3
    return float(np.clip(g, 1.0, g_max))


# ===========================================================================
# KALMAN FILTER
# ===========================================================================

class MeasCAKalmanFilter:
    """
    6-state Kalman filter with measurement-derived velocity / acceleration.
    H and R are passed per update() call so the caller can choose the
    appropriate sub-measurement depending on how many consecutive
    detections are available.
    """

    def __init__(self, dt, Q_base, R_pos, R_vel, R_acc):
        self.dt = dt
        self.F  = make_F(dt)

        self.Q_base = Q_base.copy()
        self.Q      = Q_base.copy()

        self.H_pos = np.zeros((2, 6)); self.H_pos[0, 0] = 1.0; self.H_pos[1, 1] = 1.0
        self.H_pos_vel = np.zeros((4, 6))
        self.H_pos_vel[0, 0] = 1.0; self.H_pos_vel[1, 1] = 1.0
        self.H_pos_vel[2, 2] = 1.0; self.H_pos_vel[3, 3] = 1.0
        self.H_full = np.eye(6)

        self.R_pos  = R_pos.copy()
        self.R_pv   = np.block([[R_pos, np.zeros((2, 2))],
                                 [np.zeros((2, 2)), R_vel]])
        self.R_full = np.block([[R_pos, np.zeros((2, 4))],
                                 [np.zeros((2, 2)), R_vel, np.zeros((2, 2))],
                                 [np.zeros((2, 4)), R_acc]])

        self.x = np.zeros(6)
        self.P = np.zeros((6, 6))   # set by initialise(); never used before that call
        self.initialised = False

    def initialise(self, pos, P_init):
        self.x[:2] = pos
        self.x[2:] = 0.0
        self.P = np.asarray(P_init, float).copy()
        self.initialised = True

    def set_modulation(self, g_vel, g_acc):
        """Scale only the velocity and acceleration Q blocks."""
        Q = self.Q_base.copy()
        Q[2, 2] *= g_vel;  Q[3, 3] *= g_vel
        Q[4, 4] *= g_acc;  Q[5, 5] *= g_acc
        self.Q = Q

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z, H, R):
        """Standard KF update (Joseph form)."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def predict_future(self, n_frames, alpha, beta, frame_w=1280.0, frame_h=720.0):
        """
        Multi-step future rollout. Before each F propagation velocity is
        scaled by beta and acceleration by alpha.

        Returns (positions, outside, edge_pos, ellipses):
            positions : (n_frames, 2) raw predicted [x, y]
            outside   : (n_frames,) bool, True when prediction left the frame
            edge_pos  : (n_frames, 2) position clamped to frame boundary
            ellipses  : list of (a_px, b_px, angle_deg) — 80% confidence ellipse
                        semi-axes and rotation angle for each step
        """
        S     = np.diag([1.0, 1.0, beta, beta, alpha, alpha])
        F_eff = self.F @ S
        x = self.x.copy()
        P = self.P.copy()
        positions, outside, edge_pos, ellipses = [], [], [], []
        for _ in range(n_frames):
            x = F_eff @ x
            P = F_eff @ P @ F_eff.T + self.Q
            pos = x[:2].copy()
            out = not (0 <= pos[0] <= frame_w and 0 <= pos[1] <= frame_h)
            positions.append(pos)
            outside.append(out)
            edge_pos.append(np.clip(pos, [0.0, 0.0], [frame_w, frame_h]))
            evals, evecs = np.linalg.eigh(P[:2, :2])
            a     = float(np.sqrt(CHI2_90 * max(evals[1], 0.0)))
            b     = float(np.sqrt(CHI2_90 * max(evals[0], 0.0)))
            angle = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))
            ellipses.append((a, b, angle))
        return (np.array(positions),
                np.array(outside, dtype=bool),
                np.array(edge_pos),
                ellipses)


# ===========================================================================
# SINGLE DRONE TRACK
# ===========================================================================

class MeasCADroneTrack:
    """
    One drone, measurement-derived CA filter.

    step() runs one predict+update (or predict+miss) cycle and is used both
    standalone (single drone, and for parity with the optimizer's
    simulate_ca_meas) and by MultiDroneTracker.
    """

    def __init__(self, initial_pos, dt, Q_base, R_pos, R_vel, R_acc, alpha, beta,
                 initial_P,
                 vel_coeffs=(0., 0., 0.), acc_coeffs=(0., 0., 0.),
                 future_frames=24, max_missing=5, g_max=20.0,
                 frame_w=1280.0, frame_h=720.0):
        self.dt            = dt
        self.vel_coeffs    = np.asarray(vel_coeffs, float)
        self.acc_coeffs    = np.asarray(acc_coeffs, float)
        self.alpha         = float(alpha)
        self.beta          = float(beta)
        self.future_frames = future_frames
        self.max_missing   = max_missing
        self.g_max         = g_max
        self.frame_w       = float(frame_w)
        self.frame_h       = float(frame_h)

        self.kf = MeasCAKalmanFilter(dt, Q_base, R_pos, R_vel, R_acc)
        self.kf.initialise(initial_pos, initial_P)

        self._prev_pos_meas = None
        self._prev_vel_meas = None
        self._missing = 0

    # -- predict phase: modulate noise + propagate one frame ------------
    def predict_step(self):
        self.kf.set_modulation(1.0, 1.0)
        self.kf.predict()

    # -- update phase with a matched detection --------------------------
    def update_step(self, detection):
        pos = np.asarray(detection[:2], float)
        if self._prev_pos_meas is not None:
            vel_meas = (pos - self._prev_pos_meas) / self.dt
            if self._prev_vel_meas is not None:
                acc_meas   = (vel_meas - self._prev_vel_meas) / self.dt
                a_mag_meas = float(np.hypot(acc_meas[0], acc_meas[1]))
                g_r_acc    = modulation_factor(a_mag_meas, self.acc_coeffs, self.g_max)
                R_mod      = self.kf.R_full.copy()
                R_mod[4:, 4:] *= g_r_acc
                z = np.concatenate([pos, vel_meas, acc_meas])
                self.kf.update(z, self.kf.H_full, R_mod)
            else:
                z = np.concatenate([pos, vel_meas])
                self.kf.update(z, self.kf.H_pos_vel, self.kf.R_pv)
            self._prev_vel_meas = vel_meas
        else:
            self.kf.update(pos, self.kf.H_pos, self.kf.R_pos)
        self._prev_pos_meas = pos
        self._missing = 0

    # -- missed detection this frame ------------------------------------
    def miss_step(self):
        # Apply same velocity/acceleration decay as predict_future.
        # predict_step() has already propagated x = F @ x, so we damp the
        # resulting velocity and acceleration to model the drone slowing down.
        self.kf.x[2] *= self.beta
        self.kf.x[3] *= self.beta
        self.kf.x[4] *= self.alpha
        self.kf.x[5] *= self.alpha
        # Break the finite-difference chain (gap makes next velocity invalid).
        self._prev_pos_meas = None
        self._prev_vel_meas = None
        self._missing += 1
        # Instant kill if predicted position left the frame.
        x, y = self.kf.x[0], self.kf.x[1]
        if not (0 <= x <= self.frame_w and 0 <= y <= self.frame_h):
            self._missing = self.max_missing

    # -- standalone single-drone convenience ----------------------------
    def step(self, detection, bbox_size=0.0):
        """One full frame: predict, then update or miss."""
        self.predict_step()
        if detection is not None:
            self.update_step(detection)
        else:
            self.miss_step()

    @property
    def is_lost(self):
        return self._missing >= self.max_missing

    @property
    def position(self):
        return self.kf.x[:2].copy()

    @property
    def velocity(self):
        return self.kf.x[2:4].copy()

    @property
    def acceleration(self):
        return self.kf.x[4:6].copy()

    def predict_future(self):
        return self.kf.predict_future(self.future_frames, self.alpha, self.beta)


# ===========================================================================
# MULTI-DRONE TRACKER  (simple fixed-radius association)
# ===========================================================================

class MultiDroneTracker:
    """
    Frame-by-frame multi-drone tracker.

    Association rule (deliberately simple):
      - Each active track predicts its next position.
      - A detection matches a track if it lies within MATCH_RADIUS pixels of
        that track's predicted position (a plain circle, Euclidean distance).
      - If several detections fall inside several tracks' circles, a Hungarian
        assignment picks the globally-closest consistent pairing -- but the
        ONLY gate is the fixed circle. No Mahalanobis, no uncertainty scaling.

    Warmup rule (even simpler):
      - A detection that matches NO active track is given to the nearest
        warming-up track (plain distance, no radius limit). If there is no
        warming-up track, it starts a new one.
      - After WARMUP consecutive detections a warming-up track becomes active.

    Death rule:
      - An active track that misses MAX_MISSING consecutive frames is killed.

    No assumptions about drone speed, turn rate, or acceleration are baked
    into the matching beyond the single MATCH_RADIUS, which is generous
    enough (50 px default) to cover fast motion at 30 fps while still
    separating distinct drones.
    """

    WARMUP       = 5      # consecutive detections to promote warmup -> active
    MAX_MISSING  = 12     # consecutive misses to kill an active track
    MATCH_RADIUS = 100.0   # px -- association circle radius

    def __init__(self, dt, Q_base, R_pos, R_vel, R_acc, alpha, beta,
                 vel_coeffs=(0., 0., 0.), acc_coeffs=(0., 0., 0.),
                 g_max=20.0, future_frames=24,
                 frame_w=1280.0, frame_h=720.0):
        self._dt      = dt
        self._Q_base  = Q_base
        self._R_pos   = R_pos
        self._R_vel   = R_vel
        self._R_acc   = R_acc
        self._alpha   = alpha
        self._beta    = beta
        self._vel_c   = np.asarray(vel_coeffs, float)
        self._acc_c   = np.asarray(acc_coeffs, float)
        self._g_max   = g_max
        self._nfut    = future_frames
        self._frame_w = frame_w
        self._frame_h = frame_h

        # active tracks: list of [track_id, MeasCADroneTrack]
        self.active = []
        # warming-up tracks: list of {'buf': [pos,...], 'miss': int}
        self.warmup = []
        self._next_id = 0

    # ------------------------------------------------------------------
    def _spawn_track(self, pos, initial_P):
        return MeasCADroneTrack(
            initial_pos=pos, dt=self._dt,
            Q_base=self._Q_base, R_pos=self._R_pos,
            R_vel=self._R_vel, R_acc=self._R_acc,
            alpha=self._alpha, beta=self._beta, initial_P=initial_P,
            vel_coeffs=self._vel_c, acc_coeffs=self._acc_c,
            future_frames=self._nfut, max_missing=self.MAX_MISSING,
            g_max=self._g_max, frame_w=self._frame_w, frame_h=self._frame_h,
        )

    # ------------------------------------------------------------------
    def step(self, detections):
        """
        Advance all tracks by one frame.

        detections : list of np.ndarray([cx, cy]) (may be empty)

        Returns list of (track_id, position, velocity, predict_future_result),
        predict_future_result = (positions, outside, edge_pos).
        """
        from scipy.optimize import linear_sum_assignment

        dets = [np.asarray(d[:2], float) for d in detections]

        # ---- 1. Predict every active track -----------------------------
        for _, trk in self.active:
            trk.predict_step()

        # ---- 2. Match detections to active tracks (fixed circle) -------
        # cost = Euclidean distance; gate = MATCH_RADIUS. Anything outside
        # the circle gets an infinite cost so it can never be matched.
        match_t2d = {}
        matched_det = set()
        n_trk, n_det = len(self.active), len(dets)

        if n_trk > 0 and n_det > 0:
            cost = np.full((n_trk, n_det), 1e9)
            for i, (_, trk) in enumerate(self.active):
                pred = trk.kf.x[:2]
                for j, d in enumerate(dets):
                    dist = float(np.linalg.norm(d - pred))
                    if dist <= self.MATCH_RADIUS:
                        cost[i, j] = dist
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.MATCH_RADIUS:
                    match_t2d[r] = c
                    matched_det.add(c)

        # ---- 3. Update matched tracks; miss / kill the rest ------------
        dead = []
        for i, (tid, trk) in enumerate(self.active):
            if i in match_t2d:
                trk.update_step(dets[match_t2d[i]])
            else:
                trk.miss_step()
                if trk.is_lost:
                    dead.append(i)
        for i in sorted(dead, reverse=True):
            self.active.pop(i)

        # ---- 4. Unmatched detections -> nearest warmup track -----------
        # A detection that matched no active track joins the nearest
        # warming-up track BY PLAIN DISTANCE -- but only if it is within
        # MATCH_RADIUS of that track's last position. Two drones that appear
        # at the same time are far apart, so each correctly starts/keeps its
        # own warming-up track instead of being merged into one buffer (which
        # would corrupt the velocity estimate at promotion). If no warming-up
        # track is close enough, the detection starts a new one.
        for w in self.warmup:
            w['got_det'] = False
        for j, d in enumerate(dets):
            if j in matched_det:
                continue
            joined = False
            if self.warmup:
                dists = [np.linalg.norm(d - np.asarray(w['buf'][-1], float))
                         for w in self.warmup]
                nearest = int(np.argmin(dists))
                if dists[nearest] <= self.MATCH_RADIUS:
                    self.warmup[nearest]['buf'].append(d)
                    self.warmup[nearest]['got_det'] = True
                    joined = True
            if not joined:
                self.warmup.append({'buf': [d], 'got_det': True})

        # ---- 5. Promote warmed-up tracks; kill any that missed this frame --
        new_warmup = []
        for w in self.warmup:
            if len(w['buf']) >= self.WARMUP:
                # Compute P_init from R values before creating the track.
                r_p  = self._R_pos[0, 0]
                r_a  = self._R_acc[0, 0]
                span = (len(w['buf']) - 1) * self._dt
                sigma2_v = (2.0 * r_p / span**2) if span > 0 else self._R_vel[0, 0]
                P_init = np.diag([r_p, r_p, sigma2_v, sigma2_v, r_a, r_a])

                trk = self._spawn_track(w['buf'][-1], P_init)
                # Seed velocity from first/last buffered positions.
                if span > 0:
                    v_est = (np.asarray(w['buf'][-1], float) -
                             np.asarray(w['buf'][0], float)) / span
                    trk.kf.x[2:4] = v_est
                    trk._prev_pos_meas = np.asarray(w['buf'][-1], float)
                self.active.append([self._next_id, trk])
                self._next_id += 1
            elif w['got_det']:
                new_warmup.append(w)
            # else: missed this frame → kill immediately
        self.warmup = new_warmup

        # ---- 6. Build output -------------------------------------------
        results = []
        for tid, trk in self.active:
            results.append((tid,
                            trk.position.copy(),
                            trk.velocity.copy(),
                            trk.kf.predict_future(self._nfut, self._alpha, self._beta,
                                                  self._frame_w, self._frame_h)))
        return results


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(0)
    dt = 1 / 30

    Q_base = np.diag([0.5, 0.5, 1.0, 1.0, 0.1, 0.1])
    R_pos  = np.diag([2.0, 2.0])
    R_vel  = np.diag([10.0, 10.0])
    R_acc  = np.diag([100.0, 100.0])

    def mk():
        return MultiDroneTracker(
            dt=dt, Q_base=Q_base, R_pos=R_pos, R_vel=R_vel, R_acc=R_acc,
            alpha=0.9, beta=0.95, vel_coeffs=[0.5, 0.2, 0.05],
            acc_coeffs=[0.3, 0.1, 0.02], g_max=20.0, future_frames=15)

    def run(label, gen, n, n_drones_ideal):
        np.random.seed(0)
        tr = mk()
        for i in range(n):
            tr.step(gen(i))
        print(f"{label:48s} IDs={tr._next_id:2d}  (ideal {n_drones_ideal})")

    def g1(i):
        return [np.array([100 + 8*i, 200.]) + np.random.randn(2)]
    run("1. straight drone", g1, 100, 1)

    def g2(i):
        if i % 7 == 6: return []
        return [np.array([100 + 8*i, 200.]) + np.random.randn(2)]
    run("2. straight drone + periodic misses", g2, 100, 1)
