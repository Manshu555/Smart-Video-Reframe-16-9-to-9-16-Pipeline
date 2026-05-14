"""
STEP 3 — TRAJECTORY OPTIMISATION + KALMAN SMOOTHING
=====================================================
Input  : analysis/detections.json  (from Step 2)
         analysis/flow.json
         frames/metadata.json       (from Step 1)

Output : trajectory.json  — smoothed (x_crop_center, y_crop_center) per frame_id

Expected Output (trajectory.json):
[
  {"frame_id": 0,   "raw_x": 960.0, "raw_y": 540.0, "smooth_x": 960.0, "smooth_y": 540.0},
  {"frame_id": 6,   "raw_x": 972.1, "raw_y": 535.8, "smooth_x": 963.2, "smooth_y": 539.1},
  ...
]

The smooth_x / smooth_y values are what get fed into Step 4 (easing + rendering).
"""

import json
import os
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from filterpy.kalman import KalmanFilter


# ─────────────────────────────────────────────────────────────────────────────
#  PART A — Compute the valid crop half-width
#  For a 9:16 crop of a 1080p-tall frame: crop_w = 1080*(9/16) = 607.5 ≈ 608
#  For a 1920x1080 source, this means the horizontal centre x must stay in
#  [304, 1616] so the crop window never exceeds the frame edges.
# ─────────────────────────────────────────────────────────────────────────────

def compute_crop_half_width(frame_w: int, frame_h: int) -> float:
    """
    The 9:16 crop must fit inside the original frame height.
    Crop width = frame_height * (9/16).
    Returns half of that (used to clamp the crop centre x).
    """
    crop_w = frame_h * (9 / 16)
    # Can't exceed the original width either
    crop_w = min(crop_w, frame_w)
    return crop_w / 2


# ─────────────────────────────────────────────────────────────────────────────
#  PART B — Global trajectory optimisation
#  We minimise a cost across ALL frames at once.
#  Variables : x[i] = proposed crop-centre-x for frame i  (y is not optimised
#              because we keep it centred vertically / at focus_y)
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_JITTER  = 20.0   # penalise fast movement between frames
LAMBDA_DIST    = 0.2   # penalise distance from the raw focus point


def build_cost_function(raw_x: np.ndarray, x_min: float, x_max: float):
    """
    Returns a scipy-compatible scalar cost function over x[i] values.

    Cost = Σ_i [ lambda_dist  * (x[i] - raw_x[i])^2
               + lambda_jitter * (x[i] - x[i-1])^2 ]
    """
    n = len(raw_x)

    def cost(x_vec):
        x = x_vec.reshape(n)
        dist_cost   = LAMBDA_DIST   * np.sum((x - raw_x) ** 2)
        jitter_cost = LAMBDA_JITTER * np.sum(np.diff(x) ** 2)
        return dist_cost + jitter_cost

    # Bounds: crop centre must stay inside valid range
    bounds = [(x_min, x_max)] * n

    return cost, bounds


def optimise_trajectory(raw_x: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Run global L-BFGS-B optimisation on the raw x-trajectory.

    Returns the optimised x array (same length as raw_x).
    """
    half_w = compute_crop_half_width(frame_w, frame_h)
    x_min  = half_w
    x_max  = frame_w - half_w

    # Initial guess: clamped raw trajectory
    x0 = np.clip(raw_x, x_min, x_max)

    cost_fn, bounds = build_cost_function(raw_x, x_min, x_max)

    print("[STEP 3] Running global trajectory optimisation (L-BFGS-B)...")
    result = minimize(
        cost_fn,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    if result.success:
        print(f"[STEP 3] Optimisation converged. Final cost: {result.fun:.4f}")
    else:
        print(f"[STEP 3] Warning: optimisation did not fully converge — {result.message}")

    return result.x


# ─────────────────────────────────────────────────────────────────────────────
#  PART C — Kalman filter smoothing
#  State vector : [x, vx]  (position + velocity in x)
#  Treats crop movement as a physical system with inertia.
# ─────────────────────────────────────────────────────────────────────────────

def build_kalman_filter(initial_x: float, process_noise: float = 0.5,
                        measurement_noise: float = 2.0) -> KalmanFilter:
    """
    1D constant-velocity Kalman filter.

    State    : [x, vx]
    Measure  : [x]
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)

    kf.F = np.array([[1, 1],     # state transition: x += vx
                     [0, 1]])

    kf.H = np.array([[1, 0]])    # measurement: we observe x directly

    kf.P *= 10.0                 # initial uncertainty

    kf.R = np.array([[measurement_noise ** 2]])   # measurement noise
    kf.Q = np.eye(2) * process_noise              # process noise

    kf.x = np.array([[initial_x], [0.0]])         # initial state [x, vx=0]

    return kf


def kalman_smooth(x_optimised: np.ndarray,
                  process_noise: float = 0.5,
                  measurement_noise: float = 2.0) -> np.ndarray:
    """
    Run Kalman forward pass over optimised x values.
    Returns smoothed x array.
    """
    print("[STEP 3] Applying Kalman filter...")
    kf = build_kalman_filter(x_optimised[0], process_noise, measurement_noise)

    smoothed = []
    for measurement in x_optimised:
        kf.predict()
        kf.update(np.array([[measurement]]))
        smoothed.append(float(kf.x[0, 0]))

    return np.array(smoothed)


# ─────────────────────────────────────────────────────────────────────────────
#  PART D — Main: stitch it all together
# ─────────────────────────────────────────────────────────────────────────────

def build_trajectory(frames_dir: str, analysis_dir: str, output_dir: str) -> list:
    """
    Full trajectory pipeline: raw focus → optimise → Kalman smooth → save.

    Returns list of dicts with per-frame crop centres.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    with open(os.path.join(frames_dir, "metadata.json")) as f:
        meta = json.load(f)
    frame_w = meta["width"]
    frame_h = meta["height"]

    # Load detections
    with open(os.path.join(analysis_dir, "detections.json")) as f:
        detections = json.load(f)

    # Load flow (optional — used to add velocity bias to raw trajectory)
    with open(os.path.join(analysis_dir, "flow.json")) as f:
        flow_data = json.load(f)
    flow_map = {d["frame_id"]: d for d in flow_data}

    # ── Extract raw_x and raw_y from focus points ─────────────────────────
    frame_ids = [d["frame_id"] for d in detections]
    raw_x     = np.array([d["focus_raw"][0] for d in detections])
    raw_y     = np.array([d["focus_raw"][1] for d in detections])

    # ── Add slight optical flow bias to raw_x ─────────────────────────────
    # If subjects are moving right, shift the raw focus slightly ahead of them
    FLOW_LEAD_FACTOR = 0.0   # Disabled optical flow bias to reduce jitter
    for i, fid in enumerate(frame_ids):
        dx = flow_map.get(fid, {}).get("dx", 0.0)
        raw_x[i] += dx * FLOW_LEAD_FACTOR

    # ── Clamp raw trajectory to valid range ───────────────────────────────
    half_w = compute_crop_half_width(frame_w, frame_h)
    raw_x  = np.clip(raw_x, half_w, frame_w - half_w)

    # y: keep at focus_y but clamp to keep crop inside frame
    crop_h = frame_w * (16 / 9)   # if we use full width
    crop_h = min(crop_h, frame_h)
    half_h = crop_h / 2
    raw_y  = np.clip(raw_y, half_h, frame_h - half_h)

    # ── Global optimisation (on x axis where most movement happens) ───────
    opt_x = optimise_trajectory(raw_x, frame_w, frame_h)

    # ── Gaussian smooth as a first pass (sigma = 5 frames) ───────────────
    gauss_x = gaussian_filter1d(opt_x, sigma=5.0)
    gauss_y = gaussian_filter1d(raw_y, sigma=5.0)

    # ── Kalman filter as final pass ───────────────────────────────────────
    smooth_x = kalman_smooth(gauss_x, process_noise=0.05, measurement_noise=5.0)
    smooth_y = kalman_smooth(gauss_y, process_noise=0.05, measurement_noise=5.0)

    # ── Assemble trajectory ───────────────────────────────────────────────
    trajectory = []
    for i, fid in enumerate(frame_ids):
        trajectory.append({
            "frame_id" : fid,
            "raw_x"    : float(raw_x[i]),
            "raw_y"    : float(raw_y[i]),
            "opt_x"    : float(opt_x[i]),
            "smooth_x" : float(smooth_x[i]),
            "smooth_y" : float(smooth_y[i]),
        })

    # Save
    traj_path = os.path.join(output_dir, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"[STEP 3] Trajectory saved: {traj_path}")
    print(f"[STEP 3] Sample entry    : {trajectory[0]}")

    return trajectory


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    traj = build_trajectory(
        frames_dir   = "frames",
        analysis_dir = "analysis",
        output_dir   = "trajectory"
    )
    print(f"Total keyframes in trajectory: {len(traj)}")
