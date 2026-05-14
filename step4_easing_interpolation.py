"""
STEP 4 — SCENE CUT DETECTION + EASING + FULL-FPS INTERPOLATION
===============================================================
Input  : trajectory/trajectory.json  (from Step 3)
         frames/metadata.json         (from Step 1)

Output : crop_commands.json — one entry per ORIGINAL frame with exact
         crop (x, y, w, h) values ready for FFmpeg

Expected Output (crop_commands.json):
[
  {"frame_id": 0,  "cx": 960.0, "cy": 540.0, "crop_x": 656, "crop_y": 0, "crop_w": 608, "crop_h": 1080},
  {"frame_id": 1,  "cx": 960.3, "cy": 540.0, "crop_x": 656, "crop_y": 0, "crop_w": 608, "crop_h": 1080},
  ...
]

crop_x, crop_y = top-left corner of the crop window in the original frame
crop_w, crop_h = width and height of the crop window (constant 9:16 size)
"""

import json
import os
import numpy as np
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────────────────────────────────────
#  PART A — Scene cut detection
#  Compare consecutive sampled frames by histogram difference.
#  A large jump = scene cut.
# ─────────────────────────────────────────────────────────────────────────────

import cv2


CUT_THRESHOLD = 0.45   # histogram correlation below this → cut detected
                       # range 0-1; lower = more sensitive


def detect_scene_cuts(frames_dir: str, sampled_ids: list) -> list:
    """
    Compare histograms of consecutive sampled frames.
    Returns list of frame_ids where a scene cut was detected BEFORE that frame.
    """
    cuts = []
    prev_hist = None

    for fid in sampled_ids:
        img_path = os.path.join(frames_dir, f"frame_{fid:06d}.jpg")
        if not os.path.exists(img_path):
            continue

        frame = cv2.imread(img_path)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist  = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist  = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            corr = cv2.compareHist(
                prev_hist.reshape(-1, 1).astype(np.float32),
                hist.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            if corr < CUT_THRESHOLD:
                cuts.append(fid)
                print(f"[STEP 4] Scene cut detected before frame {fid} (corr={corr:.3f})")

        prev_hist = hist

    print(f"[STEP 4] Total scene cuts found: {len(cuts)}")
    return cuts


# ─────────────────────────────────────────────────────────────────────────────
#  PART B — Easing functions
#  Applied within each scene (between cuts) to give cinematic feel.
# ─────────────────────────────────────────────────────────────────────────────

def ease_in_out_cubic(t: np.ndarray) -> np.ndarray:
    """
    t in [0, 1] → eased t in [0, 1].
    Slow at start, fast in middle, slow at end.
    """
    return np.where(t < 0.5, 4 * t**3, 1 - (-2*t + 2)**3 / 2)


def apply_easing_within_scenes(
    frame_ids: np.ndarray,
    smooth_x: np.ndarray,
    smooth_y: np.ndarray,
    cut_frame_ids: list,
    max_speed_px: float = 8.0
) -> tuple:
    """
    For each scene segment between cuts:
      1. Apply ease-in-out on the DELTA motion so transitions feel natural
      2. Cap speed so no single frame jump exceeds max_speed_px

    Returns eased_x, eased_y arrays (same shape as input).
    """
    cut_set    = set(cut_frame_ids)
    n          = len(frame_ids)
    eased_x    = smooth_x.copy()
    eased_y    = smooth_y.copy()

    # Find scene segment boundaries
    scene_starts = [0]
    for i, fid in enumerate(frame_ids):
        if fid in cut_set and i > 0:
            scene_starts.append(i)
    scene_starts.append(n)   # sentinel

    for seg in range(len(scene_starts) - 1):
        s = scene_starts[seg]
        e = scene_starts[seg + 1]

        if e - s < 2:
            continue

        seg_x = smooth_x[s:e]
        seg_y = smooth_y[s:e]

        # t goes 0→1 across the segment
        t = np.linspace(0, 1, e - s)
        t_eased = ease_in_out_cubic(t)

        # Remap positions along the eased time axis
        # This stretches the motion curve so it starts and ends gently
        seg_x_new = np.interp(t_eased, t, seg_x)
        seg_y_new = np.interp(t_eased, t, seg_y)

        eased_x[s:e] = seg_x_new
        eased_y[s:e] = seg_y_new

    # ── Speed limiter ─────────────────────────────────────────────────────
    for i in range(1, n):
        dx = eased_x[i] - eased_x[i-1]
        dy = eased_y[i] - eased_y[i-1]
        speed = np.sqrt(dx**2 + dy**2)

        if speed > max_speed_px:
            scale      = max_speed_px / speed
            eased_x[i] = eased_x[i-1] + dx * scale
            eased_y[i] = eased_y[i-1] + dy * scale

    return eased_x, eased_y


# ─────────────────────────────────────────────────────────────────────────────
#  PART C — Cubic spline interpolation to full FPS
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_to_full_fps(
    sampled_frame_ids: np.ndarray,
    eased_x: np.ndarray,
    eased_y: np.ndarray,
    total_frames: int
) -> tuple:
    """
    Fit a cubic spline through keyframe crop centres and evaluate at every frame.

    Returns:
        all_x, all_y : arrays of length total_frames
    """
    print("[STEP 4] Interpolating to full FPS with cubic spline...")

    cs_x = CubicSpline(sampled_frame_ids, eased_x, bc_type="natural")
    cs_y = CubicSpline(sampled_frame_ids, eased_y, bc_type="natural")

    all_frames = np.arange(total_frames)
    all_x      = cs_x(all_frames)
    all_y      = cs_y(all_frames)

    return all_x, all_y


# ─────────────────────────────────────────────────────────────────────────────
#  PART D — Convert (cx, cy) to crop (x, y, w, h)
# ─────────────────────────────────────────────────────────────────────────────

def centre_to_crop(cx: float, cy: float,
                   frame_w: int, frame_h: int) -> dict:
    """
    Given crop centre (cx, cy) and frame dimensions, compute the
    top-left corner (crop_x, crop_y) of the 9:16 window.

    The crop window size is always frame_h * (9/16) × frame_h.
    """
    crop_w = int(frame_h * 9 / 16)
    crop_h = frame_h

    crop_x = int(cx - crop_w / 2)
    crop_y = int(cy - crop_h / 2)

    # Clamp so window stays inside the original frame
    crop_x = max(0, min(crop_x, frame_w - crop_w))
    crop_y = max(0, min(crop_y, frame_h - crop_h))

    return {"crop_x": crop_x, "crop_y": crop_y, "crop_w": crop_w, "crop_h": crop_h}


# ─────────────────────────────────────────────────────────────────────────────
#  PART E — Main
# ─────────────────────────────────────────────────────────────────────────────

def build_crop_commands(frames_dir: str, trajectory_dir: str, output_dir: str) -> list:
    """
    Full Step 4 pipeline.

    Returns list of per-frame crop commands.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    with open(os.path.join(frames_dir, "metadata.json")) as f:
        meta = json.load(f)
    frame_w      = meta["width"]
    frame_h      = meta["height"]
    total_frames = meta["total_frames"]
    sampled_ids  = meta["sampled_ids"]

    # Load trajectory
    with open(os.path.join(trajectory_dir, "trajectory.json")) as f:
        traj = json.load(f)

    sampled_frame_ids = np.array([t["frame_id"] for t in traj])
    smooth_x          = np.array([t["smooth_x"]  for t in traj])
    smooth_y          = np.array([t["smooth_y"]  for t in traj])

    # ── Scene cut detection ───────────────────────────────────────────────
    cuts = detect_scene_cuts(frames_dir, sampled_ids)

    # ── Apply easing within scenes ────────────────────────────────────────
    print("[STEP 4] Applying cinematic easing...")
    eased_x, eased_y = apply_easing_within_scenes(
        sampled_frame_ids, smooth_x, smooth_y, cuts,
        max_speed_px=12.0
    )

    # ── Interpolate to every frame ────────────────────────────────────────
    all_x, all_y = interpolate_to_full_fps(
        sampled_frame_ids, eased_x, eased_y, total_frames
    )

    # Re-clamp after interpolation (spline can overshoot at boundaries)
    half_w = (frame_h * 9 / 16) / 2
    half_h = frame_h / 2
    all_x  = np.clip(all_x, half_w, frame_w - half_w)
    all_y  = np.clip(all_y, half_h, frame_h - half_h)

    # ── Build crop command list ───────────────────────────────────────────
    crop_commands = []
    for fid in range(total_frames):
        cx = float(all_x[fid])
        cy = float(all_y[fid])
        crop = centre_to_crop(cx, cy, frame_w, frame_h)
        crop_commands.append({
            "frame_id" : fid,
            "cx"       : cx,
            "cy"       : cy,
            **crop
        })

    # Save
    cmd_path = os.path.join(output_dir, "crop_commands.json")
    with open(cmd_path, "w") as f:
        json.dump(crop_commands, f, indent=2)

    print(f"[STEP 4] Crop commands saved: {cmd_path}")
    print(f"[STEP 4] Total frames written: {len(crop_commands)}")
    print(f"[STEP 4] Sample: {crop_commands[0]}")

    return crop_commands


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cmds = build_crop_commands(
        frames_dir    = "frames",
        trajectory_dir = "trajectory",
        output_dir    = "crop_data"
    )
