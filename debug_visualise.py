"""
DEBUG UTILITY — Visualise crop trajectory on frames
====================================================
Run this AFTER the pipeline to verify results visually.

Output: debug_frames/  — original frames with crop rectangle drawn
        trajectory_plot.png — graph of raw vs smooth x-trajectory
"""

import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def draw_crop_overlays(frames_dir: str, crop_data_dir: str, output_dir: str,
                       max_frames: int = 50):
    """
    Draw the crop rectangle on sampled frames and save debug images.
    Helps you visually verify the trajectory looks correct.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(frames_dir, "metadata.json")) as f:
        meta = json.load(f)
    sampled_ids = meta["sampled_ids"]

    with open(os.path.join(crop_data_dir, "crop_commands.json")) as f:
        all_cmds = json.load(f)
    crop_map = {c["frame_id"]: c for c in all_cmds}

    print(f"[DEBUG] Drawing crop overlays on {min(len(sampled_ids), max_frames)} frames...")

    for fid in tqdm(sampled_ids[:max_frames], desc="Debug frames"):
        img_path = os.path.join(frames_dir, f"frame_{fid:06d}.jpg")
        if not os.path.exists(img_path):
            continue

        frame = cv2.imread(img_path)
        crop  = crop_map.get(fid)
        if not crop:
            continue

        x, y, w, h = crop["crop_x"], crop["crop_y"], crop["crop_w"], crop["crop_h"]
        cx, cy = int(crop["cx"]), int(crop["cy"])

        # Draw crop rectangle (green)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Draw centre point (red dot)
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

        # Label
        label = f"Frame {fid} | crop({x},{y},{w},{h})"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 0), 2)

        out_path = os.path.join(output_dir, f"debug_{fid:06d}.jpg")
        cv2.imwrite(out_path, frame)

    print(f"[DEBUG] Debug frames saved to: {output_dir}/")


def plot_trajectory(trajectory_dir: str, output_path: str = "trajectory_plot.png"):
    """
    Plot raw vs optimised vs smoothed x-trajectory over time.
    Shows how each processing step progressively smooths the path.
    """
    with open(os.path.join(trajectory_dir, "trajectory.json")) as f:
        traj = json.load(f)

    frame_ids = [t["frame_id"] for t in traj]
    raw_x     = [t["raw_x"]    for t in traj]
    opt_x     = [t["opt_x"]    for t in traj]
    smooth_x  = [t["smooth_x"] for t in traj]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame_ids, raw_x,    label="Raw focus_x",        alpha=0.5, linestyle="--", color="red")
    ax.plot(frame_ids, opt_x,    label="Optimised x",         alpha=0.7, linestyle="-",  color="orange")
    ax.plot(frame_ids, smooth_x, label="Kalman smoothed x",   alpha=1.0, linestyle="-",  color="blue", linewidth=2)

    ax.set_title("Crop Centre X — Trajectory Comparison")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("X position (pixels)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[DEBUG] Trajectory plot saved: {output_path}")


if __name__ == "__main__":
    draw_crop_overlays(
        frames_dir    = "workspace/frames",
        crop_data_dir = "workspace/crop_data",
        output_dir    = "workspace/debug_frames"
    )
    plot_trajectory(
        trajectory_dir = "workspace/trajectory",
        output_path    = "workspace/trajectory_plot.png"
    )
