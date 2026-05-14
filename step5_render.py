"""
STEP 5 — FFMPEG RENDERING: APPLY CROP COMMANDS → FINAL 9:16 VIDEO
==================================================================
Input  : crop_data/crop_commands.json  (from Step 4)
         frames/metadata.json           (from Step 1)
         original input video file

Output : output_9x16.mp4  — final vertically cropped video

How it works:
  FFmpeg's 'crop' filter normally takes fixed values.
  For per-frame variable crops we use two approaches (choose one):

  APPROACH A (fast, simpler) — sendcmd filter
    Write a .txt file of timestamped crop commands, pass it to FFmpeg.
    Ideal for small/medium videos.

  APPROACH B (robust, full control) — frame-by-frame OpenCV render
    Read each original frame, apply crop, write to a temp video, then
    re-mux audio from original. Best for very long or complex videos.

  This file implements BOTH. Default is Approach B (more reliable).
"""

import json
import os
import subprocess
import cv2
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  APPROACH A — FFmpeg sendcmd (no Python frame loop, very fast)
# ─────────────────────────────────────────────────────────────────────────────

def render_with_ffmpeg_sendcmd(
    input_video: str,
    crop_commands: list,
    fps: float,
    output_path: str,
    target_w: int = 1080,
    target_h: int = 1920
):
    """
    Generate an FFmpeg sendcmd script for variable crop, then run FFmpeg.
    Fastest method — no Python frame loop needed.

    Limitation: FFmpeg's crop filter can only change parameters at
    keyframe boundaries. Works perfectly for 5–10fps keyframe intervals.
    """
    print("[STEP 5A] Building FFmpeg sendcmd script...")

    cmd_lines = []
    prev_crop = None

    for entry in crop_commands:
        # Only write a command when crop actually changes
        # (reduces file size and avoids redundant updates)
        crop = (entry["crop_x"], entry["crop_y"], entry["crop_w"], entry["crop_h"])
        if crop == prev_crop:
            continue

        timestamp = entry["frame_id"] / fps    # seconds
        x, y, w, h = crop
        # FFmpeg sendcmd format:
        # <time> [VALUE <x>] crop x;  <time> [VALUE <y>] crop y; ...
        line = (
            f"{timestamp:.4f} [enter] crop x {x},"
            f"{timestamp:.4f} [enter] crop y {y},"
            f"{timestamp:.4f} [enter] crop w {w},"
            f"{timestamp:.4f} [enter] crop h {h};"
        )
        cmd_lines.append(line)
        prev_crop = crop

    script_path = "crop_sendcmd.txt"
    with open(script_path, "w") as f:
        f.write("\n".join(cmd_lines))

    print(f"[STEP 5A] sendcmd script written: {script_path} ({len(cmd_lines)} commands)")

    # Build FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", (
            f"sendcmd=f={script_path},"
            f"crop=w={crop_commands[0]['crop_w']}:h={crop_commands[0]['crop_h']}"
            f":x={crop_commands[0]['crop_x']}:y={crop_commands[0]['crop_y']},"
            f"scale={target_w}:{target_h}:flags=lanczos"
        ),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]

    print(f"[STEP 5A] Running FFmpeg...")
    print("   " + " ".join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"[STEP 5A] Done -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  APPROACH B — OpenCV frame-by-frame (reliable, full control)
# ─────────────────────────────────────────────────────────────────────────────

def render_with_opencv(
    input_video: str,
    crop_commands: list,
    fps: float,
    output_path: str,
    target_w: int = 1080,
    target_h: int = 1920,
    temp_video: str = "temp_video_noaudio.mp4"
):
    """
    Open the original video, apply per-frame crop, write to temp file,
    then mux the original audio back in with FFmpeg.

    This is the most reliable approach — works regardless of codec, container.
    """
    print("[STEP 5B] Starting frame-by-frame render...")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer — H.264 MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (target_w, target_h))

    # Build a fast lookup: frame_id → crop params
    crop_map = {c["frame_id"]: c for c in crop_commands}

    last_crop = crop_commands[0]   # fallback for any gaps

    frame_idx = 0
    print(f"[STEP 5B] Rendering {total} frames...")

    with tqdm(total=total, desc="Rendering frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            crop = crop_map.get(frame_idx, last_crop)
            last_crop = crop

            x  = crop["crop_x"]
            y  = crop["crop_y"]
            w  = crop["crop_w"]
            h  = crop["crop_h"]

            # Apply crop
            cropped = frame[y : y + h, x : x + w]

            # Upscale to target resolution with Lanczos
            resized = cv2.resize(
                cropped, (target_w, target_h),
                interpolation=cv2.INTER_LANCZOS4
            )

            writer.write(resized)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    print(f"[STEP 5B] Video frames written to: {temp_video}")

    # ── Mux audio from original ───────────────────────────────────────────
    print("[STEP 5B] Muxing audio from original...")
    ffmpeg_mux = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", input_video,
        "-map", "0:v",           # video from our rendered file
        "-map", "1:a?",          # audio from original (? = optional, don't fail if no audio)
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path
    ]
    subprocess.run(ffmpeg_mux, check=True)

    # Clean up temp
    if os.path.exists(temp_video):
        os.remove(temp_video)

    print(f"[STEP 5B] Final video: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — choose approach and run
# ─────────────────────────────────────────────────────────────────────────────

def render_final_video(
    input_video: str,
    crop_data_dir: str,
    frames_dir: str,
    output_path: str = "output_9x16.mp4",
    method: str = "opencv",          # "opencv" or "sendcmd"
    target_w: int = 1080,
    target_h: int = 1920
):
    """
    Main Step 5 entry point.

    Args:
        input_video   : original 16:9 video
        crop_data_dir : folder containing crop_commands.json
        frames_dir    : folder containing metadata.json
        output_path   : where to save the 9:16 output
        method        : "opencv" (recommended) or "sendcmd"
        target_w/h    : output resolution (default 1080x1920 = Full HD portrait)
    """
    # Load data
    with open(os.path.join(frames_dir, "metadata.json")) as f:
        meta = json.load(f)
    fps = meta["fps"]

    with open(os.path.join(crop_data_dir, "crop_commands.json")) as f:
        crop_commands = json.load(f)

    print(f"[STEP 5] Rendering {len(crop_commands)} frames -> {output_path}")
    print(f"[STEP 5] Target resolution: {target_w}x{target_h}")
    print(f"[STEP 5] Method: {method}")

    if method == "sendcmd":
        render_with_ffmpeg_sendcmd(input_video, crop_commands, fps, output_path, target_w, target_h)
    else:
        render_with_opencv(input_video, crop_commands, fps, output_path, target_w, target_h)

    print(f"\n[STEP 5] [DONE] Output ready: {output_path}")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    render_final_video(
        input_video   = video,
        crop_data_dir = "crop_data",
        frames_dir    = "frames",
        output_path   = "output_9x16.mp4",
        method        = "opencv"
    )
