"""
STEP 1 — FRAME EXTRACTION
==========================
Input  : path to any video file (mp4, mov, avi, etc.)
Output : folder of JPEG frames + a metadata dict

Expected Output:
  frames/
    frame_0000.jpg
    frame_0005.jpg
    frame_0010.jpg
    ...
  metadata = {
    "fps"          : 30.0,
    "total_frames" : 900,
    "width"        : 1920,
    "height"       : 1080,
    "duration_sec" : 30.0,
    "sample_fps"   : 5,
    "sampled_ids"  : [0, 6, 12, 18, ...]   # original frame indices that were saved
  }
"""

import cv2
import os
import json
import numpy as np
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str, sample_fps: int = 5) -> dict:
    """
    Extract frames from a video at a reduced FPS for analysis.

    Args:
        video_path  : path to input 16:9 video
        output_dir  : folder where frames will be saved
        sample_fps  : how many frames per second to sample (5–10 is enough)

    Returns:
        metadata dict describing the video and which frames were saved
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # --- Read basic video properties ---
    original_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec   = total_frames / original_fps

    print(f"[STEP 1] Video Info:")
    print(f"         Resolution  : {width}x{height}")
    print(f"         FPS         : {original_fps:.2f}")
    print(f"         Duration    : {duration_sec:.2f}s")
    print(f"         Total frames: {total_frames}")

    # --- Determine which frame indices to sample ---
    # e.g. if video is 30fps and sample_fps=5, sample every 6th frame
    sample_interval = max(1, int(round(original_fps / sample_fps)))
    sampled_indices = list(range(0, total_frames, sample_interval))

    print(f"[STEP 1] Sampling every {sample_interval} frames → {len(sampled_indices)} frames to extract")

    # --- Extract and save frames ---
    saved_frames = []
    frame_idx    = 0
    sample_set   = set(sampled_indices)

    with tqdm(total=len(sampled_indices), desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in sample_set:
                # Save as high-quality JPEG
                out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_frames.append(frame_idx)
                pbar.update(1)

            frame_idx += 1

    cap.release()

    # --- Build metadata ---
    metadata = {
        "video_path"    : video_path,
        "fps"           : original_fps,
        "total_frames"  : total_frames,
        "width"         : width,
        "height"        : height,
        "duration_sec"  : duration_sec,
        "sample_fps"    : sample_fps,
        "sample_interval": sample_interval,
        "sampled_ids"   : saved_frames,
        "frames_dir"    : output_dir,
    }

    # Save metadata to disk so next steps can load it
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[STEP 1] Done. {len(saved_frames)} frames saved to: {output_dir}")
    print(f"[STEP 1] Metadata saved to: {meta_path}")

    return metadata


# ── Quick sanity-check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    meta  = extract_frames(video, output_dir="frames", sample_fps=5)
    print(json.dumps(meta, indent=2))
