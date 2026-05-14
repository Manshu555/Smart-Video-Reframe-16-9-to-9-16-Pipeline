"""
MASTER PIPELINE — 16:9 → 9:16 Smart Reframe (Approach 3)
=========================================================

Usage:
    python run_pipeline.py input_video.mp4
    python run_pipeline.py input_video.mp4 --sample-fps 10 --output output.mp4

Pipeline Steps:
    Step 1 → Extract frames at reduced FPS
    Step 2 → Detect subjects + compute saliency + optical flow
    Step 3 → Optimise crop trajectory + Kalman smooth
    Step 4 → Scene cut detection + easing + full-FPS interpolation
    Step 5 → Render final 9:16 video

Output structure:
    workspace/
    ├── frames/             ← sampled frames + metadata.json
    ├── analysis/           ← detections.json, flow.json, saliency/
    ├── trajectory/         ← trajectory.json
    ├── crop_data/          ← crop_commands.json
    └── output_9x16.mp4     ← FINAL OUTPUT
"""

import argparse
import os
import time

from step1_extract_frames      import extract_frames
from step2_analyze_frames      import analyze_frames
from step3_trajectory          import build_trajectory
from step4_easing_interpolation import build_crop_commands
from step5_render              import render_final_video


def run_pipeline(
    input_video : str,
    workspace   : str = "workspace",
    sample_fps  : int = 5,
    output_name : str = "output_9x16.mp4",
    render_method: str = "opencv"
):
    os.makedirs(workspace, exist_ok=True)

    frames_dir    = os.path.join(workspace, "frames")
    analysis_dir  = os.path.join(workspace, "analysis")
    trajectory_dir = os.path.join(workspace, "trajectory")
    crop_data_dir = os.path.join(workspace, "crop_data")
    output_path   = os.path.join(workspace, output_name)

    total_start = time.time()

    # ── STEP 1 ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1/5 - Frame Extraction")
    print("="*60)
    t = time.time()
    meta = extract_frames(input_video, frames_dir, sample_fps=sample_fps)
    print(f"  [TIME] {time.time()-t:.1f}s")

    # ── STEP 2 ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2/5 - Detection + Saliency + Optical Flow")
    print("="*60)
    t = time.time()
    detections, flow = analyze_frames(frames_dir, analysis_dir)
    print(f"  [TIME] {time.time()-t:.1f}s")

    # ── STEP 3 ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3/5 - Trajectory Optimisation + Kalman Smoothing")
    print("="*60)
    t = time.time()
    trajectory = build_trajectory(frames_dir, analysis_dir, trajectory_dir)
    print(f"  [TIME] {time.time()-t:.1f}s")

    # ── STEP 4 ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4/5 - Scene Cuts + Easing + Full-FPS Interpolation")
    print("="*60)
    t = time.time()
    crop_commands = build_crop_commands(frames_dir, trajectory_dir, crop_data_dir)
    print(f"  [TIME] {time.time()-t:.1f}s")

    # ── STEP 5 ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 5/5 - Rendering Final 9:16 Video")
    print("="*60)
    t = time.time()
    render_final_video(
        input_video   = input_video,
        crop_data_dir = crop_data_dir,
        frames_dir    = frames_dir,
        output_path   = output_path,
        method        = render_method
    )
    print(f"  [TIME] {time.time()-t:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("  [DONE] PIPELINE COMPLETE")
    print("="*60)
    print(f"  Input          : {input_video}")
    print(f"  Output         : {output_path}")
    print(f"  Total time     : {total_time:.1f}s")
    print(f"  Frames analysed: {len(detections)}")
    print(f"  Scene cuts     : {sum(1 for e in crop_commands if e['frame_id'] > 0 and crop_commands[e['frame_id']-1]['crop_x'] != e['crop_x'])}")
    print("="*60)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart 16:9 -> 9:16 Video Reframe (Approach 3)")
    parser.add_argument("input",         type=str,            help="Path to input 16:9 video")
    parser.add_argument("--workspace",   type=str,            default="workspace")
    parser.add_argument("--sample-fps",  type=int,            default=5)
    parser.add_argument("--output",      type=str,            default="output_9x16.mp4")
    parser.add_argument("--method",      type=str,            default="opencv",
                        choices=["opencv", "sendcmd"],        help="Render method")
    args = parser.parse_args()

    run_pipeline(
        input_video   = args.input,
        workspace     = args.workspace,
        sample_fps    = args.sample_fps,
        output_name   = args.output,
        render_method = args.method
    )
