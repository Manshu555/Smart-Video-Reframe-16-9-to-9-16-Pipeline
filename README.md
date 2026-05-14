# OpenCV Video Processing Pipeline

This repository contains an OpenCV-based assignment/pipeline for processing videos. The pipeline performs object detection, trajectory calculation, easing interpolation, and final rendering of the video.

## Prerequisites

- Python 3.x
- Install dependencies via `pip install -r requirements.txt`
- Requires `yolov8n.pt` for object detection.

## Pipeline Steps

1. **`step1_extract_frames.py`**: Extracts frames from the input video (`your_video.mp4`) for further processing.
2. **`step2_analyze_frames.py`**: Analyzes the extracted frames, likely utilizing the YOLOv8 model for object tracking/detection.
3. **`step3_trajectory.py`**: Computes the trajectory of the tracked objects across frames.
4. **`step4_easing_interpolation.py`**: Applies easing functions and interpolation to smooth out trajectories and movements.
5. **`step5_render.py`**: Renders the final output video incorporating the analysis and computed trajectories.

## Other Scripts
- **`run_pipeline.py`**: An overarching script to run the steps sequentially.
- **`generate_docx.py`**: A utility script to generate documentation/reports (e.g., `approach.docx`).
- **`debug_visualise.py`**: Helps in debugging by visualizing intermediate results.

## Usage

You can likely run the complete pipeline using:
```bash
python run_pipeline.py
```
Or execute the steps individually.
