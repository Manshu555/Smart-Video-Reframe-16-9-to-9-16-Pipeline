"""
STEP 2 — DETECTION + SALIENCY + OPTICAL FLOW
=============================================
Input  : frames/ folder + metadata.json from Step 1
Output : detections.json  — per-frame bounding boxes + confidences
         saliency/         — per-frame grayscale heatmap images
         flow.json         — per-frame dominant motion vector (dx, dy)

Expected Output (detections.json):
[
  {
    "frame_id" : 0,
    "faces"    : [{"x1":120,"y1":80,"x2":300,"y2":290,"conf":0.97}],
    "persons"  : [{"x1":100,"y1":60,"x2":340,"y2":900,"conf":0.91}],
    "focus_raw": [220, 475]     <-- weighted centre of all detections
  },
  ...
]

Expected Output (saliency/):
  saliency/frame_000000_sal.png   — 8-bit grayscale, bright=important

Expected Output (flow.json):
[
  {"frame_id": 6, "dx": 3.2, "dy": -0.4},   <-- dominant pixel motion
  ...
]
"""

import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# ── Tiny built-in saliency (no extra model download needed) ──────────────────
# Uses OpenCV's spectral-residual saliency — fast and dependency-free.
# Swap with TranSalNet / UNISAL for better quality if GPU is available.

def compute_saliency_map(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a single-channel uint8 saliency map (0-255).
    Bright pixels = visually salient regions.
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, sal_map = saliency.computeSaliency(frame_bgr)
    sal_uint8 = (sal_map * 255).astype(np.uint8)
    # Blur lightly to get smooth heatmap
    sal_uint8 = cv2.GaussianBlur(sal_uint8, (21, 21), 0)
    return sal_uint8


def peak_saliency_point(sal_map: np.ndarray):
    """Return (x, y) of the brightest point in the saliency map."""
    _, _, _, max_loc = cv2.minMaxLoc(sal_map)
    return list(max_loc)   # (x, y)


# ── Optical flow ─────────────────────────────────────────────────────────────

def compute_dominant_flow(prev_gray: np.ndarray, curr_gray: np.ndarray):
    """
    Compute dense optical flow between two greyscale frames using
    Farneback method (ships with OpenCV, no extra install).
    Returns dominant (dx, dy) — median motion of the frame.

    For production use RAFT from torchvision for sub-pixel accuracy:
      from torchvision.models.optical_flow import raft_large
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    # flow shape: (H, W, 2)  where [:,:,0]=dx, [:,:,1]=dy
    dx = float(np.median(flow[:, :, 0]))
    dy = float(np.median(flow[:, :, 1]))
    return dx, dy


# ── Weighted focus-point calculator ─────────────────────────────────────────

WEIGHT_FACE   = 0.45
WEIGHT_PERSON = 0.45
WEIGHT_SAL    = 0.10


def compute_focus_point(faces, persons, sal_peak, frame_w, frame_h):
    """
    Blend face centres + body centres + saliency peak into one (x, y).
    Falls back gracefully when detections are absent.
    """
    points  = []
    weights = []

    # Face centres (highest priority)
    for f in faces:
        cx = (f["x1"] + f["x2"]) / 2
        cy = (f["y1"] + f["y2"]) / 2
        points.append([cx, cy])
        weights.append(WEIGHT_FACE * f["conf"])

    # Person body centres
    for p in persons:
        cx = (p["x1"] + p["x2"]) / 2
        cy = (p["y1"] + p["y2"]) / 2
        points.append([cx, cy])
        weights.append(WEIGHT_PERSON * p["conf"])

    # Saliency peak (always present, lowest weight)
    points.append(sal_peak)
    weights.append(WEIGHT_SAL)

    # Weighted average
    pts = np.array(points, dtype=float)
    wts = np.array(weights, dtype=float)
    wts /= wts.sum()                              # normalise to sum=1
    focus = (pts * wts[:, None]).sum(axis=0)      # (x, y)

    # Clamp to frame
    focus[0] = float(np.clip(focus[0], 0, frame_w - 1))
    focus[1] = float(np.clip(focus[1], 0, frame_h - 1))

    return focus.tolist()


# ── Main analysis function ───────────────────────────────────────────────────

def analyze_frames(frames_dir: str, output_dir: str) -> tuple:
    """
    Run detection + saliency + optical flow on all extracted frames.

    Args:
        frames_dir : path to frames/ folder (from Step 1)
        output_dir : where to save detections.json, flow.json, saliency/

    Returns:
        (detections, flow_data)  — also written to disk as JSON
    """
    # Load metadata from Step 1
    meta_path = os.path.join(frames_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    frame_w      = meta["width"]
    frame_h      = meta["height"]
    sampled_ids  = meta["sampled_ids"]

    os.makedirs(output_dir, exist_ok=True)
    sal_dir = os.path.join(output_dir, "saliency")
    os.makedirs(sal_dir, exist_ok=True)

    # Load YOLOv8 (downloads weights on first run ~6MB for nano)
    # Use yolov8s.pt or yolov8m.pt for better accuracy on GPU
    print("[STEP 2] Loading YOLOv8 model...")
    yolo = YOLO("yolov8n.pt")

    # Face detection using OpenCV's built-in Haar cascade (no extra download needed)
    print("[STEP 2] Loading OpenCV Haar cascade face detector...")
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    detections = []
    flow_data  = []
    prev_gray  = None

    print("[STEP 2] Analysing frames...")
    for frame_id in tqdm(sampled_ids, desc="Detecting + Saliency + Flow"):
        img_path = os.path.join(frames_dir, f"frame_{frame_id:06d}.jpg")
        if not os.path.exists(img_path):
            continue

        frame_bgr  = cv2.imread(img_path)
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        curr_gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ── Saliency ──────────────────────────────────────────────────────
        sal_map  = compute_saliency_map(frame_bgr)
        sal_peak = peak_saliency_point(sal_map)
        sal_path = os.path.join(sal_dir, f"frame_{frame_id:06d}_sal.png")
        cv2.imwrite(sal_path, sal_map)

        # ── YOLO person detection ──────────────────────────────────────────
        yolo_results = yolo(frame_bgr, verbose=False, classes=[0])  # class 0 = person
        persons = []
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if conf > 0.4:
                persons.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})

        # ── Haar cascade face detection ────────────────────────────────────
        faces = []
        face_rects = face_cascade.detectMultiScale(
            curr_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for (fx, fy, fw, fh) in face_rects:
            faces.append({
                "x1": float(fx), "y1": float(fy),
                "x2": float(fx + fw), "y2": float(fy + fh),
                "conf": 0.85   # Haar doesn't give a confidence, use a reasonable default
            })

        # ── Weighted focus point ───────────────────────────────────────────
        focus = compute_focus_point(faces, persons, sal_peak, frame_w, frame_h)

        detections.append({
            "frame_id"  : frame_id,
            "faces"     : faces,
            "persons"   : persons,
            "sal_peak"  : sal_peak,
            "focus_raw" : focus
        })

        # ── Optical flow ───────────────────────────────────────────────────
        if prev_gray is not None:
            dx, dy = compute_dominant_flow(prev_gray, curr_gray)
            flow_data.append({"frame_id": frame_id, "dx": dx, "dy": dy})
        else:
            flow_data.append({"frame_id": frame_id, "dx": 0.0, "dy": 0.0})

        prev_gray = curr_gray


    # Save to disk
    det_path  = os.path.join(output_dir, "detections.json")
    flow_path = os.path.join(output_dir, "flow.json")

    with open(det_path, "w")  as f: json.dump(detections, f, indent=2)
    with open(flow_path, "w") as f: json.dump(flow_data,  f, indent=2)

    print(f"[STEP 2] Detections saved : {det_path}")
    print(f"[STEP 2] Flow data saved  : {flow_path}")
    print(f"[STEP 2] Saliency maps    : {sal_dir}/")

    return detections, flow_data


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dets, flow = analyze_frames(frames_dir="frames", output_dir="analysis")
    print(f"Sample detection: {dets[0]}")
    print(f"Sample flow     : {flow[1]}")
