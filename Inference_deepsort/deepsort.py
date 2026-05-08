import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort # New Import
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = "traffic.mp4"
OUTPUT_PATH = "output.mp4"
YOLO_MODEL = "best.pt"
CONF_THRES = 0.15
VEHICLE_CLASSES = [3, 4, 5, 8]
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# MODELS
# ==========================
model = YOLO(YOLO_MODEL).to(device)

# Initialize DeepSort Realtime
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet", # Use "mobilenet" if you don't have a custom ReID model
    half=True,
    bgr=True
)

# ==========================
# VIDEO SETUP
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# ==========================
# LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO inference
    results = model(
        frame,
        conf=CONF_THRES,
        classes=VEHICLE_CLASSES,
        imgsz=1280,
        verbose=False
    )[0]

    # 2. Prepare detections for DeepSort Realtime
    # Format: [([x, y, w, h], conf, cls), ...]
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = r
        
        # Convert xyxy to [left, top, w, h]
        w = x2 - x1
        h = y2 - y1
        detections.append(([x1, y1, w, h], score, int(cls)))

    # 3. DeepSort Update
    tracks = tracker.update_tracks(detections, frame=frame) 

    # ==========================
    # DRAWING
    # ==========================
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb() # Returns [left, top, right, bottom]
        cls = track.get_det_class()
        
        x1, y1, x2, y2 = map(int, ltrb)

        label_map = {2: "Car", 3: "Car", 4: "Van", 5: "Truck", 7: "Truck", 8: "Bus"}
        label = label_map.get(cls, "Vehicle")
        text = f"{label} ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("DeepSort Realtime", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
