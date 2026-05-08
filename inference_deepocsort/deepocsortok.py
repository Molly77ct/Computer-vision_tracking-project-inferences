import cv2
import numpy as np
##import torch
from pathlib import Path
from ultralytics import YOLO
##from boxmot import DeepOcSort
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
#import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU

import torch


# ==========================
# CONFIG
# ==========================

VIDEO_PATH = "traffic.mp4"
OUTPUT_PATH = "output.mp4"

#YOLO_MODEL = "yolov8x.pt"
YOLO_MODEL = "runs/detect/train/weights/best.pt"

##CONF_THRES = 0.3
##CONF_THRES = 0.1
CONF_THRES = 0.15

# COCO: car=2, bus=5, truck=7
##VEHICLE_CLASSES = [2, 5, 7]
## car=3, van=4, truck=5, bus=8
VEHICLE_CLASSES = [3, 4, 5, 8]


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ",device)


# ==========================
# MODELOS
# ==========================

model = YOLO(YOLO_MODEL)

tracker = DeepOcSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    ##device=device,
    device="cuda:0",
    ##fp16=torch.cuda.is_available()
    half=torch.cuda.is_available()
)


# ==========================
# VIDEO
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

    # YOLO inference
    ##results = model(frame, conf=CONF_THRES, verbose=False)[0]
    results = model(
        frame,
        conf=CONF_THRES,
        classes= VEHICLE_CLASSES,
        imgsz=1280,
        verbose=False
        )[0]

    dets = []

    if results.boxes is not None:
        for b in results.boxes:
            cls = int(b.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0])

            # formato esperado por BoxMOT:
            # [x1, y1, x2, y2, score, class]
            dets.append([x1, y1, x2, y2, conf, cls])

    dets = np.array(dets) if len(dets) else np.empty((0, 6))
    print("Deteccion YOLO: ", len(dets))

    # TRACKING (nuevo formato)
    tracks = tracker.update(dets, frame)

    # ==========================
    # DIBUJAR
    # ==========================

    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, track_id, conf, cls, _ = t

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            cls = int(cls)

            label_map = {2: "Car", 5: "Bus", 7: "Truck"}
            label = label_map.get(cls, "Vehicle")

            text = f"{label} ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
out.release()
cv2.destroyAllWindows()
