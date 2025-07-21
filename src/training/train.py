"""
Example training script
"""

import os

from ultralytics import YOLO

data_path = "/projectnb/npbssmic/ac25/Defect-Detection/src/training/data.yaml"

model = YOLO("yolov8n.pt")

results = model.train(
    data=data_path,
    epochs=100,
    patience=20,
    imgsz=640,
    save=True,
    plots=True,
    workers=8,
    device=os.getenv("CUDA_VISIBLE_DEVICES"),
)
