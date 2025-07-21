"""
Example of tuning hyperparameters
"""

import os

from ultralytics import YOLO

data_path = "/projectnb/npbssmic/ac25/Defect-Detection/src/training/data.yaml"

model = YOLO("yolov8n.pt")

model.tune(
    data=data_path,
    epochs=60,
    iterations=100,
    optimizer="AdamW",
    plots=True,
    save=True,
    val=True,
    imgsz=640,
    device=os.getenv("CUDA_VISIBLE_DEVICES"),
    workers=8,
)
