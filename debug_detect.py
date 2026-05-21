import sys
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

weights = 'yolo_dataset/runs/seg_v8_angle18/weights/best.pt'
if not os.path.exists(weights):
    print("Weights not found!")
    sys.exit(1)

model = YOLO(weights)
print("Model loaded.")

# Let's take an image from the dataset or use a dummy.
# Actually, I can't grab a live image easily without RealSense.
# Let's just inspect the model's imgsz.
print(model.model.args)
