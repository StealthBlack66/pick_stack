import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# init realsense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

# load model
weights = str(Path('yolo_dataset/runs/seg_v8_angle18/weights/best.pt').absolute())
model = YOLO(weights)

for _ in range(15):  # wait for auto-exposure
    frames = align.process(pipeline.wait_for_frames())

frames = align.process(pipeline.wait_for_frames())
cf = frames.get_color_frame()
df = frames.get_depth_frame()
color = np.asanyarray(cf.get_data())

r = model.predict(color, conf=0.10, imgsz=640, verbose=False)[0]

if r.boxes is None or len(r.boxes) == 0:
    print("YOLO found 0 boxes.")
else:
    xywh = r.boxes.xywh.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    print(f"YOLO found {len(xywh)} boxes (conf>=0.10).")
    for i in range(len(xywh)):
        cx, cy, w, h = xywh[i]
        c = confs[i]
        z = df.get_distance(int(cx), int(cy))
        print(f"Box {i}: conf={c:.2f}, cx={cx:.1f}, cy={cy:.1f}, z_center={z:.3f}m")

pipeline.stop()
