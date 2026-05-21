"""
seg_v8_angle6 모델로 큐브 + 회전각도 라이브 검출.

각 큐브 검출에 대해:
  - YOLO 가 예측한 클래스 (15도 bin: '0-15', '15-30', ...)
  - 예측 마스크에서 cv2.minAreaRect 로 정확한 각도(° in 0~90, 큐브 대칭) 계산
  - 화면에 "bin / exact°" 같이 표시

조작:
  q : 종료
  c : 캡처 저장
  + / - : conf threshold
"""
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()
WEIGHTS = ROOT.parent / "seg_v8_angle6" / "weights" / "best.pt"

CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
IMGSZ = 640

# 클래스 라벨 (data.yaml 과 동일 순서)
CLASS_NAMES = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90']
# 클래스별 색상
COLORS = [
    (0, 0, 255),     # red    0-15
    (0, 165, 255),   # orange 15-30
    (0, 255, 255),   # yellow 30-45
    (0, 255, 0),     # green  45-60
    (255, 200, 0),   # cyan   60-75
    (255, 0, 200),   # magenta 75-90
]


def mask_to_angle(mask: np.ndarray) -> tuple[float, np.ndarray] | None:
    """이진 마스크 → (angle deg in [0,90), 회전 박스 4점)."""
    m = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    rect = cv2.minAreaRect(cnt)
    (_, _), (w, h), ang = rect
    if w < h:
        ang = ang + 90
    ang = ang % 90
    box = cv2.boxPoints(rect).astype(np.int32)
    return ang, box


def main():
    if not WEIGHTS.exists():
        print(f"!! weights not found: {WEIGHTS}")
        sys.exit(1)

    print(f"loading: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"!! camera {CAM_INDEX} open fail")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"cam {CAM_INDEX} opened {int(cap.get(3))}x{int(cap.get(4))}")
    print("q/ESC quit | c capture | +/- conf | f fullscreen toggle")

    win = "seg_v8 angle (bin | exact deg)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    fullscreen = False

    conf = 0.25
    cap_idx = 0
    fps_t = time.time()
    fps_n = 0
    fps_show = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = model.predict(frame, imgsz=IMGSZ, conf=conf, verbose=False)[0]

        n = 0
        if r.boxes is not None and r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            fh, fw = frame.shape[:2]
            for mk, cls, cf in zip(masks, classes, confs):
                if mk.shape != (fh, fw):
                    mk = cv2.resize(mk, (fw, fh), interpolation=cv2.INTER_NEAREST)
                res = mask_to_angle(mk)
                if res is None:
                    continue
                ang, box = res
                color = COLORS[int(cls) % len(COLORS)]
                # 마스크 오버레이
                overlay = frame.copy()
                overlay[mk > 0.5] = color
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                # 회전 박스
                cv2.drawContours(frame, [box], 0, color, 2)
                # 텍스트: 클래스명 / 정밀 각도
                label = f"{CLASS_NAMES[int(cls)]} | {ang:.1f}°  ({cf:.2f})"
                # 박스의 가장 위쪽 점 근처에 표시
                top_pt = tuple(box[box[:, 1].argmin()])
                tx = max(0, top_pt[0] - 20)
                ty = max(15, top_pt[1] - 5)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (tx, ty - th - 4), (tx + tw + 6, ty + 2), color, -1)
                cv2.putText(frame, label, (tx + 3, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                n += 1

        # HUD
        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps_show = fps_n / (time.time() - fps_t)
            fps_t = time.time()
            fps_n = 0
        cv2.putText(frame, f"conf={conf:.2f} fps={fps_show:.1f} cubes={n}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(win, frame)
        # 윈도우의 X 버튼으로 닫혔는지도 확인 → break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):       # 27 = ESC
            break
        elif key == ord("c"):
            cap_idx += 1
            out = ROOT / f"angle_capture_{cap_idx:03d}.png"
            cv2.imwrite(str(out), frame)
            print(f"saved {out}")
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
        elif key in (ord("-"), ord("_")):
            conf = max(0.05, conf - 0.05)
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                win, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if not fullscreen:
                cv2.resizeWindow(win, 1600, 900)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
