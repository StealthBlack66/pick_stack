"""
seg_v8 cls1 / cls2 두 모델을 웹캠 라이브로 띄워서 양쪽 클래스 동시에 감지/세그.

조작:
  q : 종료
  c : 캡처 저장 (현재 폴더에 capture_NNN.png)
  + / - : confidence threshold 조정 (0.05 step)

기본:
  cam_index = 0
  imgsz = 320 (학습 해상도와 동일)
  conf = 0.25
"""
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()
RUNS = ROOT.parent

W1 = RUNS / "seg_v8_cls1" / "weights" / "best.pt"
W2 = RUNS / "seg_v8_cls2" / "weights" / "best.pt"

CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IMGSZ = 320
COLOR1 = (0, 0, 255)     # 빨강 (BGR) - class 1
COLOR2 = (255, 100, 0)   # 파랑 - class 2


def draw_results(frame: np.ndarray, result, color: tuple[int, int, int], label: str) -> None:
    """결과의 mask/box 를 frame 에 in-place 로 그린다."""
    if result.masks is None and result.boxes is None:
        return

    # 마스크 (있으면) — 컬러 오버레이
    if result.masks is not None:
        for m in result.masks.data.cpu().numpy():
            mh, mw = m.shape
            fh, fw = frame.shape[:2]
            if (mh, mw) != (fh, fw):
                m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
            mask_bool = m > 0.5
            overlay = frame.copy()
            overlay[mask_bool] = color
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # bbox + 라벨 + conf
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {c:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main() -> None:
    if not W1.exists() or not W2.exists():
        print(f"!! 가중치 누락: {W1} 또는 {W2}")
        sys.exit(1)

    print(f"loading models...")
    m1 = YOLO(str(W1))
    m2 = YOLO(str(W2))
    print(f"  cls1: {W1}")
    print(f"  cls2: {W2}")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"!! 카메라 {CAM_INDEX} 열기 실패")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"  cam {CAM_INDEX} opened {int(cap.get(3))}x{int(cap.get(4))}")
    print("  q: quit, c: capture, +/-: conf")

    conf = 0.25
    cap_idx = 0
    fps_t = time.time()
    fps_n = 0
    fps_show = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read fail")
            break

        # 두 모델 별도 추론
        r1 = m1.predict(frame, imgsz=IMGSZ, conf=conf, verbose=False)[0]
        r2 = m2.predict(frame, imgsz=IMGSZ, conf=conf, verbose=False)[0]

        draw_results(frame, r1, COLOR1, "1")
        draw_results(frame, r2, COLOR2, "2")

        # HUD
        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps_show = fps_n / (time.time() - fps_t)
            fps_t = time.time()
            fps_n = 0
        n1 = len(r1.boxes) if r1.boxes is not None else 0
        n2 = len(r2.boxes) if r2.boxes is not None else 0
        hud = f"conf={conf:.2f}  fps={fps_show:.1f}  cls1={n1} cls2={n2}"
        cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.imshow("seg_v8 cls1(red)+cls2(blue)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            cap_idx += 1
            out = ROOT / f"capture_{cap_idx:03d}.png"
            cv2.imwrite(str(out), frame)
            print(f"saved {out}")
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
        elif key in (ord("-"), ord("_")):
            conf = max(0.05, conf - 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
