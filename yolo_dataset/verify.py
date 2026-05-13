"""
RealSense 라이브로 best.pt 모델 검증.

키:
  q / ESC : 종료
  s       : 현재 frame screenshot 저장 (verify_shots/)
  +/-     : conf 임계값 ±0.05

표시:
  bbox + mask + 클래스명 + conf
  FPS, 검출 개수, 임계값
"""
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()
BEST_PT = SCRIPT_DIR / 'runs' / 'detect_v2' / 'weights' / 'best.pt'
SHOT_DIR = SCRIPT_DIR / 'verify_shots'

WIDTH, HEIGHT, FPS = 1280, 720, 30
INITIAL_CONF = 0.30


def main():
    if not BEST_PT.exists():
        raise SystemExit(f'!! best.pt 없음: {BEST_PT}')
    SHOT_DIR.mkdir(exist_ok=True)

    print(f'  모델 로딩: {BEST_PT}')
    model = YOLO(str(BEST_PT))
    # 첫 추론 워밍업 (CUDA kernel 컴파일)
    dummy = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    model.predict(dummy, conf=INITIAL_CONF, verbose=False)
    print('  워밍업 완료')

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(cfg)
    print(f'  RealSense 시작 ({WIDTH}x{HEIGHT}@{FPS})')
    print('  q/ESC=종료, s=스크린샷, +/-=conf 조정')

    conf = INITIAL_CONF
    fps_t0 = time.time()
    fps_n = 0
    fps_disp = 0.0
    shot_n = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())

            res = model.predict(img, conf=conf, verbose=False)[0]
            vis = res.plot()   # mask + bbox + label 자동 시각화

            n_det = 0 if res.boxes is None else len(res.boxes)

            fps_n += 1
            if fps_n >= 10:
                t1 = time.time()
                fps_disp = fps_n / (t1 - fps_t0)
                fps_t0 = t1
                fps_n = 0

            cv2.putText(vis, f'conf>={conf:.2f}  det={n_det}  fps={fps_disp:.1f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('YOLO seg verify (q=quit, s=shot, +/-=conf)', vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('s'):
                p = SHOT_DIR / f'shot_{shot_n:03d}.png'
                cv2.imwrite(str(p), vis)
                print(f'  saved {p.name}  (det={n_det})')
                shot_n += 1
            elif k in (ord('+'), ord('=')):
                conf = min(0.95, conf + 0.05)
                print(f'  conf -> {conf:.2f}')
            elif k in (ord('-'), ord('_')):
                conf = max(0.05, conf - 0.05)
                print(f'  conf -> {conf:.2f}')
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print('  종료')


if __name__ == '__main__':
    main()
