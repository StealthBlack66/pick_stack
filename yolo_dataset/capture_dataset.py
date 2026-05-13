"""
RealSense 라이브 → 학습용 사진 캡쳐 (raw/shot_NNN.jpg).

키:
  s / SPACE : 현재 frame 캡쳐
  q / ESC   : 종료

다양성 가이드 (반드시 따라주세요):
  - 큐브 위치를 워크스페이스 전체에 분산 (좌상/우상/좌하/우하/중앙)
  - 큐브 회전·기울기 다양하게 (각종 각도)
  - 조명 변화 (창문 빛 on/off, 형광등)
  - 주변 잡음 객체 같이 (그리퍼 일부, 케이블, 다른 물체)
  - cube 개수도 다양 (1·3·5·10개)
  - 1 위치당 1장만 — augmentation 이 변형 만듦
  - 권장 100~150장
"""
from pathlib import Path
import re
import time

import cv2
import numpy as np
import pyrealsense2 as rs

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'

WIDTH, HEIGHT, FPS = 1280, 720, 30


def next_shot_index():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in RAW_DIR.glob('shot_*.jpg'):
        m = re.match(r'shot_(\d+)\.jpg', p.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 0


def main():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(cfg)
    print(f'  RealSense 시작 ({WIDTH}x{HEIGHT}@{FPS})')

    idx = next_shot_index()
    if idx > 0:
        print(f'  기존 shot 발견 → shot_{idx:03d} 부터 이어 저장')
    print('  키: s/SPACE=캡쳐, q/ESC=종료')
    print('  목표: 100~150장. 매 캡쳐 후 위치/각도/조명/주변 객체 바꾸기')

    saved = 0
    flash_until = 0.0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            vis = img.copy()

            now = time.time()
            if now < flash_until:
                cv2.rectangle(vis, (0, 0), (WIDTH - 1, HEIGHT - 1), (255, 255, 255), 20)
            cv2.putText(vis, f'saved={saved}  next=shot_{idx:03d}.jpg',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, 's/SPACE=capture  q/ESC=quit',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Capture dataset', vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k in (ord('s'), ord(' ')):
                out = RAW_DIR / f'shot_{idx:03d}.jpg'
                cv2.imwrite(str(out), img)
                print(f'  saved {out.name}  (총 {saved + 1}장)')
                saved += 1
                idx += 1
                flash_until = now + 0.15
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f'\n  완료: 이번 세션 {saved}장 저장 → {RAW_DIR}')
        print('  다음: python3 auto_label.py')


if __name__ == '__main__':
    main()
