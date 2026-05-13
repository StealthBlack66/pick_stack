"""
실전 작업대 데이터 캡쳐 — RealSense 라이브 → raw/ 폴더에 사진 저장.

목적: 1차 학습이 흰 종이 + 단독 cube 만 학습되어 실전 작업대에서 못 잡음.
실제 환경(회로기판/케이블/펜이 함께 있는 작업대) 사진 50-100장을 추가 수집.

키:
  s / SPACE : 현재 frame 캡쳐 (raw/ 에 shot_NNN.jpg 저장)
  q / ESC   : 종료
  +/-       : 화면 zoom (UI 만, 저장은 원본)

다양성 가이드 (반드시 따라주세요):
  - 큐브 위치를 워크스페이스 전체에 분산 (좌상/우상/좌하/우하/중앙)
  - 큐브 회전·기울기 다양하게
  - 조명 변화 (창문 빛, 형광등 on/off)
  - **주변 잡음 객체 같이** — 파란 사각형, 회로기판, 케이블, 펜이 같이 보이는 사진이 핵심
  - 1 위치에 1장만 — Roboflow augmentation 이 어차피 변형
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
    """raw/ 의 기존 shot_NNN.jpg 중 최대값 + 1."""
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
        print(f'  raw/ 에 기존 shot_*.jpg 발견 → shot_{idx:03d} 부터 이어 저장')
    print('  키: s/SPACE=캡쳐, q/ESC=종료')
    print('  목표: 50-100장. 매 캡쳐 후 큐브 위치/각도/주변 객체 바꾸기')

    saved = 0
    flash_until = 0.0   # 캡쳐 직후 화면 깜빡임용
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            vis = img.copy()

            # HUD
            now = time.time()
            if now < flash_until:
                # 캡쳐 직후 흰 테두리 깜빡임
                cv2.rectangle(vis, (0, 0), (WIDTH - 1, HEIGHT - 1), (255, 255, 255), 20)
            cv2.putText(vis, f'saved={saved}  next=shot_{idx:03d}.jpg',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, 's/SPACE=capture  q/ESC=quit',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Capture dataset (s=shot, q=quit)', vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k in (ord('s'), ord(' ')):
                out = RAW_DIR / f'shot_{idx:03d}.jpg'
                cv2.imwrite(str(out), img)   # 원본(HUD 없음) 저장
                print(f'  saved {out.name}  (총 {saved + 1}장)')
                saved += 1
                idx += 1
                flash_until = now + 0.15
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f'\n  완료: 이번 세션 {saved}장 저장 → {RAW_DIR}')
        print('  다음: auto_label.py 의 CLASSES 를 [\'wood cube\'] 로 바꾸고 실행')


if __name__ == '__main__':
    main()
