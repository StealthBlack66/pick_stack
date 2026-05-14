"""
YOLO-World + SAM2 자동 prefill → raw/shot_NNN.txt (정사각형 polygon).

흐름:
  1) YOLO-World 로 cube bbox 자동 검출 (imgsz=1280, CPU 강제)
  2) 각 bbox center 픽셀을 SAM2 점 prompt → cube top 후보 mask
  3) mask 의 minAreaRect 짧은 변 = side. **center 는 mask centroid** (개선:
     이전엔 bbox center 를 강제해서 모서리 클릭 시 어긋났음)
  4) 정사각형 polygon → 정규화 → raw/shot_NNN.txt

이미 .txt 있는 사진은 skip (--overwrite 로 덮어쓰기).

다음: python3 sam2_clicker_seg.py 로 자동 결과 검토/수정.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM, YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'

YOLO_WORLD_WEIGHTS = '/home/fastcampus/Downloads/test/yolov8s-world.pt'
SAM_WEIGHTS = '/home/fastcampus/Downloads/test/sam2.1_b.pt'

PROMPT_TEXTS = [
    'wood cube', 'wooden cube', 'wooden block', 'cube', 'block',
]
DETECT_CONF_THR = 0.05
DETECT_IMGSZ = 1280
ASPECT_REJECT = 2.5
CLASS_ID = 0


def mask_to_square_polygon(mask, W, H):
    """SAM2 mask → 짧은 변 정사각형 polygon (4꼭짓점, 픽셀 좌표 int32).
    center 는 mask 의 centroid (moments) 로 — bbox center 보다 정확."""
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    rect = cv2.minAreaRect(cnt)
    (rcx, rcy), (rw, rh), angle = rect
    if min(rw, rh) <= 1:
        return None
    if max(rw, rh) / min(rw, rh) > ASPECT_REJECT:
        return None
    # centroid from moments (mask 의 무게중심 — 측면 영향 평균)
    M = cv2.moments(cnt)
    if M['m00'] > 0:
        cu = M['m10'] / M['m00']
        cv_ = M['m01'] / M['m00']
    else:
        cu, cv_ = rcx, rcy
    side = float(min(rw, rh))
    box = cv2.boxPoints(((cu, cv_), (side, side), float(angle)))
    box[:, 0] = np.clip(box[:, 0], 0, W - 1)
    box[:, 1] = np.clip(box[:, 1], 0, H - 1)
    return box.astype(np.int32)


def polygon_to_seg_line(box_int, W, H):
    poly = box_int.astype(np.float32)
    poly[:, 0] /= W
    poly[:, 1] /= H
    coords = ' '.join(f'{x:.6f} {y:.6f}' for (x, y) in poly)
    return f'{CLASS_ID} {coords}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true',
                    help='기존 .txt 도 덮어쓰기 (기본은 skip)')
    ap.add_argument('--conf', type=float, default=DETECT_CONF_THR)
    args = ap.parse_args()

    if not Path(YOLO_WORLD_WEIGHTS).exists():
        sys.exit(f'!! YOLO-World 없음: {YOLO_WORLD_WEIGHTS}')
    if not Path(SAM_WEIGHTS).exists():
        sys.exit(f'!! SAM2 없음: {SAM_WEIGHTS}')

    print(f'  YOLO-World 로딩: {YOLO_WORLD_WEIGHTS} (CPU 강제)')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_default_device('cpu')
    yolo = YOLO(YOLO_WORLD_WEIGHTS)
    yolo.set_classes(PROMPT_TEXTS)
    print(f'  prompt: {PROMPT_TEXTS}')

    print(f'  SAM2 로딩: {SAM_WEIGHTS}')
    sam = SAM(SAM_WEIGHTS)
    sam(np.zeros((720, 1280, 3), dtype=np.uint8),
        points=[[640, 360]], labels=[1], verbose=False)

    shots = sorted(RAW_DIR.glob('shot_*.jpg'))
    if not shots:
        sys.exit(f'!! shot_*.jpg 없음: {RAW_DIR}')
    print(f'  대상: {len(shots)}장 (conf={args.conf}, imgsz={DETECT_IMGSZ})')

    n_new = n_skip_existing = n_no_det = 0
    for i, ip in enumerate(shots):
        lp = ip.with_suffix('.txt')
        if lp.exists() and not args.overwrite:
            n_skip_existing += 1
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        H, W = img.shape[:2]
        r = yolo.predict(img, conf=args.conf, verbose=False, device='cpu',
                         imgsz=DETECT_IMGSZ)[0]
        if r.boxes is None or len(r.boxes) == 0:
            print(f'  [{i+1}/{len(shots)}] {ip.name} — YOLO-World 검출 0')
            n_no_det += 1
            continue
        lines = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            try:
                sres = sam(img, points=[[u, v]], labels=[1], verbose=False)
            except Exception as e:
                print(f'    SAM2 fail @ ({u:.0f},{v:.0f}): {e}')
                continue
            if not sres or sres[0].masks is None or len(sres[0].masks.data) == 0:
                continue
            mask = sres[0].masks.data[0].cpu().numpy()
            poly = mask_to_square_polygon(mask, W, H)
            if poly is None:
                continue
            lines.append(polygon_to_seg_line(poly, W, H))
        if not lines:
            print(f'  [{i+1}/{len(shots)}] {ip.name} — 유효 polygon 0')
            n_no_det += 1
            continue
        lp.write_text('\n'.join(lines), encoding='utf-8')
        n_new += 1
        print(f'  [{i+1}/{len(shots)}] {ip.name} → {len(lines)} polygon')

    print(f'\n=== 완료 ===')
    print(f'  새/덮어쓴 라벨: {n_new}장')
    print(f'  기존 라벨 유지: {n_skip_existing}장 (--overwrite 로 덮어쓰기 가능)')
    print(f'  검출 0 skip: {n_no_det}장')
    print(f'\n  다음: python3 sam2_clicker_seg.py  (자동 결과 검토/수정)')


if __name__ == '__main__':
    main()
