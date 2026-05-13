"""
detect bbox 라벨 → SAM2 polygon segmentation 라벨 자동 변환.

입력:
  yolo_dataset/images/{train,val}/*.jpg
  yolo_dataset/labels/{train,val}/*.txt   (YOLO detect: cls cx cy w h)

출력:
  yolo_dataset_seg/images/{train,val}/*.jpg    (hardlink — 디스크 안 늘어남)
  yolo_dataset_seg/labels/{train,val}/*.txt    (YOLO seg: cls x1 y1 x2 y2 ... xn yn)
  yolo_dataset_seg/data.yaml

원리:
  각 bbox 를 SAM2 의 bbox prompt 로 넣으면 cube top 면 mask 가 정확히 나옴.
  mask → 외곽 contour → approxPolyDP 로 점 개수 축소 → 정규화 좌표.

사용:
  python3 bbox_to_seg_label.py
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR                 # yolo_dataset/
DST_DIR = SCRIPT_DIR.parent / 'yolo_dataset_seg'

SAM_WEIGHTS = '/home/fastcampus/Downloads/test/sam2.1_b.pt'
CLASSES = ['wood cube']

# polygon 단순화 (점 개수 줄이기). 너무 작으면 거친 외곽, 너무 크면 점 폭주.
APPROX_EPS_RATIO = 0.003   # arcLength * 비율
MIN_POLY_POINTS = 6
MAX_POLY_POINTS = 50


def read_bbox_labels(label_path: Path):
    """detect 라벨 파일 → [(cls, cx, cy, w, h), ...] (normalized)."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        out.append((c, cx, cy, w, h))
    return out


def yolo_bbox_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return [float(x1), float(y1), float(x2), float(y2)]


def mask_to_polygon(mask, W, H):
    """boolean HxW mask → normalized polygon points (Nx2) 또는 None."""
    m = mask.astype(np.uint8) * 255
    # 작은 구멍/노이즈 정리
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    eps = max(1.0, APPROX_EPS_RATIO * cv2.arcLength(cnt, True))
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    # 점 개수 가드
    if len(poly) < MIN_POLY_POINTS:
        # 너무 적으면 epsilon 줄여 재시도
        poly = cv2.approxPolyDP(cnt, eps * 0.3, True).reshape(-1, 2)
    if len(poly) > MAX_POLY_POINTS:
        # 너무 많으면 등간격 샘플링
        idx = np.linspace(0, len(poly) - 1, MAX_POLY_POINTS, dtype=int)
        poly = poly[idx]
    if len(poly) < 3:
        return None
    poly = poly.astype(np.float32)
    poly[:, 0] /= W
    poly[:, 1] /= H
    poly = np.clip(poly, 0.0, 1.0)
    return poly


def polygon_to_yolo_seg_line(cls_id, poly):
    coords = ' '.join(f'{x:.6f} {y:.6f}' for (x, y) in poly)
    return f'{cls_id} {coords}'


def hardlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        # cross-device → copy
        import shutil
        shutil.copy2(src, dst)


def process_split(sam, split: str):
    src_img = SRC_DIR / 'images' / split
    src_lbl = SRC_DIR / 'labels' / split
    dst_img = DST_DIR / 'images' / split
    dst_lbl = DST_DIR / 'labels' / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in src_img.iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    n_ok = n_skip = n_total_polys = 0
    for i, ip in enumerate(imgs):
        lp = src_lbl / (ip.stem + '.txt')
        bboxes = read_bbox_labels(lp)
        if not bboxes:
            print(f'  [{i + 1}/{len(imgs)}] {ip.name} — bbox 0개, skip')
            n_skip += 1
            continue

        img = cv2.imread(str(ip))
        if img is None:
            print(f'  [{i + 1}/{len(imgs)}] {ip.name} — 읽기 실패, skip')
            n_skip += 1
            continue
        H, W = img.shape[:2]

        xyxy_list = [yolo_bbox_to_xyxy(cx, cy, w, h, W, H)
                     for (_c, cx, cy, w, h) in bboxes]
        try:
            res = sam(img, bboxes=xyxy_list, verbose=False)
        except Exception as e:
            print(f'  [{i + 1}/{len(imgs)}] {ip.name} — SAM2 fail: {e}')
            n_skip += 1
            continue
        if not res or res[0].masks is None:
            print(f'  [{i + 1}/{len(imgs)}] {ip.name} — mask 없음, skip')
            n_skip += 1
            continue
        masks = res[0].masks.data.cpu().numpy()  # (N, H, W)

        lines = []
        for (cls_id, *_), m in zip(bboxes, masks):
            poly = mask_to_polygon(m, W, H)
            if poly is None:
                continue
            lines.append(polygon_to_yolo_seg_line(cls_id, poly))

        if not lines:
            print(f'  [{i + 1}/{len(imgs)}] {ip.name} — polygon 0개, skip')
            n_skip += 1
            continue

        hardlink_or_copy(ip, dst_img / ip.name)
        (dst_lbl / (ip.stem + '.txt')).write_text('\n'.join(lines), encoding='utf-8')
        n_ok += 1
        n_total_polys += len(lines)
        print(f'  [{i + 1}/{len(imgs)}] {ip.name} → {len(lines)} polygon')

    return n_ok, n_skip, n_total_polys


def write_data_yaml():
    content = f"""# YOLO seg 학습 설정 (bbox_to_seg_label.py 가 생성)
path: {DST_DIR}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (DST_DIR / 'data.yaml').write_text(content, encoding='utf-8')
    print(f'  data.yaml: {DST_DIR / "data.yaml"}')


def main():
    if not Path(SAM_WEIGHTS).exists():
        sys.exit(f'!! SAM2 weights 없음: {SAM_WEIGHTS}')

    print(f'=== bbox → seg polygon ===')
    print(f'  src: {SRC_DIR}')
    print(f'  dst: {DST_DIR}')
    print(f'  SAM2 로드 중...')
    sam = SAM(SAM_WEIGHTS)
    # warmup
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    sam(dummy, bboxes=[[100, 100, 200, 200]], verbose=False)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    total_ok = total_skip = total_poly = 0
    for split in ('train', 'val'):
        print(f'\n--- split={split} ---')
        ok, sk, np_ = process_split(sam, split)
        total_ok += ok; total_skip += sk; total_poly += np_

    write_data_yaml()
    print(f'\n=== 완료 ===')
    print(f'  처리 성공: {total_ok}장, skip: {total_skip}장, polygon: {total_poly}개')
    print(f'  다음: python3 train_seg.py')


if __name__ == '__main__':
    main()
