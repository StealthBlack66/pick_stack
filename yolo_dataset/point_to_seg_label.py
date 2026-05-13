"""
SAM2 점 prompt + 정사각형 강제 → cube **상단면**만 정확히 segment 하는 라벨링.

기존 bbox_to_seg_label.py 의 문제: SAM2 bbox prompt 는 cube 전체(top+측면)를
mask 로 줘서 학습 polygon aspect ratio 가 평균 2:1 로 길쭉. 모델이 그대로
학습 → 추론 mask 도 cube 전체.

이 스크립트:
  1) 기존 detect bbox 라벨의 cube 중심 픽셀 (cx*W, cy*H) 를 점 prompt 로 SAM2 호출
  2) 반환 mask 의 minAreaRect → 짧은 변 = cube top side
  3) **정사각형 강제**: side x side 사각형, center=점 prompt 좌표, angle 유지
  4) 정사각형 polygon → 정규화 → seg 라벨 작성

기존 라벨은 yolo_dataset_seg/labels_v2_bbox_backup/ 로 자동 백업.

사용:
  python3 point_to_seg_label.py
"""
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR                     # yolo_dataset/
DST_DIR = SCRIPT_DIR.parent / 'yolo_dataset_seg'

SAM_WEIGHTS = '/home/fastcampus/Downloads/test/sam2.1_b.pt'
CLASSES = ['wood cube']

# 정사각형 검증 임계: mask aspect ratio 가 이보다 크면 SAM2 가 cube 외 영역까지
# 잡았다고 보고 그 instance 는 라벨에서 제외 (학습 손상 방지).
ASPECT_REJECT = 2.5

# polygon 4꼭짓점만 사용 (정사각형 강제했으므로 더 많은 점 불필요).
# YOLOv8-seg 학습에는 polygon 최소 3점 이상이면 OK.


def read_bbox_labels(label_path: Path):
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


def mask_to_square_polygon(mask, point_uv, W, H):
    """SAM2 mask → minAreaRect 짧은 변으로 정사각형 polygon (normalized 4점).
    실패 시 None.
    """
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
    (_rcx, _rcy), (rw, rh), angle = rect
    if min(rw, rh) <= 1:
        return None
    aspect = max(rw, rh) / min(rw, rh)
    if aspect > ASPECT_REJECT:
        return None  # cube 외 영역까지 잡힘 — 라벨 손상 방지로 reject

    # 정사각형 강제: 짧은 변 길이 = cube top side. center 는 점 prompt 좌표
    # (사용자가 cube top 중앙으로 본 점) 사용 — minAreaRect center 가
    # 측면 영향으로 cube 중심에서 벗어나도 점 prompt 가 더 신뢰적.
    side = float(min(rw, rh))
    u, v = float(point_uv[0]), float(point_uv[1])
    box = cv2.boxPoints(((u, v), (side, side), float(angle)))
    box[:, 0] = np.clip(box[:, 0], 0, W - 1) / W
    box[:, 1] = np.clip(box[:, 1], 0, H - 1) / H
    return box.astype(np.float32)


def polygon_to_yolo_seg_line(cls_id, poly):
    coords = ' '.join(f'{x:.6f} {y:.6f}' for (x, y) in poly)
    return f'{cls_id} {coords}'


def hardlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def backup_existing_labels():
    """기존 seg 라벨이 있으면 labels_v2_bbox_backup/ 으로 이동."""
    seg_lbl = DST_DIR / 'labels'
    if not seg_lbl.exists():
        return
    backup = DST_DIR / 'labels_v2_bbox_backup'
    if backup.exists():
        print(f'  기존 백업 발견 → 그대로 유지: {backup}')
        # 새 라벨 디렉토리 비우기
        for split in ('train', 'val'):
            d = seg_lbl / split
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
        return
    print(f'  기존 seg 라벨 → {backup} 로 백업')
    shutil.move(str(seg_lbl), str(backup))


def process_split(sam, split: str):
    src_img = SRC_DIR / 'images' / split
    src_lbl = SRC_DIR / 'labels' / split
    dst_img = DST_DIR / 'images' / split
    dst_lbl = DST_DIR / 'labels' / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in src_img.iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    n_ok = n_skip = n_total = n_reject = 0
    for i, ip in enumerate(imgs):
        lp = src_lbl / (ip.stem + '.txt')
        bboxes = read_bbox_labels(lp)
        if not bboxes:
            n_skip += 1
            print(f'  [{i + 1}/{len(imgs)}] {ip.name} — bbox 0개, skip')
            continue
        img = cv2.imread(str(ip))
        if img is None:
            n_skip += 1
            continue
        H, W = img.shape[:2]

        lines = []
        for (cls_id, cx, cy, _w, _h) in bboxes:
            u, v = cx * W, cy * H
            try:
                res = sam(img, points=[[u, v]], labels=[1], verbose=False)
            except Exception as e:
                print(f'    SAM2 fail @ ({u:.0f},{v:.0f}): {e}')
                continue
            if not res or res[0].masks is None or len(res[0].masks.data) == 0:
                continue
            mask = res[0].masks.data[0].cpu().numpy()
            poly = mask_to_square_polygon(mask, (u, v), W, H)
            if poly is None:
                n_reject += 1
                continue
            lines.append(polygon_to_yolo_seg_line(cls_id, poly))
            n_total += 1

        if not lines:
            n_skip += 1
            print(f'  [{i + 1}/{len(imgs)}] {ip.name} — 유효 polygon 0개, skip')
            continue

        hardlink_or_copy(ip, dst_img / ip.name)
        (dst_lbl / (ip.stem + '.txt')).write_text('\n'.join(lines), encoding='utf-8')
        n_ok += 1
        print(f'  [{i + 1}/{len(imgs)}] {ip.name} → {len(lines)} polygon')

    return n_ok, n_skip, n_total, n_reject


def write_data_yaml():
    content = f"""# YOLO seg 학습 설정 (point_to_seg_label.py 가 생성)
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

    print(f'=== Point prompt + 정사각형 강제 → seg 라벨 ===')
    print(f'  src: {SRC_DIR}')
    print(f'  dst: {DST_DIR}')

    backup_existing_labels()

    print(f'  SAM2 로드...')
    sam = SAM(SAM_WEIGHTS)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    sam(dummy, points=[[320, 240]], labels=[1], verbose=False)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    total_ok = total_skip = total_poly = total_reject = 0
    for split in ('train', 'val'):
        print(f'\n--- split={split} ---')
        ok, sk, tot, rj = process_split(sam, split)
        total_ok += ok; total_skip += sk; total_poly += tot; total_reject += rj

    write_data_yaml()
    print(f'\n=== 완료 ===')
    print(f'  이미지 OK: {total_ok}장, skip: {total_skip}장')
    print(f'  polygon: {total_poly}개 (reject by aspect>2.5: {total_reject}개)')
    print(f'  다음: python3 train_seg.py  (NAME=seg_v3 으로 미리 변경)')


if __name__ == '__main__':
    main()
