"""
YOLO-World 기반 자동 라벨링.

사용법:
  1. raw/ 폴더에 사진(.jpg/.png) 넣기 (50-200장 권장)
  2. 아래 CLASSES 리스트를 학습할 객체 이름(영문)으로 수정
  3. python3 auto_label.py
     → images/{train,val}/, labels/{train,val}/, data.yaml 자동 생성
  4. python3 train.py 로 학습

원리:
  YOLO-World 는 open-vocab. CLASSES 의 텍스트 프롬프트로 zero-shot detect →
  bbox 결과를 YOLO 포맷(.txt, class_id cx cy w h normalized) 으로 저장.
  검출 안 된 사진은 train 셋에서 자동 제외 (빈 라벨은 학습 손상).

12번 코드와 동일하게 CPU 강제 (CLIP encoder device mismatch 회피).
"""

import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ===== 사용자 설정 =====
# 학습할 클래스 (영문, YOLO-World 텍스트 프롬프트로 직접 사용됨)
# 사진에 있는 객체 이름과 정확히 매칭되어야 검출됨.
CLASSES = [
    'wood cube',
]

DETECT_CONF_THR = 0.10    # YOLO-World conf 임계값 (12번과 동일)
TRAIN_RATIO = 0.8         # train:val = 8:2
RANDOM_SEED = 42
MIN_BOX_PER_IMAGE = 1     # 검출 0개인 사진은 자동 제외

# 경로
SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'
IMG_TRAIN = SCRIPT_DIR / 'images' / 'train'
IMG_VAL = SCRIPT_DIR / 'images' / 'val'
LBL_TRAIN = SCRIPT_DIR / 'labels' / 'train'
LBL_VAL = SCRIPT_DIR / 'labels' / 'val'
DATA_YAML = SCRIPT_DIR / 'data.yaml'

YOLO_WORLD_WEIGHTS = '/home/fastcampus/Downloads/test/yolov8s-world.pt'

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def list_raw_images():
    if not RAW_DIR.exists():
        print(f'!! raw 폴더 없음: {RAW_DIR}')
        sys.exit(1)
    files = sorted([p for p in RAW_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in VALID_EXT])
    return files


def clean_split_dirs():
    """기존 split 결과 비우기 (재실행 시 깨끗하게)."""
    for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.is_file():
                f.unlink()


def detect_and_label(model, img_path: Path):
    """한 사진 → YOLO-World 검출 → YOLO 포맷 라벨 라인 리스트."""
    img = cv2.imread(str(img_path))
    if img is None:
        return [], None
    h, w = img.shape[:2]
    results = model.predict(img, conf=DETECT_CONF_THR, verbose=False, device='cpu')
    lines = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls.item())
            if cls_id < 0 or cls_id >= len(CLASSES):
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    return lines, (w, h)


def write_data_yaml():
    content = f"""# YOLO 학습 설정 (auto_label.py 가 생성)
path: {SCRIPT_DIR}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    DATA_YAML.write_text(content, encoding='utf-8')
    print(f'  data.yaml 작성: {DATA_YAML}')


def main():
    print(f'=== Auto-label (YOLO-World) ===')
    print(f'  클래스: {CLASSES}')
    raw_imgs = list_raw_images()
    print(f'  raw/ 사진 개수: {len(raw_imgs)}')
    if not raw_imgs:
        print('  !! raw/ 가 비어있습니다. 사진 넣고 다시 실행하세요.')
        sys.exit(1)

    print(f'  YOLO-World 로딩... ({YOLO_WORLD_WEIGHTS})')
    if not Path(YOLO_WORLD_WEIGHTS).exists():
        print(f'  !! 가중치 없음: {YOLO_WORLD_WEIGHTS}')
        sys.exit(1)
    # 12번과 동일하게 CPU 강제 (set_classes 의 CUDA 텐서 device mismatch 회피)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_default_device('cpu')
    model = YOLO(YOLO_WORLD_WEIGHTS)
    model.set_classes(CLASSES)

    clean_split_dirs()

    # 라벨링 + train/val split
    random.seed(RANDOM_SEED)
    shuffled = raw_imgs[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * TRAIN_RATIO)

    stats = {'labeled': 0, 'empty': 0, 'train': 0, 'val': 0}
    for i, img_path in enumerate(shuffled):
        lines, _ = detect_and_label(model, img_path)
        if len(lines) < MIN_BOX_PER_IMAGE:
            stats['empty'] += 1
            print(f'  [{i+1}/{len(shuffled)}] {img_path.name} — 검출 0개, 제외')
            continue

        is_train = i < split_idx
        dst_img = (IMG_TRAIN if is_train else IMG_VAL) / img_path.name
        dst_lbl = (LBL_TRAIN if is_train else LBL_VAL) / (img_path.stem + '.txt')
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text('\n'.join(lines), encoding='utf-8')
        stats['labeled'] += 1
        stats['train' if is_train else 'val'] += 1
        print(f'  [{i+1}/{len(shuffled)}] {img_path.name} — {len(lines)} box → {"train" if is_train else "val"}')

    write_data_yaml()
    print(f'\n=== 완료 ===')
    print(f'  라벨링 성공: {stats["labeled"]}장 (train {stats["train"]} / val {stats["val"]})')
    print(f'  제외(검출 0): {stats["empty"]}장')
    print(f'\n  다음: python3 train.py')


if __name__ == '__main__':
    main()
