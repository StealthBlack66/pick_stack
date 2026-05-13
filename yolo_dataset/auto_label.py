"""
YOLO-World 기반 자동 라벨링 — raw/ → yolo_dataset/{images,labels}/{train,val} + data.yaml.

원리:
  YOLO-World 는 open-vocab. CLASSES 텍스트 프롬프트로 zero-shot detect →
  bbox 결과를 YOLO 포맷(.txt, class_id cx cy w h normalized) 으로 저장.
  검출 0개 사진은 train 셋에서 자동 제외 (빈 라벨은 학습 손상).

CLIP encoder device mismatch 회피를 위해 CPU 강제.
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
# 데이터셋 클래스명 (학습용, data.yaml 에 들어감) — 1 클래스
CLASSES = [
    'wood cube',
]
# YOLO-World 검출 prompt (synonym 여러 개로 recall ↑). 모두 같은 cls_id=0 으로 라벨링.
PROMPT_TEXTS = [
    'wood cube',
    'wooden cube',
    'wooden block',
    'cube',
    'block',
]

DETECT_CONF_THR = 0.05
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
MIN_BOX_PER_IMAGE = 1

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
        sys.exit(f'!! raw 폴더 없음: {RAW_DIR}')
    return sorted([p for p in RAW_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in VALID_EXT])


def clean_split_dirs():
    for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.is_file():
                f.unlink()


def detect_and_label(model, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return [], None
    h, w = img.shape[:2]
    results = model.predict(img, conf=DETECT_CONF_THR, verbose=False, device='cpu',
                             imgsz=1280)
    lines = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            # PROMPT_TEXTS 어느 텍스트로 검출됐든 모두 wood cube (cls_id=0) 로 라벨
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    return lines, (w, h)


def write_data_yaml():
    DATA_YAML.write_text(f"""# YOLO 학습 설정 (auto_label.py 가 생성)
path: {SCRIPT_DIR}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
""", encoding='utf-8')
    print(f'  data.yaml 작성: {DATA_YAML}')


def main():
    print(f'=== Auto-label (YOLO-World) ===')
    print(f'  클래스: {CLASSES}')
    raw_imgs = list_raw_images()
    print(f'  raw/ 사진: {len(raw_imgs)}장')
    if not raw_imgs:
        sys.exit('!! raw/ 비어있음 — 먼저 capture_dataset.py 실행')

    if not Path(YOLO_WORLD_WEIGHTS).exists():
        sys.exit(f'!! YOLO-World weights 없음: {YOLO_WORLD_WEIGHTS}')
    print(f'  YOLO-World 로딩: {YOLO_WORLD_WEIGHTS}')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_default_device('cpu')
    model = YOLO(YOLO_WORLD_WEIGHTS)
    model.set_classes(PROMPT_TEXTS)
    print(f'  prompt 텍스트: {PROMPT_TEXTS}')

    clean_split_dirs()

    random.seed(RANDOM_SEED)
    shuffled = raw_imgs[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * TRAIN_RATIO)

    stats = {'labeled': 0, 'empty': 0, 'train': 0, 'val': 0}
    for i, img_path in enumerate(shuffled):
        lines, _ = detect_and_label(model, img_path)
        if len(lines) < MIN_BOX_PER_IMAGE:
            stats['empty'] += 1
            print(f'  [{i+1}/{len(shuffled)}] {img_path.name} — 검출 0, 제외')
            continue
        is_train = i < split_idx
        dst_img = (IMG_TRAIN if is_train else IMG_VAL) / img_path.name
        dst_lbl = (LBL_TRAIN if is_train else LBL_VAL) / (img_path.stem + '.txt')
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text('\n'.join(lines), encoding='utf-8')
        stats['labeled'] += 1
        stats['train' if is_train else 'val'] += 1
        print(f'  [{i+1}/{len(shuffled)}] {img_path.name} — {len(lines)} box → '
              f'{"train" if is_train else "val"}')

    write_data_yaml()
    print(f'\n=== 완료 ===')
    print(f'  라벨링 성공: {stats["labeled"]}장 (train {stats["train"]} / val {stats["val"]})')
    print(f'  제외(검출 0): {stats["empty"]}장')
    print(f'\n  다음: python3 point_to_seg_label.py')


if __name__ == '__main__':
    main()
