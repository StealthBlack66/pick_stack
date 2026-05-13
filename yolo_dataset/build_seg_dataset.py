"""
raw/*.jpg + raw/*.txt (seg polygon) → yolo_dataset_seg/{images,labels}/{train,val}/
+ data.yaml 8:2 split.

sam2_clicker_seg.py 가 만든 정사각형 polygon 라벨을 학습용 디렉토리로 통합.
"""
import os
import random
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'
DST_DIR = SCRIPT_DIR.parent / 'yolo_dataset_seg'

CLASSES = ['wood cube']
TRAIN_RATIO = 0.8
SEED = 42

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def collect_labeled_pairs():
    """[(img_path, label_text), ...] — raw/ 의 사진 중 .txt 가 있는 것만."""
    pairs = []
    for ip in sorted(RAW_DIR.iterdir()):
        if not ip.is_file() or ip.suffix.lower() not in VALID_EXT:
            continue
        lp = ip.with_suffix('.txt')
        if not lp.exists():
            continue
        text = lp.read_text(encoding='utf-8').strip()
        if not text:
            continue
        pairs.append((ip, text))
    return pairs


def hardlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clean_split_dirs():
    for split in ('train', 'val'):
        for sub in ('images', 'labels'):
            d = DST_DIR / sub / split
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()


def write_data_yaml():
    (DST_DIR / 'data.yaml').write_text(f"""# YOLO seg 학습 설정 (build_seg_dataset.py 가 생성)
path: {DST_DIR}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
""", encoding='utf-8')
    print(f'  data.yaml: {DST_DIR / "data.yaml"}')


def main():
    pairs = collect_labeled_pairs()
    if not pairs:
        sys.exit(f'!! 라벨된 사진 없음 (raw/*.jpg + raw/*.txt). '
                 f'먼저 sam2_clicker_seg.py 실행하세요.')
    print(f'  라벨된 사진: {len(pairs)}장')

    DST_DIR.mkdir(parents=True, exist_ok=True)
    clean_split_dirs()

    random.seed(SEED)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * TRAIN_RATIO))
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]
    # 데이터가 너무 적어 val 이 비면 train 의 마지막 1장을 val 로
    if not val and len(train) >= 2:
        val = [train.pop()]
    print(f'  split: train {len(train)} / val {len(val)}')

    for split_name, lst in [('train', train), ('val', val)]:
        img_dir = DST_DIR / 'images' / split_name
        lbl_dir = DST_DIR / 'labels' / split_name
        for ip, text in lst:
            hardlink_or_copy(ip, img_dir / ip.name)
            (lbl_dir / (ip.stem + '.txt')).write_text(text, encoding='utf-8')

    write_data_yaml()
    print(f'\n=== 완료 ===')
    print(f'  train: {len(train)}장, val: {len(val)}장')
    print(f'  다음: python3 train_seg.py')


if __name__ == '__main__':
    main()
