"""
sam2_clicker.py 로 라벨링한 shot + 기존 Roboflow IMG 통합 → train/val 8:2 split + data.yaml.

소스:
  1. raw/shot_*.jpg + raw/shot_*.txt (사용자 SAM2 라벨링, bbox)
  2. _zip/train/images/IMG_*.jpg + _zip/train/labels/IMG_*.txt (Roboflow polygon → bbox 변환)

출력:
  images/train, images/val, labels/train, labels/val, data.yaml
"""
import random
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'
ZIP_IMG_DIR = SCRIPT_DIR / '_zip' / 'train' / 'images'
ZIP_LBL_DIR = SCRIPT_DIR / '_zip' / 'train' / 'labels'
OUT_IMG_TRAIN = SCRIPT_DIR / 'images' / 'train'
OUT_IMG_VAL = SCRIPT_DIR / 'images' / 'val'
OUT_LBL_TRAIN = SCRIPT_DIR / 'labels' / 'train'
OUT_LBL_VAL = SCRIPT_DIR / 'labels' / 'val'
DATA_YAML = SCRIPT_DIR / 'data.yaml'

CLASS_ID = 0
TRAIN_RATIO = 0.8
SEED = 42


def polygon_to_bbox_line(line: str) -> str:
    """polygon YOLO line → bbox YOLO line."""
    toks = line.strip().split()
    if len(toks) < 7:
        return None   # 폴리곤은 최소 3 점 = 7 토큰
    cls = toks[0]
    coords = [float(t) for t in toks[1:]]
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    bw = x_max - x_min
    bh = y_max - y_min
    return f'{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'


def convert_label_text(text: str) -> str:
    """텍스트 안의 모든 라인을 polygon→bbox 변환 (이미 bbox 인 라인은 그대로)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) == 5:
            out.append(line)   # 이미 bbox
        else:
            converted = polygon_to_bbox_line(line)
            if converted:
                out.append(converted)
    return '\n'.join(out)


def clean_out_dirs():
    for d in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LBL_TRAIN, OUT_LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.is_file():
                f.unlink()


def collect_pairs():
    """[(img_path, label_text), ...]"""
    pairs = []

    # 1) SAM2 라벨링된 shot
    for img in sorted(RAW_DIR.glob('shot_*.jpg')):
        lbl = img.with_suffix('.txt')
        if not lbl.exists():
            continue
        text = convert_label_text(lbl.read_text(encoding='utf-8'))
        if text:
            pairs.append((img, text))
    n_shot = len(pairs)

    # 2) Roboflow IMG
    if ZIP_IMG_DIR.exists():
        for img in sorted(ZIP_IMG_DIR.glob('IMG_*.jpg')):
            lbl = ZIP_LBL_DIR / (img.stem + '.txt')
            if not lbl.exists():
                continue
            text = convert_label_text(lbl.read_text(encoding='utf-8'))
            if text:
                pairs.append((img, text))
    n_img = len(pairs) - n_shot

    print(f'  소스: shot {n_shot}장 + IMG {n_img}장 = {len(pairs)}장')
    return pairs


def write_data_yaml(nc=1, names=None):
    names = names or ['wood cube']
    DATA_YAML.write_text(f"""path: {SCRIPT_DIR}
train: images/train
val: images/val

nc: {nc}
names: {names}
""", encoding='utf-8')
    print(f'  data.yaml 작성')


def main():
    clean_out_dirs()
    pairs = collect_pairs()
    if not pairs:
        raise SystemExit('!! 라벨링된 사진이 없습니다. sam2_clicker.py 먼저 실행.')

    random.seed(SEED)
    random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_RATIO)
    train, val = pairs[:split], pairs[split:]
    print(f'  split: train {len(train)} / val {len(val)}')

    for split_name, lst in [('train', train), ('val', val)]:
        img_dir = OUT_IMG_TRAIN if split_name == 'train' else OUT_IMG_VAL
        lbl_dir = OUT_LBL_TRAIN if split_name == 'train' else OUT_LBL_VAL
        for img, text in lst:
            shutil.copy2(img, img_dir / img.name)
            (lbl_dir / (img.stem + '.txt')).write_text(text, encoding='utf-8')

    write_data_yaml()
    print('\n  다음: python3 train.py')


if __name__ == '__main__':
    main()
