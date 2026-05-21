"""
기존 wood cube 폴리곤 라벨에서 각도를 계산해서
BIN_DEG 단위 다중 클래스 데이터셋을 만든다 (5° → 18클래스, 10° → 9클래스 등).

소스:
  - yolo_dataset/raw/{shot_*.jpg, shot_*.txt}
  - yolo_dataset/images/train/{shot_*.jpg}  + yolo_dataset/labels/train/{shot_*.txt}

각도 계산:
  cv2.minAreaRect(polygon_pts) -> ((cx,cy), (w,h), angle)
  - opencv angle 은 [-90, 0)
  - w < h 면 angle += 90
  - 큐브 90도 대칭이므로 angle %= 90
  - class_id = int(angle / BIN_DEG)

출력:
  seg_v8/multiclass/
    images/train, images/val, labels/train, labels/val
    data.yaml  (nc=N_CLASSES, names=['0-5','5-10',...] 등)
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.resolve()
DATASET = ROOT.parent.parent

SOURCES = [
    (DATASET / "raw", DATASET / "raw"),                              # 이미지와 라벨이 같은 폴더
    (DATASET / "images" / "train", DATASET / "labels" / "train"),    # 이미지/라벨 분리
]

OUT = ROOT / "multiclass"
IMG_TRAIN = OUT / "images" / "train"
IMG_VAL = OUT / "images" / "val"
LBL_TRAIN = OUT / "labels" / "train"
LBL_VAL = OUT / "labels" / "val"
for d in (IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL):
    d.mkdir(parents=True, exist_ok=True)

VAL_RATIO = 0.2
SEED = 11
BIN_DEG = 5
N_CLASSES = 90 // BIN_DEG          # 18 when BIN_DEG=5
CLASS_NAMES = [f"{i*BIN_DEG}-{(i+1)*BIN_DEG}" for i in range(N_CLASSES)]


def poly_angle_deg(coords_norm: list[float], img_w: int, img_h: int) -> float:
    """정규화 폴리곤 → minAreaRect 각도 (0~90, 큐브 90도 대칭 반영)."""
    pts = np.array([[coords_norm[i] * img_w, coords_norm[i + 1] * img_h]
                    for i in range(0, len(coords_norm), 2)], dtype=np.float32)
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    if w < h:
        ang = ang + 90
    return ang % 90


def process_pair(img_p: Path, lbl_p: Path) -> tuple[list[str], int] | None:
    """이미지+라벨 한 쌍 → (new_label_lines, count)."""
    img = cv2.imread(str(img_p))
    if img is None or not lbl_p.exists():
        return None
    h, w = img.shape[:2]
    new_lines = []
    cnt = 0
    for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = list(map(float, parts[1:]))
        if (len(coords)) % 2 != 0:
            continue
        ang = poly_angle_deg(coords, w, h)
        cls = min(int(ang / BIN_DEG), N_CLASSES - 1)
        new_lines.append(f"{cls} " + " ".join(parts[1:]))
        cnt += 1
    return new_lines, cnt


def main():
    # 기존 파일 정리 (.gitkeep 보존)
    for d in (IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL):
        for p in d.iterdir():
            if p.name == ".gitkeep":
                continue
            p.unlink()

    rng = random.Random(SEED)

    # 모든 (img, lbl) 페어 모으기
    pairs: list[tuple[Path, Path]] = []
    for img_dir, lbl_dir in SOURCES:
        for img_p in sorted(img_dir.glob("*.jpg")):
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            if lbl_p.exists():
                pairs.append((img_p, lbl_p))
    print(f"총 페어: {len(pairs)}")
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO))

    per_class = [0] * N_CLASSES
    total_pos = 0
    for i, (img_p, lbl_p) in enumerate(pairs):
        split_img_dir = IMG_VAL if i < n_val else IMG_TRAIN
        split_lbl_dir = LBL_VAL if i < n_val else LBL_TRAIN
        r = process_pair(img_p, lbl_p)
        if r is None:
            continue
        new_lines, cnt = r
        if cnt == 0:
            continue
        # 클래스별 카운트
        for line in new_lines:
            cls = int(line.split()[0])
            per_class[cls] += 1
        # 복사 + 라벨 쓰기 (확장자는 그대로 .jpg)
        out_img = split_img_dir / f"{img_p.stem}.jpg"
        shutil.copy2(str(img_p), str(out_img))
        out_lbl = split_lbl_dir / f"{img_p.stem}.txt"
        out_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        total_pos += 1

    print(f"\n쓴 이미지: train={len(list(IMG_TRAIN.glob('*.jpg')))}, val={len(list(IMG_VAL.glob('*.jpg')))}")
    print(f"클래스별 인스턴스 수:")
    for c, n in enumerate(per_class):
        bar = "#" * min(n, 60)
        print(f"  {c} ({CLASS_NAMES[c]}°): {bar} {n}")

    # data.yaml
    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"# {N_CLASSES}-class angle classification ({BIN_DEG}deg bins, cube 90deg symmetry)\n"
        f"path: {OUT}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {N_CLASSES}\n"
        f"names: {CLASS_NAMES}\n",
        encoding="utf-8",
    )
    print(f"\ndata.yaml: {yaml}")


if __name__ == "__main__":
    main()
