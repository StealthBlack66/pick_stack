"""
seg_v8 클래스 1, 2 데이터셋 자동 라벨링 + train/val 분할.

흐름:
  1. 각 클래스 폴더의 images/train 에서 prefix 가 '원본' 또는 '점선윤곽' 인 PNG만 대상
  2. RGBA 알파 채널 → 최대 contour → YOLO seg polygon (normalized) 라벨 생성
  3. 이미지는 RGB 로 변환 후 덮어쓰기 (YOLO 는 3채널 가정)
  4. 같은 prefix 제외 셔플 후 20% 를 val 로 이동 (이미지+라벨 같이)

라벨 포맷: 한 줄당 `class_id x1 y1 x2 y2 ... xN yN` (모두 0~1 normalized)
클래스 ID 는 각 클래스 폴더 내부적으로 항상 0 (data.yaml 의 names 가 단일).
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.resolve()   # .../runs/seg_v8
ALLOWED_PREFIXES = {"원본", "점선윤곽"}
VAL_RATIO = 0.2
SEED = 42
MIN_CONTOUR_AREA = 5         # 너무 작은 잡음 contour 무시
EPS_RATIO = 0.005            # approxPolyDP epsilon = perimeter * EPS_RATIO


def alpha_to_polygon(alpha: np.ndarray) -> list[tuple[float, float]] | None:
    """알파(uint8 mask) → 최대 contour → 정규화 안 된 (x,y) 리스트.

    여러 contour 중 면적 최대만 사용. 너무 작으면 None.
    """
    mask = (alpha > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        return None
    peri = cv2.arcLength(cnt, closed=True)
    approx = cv2.approxPolyDP(cnt, EPS_RATIO * peri, closed=True)
    pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
    if len(pts) < 3:
        return None
    return pts


def process_image(img_path: Path, label_path: Path) -> bool:
    """이미지 1장 처리: 라벨 .txt 작성 + RGB 변환 후 덮어쓰기.

    return: 성공 여부
    """
    arr = np.array(Image.open(img_path))
    if arr.ndim != 3 or arr.shape[2] != 4:
        print(f"  skip (no alpha): {img_path.name}")
        return False
    h, w = arr.shape[:2]
    poly = alpha_to_polygon(arr[..., 3])
    if poly is None:
        print(f"  skip (no contour): {img_path.name}")
        return False
    # 정규화
    norm = []
    for x, y in poly:
        norm.append(f"{x / w:.6f}")
        norm.append(f"{y / h:.6f}")
    line = "0 " + " ".join(norm)
    label_path.write_text(line + "\n", encoding="utf-8")
    # RGBA → RGB (알파 흰배경 합성, 모델 입력 일관성)
    rgba = arr.astype(np.float32)
    a = rgba[..., 3:4] / 255.0
    rgb = (rgba[..., :3] * a + 255.0 * (1 - a)).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(img_path)
    return True


def process_class(cls: str) -> None:
    print(f"\n=== class {cls} ===")
    cls_dir = ROOT / cls
    img_train = cls_dir / "images" / "train"
    img_val = cls_dir / "images" / "val"
    lbl_train = cls_dir / "labels" / "train"
    lbl_val = cls_dir / "labels" / "val"

    # 대상 이미지 수집
    candidates = []
    skipped_prefix = 0
    for p in sorted(img_train.iterdir()):
        if p.suffix.lower() != ".png":
            continue
        prefix = p.name.split("_")[0]
        if prefix not in ALLOWED_PREFIXES:
            skipped_prefix += 1
            # 사용 안 할 이미지는 제거해서 dataset 깨끗하게
            p.unlink()
            continue
        candidates.append(p)
    print(f"  대상={len(candidates)}장,  prefix 제외 삭제={skipped_prefix}장")

    # 라벨 생성
    ok = 0
    bad = []
    for img_p in candidates:
        lbl_p = lbl_train / (img_p.stem + ".txt")
        if process_image(img_p, lbl_p):
            ok += 1
        else:
            bad.append(img_p)
    # 실패 이미지는 라벨 없이 두면 학습 손해 → 같이 제거
    for img_p in bad:
        img_p.unlink(missing_ok=True)
    print(f"  라벨 생성={ok}장,  실패 제거={len(bad)}장")

    # train/val 분할 (셔플 후 20%)
    paired = [p for p in img_train.iterdir() if p.suffix.lower() == ".png"]
    rng = random.Random(SEED)
    rng.shuffle(paired)
    n_val = max(1, int(len(paired) * VAL_RATIO))
    val_set = paired[:n_val]
    for img_p in val_set:
        lbl_src = lbl_train / (img_p.stem + ".txt")
        shutil.move(str(img_p), str(img_val / img_p.name))
        if lbl_src.exists():
            shutil.move(str(lbl_src), str(lbl_val / lbl_src.name))
    print(f"  val 이동={n_val}장, 최종 train={len(paired) - n_val}장")


def main() -> None:
    for cls in ("1", "2"):
        process_class(cls)
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
