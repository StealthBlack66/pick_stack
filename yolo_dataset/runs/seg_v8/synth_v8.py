"""
seg_v8 copy-paste 합성:
  - 현재 seg_v8/{1,2}/images/{train,val} + labels/{train,val} 의 패치 + 폴리곤 라벨을 소스로 사용
  - 알파 마스크는 폴리곤 라벨에서 cv2.fillPoly 로 재구성
  - 배경: yolo_dataset/raw/*.jpg + yolo_dataset/images/train/*.jpg (총 29장 내외)
  - 합성 규칙: 회전 없음 (사용자 요구), 스케일 0.8~3.5x, 임의 위치, 패치 1~3개
  - 음성 샘플(라벨 없음) 일정 비율 포함 → false positive 억제
  - 결과는 seg_v8/{1,2}/images/{train,val} + labels/{train,val} 에 **덮어쓰기**
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.resolve()              # .../runs/seg_v8
DATASET = ROOT.parent.parent                        # .../yolo_dataset
BG_DIRS = [DATASET / "raw", DATASET / "images" / "train"]

N_POS_PER_CLASS = 240        # 클래스당 합성 양성 샘플
N_NEG_PER_CLASS = 60         # 클래스당 배경만 (라벨 0줄)
VAL_RATIO = 0.2
PASTE_PER_IMG = (1, 3)       # 한 합성 이미지에 객체 1~3개
SCALE_RANGE = (1.5, 5.0)     # 패치 원본 크기 대비
SEED = 7
TARGET_W, TARGET_H = 640, 480
JITTER_HSV = True            # 패치 색조/밝기 약간 흔들기


def load_patch_mask(img_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """RGB 패치 + 폴리곤에서 재구성한 binary mask."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    line = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not line:
        return None
    parts = line[0].split()
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        return None
    coords = list(map(float, parts[1:]))
    pts = np.array([[int(coords[i] * w), int(coords[i + 1] * h)]
                    for i in range(0, len(coords), 2)], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return img, mask


def load_class_patches(cls: str) -> list[tuple[np.ndarray, np.ndarray]]:
    base = ROOT / cls
    patches = []
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        for img_p in sorted(img_dir.glob("*.png")):
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            if not lbl_p.exists():
                continue
            r = load_patch_mask(img_p, lbl_p)
            if r is not None:
                patches.append(r)
    return patches


def load_backgrounds() -> list[np.ndarray]:
    bgs = []
    for d in BG_DIRS:
        for p in sorted(d.glob("*.jpg")):
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            # 통일된 사이즈로 리사이즈 (long side = 800 로 고정 후 letterbox 안 함)
            h, w = im.shape[:2]
            sc = min(TARGET_W / w, TARGET_H / h) * 1.3   # 살짝 크게 → 랜덤 crop
            im = cv2.resize(im, (int(w * sc), int(h * sc)))
            bgs.append(im)
    return bgs


def random_crop(bg: np.ndarray) -> np.ndarray:
    h, w = bg.shape[:2]
    if w <= TARGET_W or h <= TARGET_H:
        return cv2.resize(bg, (TARGET_W, TARGET_H))
    x0 = random.randint(0, w - TARGET_W)
    y0 = random.randint(0, h - TARGET_H)
    return bg[y0:y0 + TARGET_H, x0:x0 + TARGET_W].copy()


def jitter_hsv(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """약한 HSV jitter — 회전·기하 변형 없음."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-5, 5)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-20, 20), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + rng.randint(-20, 20), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste_one(canvas: np.ndarray,
              patch: np.ndarray, mask: np.ndarray,
              rng: random.Random) -> list[tuple[float, float]] | None:
    """canvas 에 패치 1개 합성 + 합성된 영역의 폴리곤(canvas 좌표, 정규화 안 됨) 반환."""
    ph, pw = patch.shape[:2]
    scale = rng.uniform(*SCALE_RANGE)
    nw, nh = max(8, int(pw * scale)), max(8, int(ph * scale))
    if nw >= TARGET_W or nh >= TARGET_H:
        return None
    p_scaled = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    m_scaled = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if JITTER_HSV:
        p_scaled = jitter_hsv(p_scaled, rng)

    x0 = rng.randint(0, TARGET_W - nw)
    y0 = rng.randint(0, TARGET_H - nh)

    # 알파 블렌딩
    roi = canvas[y0:y0 + nh, x0:x0 + nw]
    m3 = (m_scaled.astype(np.float32) / 255.0)[..., None]
    blended = (p_scaled.astype(np.float32) * m3 +
               roi.astype(np.float32) * (1.0 - m3)).clip(0, 255).astype(np.uint8)
    canvas[y0:y0 + nh, x0:x0 + nw] = blended

    # 합성 후 마스크의 contour → 폴리곤 (canvas 절대좌표)
    contours, _ = cv2.findContours(m_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
    if len(approx) < 3:
        return None
    poly = [(float(p[0][0] + x0), float(p[0][1] + y0)) for p in approx]
    return poly


def polys_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """간단 bbox 겹침 체크 (정확 IoU 까진 안 봄)."""
    ax = [p[0] for p in a]; ay = [p[1] for p in a]
    bx = [p[0] for p in b]; by = [p[1] for p in b]
    return not (max(ax) < min(bx) or max(bx) < min(ax)
                or max(ay) < min(by) or max(by) < min(ay))


def normalize_poly(poly: list[tuple[float, float]]) -> str:
    parts = []
    for x, y in poly:
        parts.append(f"{x / TARGET_W:.6f}")
        parts.append(f"{y / TARGET_H:.6f}")
    return "0 " + " ".join(parts)


def synth_class(cls: str, patches, bgs) -> None:
    rng = random.Random(SEED + int(cls))
    print(f"\n=== synth class {cls} (patches={len(patches)}, bgs={len(bgs)}) ===")
    cls_dir = ROOT / cls
    # 기존 train/val 의 파일들 싹 정리
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = cls_dir / sub / split
            for p in d.iterdir():
                if p.name == ".gitkeep":
                    continue
                p.unlink()
            # .gitkeep 보존

    # 합성
    samples = []   # (img_array, label_lines)
    # 양성
    for i in range(N_POS_PER_CLASS):
        bg = rng.choice(bgs)
        canvas = random_crop(bg)
        n_paste = rng.randint(*PASTE_PER_IMG)
        placed_polys = []
        labels = []
        tries = 0
        while len(placed_polys) < n_paste and tries < n_paste * 5:
            tries += 1
            patch, mask = rng.choice(patches)
            poly = paste_one(canvas, patch, mask, rng)
            if poly is None:
                continue
            # 다른 합성 객체와 겹치면 스킵 (혼란 방지)
            if any(polys_overlap(poly, q) for q in placed_polys):
                continue
            placed_polys.append(poly)
            labels.append(normalize_poly(poly))
        if not labels:
            continue
        samples.append((canvas, labels, True))
    # 음성 (배경만)
    for i in range(N_NEG_PER_CLASS):
        bg = rng.choice(bgs)
        canvas = random_crop(bg)
        samples.append((canvas, [], False))

    rng.shuffle(samples)
    n_val = int(len(samples) * VAL_RATIO)
    val_set = set(range(n_val))

    pos_train = pos_val = neg_train = neg_val = 0
    for i, (img, labels, is_pos) in enumerate(samples):
        split = "val" if i in val_set else "train"
        name = f"synth_{cls}_{i:04d}"
        cv2.imwrite(str(cls_dir / "images" / split / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        lbl_p = cls_dir / "labels" / split / f"{name}.txt"
        if labels:
            lbl_p.write_text("\n".join(labels) + "\n", encoding="utf-8")
            if split == "val": pos_val += 1
            else: pos_train += 1
        else:
            # 명시적으로 빈 라벨 파일 작성 (YOLO 가 background image 로 인식)
            lbl_p.write_text("", encoding="utf-8")
            if split == "val": neg_val += 1
            else: neg_train += 1
    print(f"  train: pos={pos_train} neg={neg_train}")
    print(f"  val  : pos={pos_val}  neg={neg_val}")


def main():
    bgs = load_backgrounds()
    if not bgs:
        print("!! 배경 이미지 없음")
        return
    print(f"backgrounds loaded: {len(bgs)}")
    for cls in ("1", "2"):
        patches = load_class_patches(cls)
        if not patches:
            print(f"!! class {cls} 패치 없음")
            continue
        synth_class(cls, patches, bgs)
    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
