"""
raw/shot_*.jpg 중 **라벨이 없는** 사진만 detect_v2 best.pt 로 자동 bbox 라벨링.

기존 sam2_clicker.py 의 수동 클릭 대체. detect_v2 는 cube 가 이미 학습돼 있어
새 사진의 bbox 검출이 정확. 자동 → 사용자 노동 최소.

기존 라벨 (raw/shot_NNN.txt) 가 있는 경우는 건드리지 않음 (수동 라벨 보호).

흐름:
  1) capture_dataset.py 로 새 사진 추가 캡쳐
  2) python3 auto_label_raw_shots.py   ← 이 스크립트
  3) python3 merge_dataset.py          → yolo_dataset/images,labels split
  4) python3 point_to_seg_label.py     → yolo_dataset_seg/labels (seg polygon)
  5) train_seg.py 의 NAME 갱신 후 python3 train_seg.py

옵션:
  --conf 0.30        : 검출 임계
  --overwrite        : 기존 라벨도 덮어쓰기 (기본은 skip)
  --weights <path>   : 다른 weight 사용 (기본 detect_v2/best.pt)
"""
import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'
DEFAULT_WEIGHTS = SCRIPT_DIR / 'runs' / 'detect_v2' / 'weights' / 'best.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--weights', type=str, default=str(DEFAULT_WEIGHTS))
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f'!! weights 없음: {weights}')
    print(f'  weights: {weights}')
    print(f'  conf: {args.conf}, overwrite: {args.overwrite}')

    model = YOLO(str(weights))

    shots = sorted(RAW_DIR.glob('shot_*.jpg'))
    if not shots:
        sys.exit(f'!! raw/shot_*.jpg 없음 — 먼저 capture_dataset.py 실행')
    print(f'  대상 사진: {len(shots)}장')

    n_new = n_skip_existing = n_no_det = 0
    for ip in shots:
        lbl = ip.with_suffix('.txt')
        if lbl.exists() and not args.overwrite:
            n_skip_existing += 1
            continue
        img = cv2.imread(str(ip))
        if img is None:
            print(f'  {ip.name} — 읽기 실패')
            continue
        H, W = img.shape[:2]
        r = model(img, conf=args.conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            print(f'  {ip.name} — 검출 0개, 라벨 안 만듦')
            n_no_det += 1
            continue
        lines = []
        for b in r.boxes:
            cls_id = int(b.cls.item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        lbl.write_text('\n'.join(lines), encoding='utf-8')
        n_new += 1
        print(f'  {ip.name} → {len(lines)} bbox')

    print(f'\n=== 완료 ===')
    print(f'  새 라벨: {n_new}장')
    print(f'  기존 라벨 유지: {n_skip_existing}장')
    print(f'  검출 0개로 skip: {n_no_det}장')
    print(f'\n  다음: python3 merge_dataset.py')


if __name__ == '__main__':
    main()
