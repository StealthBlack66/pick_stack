"""
YOLO Segmentation 학습.

사용법:
  1. bbox_to_seg_label.py 가 만든 yolo_dataset_seg/data.yaml 이 있어야 함
  2. python3 train_seg.py
  3. 학습 완료 후 best.pt 경로가 출력됨 → 15번 YOLO_WEIGHTS 로 교체

기본값:
  - 모델: yolov8n-seg.pt (자동 다운로드, segmentation 헤드 포함)
  - epochs: 80, imgsz: 640, batch: 16
  - device: GPU 자동 (CUDA 없으면 CPU)
"""
from pathlib import Path
import sys

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_YAML = SCRIPT_DIR.parent / 'yolo_dataset_seg' / 'data.yaml'

START_WEIGHTS = 'yolov8n-seg.pt'   # ultralytics 가 없으면 자동 다운로드

EPOCHS = 80
IMG_SIZE = 640
BATCH = 16
PROJECT = str(SCRIPT_DIR / 'runs')   # 기존 runs/ 와 같은 위치 (detect_v2 옆에 seg_vN 생성)


def _next_seg_name():
    """runs/seg_v* 중 최대 번호 + 1. 사용자가 매번 NAME 수정 안 해도 됨."""
    runs_dir = Path(PROJECT)
    nums = []
    if runs_dir.exists():
        for p in runs_dir.glob('seg_v*'):
            try:
                nums.append(int(p.name.split('_v')[1]))
            except (IndexError, ValueError):
                pass
    return f'seg_v{max(nums) + 1 if nums else 1}'


NAME = _next_seg_name()


def main():
    if not DATA_YAML.exists():
        print(f'!! data.yaml 없음: {DATA_YAML}')
        print('   먼저 python3 bbox_to_seg_label.py 실행하세요.')
        sys.exit(1)

    print(f'=== YOLO Segmentation 학습 ===')
    print(f'  data: {DATA_YAML}')
    print(f'  start weights: {START_WEIGHTS}')
    print(f'  epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH}')

    model = YOLO(START_WEIGHTS)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        exist_ok=False,
    )

    best_pt = Path(PROJECT) / NAME / 'weights' / 'best.pt'
    print(f'\n=== 완료 ===')
    print(f'  best.pt: {best_pt}')
    print(f'  metrics: {results.results_dict if hasattr(results, "results_dict") else results}')
    print(f'\n  15번에서 사용:')
    print(f"    YOLO_WEIGHTS = '{best_pt}'")


if __name__ == '__main__':
    main()
