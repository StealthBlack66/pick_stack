"""
YOLO Segmentation 학습 — yolov8s-seg 시작 가중치.

사용:
  1. point_to_seg_label.py 가 만든 yolo_dataset_seg/data.yaml 필요
  2. python3 train_seg.py
  3. 학습 후 best.pt 경로 출력 → 15번 YOLO_WEIGHTS 로 교체

설정:
  - 모델: yolov8s-seg.pt (n 보다 한 단계 큼, 자동 다운로드)
  - epochs: 80, imgsz: 640, batch: 16
  - NAME 은 runs/seg_v* 의 다음 번호로 자동 결정
"""
from pathlib import Path
import sys

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_YAML = SCRIPT_DIR.parent / 'yolo_dataset_seg' / 'data.yaml'

START_WEIGHTS = 'yolov8s-seg.pt'   # s 사이즈 (자동 다운로드)

EPOCHS = 80
IMG_SIZE = 640
BATCH = 16
PROJECT = str(SCRIPT_DIR / 'runs')


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
        print('   먼저 point_to_seg_label.py 실행하세요.')
        sys.exit(1)

    print(f'=== YOLO Segmentation 학습 ===')
    print(f'  data: {DATA_YAML}')
    print(f'  start weights: {START_WEIGHTS}')
    print(f'  epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH}')
    print(f'  NAME: {NAME}')

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
