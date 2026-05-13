"""
YOLO Detection 학습.

사용법:
  1. auto_label.py 가 만든 data.yaml 이 있어야 함
  2. python3 train.py
  3. 학습 완료 후 best.pt 경로가 출력됨 → 12번의 YOLO_WEIGHTS 로 교체

기본값:
  - 모델: yolov8n.pt (가장 가벼움)
  - epochs: 100, imgsz: 640, batch: 16
  - device: GPU 자동 (CUDA 없으면 CPU)
"""

from pathlib import Path
import sys

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_YAML = SCRIPT_DIR / 'data.yaml'

# 학습 시작 가중치 — auto_label.py 가 bbox 라벨 만드므로 detect 모델
START_WEIGHTS = '/home/fastcampus/Downloads/test/yolov8n.pt'

# 하이퍼파라미터
EPOCHS = 70
IMG_SIZE = 640
BATCH = 16
PROJECT = str(SCRIPT_DIR / 'runs')
NAME = 'detect_v2'


def main():
    if not DATA_YAML.exists():
        print(f'!! data.yaml 없음: {DATA_YAML}')
        print('   먼저 python3 auto_label.py 실행하세요.')
        sys.exit(1)
    if not Path(START_WEIGHTS).exists():
        print(f'!! 시작 가중치 없음: {START_WEIGHTS}')
        sys.exit(1)

    print(f'=== YOLO Detection 학습 ===')
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
    print(f'\n  12번에서 사용하려면:')
    print(f"    YOLO_WEIGHTS = '{best_pt}'")


if __name__ == '__main__':
    main()
