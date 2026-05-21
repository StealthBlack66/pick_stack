"""
seg_v8 클래스 1, 2 각각 독립 학습.

- 시작 가중치: ../../yolov8s-seg.pt (기존에 다운로드된 파일 재사용)
- 출력: runs/seg_v8_cls{N}/  (yolo_dataset/runs/ 아래)
- imgsz=320 — 원본 패치가 ~26-37px 라 320 정도가 적절 (640 까지는 과대)
- epochs=50, batch=16

사용:
  python3 train_v8.py            # 두 클래스 모두 학습
  python3 train_v8.py 1          # 클래스 1만
  python3 train_v8.py 2          # 클래스 2만
"""
from pathlib import Path
import sys

from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()              # .../runs/seg_v8
DATASET_ROOT = ROOT.parent.parent                   # .../yolo_dataset
RUNS = ROOT.parent                                   # .../runs
WEIGHTS = DATASET_ROOT / "yolov8s-seg.pt"

EPOCHS = 50
IMG_SIZE = 320
BATCH = 16


def train_class(cls: str) -> Path:
    data_yaml = ROOT / cls / "data.yaml"
    name = f"seg_v8_cls{cls}"
    print(f"\n=== 학습 시작: class {cls} ===")
    print(f"  data: {data_yaml}")
    print(f"  weights: {WEIGHTS}")
    print(f"  output: {RUNS / name}")

    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    best = RUNS / name / "weights" / "best.pt"
    print(f"  best.pt: {best}")
    return best


def main() -> None:
    classes = sys.argv[1:] or ["1", "2"]
    results = {}
    for c in classes:
        results[c] = train_class(c)
    print("\n=== 전체 완료 ===")
    for c, p in results.items():
        print(f"  class {c}: {p}")


if __name__ == "__main__":
    main()
