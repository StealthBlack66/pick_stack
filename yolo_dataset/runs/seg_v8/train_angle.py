"""
seg_v8 다중 클래스 각도 분류 + 세그먼트 모델 학습.
NAME 은 data.yaml 의 nc(클래스 수) 로 결정됨 — angle18 (5°) / angle9 (10°) / angle6 (15°) 등.
"""
import re
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()
DATASET = ROOT.parent.parent
DATA_YAML = ROOT / "multiclass" / "data.yaml"
WEIGHTS = DATASET / "yolov8s-seg.pt"
RUNS = ROOT.parent


def _name_from_yaml():
    """data.yaml 의 nc 값으로 NAME 자동 결정 — angle{nc} 형태."""
    if not DATA_YAML.exists():
        return "seg_v8_angle"
    m = re.search(r"^nc:\s*(\d+)", DATA_YAML.read_text(encoding="utf-8"), re.M)
    return f"seg_v8_angle{m.group(1)}" if m else "seg_v8_angle"


NAME = _name_from_yaml()

EPOCHS = 80
IMG_SIZE = 640        # 원본 1280x720 이라 640 권장
BATCH = 8             # 학습 이미지가 9장뿐이라 작게


def main():
    print(f"data: {DATA_YAML}")
    print(f"weights: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(RUNS),
        name=NAME,
        exist_ok=True,
        verbose=True,
        # 사용자 요구: 회전 augmentation 없음
        degrees=0.0,
        flipud=0.0,
        fliplr=0.0,    # 회전 정보를 보존해야 하니까 좌우 플립도 끔
    )
    best = RUNS / NAME / "weights" / "best.pt"
    print(f"\nbest.pt: {best}")


if __name__ == "__main__":
    main()
