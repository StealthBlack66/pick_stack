# YOLO 커스텀 학습 (auto-label + train)

## 폴더 구조

```
yolo_dataset/
  raw/               ← 여기에 RealSense 로 찍은 사진 넣기 (.jpg/.png, 50~200장 권장)
  images/{train,val} ← auto_label.py 가 자동으로 채움
  labels/{train,val} ← auto_label.py 가 자동으로 채움
  data.yaml          ← auto_label.py 가 자동 생성
  auto_label.py      ← YOLO-World 로 자동 라벨링 + split
  train.py           ← ultralytics 학습 실행
  runs/              ← 학습 결과 (best.pt 가 여기 떨어짐)
```

## 사용 순서

1. **사진 수집** — RealSense 또는 휴대폰으로 객체 사진 50~200장. `raw/` 에 넣기.
   - 다양한 각도/조명/배경 권장.
2. **클래스 설정** — `auto_label.py` 의 `CLASSES` 리스트를 학습할 객체 이름(영문)으로 수정.
   - YOLO-World 의 텍스트 프롬프트로 직접 사용되니 사진 속 객체와 매칭되는 단어로.
3. **자동 라벨링**
   ```bash
   cd ~/Downloads/test/로봇강의_예제/02_Doosan_Robot_제어/yolo_dataset
   python3 auto_label.py
   ```
4. **(선택) 라벨 검수** — `labels/train/*.txt` 일부 확인. 이상하면 그 사진은 `images/train/` 에서 삭제.
5. **학습**
   ```bash
   python3 train.py
   ```
6. **12번에 적용** — 학습 끝나면 출력된 `best.pt` 경로를 [12_비전_피크앤플레이스.py:142](../12_비전_피크앤플레이스.py#L142) `YOLO_WEIGHTS` 에 넣고 `YOLO-World` 대신 일반 `YOLO()` 로 로드하도록 init 부분 수정.

## 메모

- YOLO-World 가 못 잡는 객체는 자동 제외 → 그런 사진은 사람이 라벨링 도구로 보충해야 함.
- `train.py` 의 `EPOCHS=100` 은 데이터 양에 따라 조정. 50장이면 50, 200장 이상이면 100~150.
- CPU 만 있으면 학습이 매우 느림 — CUDA 있는 PC 권장.
