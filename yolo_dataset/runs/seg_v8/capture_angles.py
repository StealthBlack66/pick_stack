"""
각도별 다중 클래스 라벨링 캡처 도구.

흐름:
  - webcam (video6) 라이브
  - 기존 wood cube 디텍터 seg_v7/best.pt 로 큐브 자동 마스크
  - 큐브를 원하는 각도로 놓고 숫자키 1~9 누르면, 현재 프레임에 보이는 모든 큐브가 그 클래스로 저장
  - 'r' 키: 마지막 저장 취소 (실수 방지)
  - 'q' 키: 종료
  - 화면 좌상단에 클래스별 누적 카운트 표시

출력:
  로봇강의_예제/02_Doosan_Robot_제어/yolo_dataset/runs/seg_v8/multiclass/
    images/train/    capN_idx.jpg   (640x480 RGB)
    labels/train/    capN_idx.txt   (YOLO seg polygon, class_id = N-1)

train/val 분할은 캡처 끝나고 split 스크립트로 별도 처리.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()                         # .../runs/seg_v8
DETECTOR_WEIGHTS = ROOT.parent / "seg_v7" / "weights" / "best.pt"
OUT_DIR = ROOT / "multiclass"
IMG_DIR = OUT_DIR / "images" / "train"
LBL_DIR = OUT_DIR / "labels" / "train"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
IMGSZ = 640
DETECT_CONF = 0.35
COLORS = [
    (0, 0, 255), (255, 100, 0), (0, 200, 0), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255), (200, 200, 200),
]


def mask_to_polygon(mask: np.ndarray) -> list[tuple[float, float]] | None:
    """이진 마스크 → 최대 contour → approx polygon 점들 (이미지 절대 좌표)."""
    m = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 100:
        return None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
    if len(approx) < 3:
        return None
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def main() -> None:
    if not DETECTOR_WEIGHTS.exists():
        print(f"!! detector weights not found: {DETECTOR_WEIGHTS}")
        sys.exit(1)

    print(f"loading detector: {DETECTOR_WEIGHTS}")
    detector = YOLO(str(DETECTOR_WEIGHTS))

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"!! camera {CAM_INDEX} 열기 실패")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"cam {CAM_INDEX} opened {int(cap.get(3))}x{int(cap.get(4))}")
    print("=" * 50)
    print("조작:")
    print("  1~9 : 현재 화면의 모든 큐브를 해당 클래스로 저장")
    print("  r   : 마지막 저장 취소")
    print("  q   : 종료")
    print("=" * 50)

    saved_per_class: dict[int, int] = {}
    history: list[list[Path]] = []   # 각 save 의 (img_path, lbl_path) 묶음

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read fail")
            break

        # 매 프레임 디텍션
        r = detector.predict(frame, imgsz=IMGSZ, conf=DETECT_CONF, verbose=False)[0]

        # 화면용 오버레이
        disp = frame.copy()
        polys = []   # 이번 프레임 폴리곤 모음 (저장용)
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                mh, mw = m.shape
                fh, fw = frame.shape[:2]
                if (mh, mw) != (fh, fw):
                    m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
                poly = mask_to_polygon(m)
                if poly is None:
                    continue
                polys.append(poly)
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(disp, [pts], True, (255, 255, 255), 2)

        # HUD
        hud_lines = [f"cubes detected: {len(polys)}"]
        for c, n in sorted(saved_per_class.items()):
            hud_lines.append(f"class {c}: {n}")
        for i, line in enumerate(hud_lines):
            color = COLORS[(i - 1) % len(COLORS)] if i > 0 else (0, 255, 0)
            cv2.putText(disp, line, (8, 22 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(disp, "1-9 save | r undo | q quit",
                    (8, disp.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("capture_angles (white outline = detected cube)", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            if not history:
                print("undo: history empty")
                continue
            last_files = history.pop()
            for p in last_files:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            # 클래스 카운트 -1 (어떤 클래스였는지 파일명에서 추출)
            if last_files:
                # cap{cls}_xxx.* 에서 cls 추출
                fn = last_files[0].stem  # cap{cls}_xxx
                try:
                    cls = int(fn.split("_")[0].replace("cap", ""))
                    saved_per_class[cls] = max(0, saved_per_class.get(cls, 0) - 1)
                except Exception:
                    pass
            print(f"undid last save ({len(last_files)} files)")
        elif ord("1") <= key <= ord("9"):
            cls = key - ord("0")
            if not polys:
                print(f"  [class {cls}] 감지된 큐브 없음 — 스킵")
                continue
            idx = saved_per_class.get(cls, 0) + 1
            name = f"cap{cls}_{idx:03d}"
            img_path = IMG_DIR / f"{name}.jpg"
            lbl_path = LBL_DIR / f"{name}.txt"
            # 저장
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            fh, fw = frame.shape[:2]
            for poly in polys:
                parts = [f"{cls - 1}"]   # YOLO class id = cls - 1
                for x, y in poly:
                    parts.append(f"{x / fw:.6f}")
                    parts.append(f"{y / fh:.6f}")
                lines.append(" ".join(parts))
            lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            saved_per_class[cls] = idx
            history.append([img_path, lbl_path])
            print(f"  [class {cls}] saved {name}  (cubes={len(polys)})  total={idx}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== 최종 ===")
    for c, n in sorted(saved_per_class.items()):
        print(f"  class {c}: {n} images")
    print(f"  저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
