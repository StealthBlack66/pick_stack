"""
SAM2 클릭 → cube 상단면 polygon 라벨링.

raw/shot_*.jpg 를 하나씩 보여주고, 사용자가 cube 위 클릭 1번 → SAM2 mask →
minAreaRect 짧은 변으로 정사각형 polygon 생성 → 정규화하여 raw/shot_NNN.txt 에
seg 포맷 (cls x1 y1 x2 y2 x3 y3 x4 y4) 으로 저장.

키 / 마우스:
  좌클릭          : cube top 중심 클릭 → 즉시 polygon 추가
  n / ENTER       : 현재 polygon 들 저장 + 다음 사진
  r               : 마지막 클릭 취소
  R (대문자)      : 모두 reset
  s               : skip (라벨 없이 다음)
  q / ESC         : 종료
  ←/→ (a/d)       : 이전/다음 사진 (저장 안 됨)

출력: raw/shot_NNN.txt (정사각형 polygon, normalized).
cube 0개면 빈 파일 안 만듦.

다음: python3 build_seg_dataset.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'
SAM_WEIGHTS = '/home/fastcampus/Downloads/test/sam2.1_b.pt'
CLASS_ID = 0   # wood cube

ASPECT_REJECT = 2.5  # mask aspect 너무 크면 cube 외 영역 → 무시

WINDOW = 'SAM2 clicker SEG (click=add, n=save+next, r=undo, R=reset, s=skip, q=quit)'


def mask_to_square_polygon(mask, point_uv, W, H):
    """SAM2 mask → 짧은 변 기준 정사각형 polygon (4 꼭짓점, 픽셀 좌표 int32)."""
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    rect = cv2.minAreaRect(cnt)
    (_rcx, _rcy), (rw, rh), angle = rect
    if min(rw, rh) <= 1:
        return None
    if max(rw, rh) / min(rw, rh) > ASPECT_REJECT:
        return None
    side = float(min(rw, rh))
    u, v = float(point_uv[0]), float(point_uv[1])
    box = cv2.boxPoints(((u, v), (side, side), float(angle)))
    box[:, 0] = np.clip(box[:, 0], 0, W - 1)
    box[:, 1] = np.clip(box[:, 1], 0, H - 1)
    return box.astype(np.int32)


def polygon_to_seg_line(box_int, W, H):
    """정수 polygon → YOLO seg 포맷 라인 (정규화)."""
    poly = box_int.astype(np.float32)
    poly[:, 0] /= W
    poly[:, 1] /= H
    coords = ' '.join(f'{x:.6f} {y:.6f}' for (x, y) in poly)
    return f'{CLASS_ID} {coords}'


def main():
    if not Path(SAM_WEIGHTS).exists():
        sys.exit(f'!! SAM2 weights 없음: {SAM_WEIGHTS}')

    print(f'  SAM2 로드: {SAM_WEIGHTS}')
    sam = SAM(SAM_WEIGHTS)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    sam(dummy, points=[[640, 360]], labels=[1], verbose=False)
    print('  워밍업 완료')

    shots = sorted(RAW_DIR.glob('shot_*.jpg'))
    if not shots:
        sys.exit(f'!! shot_*.jpg 없음: {RAW_DIR}')
    print(f'  대상 사진: {len(shots)}장\n')

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    state = {'click': None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['click'] = (x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)

    idx = 0
    saved_n = 0
    skipped_n = 0
    while idx < len(shots):
        img_path = shots[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'  [{idx+1}/{len(shots)}] 읽기 실패 — skip')
            idx += 1
            continue
        H, W = img.shape[:2]
        polys = []  # 누적된 polygon (int32 4x2)
        click_points = []  # 클릭 위치 (시각화용)

        def render():
            vis = img.copy()
            for poly in polys:
                cv2.polylines(vis, [poly], True, (0, 255, 255), 2)
            for (px, py) in click_points:
                cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(vis,
                        f'[{idx+1}/{len(shots)}] {img_path.name}  '
                        f'polys={len(polys)}  saved={saved_n} skip={skipped_n}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, 'click=add  n/ENTER=save+next  r=undo  R=reset  s=skip  q=quit',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow(WINDOW, vis)

        render()
        action = None
        while action is None:
            k = cv2.waitKey(20) & 0xFF
            if state['click'] is not None:
                x, y = state['click']
                state['click'] = None
                try:
                    res = sam(img, points=[[x, y]], labels=[1], verbose=False)
                    masks = (res[0].masks.data.cpu().numpy()
                             if res[0].masks is not None else None)
                except Exception as e:
                    print(f'    SAM2 추론 실패: {e}')
                    masks = None
                if masks is not None and len(masks) > 0:
                    poly = mask_to_square_polygon(masks[0].astype(bool), (x, y), W, H)
                    if poly is not None:
                        polys.append(poly)
                        click_points.append((x, y))
                        print(f'    click ({x},{y}) → polygon 추가 (총 {len(polys)})')
                    else:
                        print(f'    click ({x},{y}) → mask 부적합 (aspect / 면적)')
                render()
            if k == 255:
                continue
            if k in (ord('q'), 27):
                action = 'quit'
            elif k in (ord('n'), 13, 10):
                action = 'save'
            elif k == ord('s'):
                action = 'skip'
            elif k == ord('r'):
                if polys:
                    polys.pop()
                    if click_points:
                        click_points.pop()
                    print('    undo last click')
                    render()
            elif k == ord('R'):
                polys = []
                click_points = []
                render()
            elif k in (ord('a'), 81):
                action = 'prev'
            elif k in (ord('d'), 83):
                action = 'next_noop'

        if action == 'quit':
            break
        elif action == 'skip':
            skipped_n += 1
            idx += 1
        elif action == 'prev':
            idx = max(0, idx - 1)
        elif action == 'next_noop':
            idx += 1
        elif action == 'save':
            if polys:
                lines = [polygon_to_seg_line(p, W, H) for p in polys]
                lbl_path = img_path.with_suffix('.txt')
                lbl_path.write_text('\n'.join(lines), encoding='utf-8')
                print(f'  [{idx+1}/{len(shots)}] saved {lbl_path.name} ({len(polys)} polygon)')
                saved_n += 1
            else:
                print(f'  [{idx+1}/{len(shots)}] polygon 0 — 라벨 안 저장 (skip)')
                skipped_n += 1
            idx += 1

    cv2.destroyAllWindows()
    print(f'\n=== 완료 ===  saved={saved_n}  skipped={skipped_n}  total={len(shots)}')
    print(f'  다음: python3 build_seg_dataset.py')


if __name__ == '__main__':
    main()
