"""
4점 클릭으로 cube 상단면 polygon 라벨링 + 자동 prefill 검토/수정.

흐름:
  - 시작 시 raw/shot_NNN.txt 가 있으면 polygon 로드해서 화면에 표시
    (auto_prefill_seg.py 결과를 검토하는 용도)
  - 사용자는 잘못된 polygon 만 우클릭으로 삭제, 누락된 cube 만 4점 클릭
  - n / ENTER 로 저장 → 다음 사진

키 / 마우스:
  좌클릭 1~3      : pending 꼭지점 (빨간 점)
  좌클릭 4번째    : polygon 추가 (자동 angle 정렬)
  우클릭          : 그 위치를 포함하는 polygon 삭제
  n / ENTER       : 저장 + 다음
  r               : pending 점 또는 마지막 polygon undo
  R               : 전체 reset (자동 prefill 도 사라짐)
  s               : skip
  q / ESC         : 종료
  a / d           : 이전/다음 사진 (저장 안 됨)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'
CLASS_ID = 0

WINDOW = '4-click + prefill review (L=add R=del n=save+next r=undo R=reset s=skip q=quit)'


def four_points_to_polygon(pts, W, H):
    arr = np.asarray(pts, dtype=np.float32)
    cx = float(arr[:, 0].mean())
    cy = float(arr[:, 1].mean())
    angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
    order = np.argsort(angles)
    s = arr[order]
    s[:, 0] = np.clip(s[:, 0], 0, W - 1)
    s[:, 1] = np.clip(s[:, 1], 0, H - 1)
    return s.astype(np.int32)


def polygon_to_seg_line(box_int, W, H):
    poly = box_int.astype(np.float32)
    poly[:, 0] /= W
    poly[:, 1] /= H
    coords = ' '.join(f'{x:.6f} {y:.6f}' for (x, y) in poly)
    return f'{CLASS_ID} {coords}'


def load_existing_label(lbl_path: Path, W, H):
    """기존 .txt 의 normalized polygon → 픽셀 좌표 int32 4x2 list."""
    polys = []
    if not lbl_path.exists():
        return polys
    for line in lbl_path.read_text().splitlines():
        toks = line.strip().split()
        if len(toks) < 9:   # cls + 4*(x,y)
            continue
        coords = list(map(float, toks[1:]))
        xs = coords[0::2]
        ys = coords[1::2]
        if len(xs) != 4:
            continue
        pts = np.array(list(zip(xs, ys)), dtype=np.float32)
        pts[:, 0] *= W
        pts[:, 1] *= H
        polys.append(pts.astype(np.int32))
    return polys


def main():
    shots = sorted(RAW_DIR.glob('shot_*.jpg'))
    if not shots:
        sys.exit(f'!! shot_*.jpg 없음: {RAW_DIR}')
    print(f'  대상 사진: {len(shots)}장 (기존 .txt 있으면 prefill 로 표시)\n')

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    state = {'lclick': None, 'rclick': None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['lclick'] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state['rclick'] = (x, y)

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
        polys = load_existing_label(img_path.with_suffix('.txt'), W, H)
        pending = []
        prefill_n = len(polys)
        if prefill_n:
            print(f'  [{idx+1}/{len(shots)}] {img_path.name} — prefill {prefill_n}개 로드')

        def render():
            vis = img.copy()
            for poly in polys:
                cv2.polylines(vis, [poly], True, (0, 255, 255), 2)
                for (px, py) in poly:
                    cv2.circle(vis, (int(px), int(py)), 3, (0, 200, 255), -1)
            for i, (px, py) in enumerate(pending):
                cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(vis, str(i + 1), (px + 6, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            stage = f'pick corner {len(pending) + 1}/4'
            cv2.putText(vis,
                        f'[{idx+1}/{len(shots)}] {img_path.name}  '
                        f'polys={len(polys)} (prefill={prefill_n})  '
                        f'saved={saved_n} skip={skipped_n}  [{stage}]',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(vis,
                        'L-click=add  R-click=del poly  n/ENTER=save  r=undo  R=reset  s=skip',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow(WINDOW, vis)

        render()
        action = None
        while action is None:
            k = cv2.waitKey(20) & 0xFF
            if state['lclick'] is not None:
                x, y = state['lclick']
                state['lclick'] = None
                pending.append((x, y))
                print(f'    corner {len(pending)}/4 at ({x},{y})')
                if len(pending) == 4:
                    polys.append(four_points_to_polygon(pending, W, H))
                    pending = []
                    print(f'    → polygon 추가 (총 {len(polys)})')
                render()
            if state['rclick'] is not None:
                rx, ry = state['rclick']
                state['rclick'] = None
                # 클릭 위치가 가장 안쪽인 polygon 삭제
                hit = -1
                for i, poly in enumerate(polys):
                    if cv2.pointPolygonTest(poly.astype(np.float32),
                                            (float(rx), float(ry)), False) >= 0:
                        hit = i
                        break
                if hit >= 0:
                    polys.pop(hit)
                    print(f'    polygon #{hit} 삭제 (남은 {len(polys)})')
                    render()
                else:
                    print(f'    R-click ({rx},{ry}) — 어느 polygon 안도 아님')
            if k == 255:
                continue
            if k in (ord('q'), 27):
                action = 'quit'
            elif k in (ord('n'), 13, 10):
                action = 'save'
            elif k == ord('s'):
                action = 'skip'
            elif k == ord('r'):
                if pending:
                    pending.pop()
                    print(f'    undo pending corner ({len(pending)}/4)')
                elif polys:
                    polys.pop()
                    print('    undo last polygon')
                render()
            elif k == ord('R'):
                polys = []
                pending = []
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
            lbl_path = img_path.with_suffix('.txt')
            if polys:
                lines = [polygon_to_seg_line(p, W, H) for p in polys]
                lbl_path.write_text('\n'.join(lines), encoding='utf-8')
                print(f'  [{idx+1}/{len(shots)}] saved {lbl_path.name} ({len(polys)} polygon)')
                saved_n += 1
            else:
                # polygon 0 → 라벨 파일 자체 삭제 (학습 손상 방지)
                if lbl_path.exists():
                    lbl_path.unlink()
                print(f'  [{idx+1}/{len(shots)}] polygon 0 — 라벨 파일 제거')
                skipped_n += 1
            idx += 1

    cv2.destroyAllWindows()
    print(f'\n=== 완료 ===  saved={saved_n}  skipped={skipped_n}  total={len(shots)}')
    print(f'  다음: python3 build_seg_dataset.py')


if __name__ == '__main__':
    main()
