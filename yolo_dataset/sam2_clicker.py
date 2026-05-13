"""
SAM2 보조 수동 라벨링 — raw/shot_*.jpg 순회.

사용자가 cube 위에 클릭 1번 → SAM2 가 mask 생성 → bbox 추출 → YOLO 라벨 저장.

키 / 마우스:
  좌클릭          : cube 위 클릭 → mask + bbox 즉시 생성 (여러 번 = 다중 객체)
  n / ENTER       : 현재까지 bbox 들 라벨로 저장 + 다음 사진
  r               : 마지막 클릭 취소 (다중 클릭 중 1개만)
  R (대문자)      : 모두 reset
  s               : skip (라벨 없이 다음)
  q / ESC         : 종료
  ←/→ (a/d)       : 이전/다음 사진 (저장 안 됨)

출력: raw/shot_NNN.txt — YOLO 포맷 (cls cx cy w h, normalized)
       cube 0개면 빈 파일 안 만듦.

사용 후: merge_dataset.py (또는 수동 split) 로 학습용 폴더 구성.
"""
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import SAM

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / 'raw'
SAM_WEIGHTS = '/home/fastcampus/Downloads/test/sam2.1_b.pt'
CLASS_ID = 0   # wood cube

WINDOW = 'SAM2 clicker (click=add, n=save+next, r=undo, s=skip, q=quit)'


def mask_to_bbox(mask):
    """boolean HxW → (x1, y1, x2, y2) or None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def yolo_line(bbox, w, h):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f'{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'


def main():
    if not Path(SAM_WEIGHTS).exists():
        sys.exit(f'!! SAM2 weights 없음: {SAM_WEIGHTS}')

    print(f'  SAM2 로딩... ({SAM_WEIGHTS})')
    sam = SAM(SAM_WEIGHTS)
    # warmup
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    sam(dummy, points=[[640, 360]], labels=[1], verbose=False)
    print('  워밍업 완료')

    shots = sorted(RAW_DIR.glob('shot_*.jpg'))
    if not shots:
        sys.exit(f'!! shot_*.jpg 없음: {RAW_DIR}')
    print(f'  대상: {len(shots)}장\n')

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
        h, w = img.shape[:2]
        bboxes = []   # 누적된 bbox 들

        def render():
            vis = img.copy()
            for (x1, y1, x2, y2) in bboxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, 'cube', (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis,
                        f'[{idx+1}/{len(shots)}] {img_path.name}  '
                        f'cubes={len(bboxes)}  saved={saved_n} skip={skipped_n}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, 'click=add  n/ENTER=save+next  r=undo  s=skip  q=quit',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow(WINDOW, vis)

        render()
        action = None
        while action is None:
            k = cv2.waitKey(20) & 0xFF
            if state['click'] is not None:
                x, y = state['click']
                state['click'] = None
                # SAM2 point prompt
                try:
                    res = sam(img, points=[[x, y]], labels=[1], verbose=False)
                    masks = res[0].masks.data.cpu().numpy() if res[0].masks is not None else None
                except Exception as e:
                    print(f'    SAM2 추론 실패: {e}')
                    masks = None
                if masks is not None and len(masks) > 0:
                    bb = mask_to_bbox(masks[0].astype(bool))
                    if bb is not None:
                        bboxes.append(bb)
                        print(f'    click ({x},{y}) → bbox {bb}')
                    else:
                        print(f'    click ({x},{y}) → 빈 mask, 무시')
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
                if bboxes:
                    bboxes.pop()
                    print('    undo last click')
                    render()
            elif k == ord('R'):
                bboxes = []
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
            if bboxes:
                txt = '\n'.join(yolo_line(bb, w, h) for bb in bboxes)
                lbl_path = img_path.with_suffix('.txt')
                lbl_path.write_text(txt, encoding='utf-8')
                print(f'  [{idx+1}/{len(shots)}] saved {lbl_path.name} ({len(bboxes)} cube)')
                saved_n += 1
            else:
                print(f'  [{idx+1}/{len(shots)}] cube 0개 — 라벨 안 저장 (skip 과 동일)')
                skipped_n += 1
            idx += 1

    cv2.destroyAllWindows()
    print(f'\n=== 완료 ===  saved={saved_n}  skipped={skipped_n}  total={len(shots)}')
    print(f'  다음: merge_dataset.py 로 train/val 통합')


if __name__ == '__main__':
    main()
