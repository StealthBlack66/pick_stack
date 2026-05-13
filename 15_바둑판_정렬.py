"""
나무 큐브 바둑판 정렬 — RealSense + 학습된 best.pt + 두산 e0509 + RH-P12-RN-A.

흐름:
  1) detect_all(): YOLO 검출 → 각 bbox center 의 base 좌표 (mm)
  2) 그리드 셀 4x3 (50mm 간격) 미리 계산
  3) 각 cube 에 대해:
     - approach(top-down) → 그리퍼 pre-open → descend → close → lift
     - 그리드 셀로 이동 → descend → release → lift
  4) 모든 cube 배치 완료 후 HOME

사용:
  python3 15_바둑판_정렬.py            # 실 동작
  python3 15_바둑판_정렬.py --dry-run  # 모션 없이 콘솔 출력만
  python3 15_바둑판_정렬.py --limit N  # 처음 N개만

12번을 importlib 로 재사용 (PickAndPlace, helpers).
검출은 YOLO 객체 인식만 — 상단면 추출/yaw/트래킹 등 후처리 없음.
"""
import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from ultralytics import YOLO

# 12번 모듈 로드 (한글 파일명 + 숫자 시작 → importlib.util 로 강제 load)
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
_spec = importlib.util.spec_from_file_location(
    'p12', os.path.join(_DIR, '12_비전_피크앤플레이스.py'))
p12 = importlib.util.module_from_spec(_spec)
sys.modules['p12'] = p12
_spec.loader.exec_module(p12)


# ===== 설정 =====
YOLO_WEIGHTS = ('/home/fastcampus/Downloads/test/로봇강의_예제/'
                '02_Doosan_Robot_제어/yolo_dataset/runs/seg_v3/weights/best.pt')
DETECT_CONF_THR = 0.40
NMS_IOU_THR = 0.45
RS_W, RS_H, RS_FPS = 1280, 720, 30

# 그리드 (base 좌표, mm)
GRID_ORIGIN_X = 350.0
GRID_ORIGIN_Y = -150.0
GRID_COLS = 4
GRID_ROWS = 3
GRID_SPACING = 50.0

# 그리퍼
CUBE_WIDTH_MM = 25.0
PRE_OPEN_WIDTH_MM = 50.0
RELEASE_WIDTH_MM = 27.0
GRIP_RANGE_MM = 130.0    # RH-P12-RN-A 완전 열림(POS 0)에서 약 130mm 추정

# 모션
Z_APPROACH = 80.0
Z_LIFT = 100.0


def width_mm_to_pos(width_mm: float) -> int:
    """폭(mm) → 그리퍼 POS (0=완전 열림, 700=완전 닫힘) 선형 매핑."""
    width = max(0.0, min(GRIP_RANGE_MM, width_mm))
    pos = 700.0 * (1.0 - width / GRIP_RANGE_MM)
    return int(round(pos))


# ===== Vision (YOLO 객체 인식만) =====
class CubeDetector:
    """RealSense + YOLO. bbox center 의 base 좌표만 반환."""

    def __init__(self):
        calib_path = p12.CALIB_PATH
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f'캘리브레이션 없음: {calib_path}')
        d = np.load(calib_path)
        self.T_cam2base = d['T_cam2base']
        print(f'   캘리브 로드: translation='
              f'{[round(v*1000, 1) for v in self.T_cam2base[:3, 3]]} mm')

        print(f'   YOLO 로드: {YOLO_WEIGHTS}')
        self.model = YOLO(YOLO_WEIGHTS)

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
        cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _depth_at(self, depth_frame, u, v, patch=5):
        z = depth_frame.get_distance(int(u), int(v))
        if z > 0.05:
            return z
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape
        u0, v0 = max(0, int(u) - patch), max(0, int(v) - patch)
        u1, v1 = min(W, int(u) + patch + 1), min(H, int(v) + patch + 1)
        win = arr[v0:v1, u0:u1]
        win = win[win > 0]
        return float(win.mean()) * 0.001 if win.size else 0.0

    def _pixel_to_base(self, depth_frame, u, v):
        z = self._depth_at(depth_frame, u, v)
        if z <= 0.05:
            return None
        cam = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], z)
        h = np.array([cam[0], cam[1], cam[2], 1.0])
        return (self.T_cam2base @ h)[:3] * 1000.0  # mm

    def _run_yolo(self, color):
        results = self.model(color, conf=DETECT_CONF_THR, iou=NMS_IOU_THR,
                             verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        return results[0]

    def detect_all(self, save_debug=None):
        """반환: [{'base_xyz_mm': (x,y,z), 'pixel': (u,v), 'conf': c}, ...]"""
        frames = self.align.process(self.pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return []
        color = np.asanyarray(cf.get_data())

        r = self._run_yolo(color)
        if r is None:
            return []
        xywh = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        out = []
        dbg = color.copy() if save_debug else None
        for i in range(len(xywh)):
            cx, cy, w, h = xywh[i]
            u, v = int(cx), int(cy)
            base = self._pixel_to_base(df, u, v)
            if base is None:
                continue
            out.append({
                'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                'pixel': (u, v),
                'conf': float(confs[i]),
            })
            if dbg is not None:
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(dbg, (u, v), 4, (0, 0, 255), -1)
                cv2.putText(dbg, f'{confs[i]:.2f}', (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if dbg is not None and save_debug:
            cv2.imwrite(save_debug, dbg)
        return out

    def preview_until_confirm(self, save_debug=None):
        """라이브 화면 + 검출 → 사용자 확인 후 그 시점 검출 반환.
        키:  ENTER/SPACE = 확정,  q/ESC = 취소,  s = 스크린샷
        """
        win = 'YOLO preview (ENTER=confirm, s=shot, q=quit)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print('  preview 시작 — ENTER 로 확정 (q=취소)')
        try:
            while True:
                frames = self.align.process(self.pipeline.wait_for_frames())
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if not cf or not df:
                    continue
                color = np.asanyarray(cf.get_data())

                r = self._run_yolo(color)
                dets = []
                vis = color.copy()
                if r is not None:
                    xywh = r.boxes.xywh.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for i in range(len(xywh)):
                        cx, cy, w, h = xywh[i]
                        u, v = int(cx), int(cy)
                        base = self._pixel_to_base(df, u, v)
                        if base is None:
                            continue
                        dets.append({
                            'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                            'pixel': (u, v),
                            'conf': float(confs[i]),
                        })
                        x1, y1 = int(cx - w / 2), int(cy - h / 2)
                        x2, y2 = int(cx + w / 2), int(cy + h / 2)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)
                        cv2.putText(vis, f'#{len(dets)-1} {confs[i]:.2f}',
                                    (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        bx, by, bz = dets[-1]['base_xyz_mm']
                        cv2.putText(vis, f'({bx:.0f},{by:.0f},{bz:.0f})mm',
                                    (x1, min(color.shape[0] - 5, y2 + 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

                cv2.putText(vis,
                            f'detected={len(dets)}  ENTER=confirm  s=shot  q=quit',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(win, vis)
                k = cv2.waitKey(20) & 0xFF
                if k in (ord('q'), 27):
                    cv2.destroyWindow(win)
                    cv2.waitKey(1)
                    return None
                if k in (13, 10, ord(' ')):
                    if save_debug:
                        cv2.imwrite(save_debug, vis)
                    cv2.destroyWindow(win)
                    cv2.waitKey(1)
                    return dets
                if k == ord('s'):
                    out = save_debug or '/tmp/cubes_preview.png'
                    cv2.imwrite(out, vis)
                    print(f'  screenshot: {out}')
        except Exception:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            raise


# ===== 그리드 =====
def make_grid_cells():
    """[(x_mm, y_mm), ...]  4x3 = 12개, 좌상→우→다음행."""
    cells = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cells.append((
                GRID_ORIGIN_X + c * GRID_SPACING,
                GRID_ORIGIN_Y + r * GRID_SPACING,
            ))
    return cells


def select_next_target(detections, used_indices):
    """아직 안 옮긴 cube 중 가장 이웃과 멀리 떨어진 것 우선."""
    candidates = [(i, d) for i, d in enumerate(detections) if i not in used_indices]
    if not candidates:
        return None
    best = None
    best_iso = -1.0
    for i, d in candidates:
        dx, dy, _ = d['base_xyz_mm']
        min_d = min(
            (math.hypot(d2['base_xyz_mm'][0] - dx, d2['base_xyz_mm'][1] - dy)
             for j, d2 in enumerate(detections) if j != i),
            default=float('inf'))
        if min_d > best_iso:
            best_iso = min_d
            best = i
    return best


# ===== Pick & Place 사이클 =====
# RH-P12-RN-A 는 spring-loaded 가 아니라서 토크 OFF 만으로 release 안 됨.
# 반드시 명시적 open 명령(pos < 700) 으로 풀어야 함.


def gripper_set_width(robot, width_mm: float, settle=0.6):
    """그리퍼 폭(mm) 명령. 12번 gripper_set 과 동일 구조."""
    pos = width_mm_to_pos(width_mm)
    return robot.gripper_set(pos, settle=settle, label=f'W{width_mm:.0f}mm')


def execute_one_cycle(robot, target, cell_xy, args):
    """한 cube → 한 그리드 셀. yaw 인식 없음 → top-down 고정."""
    bx, by, bz = target['base_xyz_mm']
    print(f'\n  >> cube @ ({bx:.1f}, {by:.1f}, {bz:.1f}) '
          f'→ cell ({cell_xy[0]:.0f}, {cell_xy[1]:.0f})')

    rpy = [0.0, 180.0, 0.0]

    # cube 옆면 그립을 위해 cube 중심 z 로 내려감 (검출 z = cube 윗면 가정)
    pick_z = bz - CUBE_WIDTH_MM / 2.0
    approach = [bx, by, pick_z + Z_APPROACH]
    pick = [bx, by, pick_z]
    lift = [bx, by, pick_z + Z_LIFT]

    # 내려놓는 z 는 잡았을 때 z 와 동일
    place = [cell_xy[0], cell_xy[1], pick_z]
    place_app = [cell_xy[0], cell_xy[1], pick_z + Z_APPROACH]

    # ---- Pick ----
    print('   [Pick]')
    print(f'    1) approach → {[round(v, 1) for v in approach]}')
    if not args.dry_run: robot.move_line_base(approach, rpy_deg=rpy)
    print(f'    2) pre-open ({PRE_OPEN_WIDTH_MM:.0f}mm = pos {width_mm_to_pos(PRE_OPEN_WIDTH_MM)})')
    if not args.dry_run: gripper_set_width(robot, PRE_OPEN_WIDTH_MM)
    print(f'    3) descend → {[round(v, 1) for v in pick]}')
    if not args.dry_run: robot.move_line_base(pick, rpy_deg=rpy)
    print(f'    4) close (잡기)')
    if not args.dry_run: robot.gripper_close()
    print(f'    5) lift → {[round(v, 1) for v in lift]}')
    if not args.dry_run: robot.move_line_base(lift, rpy_deg=rpy)

    # ---- Place ----
    print('   [Place]')
    print(f'    1) above cell → {[round(v, 1) for v in place_app]}')
    if not args.dry_run: robot.move_line_base(place_app, rpy_deg=rpy)
    print(f'    2) descend → {[round(v, 1) for v in place]}')
    if not args.dry_run: robot.move_line_base(place, rpy_deg=rpy)
    print(f'    3) open {RELEASE_WIDTH_MM:.0f}mm (release)')
    if not args.dry_run: gripper_set_width(robot, RELEASE_WIDTH_MM)
    print(f'    4) lift → {[round(v, 1) for v in place_app]}')
    if not args.dry_run: robot.move_line_base(place_app, rpy_deg=rpy)


# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='모션 없이 콘솔만')
    ap.add_argument('--quick', action='store_true',
                    help='드라이버 리셋 SKIP — dsr_bringup2 가 이미 살아있을 때만')
    ap.add_argument('--limit', type=int, default=None, help='처음 N개만 처리')
    ap.add_argument('--debug-img', type=str, default='/tmp/cubes_debug.png')
    args = ap.parse_args()

    print('=== 바둑판 정렬 시작 ===')
    if args.dry_run:
        print('  *** DRY-RUN 모드: 실제 모션/그리퍼 명령은 보내지 않음 ***')
    grid_cells = make_grid_cells()
    print(f'  그리드: {GRID_ROWS}x{GRID_COLS} = {len(grid_cells)}셀, 간격 {GRID_SPACING}mm')
    print(f'  시작 셀: ({grid_cells[0][0]:.0f}, {grid_cells[0][1]:.0f}) → '
          f'끝 셀: ({grid_cells[-1][0]:.0f}, {grid_cells[-1][1]:.0f})')

    if not args.quick:
        p12.reset_robot_driver()
    else:
        print('\n[준비 0/4] 드라이버 리셋 SKIP (--quick)')

    print('\n[준비 1/4] 비전 init')
    vision = CubeDetector()

    print('\n[준비 2/4] 로봇 + 그리퍼 init')
    rclpy.init()
    robot = p12.PickAndPlace()
    try:
        if not args.dry_run:
            robot.activate_robot()
            robot.gripper_init()
            robot.gripper_open()
        print('   준비 OK')

        print(f'\n[준비 3/4] HOME 자세로 이동 {p12.HOME_JOINT_DEG} deg')
        if not args.dry_run:
            try:
                robot.move_joint_deg(p12.HOME_JOINT_DEG)
                print('   HOME 도착')
            except Exception as e:
                print(f'   !! HOME 이동 실패: {e}')
        else:
            print('   (dry-run: HOME 이동 SKIP)')

        print('\n[준비 4/4] 큐브 검출 (preview 윈도우 — ENTER 로 확정)')
        dets = vision.preview_until_confirm(save_debug=args.debug_img)
        if dets is None:
            print('  사용자 취소 — 종료')
            return
        print(f'  확정 {len(dets)}개 (debug: {args.debug_img})')
        for i, d in enumerate(dets):
            x, y, z = d['base_xyz_mm']
            print(f'   #{i}: ({x:.1f}, {y:.1f}, {z:.1f}) mm  conf={d["conf"]:.2f}')

        # 작업영역 밖 cube 필터
        def _in_workspace(d):
            bx, by, bz = d['base_xyz_mm']
            pz = bz - CUBE_WIDTH_MM / 2.0
            az = pz + Z_APPROACH
            return (p12.WORK_X[0] <= bx <= p12.WORK_X[1] and
                    p12.WORK_Y[0] <= by <= p12.WORK_Y[1] and
                    p12.WORK_Z[0] <= pz <= p12.WORK_Z[1] and
                    p12.WORK_Z[0] <= az <= p12.WORK_Z[1])

        inside = [d for d in dets if _in_workspace(d)]
        skipped = len(dets) - len(inside)
        if skipped:
            print(f'  !! 작업영역 밖 cube {skipped}개 skip '
                  f'(X∈{p12.WORK_X}, Y∈{p12.WORK_Y}, Z∈{p12.WORK_Z})')
            for d in dets:
                if not _in_workspace(d):
                    x, y, z = d['base_xyz_mm']
                    print(f'     skip: ({x:.1f}, {y:.1f}, {z:.1f}) mm')
        dets = inside

        if not dets:
            print('!! 잡을 수 있는 cube 0개 — 종료')
            return

        n_target = len(dets) if args.limit is None else min(args.limit, len(dets))
        n_target = min(n_target, len(grid_cells))
        print(f'\n  {n_target}개 cube → {n_target}개 셀 매핑')

        used = set()
        for cell_idx in range(n_target):
            idx = select_next_target(dets, used)
            if idx is None:
                break
            used.add(idx)
            execute_one_cycle(robot, dets[idx], grid_cells[cell_idx], args)

        print('\n  HOME 복귀')
        if not args.dry_run:
            try:
                robot.move_joint_deg(p12.HOME_JOINT_DEG)
            except Exception as e:
                print(f'   !! HOME 복귀 실패: {e}')
        print('\n=== 완료 ===')
    finally:
        try:
            vision.stop()
        except Exception:
            pass
        try:
            robot.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
