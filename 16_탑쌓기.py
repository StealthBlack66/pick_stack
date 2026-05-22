"""
큐브 탑 쌓기 — 그리드에 정렬된 cube 들을 들어서 두 가지 탑 구성.

흐름:
  1) 15_바둑판_정렬.py 의 CubeDetector 로 현재 cube 들 검출 (preview)
  2) base XY 로 정렬한 순서대로 11개 cube 선택
  3) Tower1: 수직 1자 5층
  4) Tower2: 피라미드 3-2-1 (바닥 3개 + 중간 2개 + 꼭대기 1개)

각 pick 은 cube 현재 yaw + 90° 로 잡아서 X/Y 양축 friction 정렬 강화
(그리드 정렬 후 X 가 들쭉날쭉한 문제 해결).

사용:
  python3 16_탑쌓기.py            # 실 동작
  python3 16_탑쌓기.py --dry-run  # 모션 없이 콘솔만

12번 (PickAndPlace) + 15번 (CubeDetector / helpers) 재사용.
"""
import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np
import rclpy

# 12번, 15번 모듈 importlib 로 로드 (한글 파일명 + 숫자 시작 회피)
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p12 = _load('p12', '12_비전_피크앤플레이스.py')
p15 = _load('p15', '15_바둑판_정렬.py')


# ===== Tower 설정 =====
# 각 pick 은 cube 현재 yaw + 90° 로 잡아서 X/Y 양축 정렬
TOWER1_XY = (450.0, 100.0)      # 수직 1자 5층 base (X, Y) — 그리드 +Y 쪽
TOWER1_LAYERS = 5
TOWER2_XY = (450.0, 200.0)      # 피라미드 3-2-1 base 중심 (X, Y)
TOWER2_LAYOUT = [               # (x_offset_from_center, layer_idx)
    (-25.0, 0), (0.0, 0), (+25.0, 0),    # 바닥 3개 (touching)
    (-12.5, 1), (+12.5, 1),               # 중간 2개 (valley 안착)
    (0.0, 2),                             # 꼭대기 1개
]
STACK_PICK_YAW_OFFSET = -90.0   # pick yaw = cube 현재 yaw + 이값 (양축 정렬용). TCP offset 검증 위해 -90 시도
# Tower1 (수직 stack): 옆 cube 없음 → place_yaw=0° (pick 90° 와 다름 → rotate 중 friction 정렬)
# Tower2 (피라미드 row): 옆 cube 25mm 접촉 → 그리퍼 body 의 좁은 면이 row 방향 보게 place_yaw=90°
TOWER1_PLACE_YAW = 0.0
TOWER2_PLACE_YAW = 90.0
# 모든 cube 간 이동은 이 z 위에서 (탑 / 다른 cube 충돌 회피).
# 탑 최대 높이 ~125mm 보다 충분히 위
SAFE_TRAVEL_Z_ABOVE_TABLE = 250.0


# ===== Tower Pick & Place =====
def execute_stack_pick_place(robot, source_xyz_mm, source_yaw_deg,
                              target_center_xyz_mm, target_yaw_deg, args,
                              z_table_top, pre_open_width_mm=None,
                              duration_sec=None, descend_duration_sec=None,
                              source_tcp_z_mm=None):
    """source (cube top z) 에서 잡아 target (cube 중심 z) 에 놓기.
    모든 cube 간 이동은 safe-z 위에서 → 탑 / 다른 cube 충돌 회피.
    source_yaw / target_yaw 가 다르면 source 위 safe-z 에서 회전 (cube 양축 정렬용).

    pre_open_width_mm: pick descent 전 그리퍼 벌릴 폭(mm). None → p15.GRIPPER_OPEN_WIDTH_MM.
    duration_sec: 일반 모션 segment 시간. None → p15.MOVE_DURATION_SEC.
    descend_duration_sec: pick/place 강하 segment 시간 (느림, 정밀). None → duration_sec 와 동일.
    source_tcp_z_mm: pick 시 TCP 의 절대 z (mm). None → sz - CUBE_WIDTH/2 (옛 방식).
                     예: 15번 셋업 (모든 cube 같은 높이) 면 p15.PICK_Z 전달.
    """
    if pre_open_width_mm is None:
        pre_open_width_mm = p15.GRIPPER_OPEN_WIDTH_MM
    sx, sy, sz = source_xyz_mm
    tx, ty, tz_center = target_center_xyz_mm
    safe_z = z_table_top + SAFE_TRAVEL_Z_ABOVE_TABLE

    pick_z = source_tcp_z_mm if source_tcp_z_mm is not None else (sz - p15.CUBE_WIDTH_MM / 2.0)
    pick_rpy = [0.0, 180.0, source_yaw_deg]
    place_rpy = [0.0, 180.0, target_yaw_deg]

    transit_src_safe = [sx, sy, safe_z]
    approach = [sx, sy, pick_z + p15.Z_APPROACH]
    pick = [sx, sy, pick_z]
    transit_lift_safe = [sx, sy, safe_z]
    transit_tgt_safe = [tx, ty, safe_z]
    place_app = [tx, ty, tz_center + p15.Z_APPROACH]
    place = [tx, ty, tz_center]
    transit_after_safe = [tx, ty, safe_z]

    dur = duration_sec if duration_sec is not None else p15.MOVE_DURATION_SEC
    desc_dur = descend_duration_sec if descend_duration_sec is not None else dur

    # Group A: transit_src_safe → approach (그리퍼 동작 없는 구간 묶기 → 정지 없음)
    print(f'   [Group A] transit over source → approach @ safe z={safe_z:.0f} yaw={source_yaw_deg:+.0f}°')
    if not args.dry_run:
        robot.move_line_base_multi([
            (transit_src_safe, pick_rpy),
            (approach, pick_rpy),
        ], [dur, dur])

    # Gripper pre-open
    print(f'    pre-open {pre_open_width_mm:.0f}mm')
    if not args.dry_run: p15.gripper_set_width(robot, pre_open_width_mm)

    # Group B: descend only (single segment, no group). 강하는 정밀 위해 desc_dur 사용.
    print(f'    descend {[round(v, 1) for v in pick]}')
    if not args.dry_run: robot.move_line_base(pick, rpy_deg=pick_rpy, duration=desc_dur)

    # Gripper close — 15번 패턴: half-close (자가정렬) → 완전 close + 1.0s settle
    print(f'    close: half ({p15.HALF_CLOSE_WIDTH_MM:.0f}mm) → 자가정렬 → 완전 닫음 + 1.0s settle')
    if not args.dry_run:
        p15.gripper_set_width(robot, p15.HALF_CLOSE_WIDTH_MM, settle=p15.HALF_CLOSE_SETTLE_SEC)
        robot.gripper_set(p12.GRIP_CLOSE_POS, settle=1.0, label='CLOSE')

    # Transit (multi, place_app 위까지만) — 15번 패턴, blend 로 인한 사선 방지 위해 place 제외
    print(f'   [transit] lift → transit → place_app')
    transit_wps = [(transit_lift_safe, pick_rpy)]
    if abs(source_yaw_deg - target_yaw_deg) > 0.5:
        transit_wps.append((transit_lift_safe, place_rpy))   # 같은 XYZ 에서 yaw 만 회전
    transit_wps.append((transit_tgt_safe, place_rpy))
    transit_wps.append((place_app, place_rpy))
    if not args.dry_run:
        robot.move_line_base_multi(transit_wps, [dur] * len(transit_wps))

    # Place descend (단발 cartesian) — 수직 강하 보장 + desc_dur 로 안전 속도
    print(f'    place descend (수직) → {[round(v, 1) for v in place]}')
    if not args.dry_run:
        robot.move_line_base(place, rpy_deg=place_rpy, duration=desc_dur)

    # Gripper open + 0.5초 대기 — 15번 패턴 (cube 안착 시간)
    print(f'    open {p15.RELEASE_OPEN_WIDTH_MM:.0f}mm (release) + 0.5s 대기')
    if not args.dry_run:
        p15.gripper_set_width(robot, p15.RELEASE_OPEN_WIDTH_MM)
        time.sleep(0.5)

    # Lift after (single)
    print(f'    lift to safe z {[round(v, 1) for v in transit_after_safe]}')
    if not args.dry_run:
        robot.move_line_base(transit_after_safe, rpy_deg=place_rpy, duration=dur)


def _match_cell_to_cube(cell_xy, dets, radius=30.0):
    """grid cell 에 가장 가까운 검출 cube 반환 (radius mm 안). 없으면 None."""
    best, best_d = None, radius
    for d in dets:
        bx, by, _ = d['base_xyz_mm']
        dd = math.hypot(bx - cell_xy[0], by - cell_xy[1])
        if dd < best_d:
            best_d, best = dd, d
    return best


def build_towers(robot, dets, args):
    """그리드의 cube 11개를 Tower1 (수직 5층) + Tower2 (피라미드 3-2-1) 로 쌓음.
    그리드 cell 좌표를 직접 사용 → 검출 노이즈와 무관, deterministic 경로."""
    need = TOWER1_LAYERS + len(TOWER2_LAYOUT)   # 5 + 6 = 11
    grid_cells = p15.make_grid_cells()
    if len(grid_cells) < need:
        print(f'\n!! 그리드 cell {len(grid_cells)}개 < 필요 {need}개 — 종료')
        return

    # 전체 grid cell 중 cube 가 있는 것을 순서대로 need 개 수집 (cells 0~N-1 강제 X)
    matched = []
    matched_cell_indices = []
    for i in range(len(grid_cells)):
        if len(matched) >= need:
            break
        c = _match_cell_to_cube(grid_cells[i], dets, radius=40.0)
        if c is not None:
            matched.append(c)
            matched_cell_indices.append(i)

    if len(matched) < need:
        print(f'\n!! 그리드에서 매칭된 cube {len(matched)}개 < 필요 {need}개 (40mm 이내) — 탑 쌓기 중단')
        return

    # table 표면 z = matched cube top median - cube 두께
    sample_z = float(np.median([c['base_xyz_mm'][2] for c in matched]))
    z_table_top = sample_z - p15.CUBE_WIDTH_MM
    print(f'\n=== 탑 쌓기 시작 (cube {need}개, table z={z_table_top:.1f}mm) ===')
    for k, c in enumerate(matched):
        ci = matched_cell_indices[k]
        cx, cy = grid_cells[ci]
        print(f'  pick #{k} ← cell[{ci}]: ({cx:.0f}, {cy:.0f})  cube z={c["base_xyz_mm"][2]:.1f}mm')

    # 그리드 정렬 cube 는 yaw=0 가정. 인접 cell 에 cube 가 있으면 finger collision 위험 →
    # neighbor-aware: STACK_PICK_YAW_OFFSET 와 +90° 중 finger clearance 큰 쪽 자동 선택.
    cell_positions = [grid_cells[ci] for ci in matched_cell_indices]
    used_cells = set()

    # ---- Tower 1: 수직 5층 ----
    print(f'\n--- Tower 1 (수직 5층) at {TOWER1_XY} ---')
    for layer in range(TOWER1_LAYERS):
        cube = matched[layer]
        cx_det, cy_det, _ = cube['base_xyz_mm']
        ci = matched_cell_indices[layer]
        cx, cy = grid_cells[ci]
        drift_mm = math.hypot(cx - cx_det, cy - cy_det)
        src = (cx, cy, sample_z)
        src_yaw = p15.pick_yaw_for_grid_cell(layer, cell_positions, STACK_PICK_YAW_OFFSET, used_cells)
        target_center_z = z_table_top + p15.CUBE_WIDTH_MM / 2.0 + p15.CUBE_WIDTH_MM * layer
        target = (TOWER1_XY[0], TOWER1_XY[1], target_center_z)
        print(f'\n[Tower1 layer {layer + 1}/{TOWER1_LAYERS}] '
              f'cell[{ci}]({cx:.0f},{cy:.0f}) [det({cx_det:.0f},{cy_det:.0f}) Δ={drift_mm:.0f}mm] → '
              f'({TOWER1_XY[0]:.0f},{TOWER1_XY[1]:.0f}) center z={target_center_z:.0f}'
              f'  pick={src_yaw:+.0f}° place={TOWER1_PLACE_YAW:+.0f}°')
        execute_stack_pick_place(robot, src, src_yaw, target, TOWER1_PLACE_YAW, args,
                                 z_table_top=z_table_top)
        used_cells.add(layer)

    # ---- Tower 2: 피라미드 3-2-1 (row 충돌 회피 위해 place_yaw=90°) ----
    print(f'\n--- Tower 2 (피라미드 3-2-1) at {TOWER2_XY} '
          f'(place_yaw={TOWER2_PLACE_YAW:+.0f}°, row 충돌 회피) ---')
    next_cell = TOWER1_LAYERS
    for slot, (x_off, layer) in enumerate(TOWER2_LAYOUT):
        cube_idx = next_cell + slot
        cube = matched[cube_idx]
        cx_det, cy_det, _ = cube['base_xyz_mm']
        ci = matched_cell_indices[cube_idx]
        cx, cy = grid_cells[ci]
        drift_mm = math.hypot(cx - cx_det, cy - cy_det)
        src = (cx, cy, sample_z)
        src_yaw = p15.pick_yaw_for_grid_cell(cube_idx, cell_positions, STACK_PICK_YAW_OFFSET, used_cells)
        target_x = TOWER2_XY[0] + x_off
        target_y = TOWER2_XY[1]
        target_center_z = z_table_top + p15.CUBE_WIDTH_MM / 2.0 + p15.CUBE_WIDTH_MM * layer
        target = (target_x, target_y, target_center_z)
        print(f'\n[Tower2 slot {slot + 1}/{len(TOWER2_LAYOUT)} layer {layer}] '
              f'cell[{ci}]({cx:.0f},{cy:.0f}) [det({cx_det:.0f},{cy_det:.0f}) Δ={drift_mm:.0f}mm] → '
              f'({target_x:.0f},{target_y:.0f}) center z={target_center_z:.0f}'
              f'  pick={src_yaw:+.0f}° place={TOWER2_PLACE_YAW:+.0f}°')
        execute_stack_pick_place(robot, src, src_yaw, target, TOWER2_PLACE_YAW, args,
                                 z_table_top=z_table_top)
        used_cells.add(cube_idx)

    print('\n=== 탑 쌓기 완료 ===')


# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='모션 없이 콘솔만')
    ap.add_argument('--quick', action='store_true',
                    help='드라이버 리셋 SKIP — dsr_bringup2 가 이미 살아있을 때만')
    ap.add_argument('--conf', type=float, default=p15.DETECT_CONF_THR,
                    help=f'YOLO 검출 conf 임계 (default {p15.DETECT_CONF_THR})')
    ap.add_argument('--debug-img', type=str, default='/tmp/tower_debug.png')
    args = ap.parse_args()

    print('=== 큐브 탑 쌓기 시작 ===')
    if args.dry_run:
        print('  *** DRY-RUN 모드: 실제 모션/그리퍼 명령은 보내지 않음 ***')

    if not args.quick:
        p12.reset_robot_driver()
    else:
        print('\n[준비 0/4] 드라이버 리셋 SKIP (--quick)')

    print(f'\n[준비 1/4] 비전 init (conf={args.conf})')
    vision = p15.CubeDetector(conf_thr=args.conf)

    print('\n[준비 2/4] 로봇 + 그리퍼 init')
    rclpy.init()
    robot = p12.PickAndPlace()
    try:
        if not args.dry_run:
            robot.activate_robot()
            robot.gripper_init()
            robot.gripper_open()
        print('   준비 OK')

        print(f'\n[준비 3/4] HOME 자세 {p12.HOME_JOINT_DEG} deg')
        if not args.dry_run:
            try:
                robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=p15.MOVE_DURATION_SEC)
                print('   HOME 도착')
            except Exception as e:
                print(f'   !! HOME 이동 실패: {e}')
        else:
            print('   (dry-run: HOME 이동 SKIP)')

        print('\n[준비 4/4] cube 검출 (preview — ENTER 로 확정)')
        dets = vision.preview_until_confirm(save_debug=args.debug_img)
        if dets is None:
            print('  사용자 취소 — 종료')
            return
        print(f'  확정 {len(dets)}개')

        # 작업영역 밖 cube 필터
        def _in_workspace(d):
            bx, by, bz = d['base_xyz_mm']
            pz = bz - p15.CUBE_WIDTH_MM / 2.0
            az = pz + p15.Z_APPROACH
            return (p12.WORK_X[0] <= bx <= p12.WORK_X[1] and
                    p12.WORK_Y[0] <= by <= p12.WORK_Y[1] and
                    p12.WORK_Z[0] <= pz <= p12.WORK_Z[1] and
                    p12.WORK_Z[0] <= az <= p12.WORK_Z[1])
        inside = [d for d in dets if _in_workspace(d)]
        skipped = len(dets) - len(inside)
        if skipped:
            print(f'  !! 작업영역 밖 cube {skipped}개 skip')
        dets = inside

        # z 통일 (depth 노이즈 흡수, 같은 판 위 가정)
        if len(dets) >= 2:
            zs = [d['base_xyz_mm'][2] for d in dets]
            zmed = float(np.median(zs))
            print(f'  z 통일: median {zmed:.1f}mm 으로 모든 cube z 통일')
            dets = [{**d, 'base_xyz_mm': (d['base_xyz_mm'][0], d['base_xyz_mm'][1], zmed)}
                    for d in dets]

        if not dets:
            print('!! 잡을 수 있는 cube 0개 — 종료')
            return

        build_towers(robot, dets, args)

        print('\n  HOME 복귀')
        if not args.dry_run:
            try:
                robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=p15.MOVE_DURATION_SEC)
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
