"""
큐브 미술 쌓기 — 활용 사례 모양 조립 (계단/피라미드/십자).

지원 모양 (--shape):
  staircase   : 4+3+2+1 = 10 cubes 계단 (우측 예시 매칭)
  pyramid_1d  : 4+3+2+1 = 10 cubes 1D 피라미드 (valley 안착)
  cross_3d    : 7 cubes 3D 십자가 (수평 + + 수직 안테나)

사용:
  python3 17_미술쌓기.py --shape staircase
  python3 17_미술쌓기.py --shape pyramid_1d --dry-run
  python3 17_미술쌓기.py --shape cross_3d --quick

12번 (PickAndPlace) + 15번 (CubeDetector) + 16번 (execute_stack_pick_place) 재사용.

설계 제약:
  - 2D 큰 피라미드 (3×3 base) 는 그리퍼 body 가 사방 neighbor 와 충돌 → 1D 만 가능
  - 모든 cube 는 아래에서 받침 필요 (cantilever 불가) — 모든 모양이 이를 만족
  - 검출된 cube 의 실제 위치로 pick → 그리퍼 정확히 cube 중심에서 centering
"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np
import rclpy

# 12번/15번/16번 importlib 로 로드 (한글 파일명)
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
p16 = _load('p16', '16_탑쌓기.py')
p17t = _load('p17t', '17_test.py')   # 자동 모드 (--auto) 용 build_art_autonomous 제공


# ===== 모양 정의 =====
# 각 cube 위치 = (x_grid, y_grid, layer)  — 1 unit = CUBE_WIDTH_MM (25mm)
# 실제 mm 좌표 = SHAPE_BASE_XY + (x_grid * 25, y_grid * 25)
# layer 0 = 테이블 위, layer k = k 번 stack
SHAPES = {
    # 4-3-2-1 계단 — 각 상위 cube 가 바로 아래 cube 위에 그대로 안착 (full support)
    'staircase': [
        # 1층 row of 4
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
        # 2층 row of 3 (위치 1,2,3)
        (1, 0, 1), (2, 0, 1), (3, 0, 1),
        # 3층 row of 2 (위치 2,3)
        (2, 0, 2), (3, 0, 2),
        # 4층 row of 1 (위치 3)
        (3, 0, 3),
    ],

    # 4-3-2-1 1D 피라미드 — 상위 cube 가 아래 두 cube 의 valley 에 안착
    'pyramid_1d': [
        # 1층 4개
        (0.0, 0, 0), (1.0, 0, 0), (2.0, 0, 0), (3.0, 0, 0),
        # 2층 3개 (valley)
        (0.5, 0, 1), (1.5, 0, 1), (2.5, 0, 1),
        # 3층 2개 (valley)
        (1.0, 0, 2), (2.0, 0, 2),
        # 4층 1개 (valley 꼭대기)
        (1.5, 0, 3),
    ],

    # 3D 십자가 — 수평 + 모양 + 중앙 위로 수직 안테나
    'cross_3d': [
        # 1층 수평 + 모양 (5 cubes, 중앙 + 4방향)
        (0,  0, 0),     # 중앙
        (-1, 0, 0), (1, 0, 0),    # 좌우
        (0, -1, 0), (0, 1, 0),    # 앞뒤
        # 수직 안테나 (중앙 위로 2 cubes)
        (0, 0, 1),
        (0, 0, 2),
    ],

    # 6방향 원형/육각형
    'circle_6': [
        # 반경 1.5 grid 거리 (x_grid, y_grid, layer, yaw)
        (1.5 * math.cos(math.radians(0)),   1.5 * math.sin(math.radians(0)),   0, 90.0),
        (1.5 * math.cos(math.radians(60)),  1.5 * math.sin(math.radians(60)),  0, 90.0),
        (1.5 * math.cos(math.radians(120)), 1.5 * math.sin(math.radians(120)), 0, 90.0),
        (1.5 * math.cos(math.radians(180)), 1.5 * math.sin(math.radians(180)), 0, 90.0),
        (1.5 * math.cos(math.radians(240)), 1.5 * math.sin(math.radians(240)), 0, 90.0),
        (1.5 * math.cos(math.radians(300)), 1.5 * math.sin(math.radians(300)), 0, 90.0),
    ],

    # 바둑판 사선 피라미드 (회전 없이 대각선 칸에 배치하여 모서리 맞닿기)
    'diamond_pyramid': [
        # 1층: (0,0), (1,1), (2,2) 대각선으로 배치 (yaw 90도 유지)
        (0.0, 0.0, 0, 90.0),
        (1.0, 1.0, 0, 90.0),
        (2.0, 2.0, 0, 90.0),

        # 2층: 계곡 중앙 (0.5, 0.5) 와 (1.5, 1.5)
        (0.5, 0.5, 1, 90.0),
        (1.5, 1.5, 1, 90.0),

        # 3층: 꼭대기 (1.0, 1.0)
        (1.0, 1.0, 2, 90.0),
    ],

    # 4 cubes × 3 layers 원형 탑 — 각 cube 가 원의 중심점을 바라보도록 yaw = 자기 각도
    # 4개 cube 가 0°/90°/180°/270° 에 배치되어 면이 중심을 향함 (parallel-jaw 4-fold 대칭이라
    # 시각적으로 모두 axis-aligned 로 보이지만 의도된 회전 적용됨).
    # 반경 R=1.0 grid_unit → ART_PITCH_MM(27mm) 만큼 중심에서 떨어짐. 인접 cube 거리 ≈38mm.
    'circle_tower_4': [
        # Layer 0 (바닥)
        (math.cos(math.radians(0)),   math.sin(math.radians(0)),   0,   0.0),  # 우 (3시)
        (math.cos(math.radians(90)),  math.sin(math.radians(90)),  0,  90.0),  # 상 (12시)
        (math.cos(math.radians(180)), math.sin(math.radians(180)), 0, 180.0),  # 좌 (9시)
        (math.cos(math.radians(270)), math.sin(math.radians(270)), 0, 270.0),  # 하 (6시)
        # Layer 1 (중간)
        (math.cos(math.radians(0)),   math.sin(math.radians(0)),   1,   0.0),
        (math.cos(math.radians(90)),  math.sin(math.radians(90)),  1,  90.0),
        (math.cos(math.radians(180)), math.sin(math.radians(180)), 1, 180.0),
        (math.cos(math.radians(270)), math.sin(math.radians(270)), 1, 270.0),
        # Layer 2 (꼭대기)
        (math.cos(math.radians(0)),   math.sin(math.radians(0)),   2,   0.0),
        (math.cos(math.radians(90)),  math.sin(math.radians(90)),  2,  90.0),
        (math.cos(math.radians(180)), math.sin(math.radians(180)), 2, 180.0),
        (math.cos(math.radians(270)), math.sin(math.radians(270)), 2, 270.0),
    ],

    # 사용자 지정 정육각형 (Pointy-topped, 3층 brick-pattern, 총 18개)
    # 반경 35.15mm — 인접 cube 거리 = 반경 = 35.15mm.
    # 각 cube 면이 tangent 방향 (yaw = 위치 각도 - 90°, ±90° 등가):
    #   Layer 0 위치: 0°, 60°, 120°, 180°, 240°, 300° / yaws: -90, -30, 30, -90, -30, 30
    #   Layer 1 위치: +30° 회전 (30°, 90°, 150°, 210°, 270°, 330°) / yaws: -60, 0, 60, -60, 0, 60
    #   Layer 2 위치: +60° = Layer 0 와 동일 (60°, 120°, ..., 360°) / yaws 동일 패턴
    'hexagon_6': [
        # Layer 0 (바닥)
        ( (35.15/27.0) * math.cos(math.radians(0)),   (35.15/27.0) * math.sin(math.radians(0)),   0, -90.0),
        ( (35.15/27.0) * math.cos(math.radians(60)),  (35.15/27.0) * math.sin(math.radians(60)),  0, -30.0),
        ( (35.15/27.0) * math.cos(math.radians(120)), (35.15/27.0) * math.sin(math.radians(120)), 0,  30.0),
        ( (35.15/27.0) * math.cos(math.radians(180)), (35.15/27.0) * math.sin(math.radians(180)), 0, -90.0),
        ( (35.15/27.0) * math.cos(math.radians(240)), (35.15/27.0) * math.sin(math.radians(240)), 0, -30.0),
        ( (35.15/27.0) * math.cos(math.radians(300)), (35.15/27.0) * math.sin(math.radians(300)), 0,  30.0),

        # Layer 1 (위치 +30° 회전, brick pattern)
        ( (35.15/27.0) * math.cos(math.radians(30)),  (35.15/27.0) * math.sin(math.radians(30)),  1, -60.0),
        ( (35.15/27.0) * math.cos(math.radians(90)),  (35.15/27.0) * math.sin(math.radians(90)),  1,   0.0),
        ( (35.15/27.0) * math.cos(math.radians(150)), (35.15/27.0) * math.sin(math.radians(150)), 1,  60.0),
        ( (35.15/27.0) * math.cos(math.radians(210)), (35.15/27.0) * math.sin(math.radians(210)), 1, -60.0),
        ( (35.15/27.0) * math.cos(math.radians(270)), (35.15/27.0) * math.sin(math.radians(270)), 1,   0.0),
        ( (35.15/27.0) * math.cos(math.radians(330)), (35.15/27.0) * math.sin(math.radians(330)), 1,  60.0),

        # Layer 2 (위치 +60° = Layer 0 와 동일 XY)
        ( (35.15/27.0) * math.cos(math.radians(60)),  (35.15/27.0) * math.sin(math.radians(60)),  2, -30.0),
        ( (35.15/27.0) * math.cos(math.radians(120)), (35.15/27.0) * math.sin(math.radians(120)), 2,  30.0),
        ( (35.15/27.0) * math.cos(math.radians(180)), (35.15/27.0) * math.sin(math.radians(180)), 2, -90.0),
        ( (35.15/27.0) * math.cos(math.radians(240)), (35.15/27.0) * math.sin(math.radians(240)), 2, -30.0),
        ( (35.15/27.0) * math.cos(math.radians(300)), (35.15/27.0) * math.sin(math.radians(300)), 2,  30.0),
        ( (35.15/27.0) * math.cos(math.radians(360)), (35.15/27.0) * math.sin(math.radians(360)), 2, -90.0),
    ],
}

# 모양의 base XY (mm, robot base frame) — 그리드 옆 빈공간
SHAPE_BASE_XY = (450.0, 200.0)

# 오차 여유분 2mm 추가한 배치 간격 (기존 25 -> 27)
ART_PITCH_MM = p15.CUBE_WIDTH_MM + 2.0

# 모양별 pitch 오버라이드 — diamond_pyramid 는 대각선 안착이라 지지 면적 최대화 위해 1mm 여유로
SHAPE_PITCH_MM = {
    'diamond_pyramid': p15.CUBE_WIDTH_MM + 1.0,  # 26mm — layer1 cube 의 corner-corner overlap 12mm 확보
    'hexagon_6': p15.CUBE_WIDTH_MM + 3.0,        # 28mm — 옆 cube 충돌 회피 (마진 +1mm)
}

# place yaw — row 충돌 회피 위해 90° (16번 Tower2 와 동일 논리)
ART_PLACE_YAW = 90.0

# pick yaw — 미술쌓기는 16번 STACK_PICK_YAW_OFFSET (=-90°) 와 무관하게 항상 90°.
# (16번 탑쌓기 의 TCP offset 실험과 분리해서 미술쌓기 picking yaw 를 고정)
ART_PICK_YAW = 90.0

# pick 전 그리퍼 벌리는 폭 — 미술쌓기는 40mm 로 고정 (cube 25mm + 양옆 7.5mm 여유).
ART_PRE_OPEN_WIDTH_MM = 40.0

# 17번 자체 속도 (15번보다 느림 — 탑 쌓기는 정밀 + 안전 우선).
STACK_MOVE_DURATION_SEC = 3.0       # 일반 모션 (transit/lift 등)
STACK_DESCEND_DURATION_SEC = 4.0    # 강하 (pick / place descend — 안전 우선)


def build_art(robot, dets, shape_name, args):
    """선택된 모양을 검출 cube 들로 조립."""
    if shape_name not in SHAPES:
        print(f'!! 알 수 없는 shape: {shape_name}')
        print(f'   가능한 shape: {list(SHAPES.keys())}')
        return
    layout = SHAPES[shape_name]
    need = len(layout)
    if len(dets) < need:
        print(f'\n!! "{shape_name}" SKIP — 검출 cube {len(dets)}개 < 필요 {need}개')
        return

    # 그리드 셀 좌표를 신뢰원으로 사용 (15번이 cube 를 이 좌표에 놓음). 검출 결과는 셀별
    # cube 존재 확인 + drift 모니터링 용으로만 사용 — pick 은 항상 grid_cells[i] 로 감.
    # 전체 25 cell 중 cube 가 있는 셀을 grid 순서대로 need 개 수집 (cube 가 cell 0~N-1 에
    # 모두 있을 필요 X — 사용자가 어떤 cell 들에 놓아도 첫 need 개의 매칭만 사용).
    grid_cells = p15.make_grid_cells()
    if len(grid_cells) < need:
        print(f'\n!! 그리드 cell {len(grid_cells)}개 < 필요 {need}개 — 종료')
        return

    MATCH_RADIUS_MM = 40.0
    matched_cubes = []
    matched_cell_indices = []
    for i in range(len(grid_cells)):
        if len(matched_cubes) >= need:
            break
        cell = grid_cells[i]
        closest, best_d = None, MATCH_RADIUS_MM
        for d in dets:
            bx, by, _ = d['base_xyz_mm']
            dd = math.hypot(bx - cell[0], by - cell[1])
            if dd < best_d:
                best_d, closest = dd, d
        if closest is not None:
            matched_cubes.append(closest)
            matched_cell_indices.append(i)

    if len(matched_cubes) < need:
        print(f'\n!! 그리드에서 매칭된 cube {len(matched_cubes)}개 < 필요 {need}개 '
              f'({MATCH_RADIUS_MM:.0f}mm 이내). "{shape_name}" 조립 중단')
        return
    print(f'  ── grid → cube 매칭 결과: {len(matched_cubes)}개 cell 사용 ──')
    for k, i in enumerate(matched_cell_indices):
        cx, cy = grid_cells[i]
        bx, by, _ = matched_cubes[k]['base_xyz_mm']
        print(f'    [{k:2d}] cell[{i:2d}] ({cx:+7.1f}, {cy:+7.1f})  ← det ({bx:+7.1f}, {by:+7.1f})')

    sample_z = float(np.median([c['base_xyz_mm'][2] for c in matched_cubes]))
    z_table_top = sample_z - p15.CUBE_WIDTH_MM
    # 모양별 pitch (기본 ART_PITCH_MM, 특수 모양은 SHAPE_PITCH_MM 에서 오버라이드)
    pitch = SHAPE_PITCH_MM.get(shape_name, ART_PITCH_MM)
    print(f'\n=== "{shape_name}" 조립 시작 ({need} cubes, table z={z_table_top:.1f}, '
          f'base=({SHAPE_BASE_XY[0]:.0f},{SHAPE_BASE_XY[1]:.0f}), pitch={pitch:.1f}mm) ===')

    # 레이아웃을 layer 오름차순 → y → x 로 정렬 (바닥부터, 뒤에서 앞으로)
    layout_sorted = sorted(enumerate(layout), key=lambda e: (e[1][2], e[1][1], e[1][0]))

    # neighbor-aware pick yaw 를 위한 cell 좌표 (실제 매칭된 cell 만) + 진행 상태
    cell_positions = [grid_cells[ci] for ci in matched_cell_indices]
    used_cells = set()

    for order_idx, (_orig_i, item) in enumerate(layout_sorted):
        gx = item[0]
        gy = item[1]
        layer = item[2]
        yaw = item[3] if len(item) > 3 else ART_PLACE_YAW

        cube = matched_cubes[order_idx]
        cx_det, cy_det, _ = cube['base_xyz_mm']
        cell_idx = matched_cell_indices[order_idx]
        cx, cy = grid_cells[cell_idx]             # ← 실제 매칭된 그리드 셀 좌표로 pick
        drift_mm = math.hypot(cx - cx_det, cy - cy_det)
        # 절대 z 사용 — 15번 셋업 (모든 cube 동일 25mm, 바닥 z=-30, PICK_Z=-13) 가정.
        # 잡은 높이(PICK_Z) 와 layer 0 놓는 높이가 일치 — 같은 바닥 위 정렬 + 쌓기.
        src = (cx, cy, p15.PICK_Z)
        # 인접 cell 에 cube 가 남아있으면 finger 충돌 피하는 방향 자동 선택
        pick_yaw = p15.pick_yaw_for_grid_cell(order_idx, cell_positions, ART_PICK_YAW, used_cells)
        target_x = SHAPE_BASE_XY[0] + gx * pitch
        target_y = SHAPE_BASE_XY[1] + gy * pitch
        # 바닥면 (PICK_Z = 바닥 + 17mm) 기준 layer 별 z. 다층이면 cube 높이만큼 ↑.
        # layer 0: PICK_Z (잡은 z 와 동일 — 같은 바닥면)
        # layer N: PICK_Z + N * CUBE_WIDTH_MM
        target_center_z = p15.PICK_Z + layer * p15.CUBE_WIDTH_MM
        target = (target_x, target_y, target_center_z)
        print(f'\n[{order_idx + 1}/{need}] layer{layer} grid({gx:+.1f},{gy:+.1f}) → '
              f'mm({target_x:.0f},{target_y:.0f},z={target_center_z:.0f})  '
              f'pick cell[{cell_idx}]({cx:.0f},{cy:.0f}) yaw={pick_yaw:+.0f}° '
              f'[det({cx_det:.0f},{cy_det:.0f}) Δ={drift_mm:.0f}mm]')
        p16.execute_stack_pick_place(
            robot, src, pick_yaw, target, yaw,
            args, z_table_top=z_table_top, pre_open_width_mm=ART_PRE_OPEN_WIDTH_MM,
            duration_sec=STACK_MOVE_DURATION_SEC,
            descend_duration_sec=STACK_DESCEND_DURATION_SEC,
            source_tcp_z_mm=p15.PICK_Z)
        used_cells.add(order_idx)

    print(f'\n=== "{shape_name}" 조립 완료 ===')


def build_multi_stack(robot, dets, args):
    """2칸 블럭을 가조립하여 한 번에 잡는 멀티 그랩(Multi-Grab) 로직.
    3층짜리 3열 벽을 쌓습니다. (총 9개 소요)

    pick 좌표는 항상 grid_cells[i] (15번이 cube 를 놓은 결정적 좌표) 사용.
    검출 결과는 셀에 cube 존재 확인 + drift 모니터링 용.
    """
    need = 9
    if len(dets) < need:
        print(f'\n!! "double_stack" SKIP — 검출 cube {len(dets)}개 < 필요 {need}개')
        return

    grid_cells = p15.make_grid_cells()
    if len(grid_cells) < need:
        print(f'\n!! 그리드 cell {len(grid_cells)}개 < 필요 {need}개 — 종료')
        return

    # 전체 grid cell 중 cube 가 있는 것을 순서대로 need 개 수집 (cells 0~N-1 강제 X)
    MATCH_RADIUS_MM = 40.0
    matched_cubes = []
    matched_cell_indices = []
    for i in range(len(grid_cells)):
        if len(matched_cubes) >= need:
            break
        cell = grid_cells[i]
        closest, best_d = None, MATCH_RADIUS_MM
        for d in dets:
            bx, by, _ = d['base_xyz_mm']
            dd = math.hypot(bx - cell[0], by - cell[1])
            if dd < best_d:
                best_d, closest = dd, d
        if closest is not None:
            matched_cubes.append(closest)
            matched_cell_indices.append(i)

    if len(matched_cubes) < need:
        print(f'\n!! 매칭된 cube {len(matched_cubes)}개 < 필요 {need}개 '
              f'({MATCH_RADIUS_MM:.0f}mm 이내) — double_stack 중단')
        return

    sample_z = float(np.median([c['base_xyz_mm'][2] for c in matched_cubes]))
    z_table_top = sample_z - p15.CUBE_WIDTH_MM

    print(f'\n=== "double_stack" 멀티 그랩 조립 시작 ({need} cubes) ===')

    STAGING_X = 500.0
    STAGING_Y = SHAPE_BASE_XY[1]
    stage_z = z_table_top + p15.CUBE_WIDTH_MM / 2.0

    # neighbor-aware pick yaw 를 위한 cell 좌표 (실제 매칭된 cell 만) + 진행 상태
    cell_positions = [grid_cells[ci] for ci in matched_cell_indices]
    used_cells = set()

    def _pick_one(idx, place_xyz, place_yaw):
        cell_idx = matched_cell_indices[idx]
        x, y = grid_cells[cell_idx]
        src = (x, y, p15.PICK_Z)   # 절대 z (15번 셋업)
        pick_yaw = p15.pick_yaw_for_grid_cell(idx, cell_positions, ART_PICK_YAW, used_cells)
        p16.execute_stack_pick_place(robot, src, pick_yaw, place_xyz, place_yaw,
                                     args, z_table_top=z_table_top,
                                     pre_open_width_mm=ART_PRE_OPEN_WIDTH_MM,
                                     duration_sec=STACK_MOVE_DURATION_SEC,
                                     descend_duration_sec=STACK_DESCEND_DURATION_SEC,
                                     source_tcp_z_mm=p15.PICK_Z)
        used_cells.add(idx)

    cube_idx = 0

    for layer in range(3):
        # 바닥면 (PICK_Z) 기준 layer 별 z. layer 0 = PICK_Z, layer N = PICK_Z + N*CUBE_WIDTH.
        target_z = p15.PICK_Z + layer * p15.CUBE_WIDTH_MM

        if layer % 2 == 0:
            # 1. 가조립 2-block (left & mid)
            stage_y1 = STAGING_Y - ART_PITCH_MM / 2.0
            stage_y2 = STAGING_Y + ART_PITCH_MM / 2.0

            _pick_one(cube_idx, (STAGING_X, stage_y1, stage_z), ART_PLACE_YAW); cube_idx += 1
            _pick_one(cube_idx, (STAGING_X, stage_y2, stage_z), ART_PLACE_YAW); cube_idx += 1

            # 멀티 그랩으로 한 번에 잡아서 타겟 위치로 (중심이 타겟 좌측으로)
            multi_pick_xy = (STAGING_X, STAGING_Y)
            multi_target_xy = (SHAPE_BASE_XY[0], SHAPE_BASE_XY[1] - ART_PITCH_MM / 2.0)
            print(f'\n[Multi-Grab] layer{layer} 2-block')
            _execute_multi_grab(robot, multi_pick_xy, multi_target_xy, target_z, args, z_table_top)

            # 2. 1-block (right)
            _pick_one(cube_idx,
                      (SHAPE_BASE_XY[0], SHAPE_BASE_XY[1] + ART_PITCH_MM, target_z),
                      ART_PLACE_YAW); cube_idx += 1
        else:
            # 1. 1-block (left)
            _pick_one(cube_idx,
                      (SHAPE_BASE_XY[0], SHAPE_BASE_XY[1] - ART_PITCH_MM, target_z),
                      ART_PLACE_YAW); cube_idx += 1

            # 2. 가조립 2-block (mid & right)
            stage_y1 = STAGING_Y - ART_PITCH_MM / 2.0
            stage_y2 = STAGING_Y + ART_PITCH_MM / 2.0

            _pick_one(cube_idx, (STAGING_X, stage_y1, stage_z), ART_PLACE_YAW); cube_idx += 1
            _pick_one(cube_idx, (STAGING_X, stage_y2, stage_z), ART_PLACE_YAW); cube_idx += 1
                                         
            multi_pick_xy = (STAGING_X, STAGING_Y)
            multi_target_xy = (SHAPE_BASE_XY[0], SHAPE_BASE_XY[1] + ART_PITCH_MM / 2.0)
            print(f'\n[Multi-Grab] layer{layer} 2-block')
            _execute_multi_grab(robot, multi_pick_xy, multi_target_xy, target_z, args, z_table_top)

    print(f'\n=== "double_stack" 완료 ===')


def _execute_multi_grab(robot, src_xy, target_xy, target_z, args, z_table_top):
    if args.dry_run:
        print(f"  [DRY-RUN] Multi-grab from {src_xy} to {target_xy} at Z={target_z:.1f}")
        return
        
    src_z = z_table_top + p15.CUBE_WIDTH_MM / 2.0
    px, py = src_xy
    tx, ty = target_xy
    yaw = ART_PLACE_YAW
    
    # 1. Approach (Wide Open)
    robot.gripper_command(p15.width_mm_to_pos(100.0), settle=0.3)  # 100mm wide
    robot.move_line_base([px, py, src_z + p15.Z_APPROACH, yaw, 180, 0], duration=p15.MOVE_DURATION_SEC)
    
    # 2. Descend & Grab (squeeze 2 cubes -> 50mm + gap -> ~52mm)
    robot.move_line_base([px, py, src_z, yaw, 180, 0], duration=p15.MOVE_DURATION_SEC)
    robot.gripper_command(p15.width_mm_to_pos(45.0), settle=0.5) # grab tightly
    
    # 3. Lift
    robot.move_line_base([px, py, src_z + p15.Z_LIFT, yaw, 180, 0], duration=p15.MOVE_DURATION_SEC)
    
    # 4. Move to Target
    robot.move_line_base([tx, ty, target_z + p15.Z_APPROACH, yaw, 180, 0], duration=p15.MOVE_DURATION_SEC)
    
    # 5. Descend & Release
    robot.move_line_base([tx, ty, target_z, yaw, 180, 0], duration=p15.MOVE_DURATION_SEC)
    robot.gripper_command(p15.width_mm_to_pos(80.0), settle=0.3)
    
    # 6. Lift
    robot.move_line_base([tx, ty, target_z + p15.Z_LIFT, yaw, 180, 0], duration=p15.MOVE_DURATION_SEC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', choices=list(SHAPES.keys()) + ['double_stack'], default='staircase',
                    help='조립할 모양 (default: staircase)')
    ap.add_argument('--dry-run', action='store_true', help='모션 없이 콘솔만')
    ap.add_argument('--quick', action='store_true',
                    help='드라이버 리셋 SKIP — dsr_bringup2 가 이미 살아있을 때만')
    ap.add_argument('--conf', type=float, default=p15.DETECT_CONF_THR,
                    help=f'YOLO 검출 conf 임계 (default {p15.DETECT_CONF_THR})')
    ap.add_argument('--debug-img', type=str, default='/tmp/art_debug.png')
    ap.add_argument('--auto', action='store_true',
                    help='완전 자동 모드 — 매 cube 마다 비전 재인식 (17_test.py 의 build_art_autonomous)')
    args = ap.parse_args()

    print(f'=== 미술 쌓기 시작 (shape={args.shape}) ===')
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

        print('\n[준비 4/4] cube 검출 (카메라 화면에서 번호를 눌러 모양 선택)')
        
        shapes_list = list(SHAPES.keys()) + ['double_stack']
        msg_lines = ["Press Number to Select & Start:"]
        valid_keys = []
        key_to_shape = {}
        for i, shape in enumerate(shapes_list):
            # 1~9까지 키 매핑
            if i < 9:
                key = str(i + 1)
                valid_keys.append(key)
                key_to_shape[key] = shape
                msg_lines.append(f" {key}: {shape}")
            
        ret = vision.preview_until_confirm(save_debug=args.debug_img, custom_msg=msg_lines, valid_keys=valid_keys)
        
        if ret == (None, None) or ret is None:
            print('  사용자 취소 — 종료')
            return
            
        dets, key = ret
        args.shape = key_to_shape[key]
        print(f'  선택된 모양: {args.shape} (확정 {len(dets)}개)')

        # 작업영역 필터
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

        # z 통일 (depth 노이즈 흡수)
        if len(dets) >= 2:
            zs = [d['base_xyz_mm'][2] for d in dets]
            zmed = float(np.median(zs))
            print(f'  z 통일: median {zmed:.1f}mm 으로 모든 cube z 통일')
            dets = [{**d, 'base_xyz_mm': (d['base_xyz_mm'][0], d['base_xyz_mm'][1], zmed)}
                    for d in dets]

        if not dets:
            print('!! 잡을 수 있는 cube 0개 — 종료')
            return

        if args.shape == 'double_stack':
            if args.auto:
                print('  !! --auto 는 double_stack 미지원 — 반자동(build_multi_stack)으로 진행')
            build_multi_stack(robot, dets, args)
        elif args.auto:
            print(f'\n[AUTO] 완전 자동 모드 — 매 cube 마다 재인식 (shape={args.shape})')
            p17t.build_art_autonomous(robot, vision, args.shape, args)
        else:
            build_art(robot, dets, args.shape, args)

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
