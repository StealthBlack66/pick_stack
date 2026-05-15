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
}

# 모양의 base XY (mm, robot base frame) — 그리드 옆 빈공간
SHAPE_BASE_XY = (450.0, 200.0)

# 오차 여유분 2mm 추가한 배치 간격 (기존 25 -> 27)
ART_PITCH_MM = p15.CUBE_WIDTH_MM + 2.0

# 모양별 pitch 오버라이드 — diamond_pyramid 는 대각선 안착이라 지지 면적 최대화 위해 1mm 여유로
SHAPE_PITCH_MM = {
    'diamond_pyramid': p15.CUBE_WIDTH_MM + 1.0,  # 26mm — layer1 cube 의 corner-corner overlap 12mm 확보
}

# place yaw — row 충돌 회피 위해 90° (16번 Tower2 와 동일 논리)
ART_PLACE_YAW = 90.0


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

    # Y → X 순으로 검출 cube 정렬 (그리드 좌상→우→다음행 순서)
    sorted_dets = sorted(dets, key=lambda d: (d['base_xyz_mm'][1], d['base_xyz_mm'][0]))
    cubes = sorted_dets[:need]

    sample_z = float(np.median([c['base_xyz_mm'][2] for c in cubes]))
    z_table_top = sample_z - p15.CUBE_WIDTH_MM
    # 모양별 pitch (기본 ART_PITCH_MM, 특수 모양은 SHAPE_PITCH_MM 에서 오버라이드)
    pitch = SHAPE_PITCH_MM.get(shape_name, ART_PITCH_MM)
    print(f'\n=== "{shape_name}" 조립 시작 ({need} cubes, table z={z_table_top:.1f}, '
          f'base=({SHAPE_BASE_XY[0]:.0f},{SHAPE_BASE_XY[1]:.0f}), pitch={pitch:.1f}mm) ===')

    # 레이아웃을 layer 오름차순 → y → x 로 정렬 (바닥부터, 뒤에서 앞으로)
    layout_sorted = sorted(enumerate(layout), key=lambda e: (e[1][2], e[1][1], e[1][0]))

    for order_idx, (_orig_i, item) in enumerate(layout_sorted):
        gx = item[0]
        gy = item[1]
        layer = item[2]
        yaw = item[3] if len(item) > 3 else ART_PLACE_YAW

        cube = cubes[order_idx]
        cx, cy, _ = cube['base_xyz_mm']
        src = (cx, cy, sample_z)
        target_x = SHAPE_BASE_XY[0] + gx * pitch
        target_y = SHAPE_BASE_XY[1] + gy * pitch
        target_center_z = z_table_top + p15.CUBE_WIDTH_MM / 2.0 + p15.CUBE_WIDTH_MM * layer
        target = (target_x, target_y, target_center_z)
        print(f'\n[{order_idx + 1}/{need}] layer{layer} grid({gx:+.1f},{gy:+.1f}) → '
              f'mm({target_x:.0f},{target_y:.0f},z={target_center_z:.0f})  '
              f'pick from ({cx:.0f},{cy:.0f})')
        p16.execute_stack_pick_place(
            robot, src, p16.STACK_PICK_YAW_OFFSET, target, yaw,
            args, z_table_top=z_table_top)

    print(f'\n=== "{shape_name}" 조립 완료 ===')


def build_multi_stack(robot, dets, args):
    """2칸 블럭을 가조립하여 한 번에 잡는 멀티 그랩(Multi-Grab) 로직.
    3층짜리 3열 벽을 쌓습니다. (총 9개 소요)
    """
    need = 9
    if len(dets) < need:
        print(f'\n!! "double_stack" SKIP — 검출 cube {len(dets)}개 < 필요 {need}개')
        return

    sorted_dets = sorted(dets, key=lambda d: (d['base_xyz_mm'][1], d['base_xyz_mm'][0]))
    cubes = sorted_dets[:need]
    sample_z = float(np.median([c['base_xyz_mm'][2] for c in cubes]))
    z_table_top = sample_z - p15.CUBE_WIDTH_MM

    print(f'\n=== "double_stack" 멀티 그랩 조립 시작 ({need} cubes) ===')

    STAGING_X = 500.0
    STAGING_Y = SHAPE_BASE_XY[1]
    stage_z = z_table_top + p15.CUBE_WIDTH_MM / 2.0

    cube_idx = 0

    for layer in range(3):
        # 층마다 교차 쌓기 (Interleaved: 0,2층은 왼쪽 2블럭+우측 1블럭 / 1층은 왼쪽 1블럭+우측 2블럭)
        target_z = z_table_top + p15.CUBE_WIDTH_MM / 2.0 + p15.CUBE_WIDTH_MM * layer
        
        if layer % 2 == 0:
            # 1. 가조립 2-block (left & mid)
            c1 = cubes[cube_idx]; cube_idx += 1
            c2 = cubes[cube_idx]; cube_idx += 1
            stage_y1 = STAGING_Y - ART_PITCH_MM / 2.0
            stage_y2 = STAGING_Y + ART_PITCH_MM / 2.0
            
            p16.execute_stack_pick_place(robot, c1['base_xyz_mm'], p16.STACK_PICK_YAW_OFFSET,
                                         (STAGING_X, stage_y1, stage_z), ART_PLACE_YAW, args, z_table_top=z_table_top)
            p16.execute_stack_pick_place(robot, c2['base_xyz_mm'], p16.STACK_PICK_YAW_OFFSET,
                                         (STAGING_X, stage_y2, stage_z), ART_PLACE_YAW, args, z_table_top=z_table_top)
            
            # 멀티 그랩으로 한 번에 잡아서 타겟 위치로 (중심이 타겟 좌측으로)
            multi_pick_xy = (STAGING_X, STAGING_Y)
            multi_target_xy = (SHAPE_BASE_XY[0], SHAPE_BASE_XY[1] - ART_PITCH_MM / 2.0)
            print(f'\n[Multi-Grab] layer{layer} 2-block')
            _execute_multi_grab(robot, multi_pick_xy, multi_target_xy, target_z, args, z_table_top)
            
            # 2. 1-block (right)
            c3 = cubes[cube_idx]; cube_idx += 1
            p16.execute_stack_pick_place(robot, c3['base_xyz_mm'], p16.STACK_PICK_YAW_OFFSET,
                                         (SHAPE_BASE_XY[0], SHAPE_BASE_XY[1] + ART_PITCH_MM, target_z),
                                         ART_PLACE_YAW, args, z_table_top=z_table_top)
        else:
            # 1. 1-block (left)
            c1 = cubes[cube_idx]; cube_idx += 1
            p16.execute_stack_pick_place(robot, c1['base_xyz_mm'], p16.STACK_PICK_YAW_OFFSET,
                                         (SHAPE_BASE_XY[0], SHAPE_BASE_XY[1] - ART_PITCH_MM, target_z),
                                         ART_PLACE_YAW, args, z_table_top=z_table_top)
                                         
            # 2. 가조립 2-block (mid & right)
            c2 = cubes[cube_idx]; cube_idx += 1
            c3 = cubes[cube_idx]; cube_idx += 1
            stage_y1 = STAGING_Y - ART_PITCH_MM / 2.0
            stage_y2 = STAGING_Y + ART_PITCH_MM / 2.0
            
            p16.execute_stack_pick_place(robot, c2['base_xyz_mm'], p16.STACK_PICK_YAW_OFFSET,
                                         (STAGING_X, stage_y1, stage_z), ART_PLACE_YAW, args, z_table_top=z_table_top)
            p16.execute_stack_pick_place(robot, c3['base_xyz_mm'], p16.STACK_PICK_YAW_OFFSET,
                                         (STAGING_X, stage_y2, stage_z), ART_PLACE_YAW, args, z_table_top=z_table_top)
                                         
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
