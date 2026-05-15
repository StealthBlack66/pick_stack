"""
완전 자동 미술 쌓기 (Autonomous Art Stacking) — import-only 모듈

이 파일은 **직접 실행하는 entry point 가 아닙니다**. 다음 두 함수를 다른 스크립트
(예: 17_미술쌓기.py) 에서 import 해서 사용:
  - get_realtime_detections(vision, robot, args)
  - build_art_autonomous(robot, vision, shape_name, args)

상수 (SHAPES 등) 는 17_미술쌓기.py 에서 그대로 가져옴.

사용 예 (17_미술쌓기.py 안에서):
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('p17_test', '17_test.py')
    p17t = module_from_spec(spec); spec.loader.exec_module(p17t)
    p17t.build_art_autonomous(robot, vision, args.shape, args)
"""
import importlib.util
import os
import sys
import time

# 모듈 로드 — 17_미술쌓기.py 가 이미 _load 해뒀으면 재사용 (순환 import 방지).
# Standalone 실행은 아래 __main__ 가드로 거부되므로 sys.modules 에 'p12' 등이 없을 일 없음.
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)


def _get_or_load(name, filename):
    if name in sys.modules and sys.modules[name] is not None:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p12 = _get_or_load('p12', '12_비전_피크앤플레이스.py')
p15 = _get_or_load('p15', '15_바둑판_정렬.py')
p16 = _get_or_load('p16', '16_탑쌓기.py')

# SHAPES 등 상수는 호출자 (보통 17_미술쌓기.py = __main__) 에서 lazy 로 가져옴.
# 모듈 top-level 에서 가져오면 순환 import (17_test ↔ 17_미술쌓기) 발생.
def _shape_consts():
    import __main__ as c
    return c.SHAPES, c.SHAPE_BASE_XY, c.ART_PITCH_MM, c.SHAPE_PITCH_MM, c.ART_PLACE_YAW


def get_realtime_detections(vision, robot, args):
    """로봇을 Home으로 보내 카메라 시야를 확보한 뒤, 최신 큐브 위치를 반환합니다."""
    print("\n[비전] 최신 큐브 위치 파악을 위해 인식 위치로 이동 중...")
    if not args.dry_run:
        robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=1.5)
        time.sleep(0.5) # 진동 방지
    
    # 여러 프레임을 읽어 평균(Median)을 내어 노이즈 제거 (안정성 강화)
    all_frames = []
    for _ in range(5):
        dets = vision.detect_all()
        all_frames.append(dets)
        time.sleep(0.1)
    
    # 현재 시야에 보이는 큐브 중 작업 영역 내에 있는 것만 필터링
    valid_dets = []
    # 간단하게 마지막 프레임 기준으로 하되, 실무에선 프레임간 매칭 로직 권장
    if all_frames[-1]:
        for d in all_frames[-1]:
            bx, by, bz = d['base_xyz_mm']
            pz = bz - p15.CUBE_WIDTH_MM / 2.0
            if (p12.WORK_X[0] <= bx <= p12.WORK_X[1] and
                p12.WORK_Y[0] <= by <= p12.WORK_Y[1]):
                valid_dets.append(d)
                
    # 소모할 큐브를 Y축 기준으로 정렬 (가까운 것부터 집기)
    valid_dets.sort(key=lambda d: d['base_xyz_mm'][1])
    return valid_dets

def build_art_autonomous(robot, vision, shape_name, args):
    """자동 인식 기반 쌓기 메인 루프"""
    SHAPES, SHAPE_BASE_XY, ART_PITCH_MM, SHAPE_PITCH_MM, ART_PLACE_YAW = _shape_consts()
    layout = SHAPES[shape_name]
    need_count = len(layout)
    pitch = SHAPE_PITCH_MM.get(shape_name, ART_PITCH_MM)
    
    print(f'\n=== "{shape_name}" 완전 자동 조립 시작 (필요: {need_count}개) ===')

    # 레이아웃을 아래층부터 쌓도록 정렬
    layout_sorted = sorted(layout, key=lambda x: (x[2], x[1], x[0]))

    for i, item in enumerate(layout_sorted):
        gx, gy, layer = item[0], item[1], item[2]
        yaw = item[3] if len(item) > 3 else ART_PLACE_YAW

        # 1. 실시간 인식 실행
        dets = get_realtime_detections(vision, robot, args)
        
        if not dets:
            print(f"\n[오류] 사용할 수 있는 큐브를 찾지 못했습니다. ({i}/{need_count} 완료)")
            break
            
        # 2. 첫 번째로 검출된 큐브를 선택
        cube = dets[0]
        cx, cy, cz = cube['base_xyz_mm']
        src = (cx, cy, cz)
        
        # 3. 목표 좌표 계산
        target_x = SHAPE_BASE_XY[0] + gx * pitch
        target_y = SHAPE_BASE_XY[1] + gy * pitch
        # 테이블 바닥 높이(z_table_top)는 첫 인식 시점의 cz 기준으로 추정
        z_table_top = cz - (p15.CUBE_WIDTH_MM / 2.0)
        target_center_z = z_table_top + (p15.CUBE_WIDTH_MM / 2.0) + (p15.CUBE_WIDTH_MM * layer)
        target = (target_x, target_y, target_center_z)

        print(f"\n[{i+1}/{need_count}] 자동 매칭: 인식위치({cx:.1f}, {cy:.1f}) -> 목표(layer {layer})")
        
        # 4. 실행
        p16.execute_stack_pick_place(
            robot, src, p16.STACK_PICK_YAW_OFFSET, target, yaw,
            args, z_table_top=z_table_top
        )

    print(f'\n=== "{shape_name}" 자동 조립 프로세스 종료 ===')

if __name__ == '__main__':
    sys.stderr.write(
        '\n[17_test.py] 이 파일은 직접 실행하는 entry point 가 아닙니다.\n'
        '             17_미술쌓기.py 에서 import 해서 사용하세요.\n'
        '             제공 함수: get_realtime_detections, build_art_autonomous\n\n'
    )
    sys.exit(1)
