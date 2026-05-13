#!/usr/bin/env python3
"""
[원샷 캘리브레이션 — 로봇 연결부터 Hand-Eye 캘리브레이션까지 한 번에]

실행 순서:
  1) 기존 ROS 프로세스 모두 정리
  2) dsr_bringup2_moveit launch 백그라운드 실행 (로그: /tmp/dsr_launch.log)
  3) 핵심 ROS 서비스 활성 대기 (60초 타임아웃)
  4) Servo ON (AUTONOMOUS 모드 + 서보 켜기)
  5) 08_카메라_핸드아이_캘리브레이션.py 실행 (대화형)

사용법:
  python3 09_원샷_캘리브레이션.py

옵션:
  --skip-launch   : bringup 이 이미 떠있으면 1~3단계 스킵 (servo on 부터)
  --no-cleanup    : 시작 시 기존 프로세스 정리 생략 (위험)
  --keep-bringup  : 종료 시 bringup 프로세스 살려두기 (RViz 계속 사용 가능)
"""

import os
import sys
import time
import signal
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCH_LOG = "/tmp/dsr_launch.log"

# === 로봇 연결 설정 ===
# 모든 환경 의존 값은 doosan_config (.env / 셸 환경변수 / default 순) 에서 가져옴.
from doosan_config import (
    ROBOT_NAME, ROBOT_MODEL, ROBOT_SUBNET, ROBOT_RT_HOST, ROBOT_PORT,
    BRINGUP_PKG, BRINGUP_LAUNCH,
    detect_ros_distro, detect_doosan_ws, discover_robot_ip,
)

# 자동탐색 결과 캐시 (성공한 IP 저장 — 다음 실행 시 빠른 재연결)
IP_CACHE_FILE = os.path.join(SCRIPT_DIR, ".robot_ip_cache")

CALIB_SCRIPT = os.path.join(SCRIPT_DIR, "08_카메라_핸드아이_캘리브레이션.py")

# launch 종류 — --no-moveit 옵션 시 RViz-only 사용 (캘리브 전용 안전)
# default(MoveIt) 는 doosan_config.BRINGUP_LAUNCH 따라감 (.env 로 override 가능).
LAUNCH_FILE_MOVEIT = BRINGUP_LAUNCH                       # 기본: RViz + MoveIt 모두
LAUNCH_FILE_RVIZ_ONLY = "dsr_bringup2_rviz.launch.py"     # RViz 만 (MoveIt 없음)
LAUNCH_FILE_HEADLESS = "dsr_bringup2.launch.py"           # 시각화도 없음 (없을 수도)


def resolve_robot_host(args) -> str:
    """ROBOT_HOST 결정 우선순위: --host CLI > DSR_HOST 환경변수 > 자동탐색"""
    if args.host:
        return args.host
    env = os.environ.get("DSR_HOST")
    if env:
        return env
    return discover_robot_ip(ROBOT_SUBNET, ROBOT_PORT,
                             max_octet=100, cache_file=IP_CACHE_FILE)


# ==================== 1단계: 기존 프로세스 정리 ====================
def cleanup_existing_ros():
    """
    기존 ROS 노드/RViz/dsr 드라이버 모두 종료.
    Linux comm name 이 15자에서 잘리는 제약 때문에 pkill 기본 매칭은 신뢰 X.
    ps -ef 전체 cmdline 을 파싱해서 PID 로 직접 kill.
    """
    print("[1/5] 기존 ROS 프로세스 정리...")
    patterns = [
        'rviz2', 'move_group', 'robot_state_publisher',
        'ros2_control_node', 'run_emulator', 'DRCF',
        'dsr_bringup2_moveit', 'controller_manager/spawner',
    ]
    my_pid = os.getpid()
    my_ppid = os.getppid()
    pids = set()
    try:
        ps = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True)
        for line in ps.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            pid, cmdline = int(parts[0]), parts[1]
            if pid in (my_pid, my_ppid):
                continue
            if any(p in cmdline for p in patterns):
                pids.add(pid)
    except Exception as e:
        print(f"  [경고] ps 호출 실패: {e}")
        return

    if not pids:
        print("  ✓ 잔존 프로세스 없음")
        return

    print(f"  종료 대상: {len(pids)} 프로세스 (PIDs: {sorted(pids)[:8]}{'...' if len(pids)>8 else ''})")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"  [경고] PID {pid} 권한 부족")
    time.sleep(3)

    # 검증
    result = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True)
    remaining = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        if int(parts[0]) in (my_pid, my_ppid):
            continue
        if any(p in parts[1] for p in patterns):
            remaining.append(line)
    if remaining:
        print(f"  ⚠️ {len(remaining)} 잔존 (강제 처리 필요):")
        for r in remaining[:3]:
            print(f"    {r[:120]}")
    else:
        print("  ✓ 모두 정리됨")


# ==================== 2단계: bringup 실행 ====================
def launch_bringup(launch_file: str = LAUNCH_FILE_MOVEIT,
                   robot_host: str = None,
                   ros_distro: str = None,
                   doosan_ws: str = None):
    """선택한 launch 를 백그라운드로 실행. 프로세스 객체 반환."""
    if robot_host is None:
        raise ValueError("robot_host 가 결정되지 않음 (자동탐색 실패?)")
    if ros_distro is None:
        ros_distro = detect_ros_distro()
    if doosan_ws is None:
        doosan_ws = detect_doosan_ws()

    print(f"\n[2/5] {launch_file} (model={ROBOT_MODEL}, host={robot_host})")
    print(f"        ROS distro: {ros_distro},  doosan_ws: {doosan_ws}")
    cmd = (
        f"source /opt/ros/{ros_distro}/setup.bash && "
        f"source {doosan_ws}/install/setup.bash && "
        f"exec ros2 launch {BRINGUP_PKG} {launch_file} "
        f"name:={ROBOT_NAME} model:={ROBOT_MODEL} mode:=real "
        f"host:={robot_host} rt_host:={ROBOT_RT_HOST}"
    )
    log_file = open(LAUNCH_LOG, 'w')
    proc = subprocess.Popen(
        ['bash', '-c', cmd],
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # 새 process group → 일괄 종료 가능
    )
    print(f"  PID={proc.pid}  로그={LAUNCH_LOG}")
    return proc


# ==================== 3단계: 서비스 활성 대기 ====================
def wait_for_driver_standby(timeout_sec=60):
    """
    launch 로그에서 'ROBOT_STATE: STATE_STANDBY' 메시지 대기.
    드라이버는 INITIAL_CONTROL_SERVO_ON 시퀀스 완료 후 STANDBY 상태가 되어
    그 시점부터 외부 SetRobotMode/Control 명령을 받음.
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if os.path.exists(LAUNCH_LOG):
            with open(LAUNCH_LOG, 'r', errors='replace') as f:
                if 'ROBOT_STATE: STATE_STANDBY' in f.read():
                    return True
        time.sleep(1)
    return False


def wait_for_services(timeout_sec=60):
    """ROS 서비스 advertise 대기 + 드라이버 STATE_STANDBY 도달 + 추가 안정화."""
    print(f"\n[3/5] ROS 서비스 활성 대기 (최대 {timeout_sec}초)...")
    import rclpy
    from dsr_msgs2.srv import GetCurrentPosx, MoveLine
    from dsr_msgs2.srv import SetRobotMode, SetRobotControl

    rclpy.init()
    try:
        node = rclpy.create_node('one_shot_orchestrator_wait')
        services = [
            (GetCurrentPosx,   f'/{ROBOT_NAME}/aux_control/get_current_posx'),
            (MoveLine,         f'/{ROBOT_NAME}/motion/move_line'),
            (SetRobotMode,     f'/{ROBOT_NAME}/system/set_robot_mode'),
            (SetRobotControl,  f'/{ROBOT_NAME}/system/set_robot_control'),
        ]
        deadline = time.time() + timeout_sec
        for srv_type, srv_name in services:
            cli = node.create_client(srv_type, srv_name)
            remaining = max(1.0, deadline - time.time())
            if not cli.wait_for_service(timeout_sec=remaining):
                raise RuntimeError(f"서비스 연결 타임아웃: {srv_name}")
            print(f"  ✓ {srv_name}")
        node.destroy_node()
    finally:
        rclpy.shutdown()

    # 드라이버 자체 INITIAL_CONTROL_SERVO_ON 시퀀스 완료 대기
    print("  ⏳ 드라이버 STATE_STANDBY 도달 대기...")
    if wait_for_driver_standby(timeout_sec=30):
        print("  ✓ STATE_STANDBY 확인")
    else:
        print("  ⚠️ STATE_STANDBY 미확인 (그래도 진행)")
    # 드라이버 안정화 추가 시간 (servo on retry 완료 시간 확보)
    print("  ⏳ 드라이버 안정화 (5초)...")
    time.sleep(5)


# ==================== 4단계: Servo ON ====================
def _call_with_retry(node, cli, req, label, attempts=3, sleep_between=5.0):
    """service call 을 attempts 번 재시도. 마지막에도 실패면 RuntimeError."""
    import rclpy
    last_err = None
    for i in range(1, attempts + 1):
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            last_err = f"{label}: 응답 없음 (timeout)"
        elif not getattr(result, 'success', False):
            last_err = f"{label}: success=False (드라이버가 명령 거부 — INITIAL servo on 시퀀스 진행 중일 수 있음)"
        else:
            if i > 1:
                print(f"  ✓ {label} (재시도 {i}회 만에 성공)")
            return True
        print(f"  [재시도 {i}/{attempts}] {last_err}")
        if i < attempts:
            time.sleep(sleep_between)
    raise RuntimeError(f"{label} {attempts}회 시도 모두 실패: {last_err}")


def activate_servo():
    """
    로봇 모드 AUTONOMOUS 설정 + 서보 ON.
    이전 세션 잔재(stuck TASK_MOTION 등) 정리를 위해 SafeStopReset 우선 호출.
    드라이버 ready 안 됐을 가능성 대비 모든 호출에 재시도 적용.
    """
    print("\n[4/5] Servo ON...")
    import rclpy
    from dsr_msgs2.srv import SetRobotMode, SetRobotControl, SetSafeStopResetType

    rclpy.init()
    try:
        node = rclpy.create_node('one_shot_orchestrator_servo')

        # 0) Safe stop reset — 이전 세션의 stuck motion / alarm 클리어
        cli_reset = node.create_client(
            SetSafeStopResetType, f'/{ROBOT_NAME}/system/set_safe_stop_reset_type')
        if cli_reset.wait_for_service(timeout_sec=5.0):
            try:
                req_reset = SetSafeStopResetType.Request()
                req_reset.reset_type = 0  # PROGRAM_STOP (기본값)
                _call_with_retry(node, cli_reset, req_reset,
                                 'SetSafeStopResetType', attempts=2, sleep_between=2.0)
                print("  ✓ Safe stop reset 호출 (이전 세션 잔재 클리어)")
            except RuntimeError as e:
                # 알람 없으면 reset 이 실패할 수 있음 — 비치명
                print(f"  (Safe stop reset 비치명 실패 — 정상일 수 있음: {e})")
        else:
            print("  (Safe stop reset 서비스 없음 — 스킵)")

        # 1) Robot mode = AUTONOMOUS (1)
        cli_mode = node.create_client(SetRobotMode, f'/{ROBOT_NAME}/system/set_robot_mode')
        cli_mode.wait_for_service(timeout_sec=10.0)
        req_mode = SetRobotMode.Request()
        req_mode.robot_mode = 1
        _call_with_retry(node, cli_mode, req_mode, 'SetRobotMode AUTONOMOUS')
        print("  ✓ 로봇 모드 = AUTONOMOUS")

        # 2) Servo on
        cli_ctrl = node.create_client(SetRobotControl, f'/{ROBOT_NAME}/system/set_robot_control')
        cli_ctrl.wait_for_service(timeout_sec=10.0)
        req_ctrl = SetRobotControl.Request()
        req_ctrl.robot_control = 1
        _call_with_retry(node, cli_ctrl, req_ctrl, 'SetRobotControl SERVO_ON')
        print("  ✓ Servo ON")

        node.destroy_node()
    finally:
        rclpy.shutdown()


def check_motion_alarms_in_log(seconds_back=5):
    """
    launch 로그 마지막 N초 내에 'TASK_MOTION rejected' 알람이 있는지 검사.
    있으면 컨트롤러 박스가 stuck 상태.
    Returns: 알람 개수
    """
    if not os.path.exists(LAUNCH_LOG):
        return 0
    try:
        with open(LAUNCH_LOG, 'r', errors='replace') as f:
            tail = f.read()[-30000:]
        return tail.count('TASK_MOTION] rejected')
    except Exception:
        return 0


# ==================== 5단계: 캘리브레이션 실행 ====================
def run_calibration():
    """08_카메라_핸드아이_캘리브레이션.py 를 자식 프로세스로 실행 (대화형)."""
    print("\n[5/5] 캘리브레이션 시작 — 카메라 창에서 's', Enter, 'c', 'q' 키로 진행")
    print(f"        (스크립트: {CALIB_SCRIPT})")
    print("-" * 70)
    if not os.path.exists(CALIB_SCRIPT):
        raise FileNotFoundError(f"캘리브레이션 스크립트 없음: {CALIB_SCRIPT}")
    # 같은 파이썬 인터프리터로 실행 (cv2/realsense env 일치)
    return subprocess.call([sys.executable, CALIB_SCRIPT])


# ==================== 종료 시 정리 ====================
def stop_bringup(proc):
    """bringup 프로세스 그룹 일괄 종료."""
    if proc is None:
        return
    print("  bringup 프로세스 그룹 종료 중...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
        # SIGTERM 이 안 듣으면 KILL
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError) as e:
        print(f"  (정리 중 무시: {e})")


# ==================== main ====================
def main():
    parser = argparse.ArgumentParser(description="원샷 두산 캘리브레이션")
    parser.add_argument('--skip-launch', action='store_true',
                        help='bringup 이 이미 떠있을 때 1~3단계 스킵')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='시작 시 기존 프로세스 정리 생략')
    parser.add_argument('--keep-bringup', action='store_true',
                        help='종료 시 bringup 프로세스 살려두기 (RViz 등 계속 사용)')
    parser.add_argument('--no-moveit', action='store_true',
                        help='RViz-only launch 사용 (MoveIt 없음 — 캘리브 안 됨, 디버그용만)')
    parser.add_argument('--host', type=str, default=None,
                        help='로봇 IP 직접 지정 (default: 110.120.1.x 자동탐색)')
    parser.add_argument('--rt-host', type=str, default=None,
                        help='RT control 호스트 IP (default: 110.120.1.5)')
    args = parser.parse_args()

    # 환경 자동 감지
    try:
        ros_distro = detect_ros_distro()
        doosan_ws = detect_doosan_ws()
    except RuntimeError as e:
        print(f"[ERROR] 환경 감지 실패: {e}")
        sys.exit(1)

    # 로봇 IP 결정 (CLI > env > 자동탐색)
    print("=" * 70)
    print("  [원샷 캘리브레이션] 로봇 연결 → Servo ON → Hand-Eye Calibration")
    print(f"  ROS distro: {ros_distro}    Doosan WS: {doosan_ws}")
    print(f"  Model: {ROBOT_MODEL}    Namespace: /{ROBOT_NAME}")
    try:
        robot_host = resolve_robot_host(args)
    except RuntimeError as e:
        print(f"[ERROR] 로봇 IP 탐색 실패: {e}")
        sys.exit(1)
    rt_host = args.rt_host if args.rt_host else ROBOT_RT_HOST
    print(f"  Robot IP: {robot_host}    RT Host: {rt_host}")
    # rt_host 도 ROBOT_RT_HOST 변경
    globals()['ROBOT_RT_HOST'] = rt_host
    print("=" * 70)

    bringup_proc = None
    try:
        if args.skip_launch:
            print("\n[1~3 스킵] --skip-launch 옵션. 이미 떠있는 launch 사용.")
        else:
            if not args.no_cleanup:
                cleanup_existing_ros()
            else:
                print("[1/5] 정리 생략 (--no-cleanup)")
            # Default = MoveIt 포함 (정석). 08 은 compute_cartesian_path + execute_trajectory 사용.
            launch_file = LAUNCH_FILE_RVIZ_ONLY if args.no_moveit else LAUNCH_FILE_MOVEIT
            if args.no_moveit:
                print(f"  ⚠️ --no-moveit: RViz-only launch ({LAUNCH_FILE_RVIZ_ONLY})")
                print(f"     MoveIt 없으니 캘리브 못 함. 디버그용만.")
            else:
                print(f"  📌 MoveIt 포함 launch ({LAUNCH_FILE_MOVEIT})")
                print(f"     compute_cartesian_path + execute_trajectory 통해 안전하게 motion")
            bringup_proc = launch_bringup(launch_file, robot_host=robot_host,
                                          ros_distro=ros_distro, doosan_ws=doosan_ws)
            wait_for_services(timeout_sec=60)

        activate_servo()

        # 알람 체크 — TASK_MOTION rejected 가 누적돼 있으면 컨트롤러 박스 자체가 stuck
        n_alarms = check_motion_alarms_in_log()
        if n_alarms > 0:
            print(f"\n  ⚠️ launch 로그에 'TASK_MOTION rejected' 알람 {n_alarms}건 발견.")
            print(f"     로봇이 이전 motion stuck 상태일 수 있음. 그래도 캘리브 진행.")
            print(f"     캘리브 중 motion 실패 반복 시 두산 teach pendant ALARM CLEAR 또는")
            print(f"     컨트롤러 박스 power cycle 필요할 수 있음.")

        rc = run_calibration()
        print(f"\n캘리브레이션 종료 (exit code: {rc})")

    except KeyboardInterrupt:
        print("\n\n[중단] 사용자 Ctrl+C")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        if bringup_proc and not args.keep_bringup:
            stop_bringup(bringup_proc)
        elif bringup_proc and args.keep_bringup:
            print(f"\n[유지] bringup PID={bringup_proc.pid} 계속 실행 중 (--keep-bringup)")
            print(f"      수동 종료: kill -- -{bringup_proc.pid}")


if __name__ == "__main__":
    main()
