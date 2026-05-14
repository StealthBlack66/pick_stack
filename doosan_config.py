"""두산 강의 예제 공용 환경 설정 hub.

본 폴더의 모든 강의 예제 스크립트가 이 파일에서 환경 설정을 가져온다.
다른 사람이 자기 환경에 맞추려면 이 파일을 직접 고치거나, 같은 디렉토리에
`.env` 파일을 두거나, 셸2 환경변수로 export 하면 된다.

우선순위 (높은 것이 우선):
  1. 셸 환경변수  (예: export DSR_NAME=dsr01e0509)
  2. 본 디렉토리의 .env 파일  (있으면 자동 로드, python-dotenv 필요)
  3. 본 파일의 default 값

받는 사람용 셋업 안내는 PORTABILITY.md 참고.
"""

import math
import os
import socket
import concurrent.futures
from pathlib import Path


# ============== .env 자동 로드 ==============
_HERE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    # override=False: 셸에서 export 한 환경변수가 .env 보다 우선
    load_dotenv(_HERE / '.env', override=False)
except ImportError:
    # python-dotenv 미설치여도 환경변수 + default 로 정상 동작
    pass


# ============== 환경 설정 (env-var 또는 default) ==============

# Robot 식별
NAMESPACE   = os.environ.get('DSR_NAME',    'dsr01')
ROBOT_MODEL = os.environ.get('DSR_MODEL',   'e0509')

# 09번 호환 alias (기존 코드가 ROBOT_NAME 사용)
ROBOT_NAME = NAMESPACE

# 네트워크
ROBOT_IP    = os.environ.get('DSR_HOST',    '110.120.1.18')
RT_HOST     = os.environ.get('DSR_RT_HOST', '110.120.1.5')

# 09번 호환 alias
ROBOT_RT_HOST = RT_HOST

# 자동탐색용 서브넷 (DSR_HOST 미지정 시 이 서브넷 .1~.100 스캔)
ROBOT_SUBNET = os.environ.get('DSR_SUBNET', '110.120.1')
ROBOT_PORT = 12345    # 두산 컨트롤러 표준 포트

# 패키지 / launch 이름 (fork 다른 사람용)
BRINGUP_PKG       = os.environ.get('DSR_BRINGUP_PKG',       'dsr_bringup2')
BRINGUP_LAUNCH    = os.environ.get('DSR_BRINGUP_LAUNCH',    'dsr_bringup2_moveit.launch.py')
MOVEIT_CONTROLLER = os.environ.get('DSR_MOVEIT_CONTROLLER', 'dsr_moveit_controller')

# 워크스페이스 / ROS distro (자동 탐색은 detect_*() 함수 통해 별도 호출)
DOOSAN_WS  = os.environ.get('DOOSAN_WS',  os.path.expanduser('~/doosan_ws'))
ROS_DISTRO = os.environ.get('ROS_DISTRO', 'humble')


# ============== Namespace 자동 prefix helper ==============

def srv(path: str) -> str:
    """서비스 경로에 namespace 자동 prefix.

    예) `srv('system/set_robot_mode')` → `'/dsr01/system/set_robot_mode'`
    """
    return f'/{NAMESPACE}/{path.lstrip("/")}'


def action(path: str) -> str:
    """액션/토픽 경로에 namespace 자동 prefix (srv 와 동일하지만 의미 분리)."""
    return f'/{NAMESPACE}/{path.lstrip("/")}'


def moveit_action(suffix: str = 'follow_joint_trajectory') -> str:
    """MoveIt 컨트롤러 액션 경로 헬퍼.

    예) 기본값 → `'/dsr01/dsr_moveit_controller/follow_joint_trajectory'`
    fork 가 다른 컨트롤러 이름을 쓰면 DSR_MOVEIT_CONTROLLER 만 바꾸면 자동 반영.
    """
    return f'/{NAMESPACE}/{MOVEIT_CONTROLLER}/{suffix.lstrip("/")}'


# ============== 환경 자동 감지 (09번에서 이전) ==============

def detect_ros_distro() -> str:
    """/opt/ros/* 디렉토리에서 ROS distro 자동 감지."""
    env = os.environ.get('ROS_DISTRO')
    if env and os.path.isdir(f'/opt/ros/{env}'):
        return env
    for d in ['humble', 'iron', 'jazzy', 'rolling', 'foxy']:
        if os.path.isdir(f'/opt/ros/{d}'):
            return d
    return 'humble'  # fallback


def detect_doosan_ws() -> str:
    """doosan_ws 위치 자동 감지. install/ 가 있는 후보 디렉토리 탐색."""
    env = os.environ.get('DOOSAN_WS')
    if env and os.path.isdir(os.path.join(env, 'install')):
        return env
    home = os.path.expanduser('~')
    candidates = [
        os.path.join(home, 'doosan_ws'),
        os.path.join(home, 'ros2_ws'),
        os.path.join(home, 'dsr_ws'),
        os.path.join(home, 'ws'),
        '/opt/doosan_ws',
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'install')):
            return c
    raise RuntimeError(
        'doosan_ws 자동 감지 실패. DOOSAN_WS 환경변수 지정 필요.\n'
        f'  검색 후보: {candidates}'
    )


def check_doosan_port(ip: str, port: int = ROBOT_PORT, timeout: float = 0.5) -> bool:
    """ip:port 에 TCP 연결 가능한지 확인 (Doosan 컨트롤러 식별)."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def discover_robot_ip(subnet: str = None, port: int = ROBOT_PORT,
                      max_octet: int = 100, timeout: float = 0.5,
                      cache_file: str = None) -> str:
    """Doosan 컨트롤러 IP 자동 탐색.

    {subnet}.1 ~ {subnet}.max_octet 까지 port 12345 (Doosan 표준 포트) 스캔.
    응답하는 첫 IP 가 로봇.

    1. cache_file 에 저장된 IP 가 살아있으면 즉시 반환 (빠른 재연결)
    2. 아니면 1~max_octet 병렬 스캔
    """
    if subnet is None:
        subnet = ROBOT_SUBNET

    # 1) Cache 확인
    if cache_file and os.path.isfile(cache_file):
        try:
            with open(cache_file) as f:
                cached_ip = f.read().strip()
            if check_doosan_port(cached_ip, port, timeout):
                return cached_ip
        except Exception:
            pass

    # 2) 병렬 port scan
    print(f'  Doosan 컨트롤러 자동 탐색: {subnet}.1 ~ {subnet}.{max_octet} (port {port})')
    candidates = [f'{subnet}.{i}' for i in range(1, max_octet + 1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        results = list(ex.map(lambda ip: (ip, check_doosan_port(ip, port, timeout)),
                              candidates))
    alive = [ip for ip, ok in results if ok]
    if not alive:
        raise RuntimeError(
            f'{subnet}.1 ~ {subnet}.{max_octet} 에서 Doosan 컨트롤러 (port {port}) '
            f'응답 없음. 로봇 전원/네트워크 확인 또는 DSR_HOST 환경변수로 직접 지정.'
        )
    ip = alive[0]
    print(f'  ✓ 발견: {ip}')
    if cache_file:
        try:
            with open(cache_file, 'w') as f:
                f.write(ip)
        except Exception:
            pass
    return ip


# ============== 기존 joint offset (06번 호환, 보존) ==============

# 모든 오프셋을 0으로 초기화하여 중첩 오류를 방지합니다.
# (진동이 발생할 경우에만 손목 부분만 미세하게 조정합니다.)
JOINT_OFFSETS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def get_calibrated_radians(user_angles):
    """사용자 입력 각도(deg)에 JOINT_OFFSETS 더하고 라디안 변환."""
    calibrated = []
    for i in range(6):
        angle_with_offset = float(user_angles[i]) + JOINT_OFFSETS[i]
        calibrated.append(math.radians(angle_with_offset))
    return calibrated


# ============== 디버그용: 현재 적용 값 출력 ==============

def print_config():
    """현재 적용된 설정값 출력. `python3 doosan_config.py` 로 실행해도 됨."""
    print('=' * 60)
    print('  doosan_config — 현재 적용 설정')
    print('=' * 60)
    print(f'  NAMESPACE         = {NAMESPACE}')
    print(f'  ROBOT_MODEL       = {ROBOT_MODEL}')
    print(f'  ROBOT_IP          = {ROBOT_IP}')
    print(f'  RT_HOST           = {RT_HOST}')
    print(f'  BRINGUP_PKG       = {BRINGUP_PKG}')
    print(f'  BRINGUP_LAUNCH    = {BRINGUP_LAUNCH}')
    print(f'  MOVEIT_CONTROLLER = {MOVEIT_CONTROLLER}')
    print(f'  DOOSAN_WS         = {DOOSAN_WS}')
    print(f'  ROS_DISTRO        = {ROS_DISTRO}')
    print('  --- 예시 namespace prefix ---')
    print(f"  srv('system/set_robot_mode') = {srv('system/set_robot_mode')}")
    print(f'  moveit_action()              = {moveit_action()}')
    print('=' * 60)


if __name__ == '__main__':
    print_config()
