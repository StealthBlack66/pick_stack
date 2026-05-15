"""
[로보티스 RH-P12-RN-A 그리퍼 키보드 제어 — 통합판]

두산 e0509 + RH-P12-RN-A 통합 흐름을 한 스크립트로 실행:
  0. (default) 잔존 ROS/RViz/DRCF/realsense 정리 + dsr_bringup2_moveit.launch.py
     자동 실행 + RViz 백그라운드 띄움 + 25초 안정화 대기
  1. (default) AUTONOMOUS 모드 + Servo ON
  2. 그리퍼 시리얼 open + Goal Current(압력) + Torque Enable
  3. 키 입력 루프 — 위치 + 압력 실시간 조절

키 매핑:
  ─── 위치 제어 ───
    o : 완전 열기 (pos=0)
    c : 완전 닫기 (pos=700)
    h : 절반 위치 (pos=350)
    + : pos +50 (더 닫기, 큰 변화)
    - : pos -50 (더 열기, 큰 변화)
    > : pos +10 (중간)
    < : pos -10 (중간)
    . : pos +1  (미세)
    , : pos -1  (미세)
    s : 마지막 목표(pos/force/torque) 출력

  ─── 압력(Goal Current) 제어 ───
    1 : 매우 부드러움  (50)
    2 : 부드러움       (100)
    3 : 약함           (200)
    4 : 약중           (400)
    5 : 기본           (600)
    6 : 중강           (900)
    7 : 강함           (1300)
    8 : 매우 강함      (1700)
    9 : 최대           (2047)
    ] : 압력 +50  (미세조정)
    [ : 압력 -50  (미세조정)
    f : 현재 압력 출력
    t : 토크 ON/OFF 토글

  ─── 자동 ───
    a : Auto-grip — 닫으면서 강성 측정 → 적정 force 자동 적용
    r : Present Position 읽기 (디버그, 여러 주소 후보 시도)
    e : FC06 echo test (시리얼 패스스루 read 자체 검증)

  ─── 종료 ───
    q : 종료 (Ctrl-C 도 가능)

CLI:
  python3 11_그리퍼_키보드_제어.py             # 자동 launch + Servo ON
  python3 11_그리퍼_키보드_제어.py --no-launch  # 드라이버 이미 떠있다고 가정
  python3 11_그리퍼_키보드_제어.py --no-servo   # 모터 활성화 skip
"""

import argparse
import os
import subprocess
import sys
import termios
import time
import tty

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import (
    FlangeSerialOpen, FlangeSerialClose, FlangeSerialWrite, FlangeSerialRead,
    SetRobotMode, SetRobotControl,
)

from doosan_config import (
    NAMESPACE as NS,
    ROBOT_IP, RT_HOST, ROBOT_MODEL,
    BRINGUP_PKG, BRINGUP_LAUNCH,
    DOOSAN_WS, ROS_DISTRO,
)


# ============== 상수 ==============
PORT = 1            # 두산 툴 플랜지 시리얼 채널
BAUD = 57600        # RH-P12-RN-A 기본 보드레이트
SLAVE_ID = 1

# 위치
POS_MIN = 0         # 완전 열림
POS_MAX = 700       # 완전 닫힘
POS_STEP = 50          # 큰 변화 (+ / -)
POS_STEP_MID = 10      # 중간 변화 (> / <)
POS_STEP_FINE = 1      # 미세 변화 (. / ,)

# 압력 (Goal Current, 레지스터 275/0x0113)
FORCE_DEFAULT = 400
FORCE_MIN = 0
FORCE_MAX = 2047
FORCE_STEP = 50

# RH-P12-RN-A Modbus 레지스터 (e-Manual). 표 주소 - 40001 = wire 주소.
#   256 0x0100 1B  RW  Torque Enable      (table 40257)
#   275 0x0113 2B  RW  Goal Current       (table 40276)
#   282 0x011A 4B  RW  Goal Position      (table 40283)
#   287 0x011F 2B  R   Present Current    (table 40288, signed mA, ±1984)
#   290 0x0122 4B  R   Present Position   (table 40291, uint32, 0~700)
ADDR_PRESENT_CURRENT  = 287
ADDR_PRESENT_POSITION = 290

FORCE_PRESETS = {
    '1': (50,   '매우 부드러움'),
    '2': (100,  '부드러움'),
    '3': (200,  '약함'),
    '4': (400,  '약중'),
    '5': (600,  '기본'),
    '6': (900,  '중강'),
    '7': (1300, '강함'),
    '8': (1700, '매우 강함'),
    '9': (2047, '최대'),
}

# RH-P12-RN-A 스펙 기반 환산 (추정값):
#   - Goal Current raw: 0~2047
#   - 1 raw ≈ 1 mA (다이나믹셀 protocol 2.0 컨벤션)
#   - Max pinch force ≈ 32 N (raw 1500 ≈ 30 N 부근에서 saturate)
# 정확한 force 는 외부 force sensor 또는 Present Current read 필요.
# Present Current read 가 펌웨어 미지원 환경에서는 Goal Current 의 "최대 한계" 만 표시.
def force_to_human(goal_current: int) -> str:
    """Goal Current(raw 0~2047) → 사람이 이해하기 쉬운 문자열."""
    ma = goal_current        # 1 raw ≈ 1 mA
    # 비선형 근사 — raw 1500 에서 핀치력 약 30N saturate 가정
    pinch_n = min(32.0, goal_current * 32.0 / 1500.0)
    # 카테고리
    for thr, label in [
        (1200, '매우 강함'),
        ( 750, '강함'),
        ( 450, '중간'),
        ( 250, '약함'),
        ( 100, '부드러움'),
        (   0, '매우 부드러움'),
    ]:
        if goal_current >= thr:
            category = label
            break
    else:
        category = '미설정'
    return f'raw={goal_current:>4d}  ≈ {ma:>4d} mA  ≈ 최대 {pinch_n:>4.1f} N  ({category})'

# Auto-grip 파라미터
AUTO_PROBE_FORCE = 150        # 1단계: 부드럽게 접촉
AUTO_TEST_FORCE  = 800        # 2단계: 강하게 눌러 변형량 측정
AUTO_POLL_INTERVAL = 0.08     # 위치 폴링 주기 (s)
AUTO_STALL_COUNT = 3          # 위치가 N회 연속 변하지 않으면 접촉 판단
AUTO_STALL_TOL = 2            # 변하지 않음 판정 허용오차 (pos units)
AUTO_PROBE_TIMEOUT_S = 4.0    # 접촉 대기 최대 시간
AUTO_SETTLE_S = 0.5           # force 변경 후 안정화 대기

# Δ(P_high - P_contact) 임계값으로 강성 분류 → 최종 force
AUTO_HARDNESS_TABLE = [
    # (delta_threshold_inclusive, final_force, label)
    (50, 100, '매우 부드러움 (스폰지/풍선/딸기)'),
    (20, 200, '부드러움 (종이컵/플라스틱)'),
    ( 5, 400, '보통 (러버/소프트 플라스틱)'),
    ( 0, 600, '단단함 (나무/금속)'),
]

# 두산 드라이버 자동 launch 파라미터 (12번과 동일)
# ROBOT_IP / RT_HOST / ROBOT_MODEL 은 doosan_config 에서 import (위쪽).
DRIVER_WAIT_SEC = 25
# RViz는 dsr_bringup2_moveit.launch.py 내부 start.launch.py 가 자동으로 띄움.
# 따라서 본 스크립트에서 별도 RViz를 띄우면 창이 2개가 되므로 띄우지 않음.


# ============== Modbus RTU 프레임 빌더 ==============

def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return crc


def _make_frame(data: bytes) -> list:
    crc = _crc16(data)
    return list(data + bytes([crc & 0xFF, (crc >> 8) & 0xFF]))


def fc06_torque_enable(value: int = 1) -> list:
    """레지스터 256(0x0100) ← value (1=ON, 0=OFF)."""
    value = 1 if value else 0
    return _make_frame(bytes([SLAVE_ID, 0x06, 0x01, 0x00, 0x00, value]))


def fc06_goal_current(current: int) -> list:
    """레지스터 275(0x0113) ← Goal Current (잡는 힘)."""
    current = max(FORCE_MIN, min(FORCE_MAX, int(current)))
    return _make_frame(bytes([
        SLAVE_ID, 0x06,
        0x01, 0x13,
        (current >> 8) & 0xFF, current & 0xFF,
    ]))


def fc16_position(position: int) -> list:
    """레지스터 282(0x011A)부터 2개 ← (position, 0): 위치 명령."""
    position = max(POS_MIN, min(POS_MAX, position))
    return _make_frame(bytes([
        SLAVE_ID, 0x10,
        0x01, 0x1A,
        0x00, 0x02,
        0x04,
        (position >> 8) & 0xFF, position & 0xFF,
        0x00, 0x00,
    ]))


def fc_read(reg: int, count: int, fc: int = 0x03) -> list:
    """FC03(Read Holding) 또는 FC04(Read Input) 프레임."""
    return _make_frame(bytes([
        SLAVE_ID, fc,
        (reg >> 8) & 0xFF, reg & 0xFF,
        (count >> 8) & 0xFF, count & 0xFF,
    ]))


# Present Position 주소 후보. (FC, register, count)
# 워크스페이스 두 정의(276/2regs, 281/1reg)에 더해, Robotis 다이나믹셀 컨벤션으로
# 가능한 위치를 추가. 두산 펌웨어가 FC03 silent ignore 면 FC04도 시도.
PRESENT_POS_CANDIDATES = [
    (0x03, 276, 2),
    (0x03, 281, 1),
    (0x03, 280, 1),
    (0x03, 261, 2),    # Dynamixel Present Position 영역(132-bytes 기준)
    (0x04, 276, 2),
    (0x04, 281, 1),
]


def parse_present_position(resp: bytes, expected_fc: int = 0x03):
    """FC03/FC04 응답에서 첫 워드를 위치로 해석. 0~700 범위 검증. 실패 시 None."""
    if not resp or len(resp) < 7:
        return None
    if resp[0] != SLAVE_ID or resp[1] != expected_fc:
        return None
    byte_count = resp[2]
    if byte_count < 2 or len(resp) < 3 + byte_count + 2:
        return None
    pos = (resp[3] << 8) | resp[4]
    if not (0 <= pos <= 800):   # 범위 0~700 + 약간의 마진
        return None
    return pos


# ============== 드라이버 자동 launch + RViz ==============
def reset_robot_driver():
    """잔존 ROS/RViz/DRCF/realsense 정리 → dsr_bringup2_moveit 새로 띄움 → RViz."""
    print('\n[0/4] 기존 ROS / 카메라 / RViz 정리 + 두산 드라이버 새로 시작')

    cleanup_cmds = [
        'pkill -9 -f dsr',
        'pkill -9 -f rviz2',
        'pkill -9 -f DRCF',
        'pkill -9 -f ros2',
        'pkill -9 -f realsense',
    ]
    for c in cleanup_cmds:
        os.system(c + ' 2>/dev/null')

    # 한글 파일명은 pkill -f 패턴 매칭이 불안정해 PID 직접 추출 (자기 자신 제외)
    self_pid = str(os.getpid())
    extract = (
        "ps -ef | grep -E '(08_카메라_핸드아이|09_원샷|10_그리퍼_연결|"
        "12_비전_피크|13_|14_)' | grep -v grep | "
        f"awk '{{if($2!={self_pid}) print $2}}'"
    )
    pids = subprocess.run(['bash', '-c', extract],
                          capture_output=True, text=True).stdout.strip()
    if pids:
        os.system(f'kill -9 {pids.replace(chr(10), " ")} 2>/dev/null')

    os.system('docker stop $(docker ps -a -q) 2>/dev/null')
    os.system('docker rm   $(docker ps -a -q) 2>/dev/null')

    time.sleep(3)
    print('   기존 프로세스 정리 완료.')

    ros_cmd = (
        f'source /opt/ros/{ROS_DISTRO}/setup.bash && '
        f'source {DOOSAN_WS}/install/setup.bash && '
        f'ros2 launch {BRINGUP_PKG} {BRINGUP_LAUNCH} '
        f'name:={NS} model:={ROBOT_MODEL} mode:=real '
        f'host:={ROBOT_IP} rt_host:={RT_HOST}'
    )
    terminal_cmd = f'gnome-terminal -- bash -c "{ros_cmd}; exec bash"'
    subprocess.Popen(terminal_cmd, shell=True)
    print(f'   새 터미널에서 {BRINGUP_LAUNCH} 실행 (host={ROBOT_IP})')

    print(f'   드라이버 안정화 대기 ({DRIVER_WAIT_SEC}초)')
    for i in range(DRIVER_WAIT_SEC, 0, -1):
        sys.stdout.write(f'\r   대기 중... {i:>2}초 ')
        sys.stdout.flush()
        time.sleep(1)
    print('\n   드라이버 준비 완료.')
    print('   (RViz는 launch 가 자체적으로 띄움 — 별도 실행 안 함)')


# ============== 키보드 입력 ==============

def get_key():
    """터미널을 raw 모드로 바꿔 한 글자 즉시 읽기."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# ============== 그리퍼 + 로봇 노드 ==============

class GripperKeyController(Node):
    def __init__(self):
        super().__init__('gripper_key_controller')

        # 그리퍼 시리얼 패스스루 서비스
        gp = f'/{NS}/gripper'
        self.cli_open = self.create_client(FlangeSerialOpen, f'{gp}/flange_serial_open')
        self.cli_close = self.create_client(FlangeSerialClose, f'{gp}/flange_serial_close')
        self.cli_write = self.create_client(FlangeSerialWrite, f'{gp}/flange_serial_write')
        self.cli_read = self.create_client(FlangeSerialRead, f'{gp}/flange_serial_read')

        # 로봇 시스템 (Servo ON 용)
        self.cli_mode = self.create_client(SetRobotMode, f'/{NS}/system/set_robot_mode')
        self.cli_ctrl = self.create_client(SetRobotControl, f'/{NS}/system/set_robot_control')

        # 상태
        self.target = POS_MIN
        self.force = FORCE_DEFAULT
        self.torque_on = True
        self._serial_open = False

    def _call(self, cli, req, timeout=5.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    # ---- 로봇 활성화 ----
    def activate_robot(self):
        """AUTONOMOUS 모드 + Servo ON. 이미 활성이면 빠르게 통과(idempotent)."""
        if not self.cli_mode.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('서비스 미응답: set_robot_mode — dsr_bringup2 살아있는지 확인')
        if not self.cli_ctrl.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('서비스 미응답: set_robot_control')

        m = SetRobotMode.Request()
        m.robot_mode = 1   # AUTONOMOUS
        r = self._call(self.cli_mode, m)
        if not (r and r.success):
            print('   (이미 AUTONOMOUS 모드일 가능성 — 계속 진행)')

        c = SetRobotControl.Request()
        c.robot_control = 1   # SERVO ON
        r = self._call(self.cli_ctrl, c)
        if not (r and r.success):
            print('   (이미 Servo ON 상태일 가능성 — 계속 진행)')

    # ---- 그리퍼 시리얼 + 토크 + 초기 압력 ----
    def serial_open_and_init(self):
        for cli, name in [
            (self.cli_open, 'flange_serial_open'),
            (self.cli_close, 'flange_serial_close'),
            (self.cli_write, 'flange_serial_write'),
            (self.cli_read,  'flange_serial_read'),
        ]:
            if not cli.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f'서비스 {name} 미응답 — 두산 드라이버 실행 여부 확인')

        # stale handle 정리
        try:
            pre = FlangeSerialClose.Request()
            pre.port = PORT
            self._call(self.cli_close, pre, timeout=3.0)
        except Exception:
            pass
        time.sleep(0.3)

        # open with retry
        for attempt in range(3):
            req_o = FlangeSerialOpen.Request()
            req_o.port = PORT
            req_o.baudrate = BAUD
            req_o.bytesize = 8
            req_o.parity = 0
            req_o.stopbits = 1
            r = self._call(self.cli_open, req_o)
            if r and r.success:
                break
            try:
                rc = FlangeSerialClose.Request(); rc.port = PORT
                self._call(self.cli_close, rc, timeout=2.0)
            except Exception:
                pass
            time.sleep(0.3 + attempt * 0.2)
        else:
            raise RuntimeError('FlangeSerialOpen 실패 — 그리퍼 케이블/전원 확인')
        self._serial_open = True
        time.sleep(0.1)

        # 토크 ON
        if not self._write_frame(fc06_torque_enable(1)):
            raise RuntimeError('토크 활성화 실패')
        self.torque_on = True
        time.sleep(0.15)

        # 초기 압력 적용 (다음 위치 명령부터 효과)
        if not self._write_frame(fc06_goal_current(self.force)):
            print('   (Goal Current 초기 설정 실패 — 펌웨어 default 로 동작)')
        time.sleep(0.1)

        # Read 가용성 사전 점검 (auto-grip 가능 여부 미리 안내)
        print('   read 가용성 점검 중...')
        p0 = self.read_present_position(settle=0.3, timeout=0.8)
        if p0 is None:
            print('   ⚠ Present Position read 미응답 — auto-grip (a 키) 사용 불가.')
            print('     기본 위치/압력 명령은 정상. force preset (1-9) 으로 수동 사용 가능.')
        else:
            print(f'   ✓ Present Position read OK (현재 pos={p0}) — auto-grip 사용 가능.')

    def serial_close(self):
        if not self._serial_open:
            return
        req = FlangeSerialClose.Request()
        req.port = PORT
        self._call(self.cli_close, req)
        self._serial_open = False

    # ---- 명령 ----
    def _write_frame(self, frame) -> bool:
        req = FlangeSerialWrite.Request()
        req.port = PORT
        req.data = frame
        r = self._call(self.cli_write, req)
        return bool(r and r.success)

    def set_position(self, pos: int) -> bool:
        pos = max(POS_MIN, min(POS_MAX, pos))
        # 매 위치 명령 전 토크 enable 재호출 (sticky open 보완).
        # init 직후 첫 명령은 거짓 success 가능 → 매번 토크 상태 재확인이 안전.
        self._write_frame(fc06_torque_enable(1 if self.torque_on else 0))
        time.sleep(0.05)
        if self._write_frame(fc16_position(pos)):
            self.target = pos
            return True
        return False

    def set_force(self, current: int) -> bool:
        current = max(FORCE_MIN, min(FORCE_MAX, current))
        # 매 force 명령 전에도 토크 enable 재호출 (set_position 과 동일 패턴)
        self._write_frame(fc06_torque_enable(1 if self.torque_on else 0))
        time.sleep(0.05)
        if self._write_frame(fc06_goal_current(current)):
            self.force = current
            return True
        return False

    def toggle_torque(self) -> bool:
        new_state = not self.torque_on
        if self._write_frame(fc06_torque_enable(1 if new_state else 0)):
            self.torque_on = new_state
            return True
        return False

    # ---- 피드백 (Present Position) ----
    def _read_response(self, timeout: float = 0.5) -> bytes:
        """FlangeSerialRead 호출, raw bytes 반환. 실패 시 b''."""
        req = FlangeSerialRead.Request()
        req.port = PORT
        req.timeout = float(timeout)
        r = self._call(self.cli_read, req, timeout=max(1.0, timeout + 0.5))
        if r and r.success:
            return bytes(r.data)
        return b''

    def read_present_position(self, settle: float = 0.5, timeout: float = 1.0,
                              verbose: bool = False):
        """FC03/FC04 query → 대기 → 응답 read → 위치 파싱.
        여러 (FC, reg, count) 후보를 차례로 시도. 실패 시 None.

        settle/timeout default는 두산 공식 gripper_service_node.cpp 의 보수적 값
        (sleep 500ms, read_timeout 3.0s) 을 흉내. auto_grip 폴링 시는 빠른 값을 명시 인자로.
        """
        last_resp = b''
        for fc, reg, count in PRESENT_POS_CANDIDATES:
            if not self._write_frame(fc_read(reg, count, fc)):
                if verbose:
                    print(f'    [debug] write 실패 (FC=0x{fc:02X} reg={reg})')
                continue
            time.sleep(settle)
            resp = self._read_response(timeout=timeout)
            last_resp = resp
            pos = parse_present_position(resp, expected_fc=fc)
            if verbose:
                print(f'    [debug] FC=0x{fc:02X} reg={reg} count={count}  '
                      f'resp_len={len(resp)}  '
                      f'hex={resp.hex() if resp else "(empty)"}  parsed={pos}')
            if pos is not None:
                if PRESENT_POS_CANDIDATES[0] != (fc, reg, count):
                    PRESENT_POS_CANDIDATES.remove((fc, reg, count))
                    PRESENT_POS_CANDIDATES.insert(0, (fc, reg, count))
                return pos
        if verbose:
            print(f'    [debug] 모든 후보 실패. last_resp_len={len(last_resp)}')
        return None

    def echo_test(self):
        """FC06(write single register, register 256, value 1=Torque ON)을 보내고 응답을 읽어
        그리퍼가 write echo 를 보내는지 확인. 시리얼 패스스루 read 자체의 검증용."""
        print('  [echo] FC06 torque_enable 명령 전송...')
        if not self._write_frame(fc06_torque_enable(1)):
            print('  [echo] write 자체 실패')
            return
        # DRCF v2.10.03.00 / DRFL v1.33.03 환경에서 read 응답 지연 가능성 — 두산 공식 gripper.py 수준의 여유 부여
        print('  [echo] RX 대기 1.5초 → flange_serial_read timeout 3초로 호출...')
        time.sleep(1.5)
        resp = self._read_response(timeout=3.0)
        if resp:
            print(f'  [echo] 응답: len={len(resp)}  hex={resp.hex()}')
            if len(resp) >= 8 and resp[0] == SLAVE_ID and resp[1] == 0x06:
                print('  [echo] ✓ FC06 echo 정상 — 시리얼 read 서비스 자체는 작동.')
                print('  [echo]   Present Position read 미응답은 펌웨어가 FC03/FC04 미지원이거나')
                print('  [echo]   해당 주소를 모름. → auto-grip은 위치 read 없이 동작 예정.')
            elif len(resp) >= 5 and resp[1] == 0x86:
                print('  [echo] FC06 exception 응답 — register 256 자체에 문제. 보고서 작성 필요.')
            else:
                print('  [echo] 알 수 없는 응답 형식. 위 hex 분석 필요.')
        else:
            print('  [echo] 응답 없음 — 두산 시리얼 패스스루 read 자체가 작동 안 함.')
            print('  [echo]   /dsr01/gripper/flange_serial_read 서비스 또는 두산 controller 설정 점검 필요.')
            print('  [echo]   진단 체크리스트:')
            print('  [echo]    1) dsr_bringup2 가 emulator(mode:=virtual) 가 아닌지 확인 (mode:=real + 실기 host)')
            print('  [echo]    2) 그리퍼 RS-485 케이블 분리 후 재결합 (RX 라인 접촉 불량 확인)')
            print('  [echo]    3) 펜던트에서 두산 controller 재부팅 (시리얼 핸들/패스스루 버퍼 리셋)')
            print('  [echo]    4) doosan-robot2/dsr_controller2/scripts/gripper.py 로 동일 증상 재현 여부 확인')

    # ---- Auto-grip: 위치 피드백 기반 강성 추정 ----
    def auto_grip(self):
        """
        닫으면서 강성을 측정해서 적정 force를 자동으로 잡는 절차.
          1. force=AUTO_PROBE_FORCE (낮음) 로 닫기 시작
          2. Present Position 폴링 — 위치가 멈추면 접촉 (P_contact)
          3. force=AUTO_TEST_FORCE (높음) 로 올리고 잠깐 대기
          4. 다시 위치 read → P_high
          5. Δ = P_high - P_contact 로 강성 분류, 최종 force 적용

        반환: dict(category, final_force, p_contact, p_high, delta) 또는 None(실패)
        """
        # 사전 점검: 위치 read 한 번으로 통신 검증
        p0 = self.read_present_position()
        if p0 is None:
            print('\n  !! Present Position 읽기 실패 — auto-grip 사용 불가.')
            print('     원인: 두산 시리얼 패스스루 read 또는 그리퍼 펌웨어가 FC03/FC04 미응답.')
            print('     진단: e (echo test) 로 시리얼 read 자체 가용성 확인.')
            print()
            print('     ✋ 대신 force preset 으로 수동 잡기:')
            print('        1) 잡을 물체를 그리퍼 사이에 놓기')
            print('        2) 1~9 키로 적절한 force 선택:')
            print('             1=매우 부드러움 (스폰지/딸기)   →  50')
            print('             2=부드러움 (종이컵)            → 100')
            print('             3=약함                         → 200')
            print('             5=기본 (러버/플라스틱)          → 600')
            print('             7=강함 (단단한 물체)            →1300')
            print('             9=최대 (나무/금속)              →2047')
            print('        3) c 키 (또는 + 로 단계적) — 닫기')
            print('        4) 잡힌 정도 보고 force 조정 가능 ([ ] 키로 미세조정)')
            return None

        print(f'\n  [auto-grip] 시작 — 현재 pos={p0}')

        # 1) probe force 적용
        if not self.set_force(AUTO_PROBE_FORCE):
            print('  !! probe force 설정 실패')
            return None
        time.sleep(0.05)

        # 2) 닫기 명령 + 위치 폴링
        if not self.set_position(POS_MAX):
            print('  !! 닫기 명령 실패')
            return None

        print(f'  [auto-grip] 1/3 probe (force={AUTO_PROBE_FORCE}) — 접촉 대기...')
        t0 = time.time()
        last_pos = p0
        stall = 0
        p_contact = None
        while time.time() - t0 < AUTO_PROBE_TIMEOUT_S:
            time.sleep(AUTO_POLL_INTERVAL)
            p = self.read_present_position()
            if p is None:
                continue
            if abs(p - last_pos) <= AUTO_STALL_TOL:
                stall += 1
            else:
                stall = 0
            last_pos = p
            if stall >= AUTO_STALL_COUNT:
                p_contact = p
                break
            # 거의 다 닫혔는데 멈춤 없이 도달 → 물체 없음
            if p >= POS_MAX - 5:
                p_contact = p
                break

        if p_contact is None:
            print('  !! 접촉 감지 실패 (timeout) — 위치가 계속 변함')
            return None

        # 빈 그리퍼 케이스
        if p_contact >= POS_MAX - 10:
            print(f'  [auto-grip] 물체 없음 (p={p_contact} ≈ 완전 닫힘) — '
                  'force는 기본값으로 복귀')
            self.set_force(FORCE_DEFAULT)
            return {'category': '물체 없음', 'final_force': FORCE_DEFAULT,
                    'p_contact': p_contact, 'p_high': p_contact, 'delta': 0}

        print(f'  [auto-grip] 2/3 접촉 P1={p_contact} — 강성 측정 (force={AUTO_TEST_FORCE})')

        # 3) 강한 force로 다시 눌러보고 위치 변화 측정
        if not self.set_force(AUTO_TEST_FORCE):
            print('  !! test force 설정 실패')
            return None
        # set_position을 다시 보내야 펌웨어가 새 force로 재시도
        self.set_position(POS_MAX)
        time.sleep(AUTO_SETTLE_S)
        p_high = self.read_present_position()
        if p_high is None:
            print('  !! P_high 읽기 실패')
            self.set_force(FORCE_DEFAULT)
            return None

        delta = p_high - p_contact

        # 4) 강성 분류 (Δ가 큰 순서대로 매칭)
        for thr, force, label in AUTO_HARDNESS_TABLE:
            if delta >= thr:
                final_force = force
                category = label
                break
        else:
            final_force = FORCE_DEFAULT
            category = '판정 실패'

        print(f'  [auto-grip] 3/3 P2={p_high}  Δ={delta}  → {category}  '
              f'→ final force={final_force}')

        # 5) 최종 force 적용 + 마지막 닫기 명령으로 안정
        self.set_force(final_force)
        self.set_position(POS_MAX)
        time.sleep(0.2)

        return {'category': category, 'final_force': final_force,
                'p_contact': p_contact, 'p_high': p_high, 'delta': delta}

    def read_status_chunk(self):
        """Present Current(mA) + Present Position(0~700) 동시 읽기.

        Modbus FC03 한 트랜잭션으로 reg 287..290 (4 regs = 8 byte) 읽음.
          [0..1] Present Current   2B signed   mA
          [2..3] Present Velocity  2B signed   (사용 안 함)
          [4..7] Present Position  4B uint32   0~700

        실패 시 개별 read 폴백.
        반환: (position, current_mA), 실패 항목은 None.
        """
        # 1) 4-reg 일괄 읽기 (응답 = SID + FC + bc(=8) + 8B data + 2B CRC = 13B)
        #    DRCF v2.10.03.00 / DRFL v1.33.03 에서 RX 캡처가 느릴 가능성이 있어
        #    write→read 사이 80ms settle, read timeout 500ms 사용.
        if self._write_frame(fc_read(ADDR_PRESENT_CURRENT, 4, 0x03)):
            time.sleep(0.08)
            r = self._read_response(timeout=0.5)
            if (len(r) >= 13 and r[0] == SLAVE_ID
                    and r[1] == 0x03 and r[2] == 8):
                cur = (r[3] << 8) | r[4]
                if cur > 32767:
                    cur -= 65536
                pos = (r[7] << 24) | (r[8] << 16) | (r[9] << 8) | r[10]
                return pos, cur

        # 2) 폴백: 개별 읽기
        pos = cur = None
        if self._write_frame(fc_read(ADDR_PRESENT_CURRENT, 1, 0x03)):
            time.sleep(0.08)
            rc = self._read_response(timeout=0.5)
            if len(rc) >= 7 and rc[1] == 0x03:
                raw = (rc[3] << 8) | rc[4]
                cur = raw - 65536 if raw > 32767 else raw
        if self._write_frame(fc_read(ADDR_PRESENT_POSITION, 2, 0x03)):
            time.sleep(0.08)
            rp = self._read_response(timeout=0.5)
            if len(rp) >= 9 and rp[1] == 0x03:
                pos = (rp[3] << 24) | (rp[4] << 16) | (rp[5] << 8) | rp[6]
        return pos, cur

    def plot_material_property(self):
        print("\n  [PLOT] 물성 측정을 시작합니다.")
        print("  [PLOT] 1. 측정 준비: 그리퍼를 열고 대기합니다...")
        self.set_position(POS_MIN)
        self.set_force(800) # 충분한 힘으로 설정
        time.sleep(1.0)
        
        print("  [PLOT] 2. 측정 시작! (물체를 꽉 쥐면서 3초간 실시간 데이터 수집)")
        self.set_position(POS_MAX)
        
        t0 = time.time()
        times = []
        positions = []
        currents = []
        
        while time.time() - t0 < 3.0:
            p, c = self.read_status_chunk()
            if p is not None and c is not None:
                times.append(time.time() - t0)
                positions.append(p)
                currents.append(c)
            # 폴링 주기 조절 (FlangeSerialRead 내부 timeout 때문에 이미 약간의 딜레이가 있음)
            time.sleep(0.01)
            
        print(f"  [PLOT] 측정 완료! 총 {len(times)}개의 정상 데이터 수집.")
        if len(times) < 3:
            print("  [PLOT] ❌ 통신 문제로 데이터를 충분히 수집하지 못했습니다. (에코 테스트 'e'를 확인해보세요)")
            return
            
        print("  [PLOT] 그래프 창을 띄웁니다. (창을 닫으면 제어 모드로 돌아갑니다)")
        
        plt.figure(figsize=(12, 5))
        
        # 1. 시간 대비 위치/전류 그래프
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('Time vs Position & Current')
        ax1.set_xlabel('Time (s)')
        
        color1 = 'tab:blue'
        ax1.plot(times, positions, label='Position (0~700)', color=color1, marker='.')
        ax1.set_ylabel('Position', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True)
        
        ax2 = ax1.twinx()  
        color2 = 'tab:red'
        ax2.plot(times, currents, label='Current (mA)', color=color2, marker='.')
        ax2.set_ylabel('Current (mA)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 2. 물성 곡선 (Position vs Current)
        ax3 = plt.subplot(1, 2, 2)
        ax3.plot(positions, currents, 'o-', color='tab:green', markersize=4)
        ax3.set_title('Force-Displacement Curve (Stiffness)')
        ax3.set_xlabel('Position (0~700) -> Higher is Closed')
        ax3.set_ylabel('Current (mA) -> Force')
        ax3.grid(True)
        
        # 주석 (단단함 판별 가이드)
        ax3.text(0.05, 0.95, "Steep slope = Hard Object\nFlat slope = Soft Object", 
                 transform=ax3.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


# ============== 도움말 ==============

def print_help():
    print('\n' + '=' * 70)
    print('  [RH-P12-RN-A 그리퍼 키보드 제어 — 통합판]')
    print('=' * 70)
    print('  ── 위치 ──')
    print('    o : 완전 열기 (0)        c : 완전 닫기 (700)     h : 절반 (350)')
    print(f'    + : pos +{POS_STEP:<3d}             - : pos -{POS_STEP:<3d}            (큰 변화)')
    print(f'    > : pos +{POS_STEP_MID:<3d}             < : pos -{POS_STEP_MID:<3d}            (중간)')
    print(f'    . : pos +{POS_STEP_FINE:<3d}             , : pos -{POS_STEP_FINE:<3d}            (미세)')
    print('    s : 상태 출력')
    print('  ── 압력 (Goal Current) ──')
    for k in '123456789':
        v, label = FORCE_PRESETS[k]
        print(f'    {k} : {label:<10s} ({v})')
    print(f'    ] : 압력 +{FORCE_STEP}              [ : 압력 -{FORCE_STEP}             '
          'f : 현재 압력 출력')
    print('    t : 토크 ON/OFF 토글  (OFF 시 그리퍼가 완전 풀려 free 상태)')
    print('  ── 자동 ──')
    print('    a : Auto-grip — 닫으면서 강성 측정 → 적정 force 자동 적용')
    print('    r : Present Position 읽기 (디버그, 여러 주소 후보 시도)')
    print('    e : FC06 echo test (시리얼 패스스루 read 자체 검증)')
    print('    p : [PLOT] 물성 측정! (3초간 쥐면서 X-Y 그래프 시각화)')
    print('  ── 종료 ──')
    print('    q : 종료 (Ctrl-C 도 가능)')
    print('=' * 70)
    print('  Tip: 압력 명령은 다음 위치 명령(o/c/h/+/-)부터 적용됩니다.')
    print('=' * 70 + '\n')


def status_str(node: 'GripperKeyController') -> str:
    torque = 'ON ' if node.torque_on else 'OFF'
    return f'pos={node.target:>3d}  force={node.force:>4d}  torque={torque}'


# ============== CLI ==============

def parse_args():
    p = argparse.ArgumentParser(description='RH-P12-RN-A 그리퍼 키보드 제어 (통합판)')
    p.add_argument('--no-launch', action='store_true',
                   help='dsr_bringup2_moveit 자동 실행 skip (이미 떠있는 경우)')
    p.add_argument('--no-servo', action='store_true',
                   help='AUTONOMOUS + Servo ON skip (그리퍼만 테스트)')
    return p.parse_args()


# ============== 메인 ==============

def main():
    args = parse_args()

    if not args.no_launch:
        reset_robot_driver()
    else:
        print('[--no-launch] 드라이버 자동 실행 skip — 이미 떠있다고 가정')

    rclpy.init()
    node = GripperKeyController()
    try:
        if not args.no_servo:
            print('\n[1/3] AUTONOMOUS + Servo ON')
            node.activate_robot()
            print('   ✓ 로봇 활성화 시도 완료')
        else:
            print('\n[1/3] [--no-servo] Servo ON skip')

        print('\n[2/3] 그리퍼 시리얼 open + 토크 ON + 초기 압력')
        node.serial_open_and_init()
        print(f'   ✓ 그리퍼 준비 완료 ({status_str(node)})')

        print('\n[3/3] 키 입력 루프 시작')
        print_help()
        print(f'\r[CMD]  {status_str(node)}            ', end='', flush=True)

        while True:
            k = get_key()
            if k == 'o':
                node.set_position(POS_MIN)
                tag = '[OPEN] '
            elif k == 'c':
                node.set_position(POS_MAX)
                tag = '[CLOSE]'
            elif k == 'h':
                node.set_position((POS_MIN + POS_MAX) // 2)
                tag = '[HALF] '
            elif k == '+':
                node.set_position(node.target + POS_STEP)
                tag = f'[+{POS_STEP}]  '
            elif k == '-':
                node.set_position(node.target - POS_STEP)
                tag = f'[-{POS_STEP}]  '
            elif k == '>':
                node.set_position(node.target + POS_STEP_MID)
                tag = f'[+{POS_STEP_MID}]  '
            elif k == '<':
                node.set_position(node.target - POS_STEP_MID)
                tag = f'[-{POS_STEP_MID}]  '
            elif k == '.':
                node.set_position(node.target + POS_STEP_FINE)
                tag = f'[+{POS_STEP_FINE}]   '
            elif k == ',':
                node.set_position(node.target - POS_STEP_FINE)
                tag = f'[-{POS_STEP_FINE}]   '
            elif k == 's':
                # 전체 상태 한 줄에 자세히 출력
                print()
                print(f'  [STAT] pos(목표)={node.target}  '
                      f'force(Goal Current)={node.force}  '
                      f'torque={"ON" if node.torque_on else "OFF"}')
                tag = '[STAT] '
            elif k in FORCE_PRESETS:
                v, label = FORCE_PRESETS[k]
                node.set_force(v)
                print()
                print(f'  [F{k}:{label}] {force_to_human(v)}')
                tag = f'[F{k}:{label}]'
            elif k == ']':
                node.set_force(node.force + FORCE_STEP)
                print()
                print(f'  [F+{FORCE_STEP}] {force_to_human(node.force)}')
                tag = f'[F+{FORCE_STEP}]'
            elif k == '[':
                node.set_force(node.force - FORCE_STEP)
                print()
                print(f'  [F-{FORCE_STEP}] {force_to_human(node.force)}')
                tag = f'[F-{FORCE_STEP}]'
            elif k == 'f':
                # 현재 Goal Current 출력 + 사람이 이해하기 쉬운 환산
                print()
                print(f'  [FORCE] {force_to_human(node.force)}')
                print(f'          torque={"ON" if node.torque_on else "OFF"}  '
                      f'pos(목표)={node.target}')
                print('          ⚠ Goal Current 는 "최대 한계" 입니다 — 실제 잡는 힘은')
                print('             가벼운 물체엔 작고, 한계에 도달할 때까지 증가.')
                print('             실시간 측정은 Present Current read 필요 (펌웨어 미지원 시 불가).')
                tag = '[FORCE]'
            elif k == 't':
                node.toggle_torque()
                tag = '[TORQ]'
            elif k == 'a':
                # auto-grip 은 진행 메시지를 줄바꿈으로 출력하므로 화면 클리어 후 호출
                print()
                node.auto_grip()
                tag = '[AUTO]'
            elif k == 'r':
                print()
                p = node.read_present_position(verbose=True)
                if p is None:
                    print("  [READ] 실패 — 위 hex 덤프 확인. 'e' 키로 echo test 도 시도해보세요.")
                else:
                    print(f'  [READ] Present Position = {p}')
                tag = '[READ]'
            elif k == 'e':
                print()
                node.echo_test()
                tag = '[ECHO]'
            elif k == 'p':
                print()
                node.plot_material_property()
                tag = '[PLOT]'
            elif k in ('q', '\x03'):
                break
            else:
                tag = f'[?{repr(k)}]'

            print(f'\r{tag:<14s} {status_str(node)}            ',
                  end='', flush=True)
    except Exception as e:
        print(f'\n에러: {e}')
    finally:
        try:
            node.serial_close()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print('\n[종료]')


if __name__ == '__main__':
    main()
