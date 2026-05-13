"""
[로보티스 RH-P12-RN-A 그리퍼 연결 점검]

두산 e0509 의 툴 플랜지 RS-485 에 직결된 RH-P12-RN-A 그리퍼는,
두산 ROS 2 드라이버가 자체적으로 시리얼 패스스루 서비스를 노출합니다:
  - /dsr01/gripper/flange_serial_open
  - /dsr01/gripper/flange_serial_write
  - /dsr01/gripper/flange_serial_close

별도 launch 가 필요 없으며, 본 스크립트는 다음을 검증합니다:
  1) 그리퍼 서비스 3개의 가용성
  2) 시리얼 포트 open → 토크 활성화 → close 핑
서비스가 모두 살아있고 핑이 성공하면 11_그리퍼_키보드_제어.py 로 넘어갈 준비 완료.
"""

import time
import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import FlangeSerialOpen, FlangeSerialClose, FlangeSerialWrite

from doosan_config import NAMESPACE as NS


PORT = 1            # 두산 툴 플랜지 시리얼 채널
BAUD = 57600        # RH-P12-RN-A 기본 보드레이트
SLAVE_ID = 1        # 그리퍼 Modbus RTU 슬레이브 ID


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


def fc06_torque_enable() -> list:
    """Modbus FC06: 레지스터 256(0x0100)에 1 쓰기 → 토크 ON."""
    return _make_frame(bytes([SLAVE_ID, 0x06, 0x01, 0x00, 0x00, 0x01]))


class GripperPing(Node):
    def __init__(self):
        super().__init__('gripper_ping')
        prefix = f'/{NS}/gripper'
        self.cli_open = self.create_client(FlangeSerialOpen, f'{prefix}/flange_serial_open')
        self.cli_close = self.create_client(FlangeSerialClose, f'{prefix}/flange_serial_close')
        self.cli_write = self.create_client(FlangeSerialWrite, f'{prefix}/flange_serial_write')

    def _call(self, cli, req, timeout=5.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def check_services(self) -> bool:
        services = [
            (self.cli_open, f'/{NS}/gripper/flange_serial_open'),
            (self.cli_close, f'/{NS}/gripper/flange_serial_close'),
            (self.cli_write, f'/{NS}/gripper/flange_serial_write'),
        ]
        all_ok = True
        for cli, name in services:
            ok = cli.wait_for_service(timeout_sec=3.0)
            print(f'  [{"OK" if ok else "X "}] {name}')
            all_ok = all_ok and ok
        return all_ok

    def ping(self) -> bool:
        # 1) 시리얼 포트 열기
        req_open = FlangeSerialOpen.Request()
        req_open.port = PORT
        req_open.baudrate = BAUD
        req_open.bytesize = 8
        req_open.parity = 0
        req_open.stopbits = 1
        r = self._call(self.cli_open, req_open)
        if not (r and r.success):
            print('  [X ] FlangeSerialOpen 실패 — 그리퍼 케이블/전원 확인')
            return False
        print('  [OK] 시리얼 포트 open')

        try:
            time.sleep(0.1)
            # 2) 토크 활성화 (실제 그리퍼와 통신 성립 여부 확인)
            req_w = FlangeSerialWrite.Request()
            req_w.port = PORT
            req_w.data = fc06_torque_enable()
            r = self._call(self.cli_write, req_w)
            if r and r.success:
                print('  [OK] 토크 활성화 명령 전송 성공')
            else:
                print('  [X ] 토크 활성화 명령 실패 — 그리퍼 응답 없음')
                return False
            time.sleep(0.2)
            return True
        finally:
            req_c = FlangeSerialClose.Request()
            req_c.port = PORT
            self._call(self.cli_close, req_c)
            print('  [OK] 시리얼 포트 close')


def main():
    print('\n' + '=' * 70)
    print('  [Robotis RH-P12-RN-A 그리퍼 연결 점검]')
    print('  두산 툴 플랜지 RS-485 패스스루를 통해 그리퍼와의 통신을 검증합니다.')
    print('=' * 70)

    rclpy.init()
    node = GripperPing()

    try:
        print('\n[단계 1] 두산 그리퍼 서비스 가용성 확인')
        if not node.check_services():
            print('\n!! 일부 서비스가 응답하지 않습니다.')
            print('   - 00_두산로봇_리눅스_실물_연결.py 를 실행해 dsr_bringup2 가 살아있는지 확인하세요.')
            print('   - 00_로봇_활성화하기.py 로 Servo ON 상태인지 확인하세요.')
            return

        print('\n[단계 2] 시리얼 핑 (open → 토크 ON → close)')
        if not node.ping():
            print('\n!! 그리퍼와 통신 실패. 전원/케이블/슬레이브 ID 를 점검하세요.')
            return

        print('\n' + '=' * 70)
        print('  [그리퍼 통신 정상]')
        print('  이제 11_그리퍼_키보드_제어.py 로 키보드 제어를 시작할 수 있습니다.')
        print('=' * 70 + '\n')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
