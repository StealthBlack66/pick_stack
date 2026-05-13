"""
[작업 포즈 순회 캘리브레이션 — 거리 기반 (Cartesian 200mm)]

기존 06_로봇_관절_제어_보정형.py 는:
  - MoveIt FollowJointTrajectory 액션을 사용 (namespace `/dsr_moveit_controller/...`
    에서 호출 — 실제 액션은 `/dsr01/dsr_moveit_controller/...` 에 있어 동작 안 함)
  - 관절(degree)만 받음 → TCP 직선 거리 정확도 검증 불가

이 스크립트는 두산 드라이버 직속 service `/dsr01/motion/move_line` 을 직접 호출.
- 8개 way-point 순회 (X/Y/Z 각 ±200mm + 3D 대각)
- 모든 이동을 현재 TCP 기준 *상대* 변위 (DR_MV_MOD_REL=1) 로 지정
  → 시작 자세 좌표 알 필요 없음, 종료 시 시작 자세로 정확히 복귀

[실행 전 필수]
  1. 00_두산로봇_리눅스_실물_연결.py 로 로봇 연결
  2. 00_로봇_활성화하기.py 로 Servo ON
  3. 작업 공간에 현재 TCP 기준 모든 축 ±200mm 여유 확보
  4. 비상정지 버튼 손에 닿는 위치
"""

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveLine
import time
import sys

WAYPOINTS = [
    # (dx, dy, dz, 설명)
    (+200.0,    0.0,    0.0, "X+ 전진"),
    (-200.0,    0.0,    0.0, "X- 복귀"),
    (   0.0, +200.0,    0.0, "Y+ 좌측"),
    (   0.0, -200.0,    0.0, "Y- 복귀"),
    (   0.0,    0.0, +200.0, "Z+ 상승"),
    (   0.0,    0.0, -200.0, "Z- 하강"),
    (+100.0, +100.0, +100.0, "3D 대각 전진"),
    (-100.0, -100.0, -100.0, "3D 대각 복귀"),
]

VEL = [50.0, 30.0]   # [mm/sec, deg/sec]
ACC = [100.0, 60.0]  # [mm/sec^2, deg/sec^2]
SETTLE_SEC = 0.5     # 이동 사이 진동 가라앉히는 시간

DR_BASE = 0
DR_MV_MOD_REL = 1


class WorkposeCalibrator(Node):
    def __init__(self):
        super().__init__('workpose_calibrator')
        self.cli = self.create_client(MoveLine, '/dsr01/motion/move_line')
        if not self.cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                "/dsr01/motion/move_line 서비스를 찾을 수 없습니다.\n"
                "로봇 연결 launch 가 떠있고, namespace 가 /dsr01 인지 확인하세요."
            )

    def move_relative(self, dx, dy, dz):
        req = MoveLine.Request()
        req.pos = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0]
        req.vel = list(VEL)
        req.acc = list(ACC)
        req.ref = DR_BASE
        req.mode = DR_MV_MOD_REL
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        return res is not None and res.success


def main():
    rclpy.init()
    try:
        node = WorkposeCalibrator()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        rclpy.shutdown()
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" [작업 포즈 순회 캘리브레이션]")
    print(f" 현재 자세에서 {len(WAYPOINTS)} 개 way-point 를 순회합니다.")
    print(f" Cartesian {abs(WAYPOINTS[0][0]):.0f} mm 단위, 속도 {VEL[0]:.0f} mm/sec.")
    print(" 비상정지 버튼 위치를 손에 닿는 곳에 두세요.")
    print("=" * 70)

    try:
        input(">> 준비되면 Enter 를 눌러 시작합니다 (취소: Ctrl+C)... ")
    except KeyboardInterrupt:
        print("\n사용자 취소.")
        node.destroy_node()
        rclpy.shutdown()
        return

    completed = 0
    for i, (dx, dy, dz, label) in enumerate(WAYPOINTS, 1):
        print(f"\n[{i}/{len(WAYPOINTS)}] {label} : dx={dx:+.0f}, dy={dy:+.0f}, dz={dz:+.0f}")
        ok = node.move_relative(dx, dy, dz)
        if not ok:
            print("  !! 이동 실패 — 시퀀스 중단.")
            print("     - Servo ON 인지 (00_로봇_활성화하기.py 먼저 실행)")
            print("     - 작업 공간 한계 / 충돌 회피 영역 인지")
            print("     - ros2 service list | grep /dsr01/motion 으로 service 확인")
            break
        print("  >> 성공.")
        completed += 1
        time.sleep(SETTLE_SEC)

    print("\n" + "-" * 70)
    if completed == len(WAYPOINTS):
        print(f"[순회 완료] 8 개 way-point 통과. 시작 자세로 복귀했습니다.")
    else:
        print(f"[부분 완료] {completed}/{len(WAYPOINTS)} way-point 만 성공.")
    print("-" * 70 + "\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
