#!/usr/bin/env python3
"""
[자동 Hand-Eye Calibration — 거리(Cartesian) 기반 순회 업그레이드]

기존 sim2real/deprecated/auto_hand_eye_calibration.py 의 업그레이드 버전.

기존 한계:
  - 출력 경로가 다른 사용자(/home/fhekwn549) 로 박혀있어 시작 직후 크래시 → 로봇이 안 움직임
  - 모션이 j1/j5/j6 의 작은 회전 (±5~20°) 만 사용 → TCP 거리 변화 부재 → Hand-Eye 수학 ill-conditioned

이 버전:
  - OUTPUT_DIR 을 본 작업 디렉토리로 수정
  - MoveLine service 를 사용해 base TCP 자세 기준 *절대 Cartesian 좌표*로 이동
  - 200mm Cartesian translation + ±20° rotation 을 조합한 20개 포즈 생성
    (Hand-Eye Calibration 에 필요한 회전+병진 다양성 충족)
  - 매 포즈에서 base TCP 로 안전 복귀 (이전 포즈 누적 drift 없음)

사용법:
  1. 로봇 연결 launch 가 떠있어야 함 (00_두산로봇_리눅스_실물_연결.py)
  2. Servo ON (00_로봇_활성화하기.py)
  3. 그리퍼로 ArUco 마커 부착 (검정 테두리 포함 한 변 = ARUCO_MARKER_SIZE)
  4. 마커가 카메라에 잘 보이는 안전 위치로 로봇 이동 (RViz/teach pendant)
  5. python 08_카메라_핸드아이_캘리브레이션.py
     → 시작 시 ArUco 사전 자동 탐색 (마커 보이면 즉시 적합한 사전 적용)
  6. 's' 로 현재 자세를 base 로 저장
  7. Enter 로 자동 순회 시작
  8. 끝나면 'c' 로 calibrate

⚠️ Base TCP 위치 기준으로 ±200mm 이동 + ±20° 회전이 안전한 작업 공간 확보 필수
"""

import numpy as np
import cv2
import time
import os
import sys
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation as R

# RealSense
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    print("Warning: pyrealsense2 not installed")
    HAS_REALSENSE = False

# ROS2
try:
    import rclpy
    from rclpy.action import ActionClient
    from dsr_msgs2.srv import GetCurrentPosx, GetCurrentRotm
    from moveit_msgs.srv import GetCartesianPath
    from moveit_msgs.action import ExecuteTrajectory
    from geometry_msgs.msg import Pose, Point, Quaternion
    HAS_ROS2 = True
except ImportError:
    print("Warning: ROS2 not available")
    HAS_ROS2 = False


# ==================== 설정 ====================
# ===== ArUco 마커 설정 =====
# 사용자가 가진 마커는 ArUco fiducial. 단일 마커로도 hand-eye calibration 가능.
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_50   # 사용자 마커 확인됨 (debug 이미지로 검증). 'd' 키로 재탐색 가능.
ARUCO_MARKER_SIZE = 0.050               # 마커 검정 테두리 한 변 길이 (m). 측정 후 조정!
ARUCO_TARGET_ID = 0                     # 우선 검출 대상 ID (None=any, 0=옛 마커, 1=새 마커)
                                         # 화면에 여러 마커 잡혀도 이 ID만 사용 → 안전

# ===== 그리퍼 부착 위치 =====
# 마커가 그리퍼의 어느 면에 있는지 — 사용자 셋업 기록용.
# 수학에는 영향 없음 (캘리브레이션은 TCP 기준 마커 위치 자체를 계산하므로 이 정보 불필요).
# 다만 motion 설계 시 visibility 가정 명확화에 도움.
#   "+Z" = Tool +Z 면 (그리퍼 앞면, 보통 grasping direction 쪽)
#   "-Z" = Tool -Z 면 (그리퍼 뒷면, flange 쪽)  ← 사용자 현재 셋업
MARKER_FACE = "-Z"                      # 사용자: 그리퍼 뒷면에 부착

# 사전 자동 탐색용 후보 — 'd' 키 디버그 시 모두 시도
CANDIDATE_ARUCO_DICTS = [
    ("DICT_4X4_50",  cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_4X4_1000",cv2.aruco.DICT_4X4_1000),
    ("DICT_5X5_50",  cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_5X5_1000",cv2.aruco.DICT_5X5_1000),
    ("DICT_6X6_50",  cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_6X6_1000",cv2.aruco.DICT_6X6_1000),
    ("DICT_7X7_50",  cv2.aruco.DICT_7X7_50),
    ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
    ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
    ("DICT_7X7_1000",cv2.aruco.DICT_7X7_1000),
    ("DICT_ARUCO_ORIGINAL",   cv2.aruco.DICT_ARUCO_ORIGINAL),
    ("DICT_APRILTAG_16h5",    cv2.aruco.DICT_APRILTAG_16h5),
    ("DICT_APRILTAG_25h9",    cv2.aruco.DICT_APRILTAG_25h9),
    ("DICT_APRILTAG_36h10",   cv2.aruco.DICT_APRILTAG_36h10),
    ("DICT_APRILTAG_36h11",   cv2.aruco.DICT_APRILTAG_36h11),
]

# 본 working dir 기준 데이터 저장 (이전 버전의 /home/fhekwn549 버그 수정)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "calibration_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "calibration_result.npz")
DATA_FILE = os.path.join(OUTPUT_DIR, "calibration_data.npz")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

# 모션 파라미터 — tool-frame 기준
# (참고: 마커 면의 normal axis = MARKER_FACE 방향, 즉 사용자 셋업에선 Tool -Z.
#  Z 축 자체는 +/- 양쪽이 같은 회전 축이므로, 회전 패턴은 마커가 +Z 든 -Z 든 동일하게 동작.)
TRANS_DELTA = 120.0   # Tool 축 방향 translation (mm)
ROT_TILT = 10.0       # Tool X/Y tilt — 마커 살짝 비스듬히 (visibility 유지 한계)
ROT_ROLL = 30.0       # Tool Z roll — 마커 면 in-plane 회전 (항상 visible, 큰 각도 OK)

# === 자동 정제 (Auto-Refine) ===
# 'c' 누른 후 오차가 목표 미달이면 자동으로 추가 포즈 수집 + 재계산 반복.
# "매우 우수" 등급 도달하면 자동 종료.
AUTO_REFINE_ENABLED      = True
AUTO_REFINE_TARGET_POS_MM  = 3.0    # "매우 우수" 위치 평균 오차 (mm)
AUTO_REFINE_TARGET_ROT_DEG = 0.5    # "매우 우수" 회전 평균 오차 (deg)
AUTO_REFINE_MAX_ROUNDS   = 5        # 최대 정제 라운드 (무한 루프 방지)
AUTO_REFINE_POSES_PER_ROUND = 8     # 라운드당 추가 포즈 수
VEL_LIN = 50.0        # 직선 속도 (mm/sec)
VEL_ANG = 30.0        # 각속도 (deg/sec)
ACC_LIN = 100.0       # 직선 가속도 (mm/sec^2)
ACC_ANG = 60.0        # 각가속도 (deg/sec^2)
SETTLE_TIME = 2.5     # 이동 후 진동 가라앉히는 시간 (초)

DR_BASE = 0

# Hand-Eye 캘리브레이션은 MoveIt 의 compute_cartesian_path + execute_trajectory 사용.
# (직접 MoveLine 서비스 호출은 dsr_controller2 의 update() 와 충돌해 박스 stuck)
MOVEIT_GROUP_NAME = 'arm'
MOVEIT_TIP_LINK = 'link_6'
MOVEIT_PLANNING_FRAME = 'world'
CARTESIAN_MAX_STEP = 0.005       # 5mm — 부드러운 직선 보간
CARTESIAN_JUMP_THRESHOLD = 0.0   # 0 = jump check 비활성

# compute_cartesian_path 가 생성하는 trajectory 의 time_from_start 를 scaling.
# MoveIt2 humble 의 GetCartesianPath 는 velocity/acceleration scaling 필드가 없어서,
# 결과 trajectory 가 joint_limits.yaml 의 max 속도로 너무 공격적일 수 있음.
# 5.0 = 5배 slow (속도 20%) — 캘리브 안전 기준
TRAJECTORY_TIME_SCALE = 5.0


def generate_calibration_offsets() -> List[Tuple[float, float, float, float, float, float]]:
    """
    *Tool 좌표계 기준* offsets [(dx,dy,dz,drx,dry,drz)] 생성.
    apply_tool_offset_to_posx() 로 base 자세와 합성됨.

    설계 (정확도-visibility 동시 최적):
    - 사용자 셋업: 마커가 그리퍼 뒷면 (Tool -Z 면) 부착, 카메라 정면
    - Tool Z 축 회전 (drz_tool): 마커 면이 카메라 향한 채 in-plane 회전
      → 마커 항상 visible → ±30° 큰 각도 가능 (앞면 부착이든 뒷면 부착이든 동일)
    - Tool X/Y tilt (drx, dry): 마커 면을 살짝 기울임 → ±10° 까지 visibility 안전
    - Translation: 자세 안 바뀌므로 항상 visible

    Hand-Eye 수학 well-conditioning:
    - Z roll (in-plane) 큰 각도 → 회전 정보 풍부
    - X/Y tilt 작은 각도 → axis diversity 확보
    - Translation 다양성 → 병진 정보
    → 16개 포즈로 충분, 정확도 ↑ visibility ↑
    """
    T = TRANS_DELTA
    TI = ROT_TILT
    RO = ROT_ROLL
    return [
        # === Pure translations 6개 (visibility 100%) ===
        ( +T,    0,    0,    0,    0,    0),  # tool +X
        ( -T,    0,    0,    0,    0,    0),  # tool -X
        (   0,  +T,    0,    0,    0,    0),  # tool +Y
        (   0,  -T,    0,    0,    0,    0),  # tool -Y
        (   0,   0,  +T,    0,    0,    0),  # tool +Z (가까이)
        (   0,   0,  -T,    0,    0,    0),  # tool -Z (멀어짐)

        # === Tool +Z roll 4개 (마커 face 그대로 — visibility 100%, 큰 각도) ===
        (   0,   0,    0,    0,    0,  +RO),       # +30° roll (marker rotates in-plane)
        (   0,   0,    0,    0,    0,  -RO),       # -30° roll
        ( +T*0.5,   0, 0,    0,    0,  +RO*0.7),   # translation + 21° roll
        (-T*0.5,    0, 0,    0,    0,  -RO*0.7),   # 반대 방향

        # === Tool X tilt 2개 (위/아래 살짝 기울임 — ±10°, 마커 visible) ===
        (   0,  +T*0.4, 0,  +TI,    0,    0),
        (   0,  -T*0.4, 0,  -TI,    0,    0),

        # === Tool Y tilt 2개 (좌/우 살짝 기울임) ===
        (+T*0.4,   0,   0,    0,  +TI,    0),
        (-T*0.4,   0,   0,    0,  -TI,    0),

        # === Combined: translation + 모든 축 작은 회전 (information 최대) ===
        (+T*0.5, +T*0.3, +T*0.3,  +TI*0.7, +TI*0.7,  +RO*0.5),
        (-T*0.5, -T*0.3, +T*0.3,  -TI*0.7, +TI*0.7,  -RO*0.5),
    ]  # 총 16 — 정확도/visibility 동시 최적


class RealSenseCamera:
    """RealSense D455F 카메라 인터페이스 (변경 없음)"""

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intrinsics.coeffs)

        print(f"Camera Matrix:\n{self.camera_matrix}")
        print(f"Distortion Coefficients: {self.dist_coeffs}")

    def get_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def stop(self):
        self.pipeline.stop()


def posx_to_pose(posx_6: List[float]) -> Pose:
    """
    두산 posx [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg] → ROS Pose (m, quaternion)
    회전 표현은 **Euler ZYZ** (두산 공식 컨벤션, DRFS.h / RobotStateRt.msg 명시):
      (a, b, c) = first Z 회전 → Y 회전 → last Z 회전
    """
    p = Pose()
    p.position = Point(x=posx_6[0]/1000.0, y=posx_6[1]/1000.0, z=posx_6[2]/1000.0)
    a, b, c = np.deg2rad(posx_6[3]), np.deg2rad(posx_6[4]), np.deg2rad(posx_6[5])
    quat = R.from_euler('ZYZ', [a, b, c]).as_quat()  # (x, y, z, w)
    p.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    return p


class RobotController:
    """
    MoveIt 정석 path: compute_cartesian_path 로 trajectory 생성 →
    execute_trajectory action 으로 joint_trajectory_controller 통해 실행.

    MoveLine service 직접 호출은 dsr_controller2 의 update() 와 servoJ race
    condition 이 있어 박스 stuck 유발 — 사용 안 함.
    """

    def __init__(self, namespace='dsr01'):
        self.namespace = namespace
        self.node = rclpy.create_node('auto_hand_eye_calibration_v2')

        # 위치 조회 (Euler 포함) — move 명령 전송 시 target posx 만들 때 필요
        self.cli_get_posx = self.node.create_client(
            GetCurrentPosx, f'/{namespace}/aux_control/get_current_posx')
        # 회전 행렬 직접 조회 — Euler 변환 우회로 R_gripper_to_base 정확도 ↑
        self.cli_get_rotm = self.node.create_client(
            GetCurrentRotm, f'/{namespace}/aux_control/get_current_rotm')

        # MoveIt: 직선 trajectory 생성 + 실행
        self.cli_cartesian = self.node.create_client(
            GetCartesianPath, f'/{namespace}/compute_cartesian_path')
        self.act_execute = ActionClient(
            self.node, ExecuteTrajectory, f'/{namespace}/execute_trajectory')

        print("서비스/액션 연결 대기 중...")
        if not self.cli_get_posx.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("get_current_posx 서비스 연결 실패")
        if not self.cli_get_rotm.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("get_current_rotm 서비스 연결 실패")
        if not self.cli_cartesian.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("compute_cartesian_path 서비스 연결 실패 (MoveIt 떠있어야 함)")
        if not self.act_execute.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("execute_trajectory 액션 서버 연결 실패")
        print("서비스/액션 연결 완료 (4 endpoints)")

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        TCP pose (t, R). t 는 미터, R 은 (3,3) rotation matrix.
        Translation: GetCurrentPosx 의 [x,y,z] (mm → m)
        Rotation:    GetCurrentRotm 의 3x3 행렬 — Euler 변환 우회로 정확
        """
        # Translation: posx
        req_p = GetCurrentPosx.Request(); req_p.ref = DR_BASE
        f_p = self.cli_get_posx.call_async(req_p)
        rclpy.spin_until_future_complete(self.node, f_p, timeout_sec=5.0)
        rp = f_p.result()
        if not (rp and rp.success and len(rp.task_pos_info) > 0):
            raise RuntimeError("TCP posx 획득 실패")
        pd = rp.task_pos_info[0].data
        position = np.array([pd[0]/1000.0, pd[1]/1000.0, pd[2]/1000.0])

        # Rotation: rotm (직접 회전 행렬 — Euler convention 무관)
        req_r = GetCurrentRotm.Request(); req_r.ref = DR_BASE
        f_r = self.cli_get_rotm.call_async(req_r)
        rclpy.spin_until_future_complete(self.node, f_r, timeout_sec=5.0)
        rr = f_r.result()
        if not (rr and rr.success and len(rr.rot_matrix) == 3):
            raise RuntimeError(f"TCP rotm 획득 실패 (rows={len(rr.rot_matrix) if rr else 'None'})")
        # rot_matrix 는 3개의 Float64MultiArray, 각 row 3 원소 (row-major)
        rotation = np.array([list(row.data) for row in rr.rot_matrix], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise RuntimeError(f"예상치 못한 rotm shape: {rotation.shape}")
        return position, rotation

    def get_tcp_posx_raw(self) -> List[float]:
        """TCP raw 6값 [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]"""
        req = GetCurrentPosx.Request()
        req.ref = DR_BASE
        future = self.cli_get_posx.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        result = future.result()
        if not (result and result.success and len(result.task_pos_info) > 0):
            raise RuntimeError("TCP raw posx 획득 실패")
        return list(result.task_pos_info[0].data[:6])

    def _plan_cartesian(self, target_posx: List[float]):
        """
        현재 TCP → target_posx 까지 직선 cartesian 경로 생성.
        Returns: (RobotTrajectory, fraction) 또는 None
        """
        target_pose = posx_to_pose(target_posx)
        req = GetCartesianPath.Request()
        req.header.frame_id = MOVEIT_PLANNING_FRAME
        req.group_name = MOVEIT_GROUP_NAME
        req.link_name = MOVEIT_TIP_LINK
        req.waypoints = [target_pose]   # current 는 자동으로 start 가 됨
        req.max_step = CARTESIAN_MAX_STEP
        req.jump_threshold = CARTESIAN_JUMP_THRESHOLD
        req.avoid_collisions = False     # 캘리브에서는 검사 끔 (그리퍼+마커 등 collision 미정의)

        future = self.cli_cartesian.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            return None, 0.0
        return result.solution, float(result.fraction)

    @staticmethod
    def _scale_trajectory_time(trajectory, scale: float, min_total_sec: float = 1.0):
        """
        compute_cartesian_path 결과 trajectory 의 time_from_start 를 scale 배 늘림.
        모든 timestamp 이 0 인 비정상 trajectory 의 경우엔 등간격 분배.
        scale=5.0 → 5배 천천히. velocity/acceleration 도 비례 reduce. in-place 수정.
        """
        jt = trajectory.joint_trajectory
        if not jt.points:
            return

        # 1) 마지막 point 의 time_from_start 가 0 이면 → 등간격 시간 부여
        last_pt = jt.points[-1]
        last_ns = last_pt.time_from_start.sec * 1_000_000_000 + last_pt.time_from_start.nanosec
        if last_ns == 0:
            n = len(jt.points)
            total_ns = int(min_total_sec * scale * 1_000_000_000)
            for i, pt in enumerate(jt.points):
                t_ns = int(total_ns * (i + 1) / n) if n > 1 else 0
                pt.time_from_start.sec = t_ns // 1_000_000_000
                pt.time_from_start.nanosec = t_ns % 1_000_000_000
                # velocity/acceleration 0 으로 (controller 가 trajectory 보간으로 처리)
                pt.velocities = []
                pt.accelerations = []
            return

        # 2) 정상 trajectory 면 그냥 scale 배
        if scale <= 1.0:
            return
        for pt in jt.points:
            total_ns = pt.time_from_start.sec * 1_000_000_000 + pt.time_from_start.nanosec
            new_ns = int(total_ns * scale)
            pt.time_from_start.sec = new_ns // 1_000_000_000
            pt.time_from_start.nanosec = new_ns % 1_000_000_000
            if pt.velocities:
                pt.velocities = [v / scale for v in pt.velocities]
            if pt.accelerations:
                pt.accelerations = [a / (scale * scale) for a in pt.accelerations]

    def _execute_trajectory(self, trajectory) -> bool:
        """RobotTrajectory 를 execute_trajectory action 으로 전송 (시간 scaling 후)"""
        self._scale_trajectory_time(trajectory, TRAJECTORY_TIME_SCALE)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self.act_execute.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=120.0)
        result = result_future.result()
        if result is None:
            return False
        # MoveItErrorCodes.SUCCESS == 1
        return getattr(result.result.error_code, 'val', 0) == 1

    def move_line_absolute(self, target_posx: List[float]) -> bool:
        """
        절대 Cartesian 좌표 [x,y,z,rx,ry,rz] mm/deg 로 직선 이동.
        compute_cartesian_path + execute_trajectory 사용 (정석).
        """
        trajectory, fraction = self._plan_cartesian(target_posx)
        if trajectory is None or fraction < 0.99:
            print(f"  [경고] cartesian 경로 생성 실패 (fraction={fraction:.2f})")
            return False
        n_points = len(trajectory.joint_trajectory.points)
        if n_points < 2:
            print(f"  [경고] trajectory point 너무 적음 ({n_points})")
            return False
        # 시간 scale 적용 후 실행
        self._scale_trajectory_time(trajectory, TRAJECTORY_TIME_SCALE)
        last = trajectory.joint_trajectory.points[-1].time_from_start
        total_sec = last.sec + last.nanosec / 1e9
        print(f"  trajectory: {n_points} points, 총 {total_sec:.2f}초 (scale={TRAJECTORY_TIME_SCALE}x slower)")

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self.act_execute.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=120.0)
        result = result_future.result()
        if result is None:
            return False
        return getattr(result.result.error_code, 'val', 0) == 1

    def shutdown(self):
        self.node.destroy_node()


def make_aruco_detector(dict_id: int):
    """
    OpenCV 4.7+ 의 새 ArucoDetector API. main 에서 1회 생성 후 재사용 (성능).

    CORNER_REFINE_SUBPIX: cv2.cornerSubPix 기반 sub-pixel 코너 정밀화.
    깨끗한 인쇄 마커에 가장 정확. 코너 ±0.05px 정확도까지 수렴.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    detector_params = cv2.aruco.DetectorParameters()

    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 30
    detector_params.cornerRefinementMinAccuracy = 0.05

    return cv2.aruco.ArucoDetector(aruco_dict, detector_params)


def detect_aruco(detector, image: np.ndarray
                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    ArUco 마커 검출. ARUCO_TARGET_ID 와 일치하는 마커 우선 반환.
    일치 없으면 첫 번째 검출 마커 fallback.
    Returns: (corners (4,2) 시계방향, marker_id) 또는 (None, None)
    """
    corners_list, ids, _ = detector.detectMarkers(image)
    if ids is None or len(ids) == 0:
        return None, None
    # ARUCO_TARGET_ID 와 일치하는 마커 우선
    if ARUCO_TARGET_ID is not None:
        for i in range(len(ids)):
            if int(ids[i][0]) == ARUCO_TARGET_ID:
                return corners_list[i].reshape(4, 2), ARUCO_TARGET_ID
        # 일치 없으면 None (잘못된 마커 사용 방지)
        return None, None
    # ARUCO_TARGET_ID = None 이면 첫 번째 마커 fallback
    return corners_list[0].reshape(4, 2), int(ids[0][0])


def detect_aruco_all(detector, image: np.ndarray
                     ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """검출된 모든 마커 반환 (drawDetectedMarkers 시각화용)."""
    corners_list, ids, _ = detector.detectMarkers(image)
    return corners_list, ids


def diagnose_aruco_dict(image: np.ndarray) -> List[Tuple[str, int, list]]:
    """
    어떤 ArUco 사전이 마커를 검출하는지 모두 시도. 시작 자동 탐색 + 'd' 키에서 사용.
    Returns: [(사전명, 사전ID, [검출된 marker IDs]), ...]
    """
    found = []
    for name, dict_id in CANDIDATE_ARUCO_DICTS:
        try:
            d = make_aruco_detector(dict_id)
            _, ids, _ = d.detectMarkers(image)
            if ids is not None and len(ids) > 0:
                found.append((name, dict_id, ids.flatten().tolist()))
        except cv2.error:
            pass
    return found


def estimate_aruco_pose(corners: np.ndarray, marker_size: float,
                        camera_matrix: np.ndarray, dist_coeffs: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    단일 ArUco 마커의 카메라 frame 기준 pose 추정.

    SOLVEPNP_IPPE_SQUARE (정사각형 마커 전용 PnP):
    - 마커가 깨끗한 정사각형일 때 가장 정확
    - 두 가지 가능 해 반환 → reprojection error 최저 선택 (Z flip 회피)

    Returns: (tvec, rotation_matrix, rvec) — m / (3,3) / (3,) Rodrigues
    """
    half = marker_size / 2.0
    obj_points = np.array([
        [-half,  half, 0],   # 0: top-left
        [ half,  half, 0],   # 1: top-right
        [ half, -half, 0],   # 2: bottom-right
        [-half, -half, 0],   # 3: bottom-left
    ], dtype=np.float32)
    img_points = corners.reshape(4, 1, 2).astype(np.float32)

    # IPPE_SQUARE: ArUco 정사각형 마커에 최적화 — 항상 2 solutions 반환.
    n_sol, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if n_sol == 0:
        # Fallback: ITERATIVE (정확도 약간 ↓ 하지만 항상 동작)
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points,
                                       camera_matrix, dist_coeffs)
        if not ret:
            raise RuntimeError("ArUco PnP 풀이 실패")
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return tvec.flatten(), rotation_matrix, rvec.flatten()

    # 가장 낮은 reprojection error 해 선택 → ambiguity (Z flip) 회피
    errs = np.asarray(errors).flatten()
    best = int(np.argmin(errs))
    rvec, tvec = rvecs[best], tvecs[best]

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return tvec.flatten(), rotation_matrix, rvec.flatten()


def estimate_aruco_pose_averaged(detector, get_frame, marker_size: float,
                                  camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                                  n_frames: int = 5
                                  ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    여러 프레임에 걸쳐 마커 pose 를 평균 — sub-pixel detection noise 를 √N 만큼 감소.
    데이터 수집 시 정확도 향상용 (시각화 표시용은 single-frame estimate_aruco_pose 사용).

    Returns: (tvec_avg, R_avg, rvec_avg, marker_id) 또는 None (검출 실패)
    """
    tvecs, quats, mid = [], [], None
    for _ in range(n_frames):
        f = get_frame()
        corners_list, ids, _ = detector.detectMarkers(f)
        if ids is None or len(ids) == 0:
            continue
        # ARUCO_TARGET_ID 와 일치하는 마커만 사용
        sel_idx = None
        if ARUCO_TARGET_ID is not None:
            for i in range(len(ids)):
                if int(ids[i][0]) == ARUCO_TARGET_ID:
                    sel_idx = i
                    break
            if sel_idx is None:
                continue   # target ID 없으면 이 프레임 skip
        else:
            sel_idx = 0    # None 이면 첫 번째
        c = corners_list[sel_idx].reshape(4, 2)
        try:
            t, R_, _ = estimate_aruco_pose(c, marker_size, camera_matrix, dist_coeffs)
            tvecs.append(t)
            quats.append(R.from_matrix(R_).as_quat())
            if mid is None:
                mid = int(ids[sel_idx][0])
        except Exception:
            continue

    if len(tvecs) < max(2, n_frames // 2):
        return None  # 너무 많이 실패

    # translation 평균
    t_avg = np.mean(tvecs, axis=0)

    # quaternion 평균 (sign ambiguity 처리: 모두 첫 quaternion 과 dot > 0 이 되도록 정렬)
    q0 = quats[0]
    aligned = [q if np.dot(q, q0) > 0 else -q for q in quats]
    q_avg = np.mean(aligned, axis=0)
    q_avg /= np.linalg.norm(q_avg)
    R_avg = R.from_quat(q_avg).as_matrix()
    rvec_avg, _ = cv2.Rodrigues(R_avg)
    return t_avg, R_avg, rvec_avg.flatten(), mid


def generate_refinement_offsets(round_idx: int, n: int = 8
                                ) -> List[Tuple[float, float, float, float, float, float]]:
    """
    정제 라운드용 추가 offsets 생성. round 마다 다른 pseudo-random 패턴.
    안전 범위 내 랜덤 (visibility-safe):
      - translation: ±TRANS_DELTA × 다양한 scale
      - tilt: ±ROT_TILT (작은 X/Y 회전)
      - roll: ±ROT_ROLL (큰 Z 회전 — 항상 visible)
    """
    import random
    rng = random.Random(round_idx * 7919 + 13)  # deterministic per round
    T, TI, RO = TRANS_DELTA, ROT_TILT, ROT_ROLL
    offsets = []
    for i in range(n):
        # 다양한 magnitude scale 로 정보량 다양화
        scale = rng.choice([1.0, 0.7, 0.5, 0.3])
        dx = rng.uniform(-1, 1) * T * scale
        dy = rng.uniform(-1, 1) * T * scale
        dz = rng.uniform(-1, 1) * T * scale
        # tilt 는 항상 작게 (visibility safe)
        rx = rng.uniform(-1, 1) * TI * 0.8
        ry = rng.uniform(-1, 1) * TI * 0.8
        # roll 은 크게 (정보량 풍부 + visible)
        rz = rng.uniform(-1, 1) * RO * 0.8
        offsets.append((dx, dy, dz, rx, ry, rz))
    return offsets


def calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c, verbose: bool = True):
    """
    Hand-Eye 캘리브레이션 — 3 단계 (사격 비유):
    1) cv2.calibrateHandEye 5 method 중 best (= 사격 후 탄착군 형성)
    2) Nonlinear refinement (= 크리커 조정으로 원점 잡기)
    3) Iterative outlier rejection (= flyer 제외하고 재조정 → 정밀 사격)
       → 잔차 > 2σ 인 포즈 제외하고 다시 refinement
    """
    from scipy.optimize import least_squares

    # === 1) 초기값 — 5 method 중 best ===
    methods = [
        cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK,
        cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_ANDREFF,
        cv2.CALIB_HAND_EYE_DANIILIDIS,
    ]

    def residual_norm(R_c2b_, t_c2b_):
        T_c2b_ = np.eye(4); T_c2b_[:3,:3]=R_c2b_; T_c2b_[:3,3]=t_c2b_
        Ts = []
        for i in range(len(R_g2b)):
            T_g2b_ = np.eye(4); T_g2b_[:3,:3]=R_g2b[i]; T_g2b_[:3,3]=np.asarray(t_g2b[i]).flatten()
            T_t2c_ = np.eye(4); T_t2c_[:3,:3]=R_t2c[i]; T_t2c_[:3,3]=np.asarray(t_t2c[i]).flatten()
            Ts.append(np.linalg.inv(T_g2b_) @ T_c2b_ @ T_t2c_)
        pos = np.array([T[:3,3] for T in Ts])
        return np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean()

    best_R, best_t, best_r = None, None, np.inf
    for m in methods:
        try:
            Rc, tc = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
            r = residual_norm(Rc, tc.flatten())
            if r < best_r:
                best_r = r; best_R = Rc; best_t = tc.flatten()
        except Exception:
            continue

    if best_R is None:
        # fallback: TSAI 단독
        R_, t_ = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                      method=cv2.CALIB_HAND_EYE_TSAI)
        return R_, t_.flatten()

    # === 2) Nonlinear refinement ===
    # 자유 변수: X (R, t) — rotation 은 axis-angle (3) + translation (3) = 6 params
    from scipy.spatial.transform import Rotation as R_lib

    def x_to_T(x):
        rvec = x[:3]; tvec = x[3:6]
        T = np.eye(4)
        T[:3,:3] = R_lib.from_rotvec(rvec).as_matrix()
        T[:3,3] = tvec
        return T

    def residuals(x):
        T_c2b_ = x_to_T(x)
        Ts = []
        for i in range(len(R_g2b)):
            T_g2b_ = np.eye(4); T_g2b_[:3,:3]=R_g2b[i]; T_g2b_[:3,3]=np.asarray(t_g2b[i]).flatten()
            T_t2c_ = np.eye(4); T_t2c_[:3,:3]=R_t2c[i]; T_t2c_[:3,3]=np.asarray(t_t2c[i]).flatten()
            Ts.append(np.linalg.inv(T_g2b_) @ T_c2b_ @ T_t2c_)
        pos = np.array([T[:3,3] for T in Ts])
        # 각 pose 의 T_t2g translation 이 평균에서 얼마나 벗어나는지 (vector residual)
        return (pos - pos.mean(axis=0)).flatten()

    # 초기 x: 가장 낮은 residual 의 cv2 결과
    x0 = np.concatenate([R_lib.from_matrix(best_R).as_rotvec(), best_t])
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=200)
        T_refined = x_to_T(result.x)
        R_refined = T_refined[:3,:3]
        t_refined = T_refined[:3,3]
        r_refined = residual_norm(R_refined, t_refined)
        if r_refined < best_r:
            if verbose:
                print(f"  [refinement] residual {best_r*1000:.1f}mm → {r_refined*1000:.1f}mm")
            best_R, best_t = R_refined, t_refined
            best_r = r_refined
    except Exception as e:
        if verbose:
            print(f"  [refinement] 실패: {e} (cv2 결과 사용)")

    # === 3) Iterative outlier rejection (사격의 'flyer 제외 후 정밀 조준') ===
    # 각 pose 의 T_t2g 가 mean 에서 얼마나 벗어나는지 측정 → 2σ 이상 outlier 제외 후 재 refinement
    n_total = len(R_g2b)
    keep_idx = list(range(n_total))
    for it in range(3):  # 최대 3 iteration
        T_c2b_cur = np.eye(4); T_c2b_cur[:3,:3]=best_R; T_c2b_cur[:3,3]=best_t
        Ts = []
        for i in keep_idx:
            T_g_ = np.eye(4); T_g_[:3,:3]=R_g2b[i]; T_g_[:3,3]=np.asarray(t_g2b[i]).flatten()
            T_t_ = np.eye(4); T_t_[:3,:3]=R_t2c[i]; T_t_[:3,3]=np.asarray(t_t2c[i]).flatten()
            Ts.append(np.linalg.inv(T_g_) @ T_c2b_cur @ T_t_)
        pos = np.array([T[:3,3] for T in Ts])
        mean_p = pos.mean(axis=0)
        devs = np.linalg.norm(pos - mean_p, axis=1)  # m
        thresh = devs.mean() + 2 * devs.std()  # 2σ 기준
        new_keep = [keep_idx[k] for k in range(len(keep_idx)) if devs[k] <= thresh]
        n_removed = len(keep_idx) - len(new_keep)
        if n_removed == 0 or len(new_keep) < 6:
            # 더 제외할 outlier 없음 또는 데이터 너무 적음 → 종료
            break

        # 줄어든 데이터로 다시 refinement
        Rg = [R_g2b[i] for i in new_keep]; tg = [t_g2b[i] for i in new_keep]
        Rt = [R_t2c[i] for i in new_keep]; tt = [t_t2c[i] for i in new_keep]

        def residuals_subset(x):
            T_ = x_to_T(x)
            tlist = []
            for i in range(len(Rg)):
                Tg_ = np.eye(4); Tg_[:3,:3]=Rg[i]; Tg_[:3,3]=np.asarray(tg[i]).flatten()
                Tt_ = np.eye(4); Tt_[:3,:3]=Rt[i]; Tt_[:3,3]=np.asarray(tt[i]).flatten()
                tlist.append((np.linalg.inv(Tg_) @ T_ @ Tt_)[:3,3])
            tarr = np.array(tlist)
            return (tarr - tarr.mean(axis=0)).flatten()

        x0_iter = np.concatenate([R_lib.from_matrix(best_R).as_rotvec(), best_t])
        try:
            res = least_squares(residuals_subset, x0_iter, method='lm', max_nfev=200)
            T_iter = x_to_T(res.x)
            R_iter = T_iter[:3,:3]
            t_iter = T_iter[:3,3]
            # 전체 데이터 기준 평가는 의미 없음 (subset 으로 봐야)
            # subset 기준 residual 비교
            def subset_resid(R_, t_):
                Tcb_ = np.eye(4); Tcb_[:3,:3]=R_; Tcb_[:3,3]=t_
                Ts_ = []
                for i in range(len(Rg)):
                    Tg_ = np.eye(4); Tg_[:3,:3]=Rg[i]; Tg_[:3,3]=np.asarray(tg[i]).flatten()
                    Tt_ = np.eye(4); Tt_[:3,:3]=Rt[i]; Tt_[:3,3]=np.asarray(tt[i]).flatten()
                    Ts_.append((np.linalg.inv(Tg_) @ Tcb_ @ Tt_)[:3,3])
                pos_ = np.array(Ts_)
                return np.linalg.norm(pos_ - pos_.mean(axis=0), axis=1).mean()

            r_iter = subset_resid(R_iter, t_iter)
            r_prev = subset_resid(best_R, best_t)
            if r_iter < r_prev:
                if verbose:
                    print(f"  [outlier-iter {it+1}] {n_removed} flyer 제외 ({len(new_keep)}/{n_total} 유지). "
                          f"잔차 {r_prev*1000:.1f}mm → {r_iter*1000:.1f}mm")
                best_R, best_t = R_iter, t_iter
                keep_idx = new_keep
            else:
                # 개선 없음 → 종료
                break
        except Exception:
            break

    return best_R, best_t


def evaluate_hand_eye(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
                      R_c2b: np.ndarray, t_c2b: np.ndarray) -> dict:
    """
    Hand-Eye 캘리브레이션 결과의 오차 평가 (Eye-to-Hand setup).

    원리:
      Eye-to-Hand 에서 T_target_to_gripper (마커-그리퍼 변환) 는 일정해야 함.
      각 포즈에서 추정해 평균을 구한 뒤, 평균값으로 모든 포즈에서 T_target_to_cam 예측.
      예측 vs 측정 비교 → 위치 오차 (mm), 회전 오차 (deg).

    Returns: dict with mean/std/max of position and rotation errors
    """
    n = len(R_g2b_list)
    if n < 2:
        return {}

    def make_T(R_, t_):
        T = np.eye(4)
        T[:3, :3] = np.asarray(R_)
        T[:3, 3] = np.asarray(t_).flatten()
        return T

    T_c2b = make_T(R_c2b, t_c2b)
    T_b2c = np.linalg.inv(T_c2b)

    # 1) 각 포즈에서 T_target_to_gripper 추정
    Ts_t2g = []
    for i in range(n):
        T_g2b = make_T(R_g2b_list[i], t_g2b_list[i])
        T_t2c = make_T(R_t2c_list[i], t_t2c_list[i])
        T_t2g = np.linalg.inv(T_g2b) @ T_c2b @ T_t2c
        Ts_t2g.append(T_t2g)

    # 2) 평균 T_target_to_gripper 계산 (translation 단순 평균 + quaternion 평균)
    avg_pos = np.mean([T[:3, 3] for T in Ts_t2g], axis=0)
    quats = [R.from_matrix(T[:3, :3]).as_quat() for T in Ts_t2g]
    q0 = quats[0]
    aligned = [q if np.dot(q, q0) > 0 else -q for q in quats]  # sign ambiguity 해소
    avg_q = np.mean(aligned, axis=0)
    avg_q /= np.linalg.norm(avg_q)
    T_t2g_avg = np.eye(4)
    T_t2g_avg[:3, :3] = R.from_quat(avg_q).as_matrix()
    T_t2g_avg[:3, 3] = avg_pos

    # 3) 각 포즈에서 예측 vs 측정 오차 산출
    pos_errs_mm = []
    rot_errs_deg = []
    for i in range(n):
        T_g2b = make_T(R_g2b_list[i], t_g2b_list[i])
        T_t2c_meas = make_T(R_t2c_list[i], t_t2c_list[i])
        T_t2c_pred = T_b2c @ T_g2b @ T_t2g_avg
        # 위치 오차 (m → mm)
        pos_err = float(np.linalg.norm(T_t2c_pred[:3, 3] - T_t2c_meas[:3, 3]) * 1000.0)
        pos_errs_mm.append(pos_err)
        # 회전 오차 (deg)
        R_diff = T_t2c_pred[:3, :3] @ T_t2c_meas[:3, :3].T
        cos_theta = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
        rot_errs_deg.append(float(np.rad2deg(np.arccos(cos_theta))))

    return {
        'n_samples': n,
        'pos_err_mean_mm': float(np.mean(pos_errs_mm)),
        'pos_err_std_mm':  float(np.std(pos_errs_mm)),
        'pos_err_max_mm':  float(np.max(pos_errs_mm)),
        'pos_err_min_mm':  float(np.min(pos_errs_mm)),
        'rot_err_mean_deg': float(np.mean(rot_errs_deg)),
        'rot_err_std_deg':  float(np.std(rot_errs_deg)),
        'rot_err_max_deg':  float(np.max(rot_errs_deg)),
        'rot_err_min_deg':  float(np.min(rot_errs_deg)),
        'pos_errs_per_pose_mm': pos_errs_mm,
        'rot_errs_per_pose_deg': rot_errs_deg,
        't_target2gripper_mm': (T_t2g_avg[:3, 3] * 1000.0).tolist(),
    }


def _posx_to_matrix(posx_6: List[float]) -> np.ndarray:
    """[x_mm, y_mm, z_mm, a_deg, b_deg, c_deg] → 4x4 동차 행렬 (두산 ZYZ Euler)"""
    a, b, c = np.deg2rad(posx_6[3]), np.deg2rad(posx_6[4]), np.deg2rad(posx_6[5])
    rot = R.from_euler('ZYZ', [a, b, c]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [posx_6[0], posx_6[1], posx_6[2]]   # mm
    return T


def _matrix_to_posx(T: np.ndarray) -> List[float]:
    """4x4 동차 행렬 → [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg] (두산 ZYZ Euler)"""
    pos = T[:3, 3]
    a, b, c = R.from_matrix(T[:3, :3]).as_euler('ZYZ')
    return [pos[0], pos[1], pos[2],
            np.rad2deg(a), np.rad2deg(b), np.rad2deg(c)]


def apply_tool_offset_to_posx(base_posx: List[float], tool_offset: Tuple) -> List[float]:
    """
    base 자세에 *tool 좌표계 기준* offset 을 합성해 새 base-frame 절대 좌표 생성.
    target = base_pose @ tool_offset_pose

    이렇게 하면 회전이 **그리퍼 자체 기준** 으로 일어남:
      tool_offset = (0, 0, 0, 0, 0, +30)  → 그리퍼 Tool Z 축 기준 30° 회전 (in-plane)
      → 마커 면 (Tool +Z 또는 -Z 면) 이 카메라 향한 채 in-plane 회전 (visibility 보존)
      참고: 마커가 앞면(+Z)에 있든 뒷면(-Z)에 있든 회전 축이 같아 동일하게 동작.
    """
    base_T = _posx_to_matrix(base_posx)
    offset_T = _posx_to_matrix(list(tool_offset))
    target_T = base_T @ offset_T
    return _matrix_to_posx(target_T)


# 호환성 유지 (옛 base-frame 합성 — 사용 안 함, 백업)
def apply_offset_to_posx(base_posx: List[float], offset: Tuple) -> List[float]:
    """(deprecated) base [x,y,z,rx,ry,rz] 에 offset 단순 합 — Euler 더하기는 부정확함"""
    return [base_posx[i] + offset[i] for i in range(6)]


def main():
    print("=" * 70)
    print("  자동 Hand-Eye Calibration v2 — 거리 기반 (Cartesian)")
    print("=" * 70)

    if not HAS_REALSENSE:
        print("Error: pyrealsense2 가 필요합니다. pip install pyrealsense2")
        return
    if not HAS_ROS2:
        print("Error: ROS2 환경이 필요합니다.")
        return

    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"\n출력 경로: {OUTPUT_DIR}")

    print("\nRealSense 카메라 초기화...")
    camera = RealSenseCamera()

    rclpy.init()
    print("로봇 서비스 연결 중...")
    try:
        robot = RobotController()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        camera.stop()
        rclpy.shutdown()
        sys.exit(1)

    # MoveIt path 사용 — dsr_moveit_controller 가 active 인 상태로 진행 (정석)

    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []

    if os.path.exists(DATA_FILE):
        print(f"\n기존 데이터 발견: {DATA_FILE}")
        if input("기존 데이터 삭제하고 새로 시작? (y/n): ").lower() == 'y':
            os.remove(DATA_FILE)
            for f in os.listdir(IMAGE_DIR):
                os.remove(os.path.join(IMAGE_DIR, f))
            print("기존 데이터 삭제됨")

    print("\n" + "=" * 70)
    print("조작 방법:")
    print("  's' = 현재 자세를 base 로 저장 (이 시점의 TCP 좌표 사용)")
    print("  Enter = 자동 순회 시작 (20 포즈)")
    print("  'c' = 캘리브레이션 계산")
    print("  'h' = base 자세로 복귀")
    print("  'd' = 디버그: 현재 프레임 저장 + 모든 ArUco 사전 후보 강제 재탐색")
    print("  'q' = 종료")
    print("=" * 70)

    cv2.namedWindow("Hand-Eye Calibration v2 (ArUco)", cv2.WINDOW_NORMAL)

    # ArUco 검출기 — main 에서 1회 생성 후 재사용. 'd' 키 / 자동 탐색 시 교체.
    aruco_dict_id = ARUCO_DICT_ID
    aruco_detector = make_aruco_detector(aruco_dict_id)

    # 시작 시 자동 사전 탐색 (1회) — 디폴트 사전과 다르더라도 즉시 적합한 사전 적용
    print("\n시작 자동 사전 탐색 중...")
    for _ in range(20):  # 카메라 초기 프레임 안정화
        camera.get_frame()
    probe = camera.get_frame()
    found_initial = diagnose_aruco_dict(probe)
    if found_initial:
        name, dict_id, ids = found_initial[0]
        if dict_id != aruco_dict_id:
            print(f">> 마커 검출됨 — 사전을 '{name}' 으로 자동 변경 (IDs={ids})")
            aruco_dict_id = dict_id
            aruco_detector = make_aruco_detector(aruco_dict_id)
        else:
            print(f">> 디폴트 사전 '{name}' 으로 마커 검출됨 (IDs={ids})")
    else:
        print(">> (현재 마커 안 보임. 카메라 화면에 마커 잡히면 자동 검출됨. 'd' 키로 강제 재탐색 가능)")

    base_posx: Optional[List[float]] = None
    offsets: Optional[List[Tuple]] = None
    auto_mode = False
    cur_idx = 0

    try:
        while True:
            frame = camera.get_frame()
            display = frame.copy()
            all_corners, all_ids = detect_aruco_all(aruco_detector, frame)

            if all_ids is not None and len(all_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, all_corners, all_ids)
                # ARUCO_TARGET_ID 우선 선택
                sel = 0
                if ARUCO_TARGET_ID is not None:
                    for i in range(len(all_ids)):
                        if int(all_ids[i][0]) == ARUCO_TARGET_ID:
                            sel = i
                            break
                    else:
                        sel = -1   # target 없음
                if sel >= 0:
                    primary_corners = all_corners[sel].reshape(4, 2)
                    sel_id = int(all_ids[sel][0])
                    try:
                        t_b, _, rvec = estimate_aruco_pose(primary_corners, ARUCO_MARKER_SIZE,
                                                           camera.camera_matrix, camera.dist_coeffs)
                        cv2.drawFrameAxes(display, camera.camera_matrix, camera.dist_coeffs,
                                          rvec, t_b, ARUCO_MARKER_SIZE * 0.6)
                        cv2.putText(display, f"Distance: {np.linalg.norm(t_b):.3f}m  "
                                              f"ID:{sel_id}", (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception:
                        pass
                    cv2.putText(display, f"ArUco OK ({len(all_ids)} markers, using ID:{sel_id})",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # target ID 없음 (다른 ID 의 마커들만 검출)
                    cv2.putText(display,
                                f"target ID:{ARUCO_TARGET_ID} 없음 (검출 IDs: {all_ids.flatten().tolist()})",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(display, "ArUco NOT FOUND - press 'd' to scan dicts", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if base_posx is not None:
                cv2.putText(display,
                            f"Base [{base_posx[0]:.0f},{base_posx[1]:.0f},{base_posx[2]:.0f}]mm",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Press 's' to set base TCP pose", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(display, f"Saved samples: {len(R_g2b_list)}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if auto_mode and offsets:
                cv2.putText(display, f"AUTO {cur_idx}/{len(offsets)}", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Hand-Eye Calibration v2 (ArUco)", display)

            # 자동 순회 처리 (Cartesian MoveLine 사용)
            if auto_mode and offsets and cur_idx < len(offsets) and base_posx is not None:
                offset = offsets[cur_idx]
                # tool 좌표계 기준 offset → base frame 절대 좌표로 합성
                target = apply_tool_offset_to_posx(base_posx, offset)
                print(f"\n[{cur_idx + 1}/{len(offsets)}] tool dxyz=({offset[0]:+.0f},{offset[1]:+.0f},{offset[2]:+.0f})mm "
                      f"tool drot=({offset[3]:+.0f},{offset[4]:+.0f},{offset[5]:+.0f})deg")
                print(f"  target posx = [{target[0]:.0f}, {target[1]:.0f}, {target[2]:.0f},"
                      f" {target[3]:.1f}, {target[4]:.1f}, {target[5]:.1f}]")

                if not robot.move_line_absolute(target):
                    print("  [경고] 이동 실패 (작업 공간 한계 / 충돌 / Servo OFF?). 다음 포즈로.")
                    cur_idx += 1
                    continue

                print(f"  안정화 대기 ({SETTLE_TIME}초)...")
                time.sleep(SETTLE_TIME)

                # 5 프레임 평균으로 pose 추정 — sub-pixel noise 감소
                avg = estimate_aruco_pose_averaged(
                    aruco_detector, camera.get_frame, ARUCO_MARKER_SIZE,
                    camera.camera_matrix, camera.dist_coeffs, n_frames=5)
                if avg is not None:
                    t_b, R_b, _, marker_id = avg
                    try:
                        t_r, R_r = robot.get_tcp_pose()
                        R_g2b_list.append(R_r); t_g2b_list.append(t_r)
                        R_t2c_list.append(R_b); t_t2c_list.append(t_b)

                        # 마지막 프레임 저장 (시각화용)
                        last_frame = camera.get_frame()
                        img_path = os.path.join(IMAGE_DIR, f"pose_{len(R_g2b_list):02d}.jpg")
                        cv2.imwrite(img_path, last_frame)
                        np.savez(DATA_FILE,
                                 R_gripper2base=R_g2b_list, t_gripper2base=t_g2b_list,
                                 R_target2cam=R_t2c_list, t_target2cam=t_t2c_list)
                        print(f"  [저장] 누적 {len(R_g2b_list)} 개 (marker ID={marker_id}, 5프레임 평균)")
                    except Exception as e:
                        print(f"  [오류] 데이터 저장 실패: {e}")
                else:
                    # marker 가림 — base 로 자동 복귀 (다음 motion 이 깨끗한 상태에서 시작)
                    print("  [경고] ArUco 마커 검출 실패 (5프레임 중 부족). base 자세로 복귀.")
                    robot.move_line_absolute(base_posx)
                    time.sleep(SETTLE_TIME)

                cur_idx += 1
                if cur_idx >= len(offsets):
                    auto_mode = False
                    print("\n" + "=" * 70)
                    print(f" 자동 수집 완료. 총 {len(R_g2b_list)} 샘플")
                    print(" 'h' 로 base 복귀 후 'c' 로 캘리브레이션 계산하세요.")
                    print("=" * 70)
                continue

            key = cv2.waitKey(100) & 0xFF

            if key == ord('s'):
                try:
                    base_posx = robot.get_tcp_posx_raw()
                    offsets = generate_calibration_offsets()
                    print(f"\n[Base 저장됨] posx = [{base_posx[0]:.0f}, {base_posx[1]:.0f}, {base_posx[2]:.0f},"
                          f" {base_posx[3]:.1f}, {base_posx[4]:.1f}, {base_posx[5]:.1f}]")
                    print(f"  생성된 offset 포즈 {len(offsets)} 개. Enter 로 자동 시작.")
                except Exception as e:
                    print(f"\n[오류] base TCP 저장 실패: {e}")

            elif key == 13:  # Enter
                if base_posx is not None and offsets is not None:
                    print("\n자동 순회 시작!")
                    auto_mode = True
                    cur_idx = 0
                else:
                    print("\n먼저 's' 로 base 자세를 저장하세요.")

            elif key == ord('c'):
                if len(R_g2b_list) < 3:
                    print("\n[오류] 최소 3 샘플 필요 (현재 {} 개)".format(len(R_g2b_list)))
                    continue
                print("\n" + "=" * 70 + "\n캘리브레이션 계산 중...")
                try:
                    def _do_calibrate_save_print(label: str = ""):
                        """현재 누적 데이터로 calibrate + evaluate + 저장 + 출력. metrics 반환."""
                        R_c2b, t_c2b = calibrate_hand_eye(R_g2b_list, t_g2b_list,
                                                         R_t2c_list, t_t2c_list)
                        T = np.eye(4); T[:3, :3] = R_c2b; T[:3, 3] = t_c2b
                        if label:
                            print(f"\n[{label}]")
                        print(f"카메라 위치 (베이스 기준): "
                              f"x={t_c2b[0]*1000:.1f} y={t_c2b[1]*1000:.1f} z={t_c2b[2]*1000:.1f} mm")
                        m = evaluate_hand_eye(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
                                              R_c2b, t_c2b)
                        if m:
                            print(f"  N={m['n_samples']}  "
                                  f"위치 평균={m['pos_err_mean_mm']:5.2f}mm "
                                  f"max={m['pos_err_max_mm']:.2f}mm  "
                                  f"회전 평균={m['rot_err_mean_deg']:.2f}° "
                                  f"max={m['rot_err_max_deg']:.2f}°")
                            p, r = m['pos_err_mean_mm'], m['rot_err_mean_deg']
                            if p < 3 and r < 0.5:        q = "🟢 매우 우수"
                            elif p < 8 and r < 1.5:      q = "🟢 우수"
                            elif p < 15 and r < 3.0:     q = "🟡 양호"
                            elif p < 30 and r < 5.0:     q = "🟡 보통"
                            else:                        q = "🔴 부정확"
                            print(f"  품질 등급: {q}")
                        np.savez(OUTPUT_FILE,
                                 T_cam2base=T, R_cam2base=R_c2b, t_cam2base=t_c2b,
                                 camera_matrix=camera.camera_matrix, dist_coeffs=camera.dist_coeffs,
                                 aruco_dict_id=aruco_dict_id, marker_size=ARUCO_MARKER_SIZE,
                                 num_samples=len(R_g2b_list),
                                 pos_err_mean_mm=m.get('pos_err_mean_mm', -1),
                                 pos_err_max_mm=m.get('pos_err_max_mm', -1),
                                 rot_err_mean_deg=m.get('rot_err_mean_deg', -1),
                                 rot_err_max_deg=m.get('rot_err_max_deg', -1),
                                 pos_errs_per_pose_mm=m.get('pos_errs_per_pose_mm', []),
                                 rot_errs_per_pose_deg=m.get('rot_errs_per_pose_deg', []))
                        return m

                    # 1) 초기 캘리브레이션
                    print("-" * 70)
                    metrics = _do_calibrate_save_print("초기 캘리브레이션")
                    print("-" * 70)

                    # 2) AUTO_REFINE: 목표 미달 시 추가 포즈 수집 + 재계산 반복
                    if (AUTO_REFINE_ENABLED and metrics and base_posx is not None and
                        (metrics['pos_err_mean_mm'] >= AUTO_REFINE_TARGET_POS_MM or
                         metrics['rot_err_mean_deg'] >= AUTO_REFINE_TARGET_ROT_DEG)):

                        print(f"\n🔄 자동 정제 시작 — 목표: 위치<{AUTO_REFINE_TARGET_POS_MM}mm, "
                              f"회전<{AUTO_REFINE_TARGET_ROT_DEG}°  (최대 {AUTO_REFINE_MAX_ROUNDS}라운드)")
                        for round_idx in range(1, AUTO_REFINE_MAX_ROUNDS + 1):
                            print(f"\n=== 정제 라운드 {round_idx}/{AUTO_REFINE_MAX_ROUNDS} ===")
                            extra = generate_refinement_offsets(round_idx, AUTO_REFINE_POSES_PER_ROUND)
                            for j, eoff in enumerate(extra, 1):
                                tgt = apply_tool_offset_to_posx(base_posx, eoff)
                                print(f"  [{round_idx}.{j}/{len(extra)}] tool dxyz=({eoff[0]:+.0f},{eoff[1]:+.0f},{eoff[2]:+.0f})mm "
                                      f"drot=({eoff[3]:+.0f},{eoff[4]:+.0f},{eoff[5]:+.0f})deg")
                                if not robot.move_line_absolute(tgt):
                                    print("    이동 실패, skip")
                                    continue
                                time.sleep(SETTLE_TIME)
                                # 5 프레임 평균 (정확도 ↑)
                                avg = estimate_aruco_pose_averaged(
                                    aruco_detector, camera.get_frame, ARUCO_MARKER_SIZE,
                                    camera.camera_matrix, camera.dist_coeffs, n_frames=5)
                                if avg is None:
                                    print("    마커 미검출 — base 복귀 후 다음")
                                    robot.move_line_absolute(base_posx)
                                    time.sleep(SETTLE_TIME)
                                    continue
                                try:
                                    t_b, R_b, _, _ = avg
                                    t_r, R_r = robot.get_tcp_pose()
                                    R_g2b_list.append(R_r); t_g2b_list.append(t_r)
                                    R_t2c_list.append(R_b); t_t2c_list.append(t_b)
                                    last_f = camera.get_frame()
                                    cv2.imwrite(os.path.join(IMAGE_DIR,
                                                f"pose_{len(R_g2b_list):02d}.jpg"), last_f)
                                    np.savez(DATA_FILE,
                                             R_gripper2base=R_g2b_list, t_gripper2base=t_g2b_list,
                                             R_target2cam=R_t2c_list, t_target2cam=t_t2c_list)
                                    print(f"    ✓ 저장 (누적 {len(R_g2b_list)}, 5프레임 평균)")
                                except Exception as ex:
                                    print(f"    저장 실패: {ex}")

                            # 라운드 종료 시 base 복귀 (다음 라운드 깨끗한 시작)
                            print(f"\n  base 자세 복귀...")
                            robot.move_line_absolute(base_posx)
                            time.sleep(SETTLE_TIME)

                            # 재계산 + 평가
                            metrics = _do_calibrate_save_print(f"라운드 {round_idx} 후")

                            # 목표 도달 체크
                            if (metrics['pos_err_mean_mm'] < AUTO_REFINE_TARGET_POS_MM and
                                metrics['rot_err_mean_deg'] < AUTO_REFINE_TARGET_ROT_DEG):
                                print(f"\n🟢 매우 우수 등급 도달! 자동 정제 종료 "
                                      f"(라운드 {round_idx}, 총 {metrics['n_samples']} 샘플)")
                                break
                        else:
                            print(f"\n⚠️ 최대 라운드({AUTO_REFINE_MAX_ROUNDS}) 도달. "
                                  f"현재 best: 위치 {metrics['pos_err_mean_mm']:.2f}mm / "
                                  f"회전 {metrics['rot_err_mean_deg']:.2f}°")

                    print(f"\n[최종] 총 {len(R_g2b_list)} 샘플  결과: {OUTPUT_FILE}")
                    print("=" * 70)
                except Exception as e:
                    print(f"\n[오류] 캘리브레이션 실패: {e}")
                    import traceback; traceback.print_exc()

            elif key == ord('h'):
                if base_posx is not None:
                    print("\nbase 자세로 복귀 중...")
                    if robot.move_line_absolute(base_posx):
                        time.sleep(SETTLE_TIME)
                        print(">> 복귀 완료.")
                    else:
                        print(">> 복귀 실패.")

            elif key == ord('d'):
                # 디버그: 현재 프레임 저장 + 모든 ArUco 사전 후보 검출 시도
                debug_path = os.path.join(OUTPUT_DIR, f"debug_{int(time.time())}.png")
                cv2.imwrite(debug_path, frame)
                print(f"\n[디버그] 프레임 저장: {debug_path}")
                print(f"  ArUco 사전 후보별 검출 결과:")
                found = diagnose_aruco_dict(frame)
                if found:
                    for name, dict_id, ids in found:
                        marker = " <-- 자동 적용" if (name, dict_id, ids) == found[0] else ""
                        print(f"    ✅ {name:24s} marker IDs={ids}{marker}")
                    # 첫 번째 검출된 사전을 즉시 적용 (코드 수정 없이) + detector 재생성
                    aruco_dict_id = found[0][1]
                    aruco_detector = make_aruco_detector(aruco_dict_id)
                    print(f"\n  → 활성 사전이 '{found[0][0]}' 로 자동 변경됨. 이대로 진행 가능.")
                    print(f"     마커 한 변(검정 테두리 포함) 길이가 {ARUCO_MARKER_SIZE*1000:.0f}mm 가 맞는지 확인!")
                    print(f"     다르면 ARUCO_MARKER_SIZE 상수 수정 필요.")
                else:
                    print("    ❌ 모든 사전에서 검출 실패. 다음 점검:")
                    print("    1) 마커 전체가 화면에 들어와 있는지 (검정 테두리 잘림 X)")
                    print("    2) 조명 충분한지 (그림자/반사광 X)")
                    print("    3) 카메라 포커스 맞는지")
                    print("    4) 마커의 검정 테두리 두께가 충분한지 (cell 크기 1배 이상)")
                    print(f"       (저장된 이미지 {debug_path} 확인)")

            elif key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        camera.stop()
        robot.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
