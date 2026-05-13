"""
[통합 Pick & Place (cuRobo 기반) — 두산 e0509 + RealSense + YOLOv8 + RH-P12-RN-A]

12번을 cuRobo MotionGen 기반으로 재작성한 버전.
- dsr Ikin 분기 시도 제거
- cuRobo `MotionGen.plan_single` 으로 도달가능성·관절한계·충돌 검증 + trajectory 계획
- 모션 실행: 12번에서 검증된 `FollowJointTrajectory` 액션 (MoveIt 컨트롤러 경로).
  `/dsr01/motion/move_*` 직접 서비스는 dsr_bringup2_moveit 환경에서 거짓 success
  를 반환하므로 사용 X — trajectory action 만 실제 robot 을 움직임.
- 12번의 비전·그리퍼·안전·리셋·자세 캘리브레이션 흐름은 그대로 유지

두 단계 흐름:
  [A] 준비 (한 번만) — 실행 직후 자동 진행:
        0. 잔존 ROS/RealSense/강의 스크립트 정리 + dsr_bringup2_moveit 새로 띄움
        1. 캘리브레이션 + YOLO + RealSense + 로봇 Servo ON + 그리퍼 init + 열기
        2. cuRobo MotionGen 로드 + warmup (30~60초)
        3. HOME 관절 자세로 이동
  [B] 명령 루프:
        잡을 물체 → YOLO → 좌표 → 사용자 확인 → cuRobo plan → MoveSplineJoint 실행
        진단 명령: plan_only / world_clear / curobo_warmup / pose / state / recover / home / ...

전제 조건:
  - 12번과 동일 + cuRobo v0.7.8 / torch+cu128 / CUDA 12.8 설치 + ~/.local 의존성
  - e0509_gripper.urdf 가 ~/doosan_ws/src/e0509_gripper_description/config/curobo/ 에 존재

실행:
  python3 13_비전_피크앤플레이스_curobo.py
  python3 13_비전_피크앤플레이스_curobo.py --quick    # 드라이버 리셋 skip
  python3 13_비전_피크앤플레이스_curobo.py --dry-run  # 모션 없이 plan 만 시도
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import cv2

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from dsr_msgs2.srv import (
    SetRobotMode, SetRobotControl, GetCurrentPose,
    GetRobotState, SetSafetyMode, SetSafeStopResetType,
    FlangeSerialOpen, FlangeSerialClose, FlangeSerialWrite,
)

try:
    import pyrealsense2 as rs
except ImportError:
    print('!! pyrealsense2 미설치. pip install pyrealsense2 후 다시 실행하세요.')
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print('!! ultralytics 미설치. 다음 순서로 설치 후 다시 실행하세요:')
    print('     source /home/fastcampus/Downloads/test/vision_env/bin/activate')
    print('     pip install ultralytics')
    sys.exit(1)

try:
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig, JointState as CuroboJointState
    from curobo.types.math import Pose as CuroboPose
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from curobo.geom.types import WorldConfig, Cuboid
except ImportError as e:
    print(f'!! cuRobo 또는 torch 미설치/import 실패: {e}')
    print('   - cuRobo: cd ~/curobo && /usr/bin/python3 -m pip install --user -e . --no-build-isolation')
    print('   - PyTorch: ~/.local 에 torch (cu128) 설치 + ~/.bashrc 의 PYTHONPATH 확인')
    sys.exit(1)

try:
    from scipy.spatial.transform import Rotation as _ScipyR
except ImportError:
    print('!! scipy 미설치. pip install --user scipy 후 다시 실행하세요.')
    sys.exit(1)


# ============== 사용자 조정 가능 상수 ==============
# 환경 의존 값(NS/IP/모델/패키지)은 doosan_config.py 또는 .env 에서 가져옴.
from doosan_config import (
    NAMESPACE as NS,
    ROBOT_IP, RT_HOST, ROBOT_MODEL,
    BRINGUP_PKG, BRINGUP_LAUNCH, MOVEIT_CONTROLLER,
    DOOSAN_WS, ROS_DISTRO,
)

DRIVER_WAIT_SEC = 25
# dsr_bringup2_moveit.launch.py 가 자체로 RViz 를 띄우지 않으므로 별도 실행
# (DOOSAN_WS 안 dsr_moveit2 경로는 fork 마다 같은 구조라 가정)
RVIZ_CONFIG = os.path.join(
    DOOSAN_WS, 'src', 'doosan-robot2', 'dsr_moveit2',
    f'dsr_moveit_config_{ROBOT_MODEL}', 'launch', 'moveit.rviz'
)

# 로봇 자세
HOME_JOINT_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]   # 안전한 시작 자세 (deg)
TOP_DOWN_RPY_DEG = [0.0, 180.0, 0.0]                # 그리퍼가 -Z(아래) 향하는 자세 (deg)

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# FollowJointTrajectory: trajectory 의 t=time_from_start. 길수록 안전·느림.
MOVE_DURATION_SEC = 5.0

# 두산 e0509 펌웨어 관절 한계 (안전 마진 포함). curobo 결과 trajectory 의 모든
# 포인트가 이 범위에 들어가야 MoveSplineJoint 가 abort 안 됨.
SAFE_JOINT_LIMITS_DEG = [
    (-360.0, 360.0),    # joint_1
    ( -90.0,  90.0),    # joint_2  (펌웨어 ±95, 마진 5도)
    (-150.0, 150.0),    # joint_3  (URDF ±155, 마진 5도)
    (-360.0, 360.0),    # joint_4
    (-135.0, 135.0),    # joint_5
    (-360.0, 360.0),    # joint_6
]

# cuRobo robot/world config 파일 경로
CUROBO_URDF = os.environ.get(
    'E0509_CUROBO_URDF',
    os.path.expanduser('~/doosan_ws/src/e0509_gripper_description/config/curobo/e0509_gripper.urdf'),
)
CUROBO_BASE_LINK = 'base_link'
CUROBO_EE_LINK = 'gripper_rh_p12_rn_base'   # 플랜지 직결 (TCP z-offset 별도 적용)
# 책상(고정 장애물): 베이스 plate 기준 z=-0.02m, 두께 4cm, 1.2x1.2m
WORLD_TABLE = dict(name='table', pose=[0.0, 0.0, -0.02, 1.0, 0.0, 0.0, 0.0], dims=[1.2, 1.2, 0.04])

# Pick & Place 기하
Z_APPROACH = 100.0          # mm — 객체 위 안전 높이 (pre-grasp)
Z_LIFT = 100.0              # mm — 잡은 후 들어올림
PLACE_OFFSET_XYZ = [0.0, 200.0, 0.0]   # mm — Pick 좌표 기준 +Y 200mm

# 그리퍼 TCP 오프셋 (mm): 두산 플랜지 → 잡는 지점까지의 z 거리.
# 두 가지 프로파일을 검출 z 로 자동 선택:
#   - NORMAL: 컵·병처럼 높이 있는 물체 → 그리퍼 핑거 가운데로 잡음 (~160mm)
#   - FLAT  : 펜·동전처럼 바닥에 바짝 붙은 작은 물체 → 그리퍼 끝으로 집음 (~180mm)
GRIPPER_TCP_OFFSET_NORMAL = 160.0   # 일반 물체
GRIPPER_TCP_OFFSET_FLAT   = 161.0   # 바닥 물체 (더 깊이 내려가야 함)
FLAT_OBJECT_Z_THRESHOLD   = -20.0   # 검출 z 가 이 값 이하면 FLAT 모드 (mm, 베이스 plate 기준)
# 시작 시 default — 첫 사이클 전까지는 NORMAL, 사이클마다 자동 선택됨
GRIPPER_TCP_OFFSET_Z = GRIPPER_TCP_OFFSET_NORMAL

# Grasp dive (mm): 검출된 z(객체 윗면) 에서 추가로 더 내려갈 깊이 (TOP-DOWN 모드용).
# 객체 종류별로 정확히 분류 — 컵 같은 속 빈 물체는 입구로 깊이 빠지면 안 됨.
GRASP_DIVE_NORMAL_MM = 40.0     # 블록·박스·과일 — 측면 가운데를 잡음
GRASP_DIVE_HOLLOW_MM = 5.0      # 컵·병·머그 — 입구 가장자리만 살짝 잡음 (안 빠지게)
GRASP_DIVE_FLAT_MM   = 0.0      # 펜·동전·종이 — 표면 그대로

# HOLLOW 객체 — 속이 빈 직립 원통형 (입구 잡기). SIDE-grasp 와 같은 집합.
HOLLOW_CLASSES = {
    'cup', 'mug', 'glass', 'wine glass', 'bottle', 'can',
    'vase', 'tumbler', 'beer', 'soda can', 'water bottle',
}
# SIDE-grasp 시도 대상 (HOLLOW 와 동일). cuRobo plan 실패 시 HOLLOW TOP-DOWN 으로 fallback.
SIDE_CLASSES = set(HOLLOW_CLASSES)
# SIDE 접근 시 객체와의 거리 (XY 평면, mm)
SIDE_APPROACH_DIST_MM = 100.0

# 작업 영역 안전 한계 (베이스 좌표, mm) — 검출값이 벗어나면 거부 (클립 X).
# WORK_Z 하한이 음수인 이유: 두산 베이스 plate 가 테이블보다 위에 마운팅돼있는 경우
# 객체(=테이블 위)는 베이스 기준 음수 z 에 위치. 사용자 setup 에 맞게 set_work_z 로 조정.
WORK_X = (-600.0, 600.0)
WORK_Y = (-600.0, 600.0)
WORK_Z = (-150.0, 600.0)

# 안전: 사람/동물 등 절대 잡으면 안 되는 COCO 클래스는 항상 제외
BLOCKED_CLASSES = {
    'person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'bird',
}

# scan 명령에서 한 번에 검출할 흔한 물체 후보 (영문, 자유 편집)
COMMON_OBJECTS = [
    'pen', 'pencil', 'marker', 'eraser',
    'cup', 'mug', 'bottle', 'glass',
    'book', 'notebook',
    'phone', 'remote', 'mouse', 'keyboard',
    'wood block', 'wooden block', 'cube',
    'apple', 'banana', 'orange',
    'scissors', 'ruler', 'tape',
]

# 그리퍼
GRIPPER_PORT = 1
GRIPPER_BAUD = 57600
GRIP_OPEN_POS = 0
GRIP_CLOSE_POS = 700

# 비전
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = '/home/fastcampus/Downloads/test'
CALIB_PATH = os.path.join(SCRIPT_DIR, 'calibration_data', 'calibration_result.npz')
# YOLO-World (open-vocabulary): 임의 영문 텍스트로 검출 가능 (pen, wood block, ruler ...).
# 첫 실행 시 ultralytics 가 자동 다운로드 후 캐시. 약 50MB.
YOLO_WEIGHTS = 'yolov8s-world.pt'
DETECT_CONF_THR = 0.10        # World 는 일반 YOLO 보다 conf 가 낮게 나오는 경향 → 0.1 로 완화
NMS_IOU_THR = 0.45
RS_W, RS_H, RS_FPS = 640, 480, 30


# ============== Modbus RTU (그리퍼) ==============
def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return crc

def _make_frame(data: bytes) -> list:
    crc = _crc16(data)
    return list(data + bytes([crc & 0xFF, (crc >> 8) & 0xFF]))

def fc06_torque_enable(slave=1) -> list:
    return _make_frame(bytes([slave, 0x06, 0x01, 0x00, 0x00, 0x01]))

def _rpy_deg_to_quat_wxyz(rpy_deg):
    """두산 RPY(ZYZ Euler, deg) → 쿼터니언 (w, x, y, z). curobo Pose 입력용."""
    q = _ScipyR.from_euler('ZYZ', rpy_deg, degrees=True).as_quat()   # [x,y,z,w]
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


# ============== cuRobo Motion Planner ==============
class CuroboPlanner:
    """
    두산 e0509 + RH-P12-RN-A 용 cuRobo MotionGen 래퍼.
    참조: ~/doosan_ws/src/e0509_gripper_description/scripts/curobo_planner_node.py
    """
    def __init__(self, urdf_path=CUROBO_URDF,
                 base_link=CUROBO_BASE_LINK, ee_link=CUROBO_EE_LINK):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                f'cuRobo URDF 없음: {urdf_path}\n'
                '   E0509_CUROBO_URDF 환경변수로 경로 지정하거나 '
                'e0509_gripper_description 패키지 빌드 확인.'
            )
        self.tensor_args = TensorDeviceType(device=torch.device('cuda:0'))

        print(f'   RobotConfig.from_basic urdf={os.path.basename(urdf_path)}, '
              f'base={base_link}, ee={ee_link}')
        robot_cfg = RobotConfig.from_basic(
            urdf_path=urdf_path,
            base_link=base_link,
            ee_link=ee_link,
            tensor_args=self.tensor_args,
        )

        # 책상을 고정 장애물로
        self._base_world_cuboids = [
            Cuboid(name=WORLD_TABLE['name'],
                   pose=list(WORLD_TABLE['pose']),
                   dims=list(WORLD_TABLE['dims'])),
        ]
        world_cfg = WorldConfig(cuboid=list(self._base_world_cuboids))

        mg_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg, world_cfg,
            tensor_args=self.tensor_args,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            collision_cache={'obb': 30, 'mesh': 10},
        )
        self.motion_gen = MotionGen(mg_cfg)
        print('   cuRobo MotionGen warmup 시작 (30~60초 소요)...')
        t0 = time.time()
        self.motion_gen.warmup(warmup_js_trajopt=False)
        print(f'   cuRobo warmup 완료 ({time.time()-t0:.1f}s)')

    def warmup(self):
        """수동 재warmup (CUDA 캐시 reset 후 디버그용)."""
        t0 = time.time()
        self.motion_gen.warmup(warmup_js_trajopt=False)
        print(f'   재warmup 완료 ({time.time()-t0:.1f}s)')

    def world_clear(self):
        """검출 객체 누적 cuboid 제거. 책상만 남김."""
        world_cfg = WorldConfig(cuboid=list(self._base_world_cuboids))
        self.motion_gen.update_world(world_cfg)
        print('   world cleared (table만 남음)')

    def plan(self, start_joints_rad, target_xyz_m, target_quat_wxyz):
        """
        start_joints_rad: list[6] in rad
        target_xyz_m:    list[3] in meter (base 좌표, 플랜지 보정 적용된 값)
        target_quat_wxyz: list[4]  (w,x,y,z) — curobo 표준
        반환: (positions ndarray (N,6) in rad, plan_time_ms) 또는 (None, plan_time_ms)
        """
        t0 = time.time()
        start_state = CuroboJointState.from_position(
            position=torch.tensor([start_joints_rad], device='cuda:0', dtype=torch.float32),
            joint_names=JOINT_NAMES,
        )
        target_pose = CuroboPose(
            position=torch.tensor([target_xyz_m], device='cuda:0', dtype=torch.float32),
            quaternion=torch.tensor([target_quat_wxyz], device='cuda:0', dtype=torch.float32),
        )
        result = self.motion_gen.plan_single(start_state, target_pose)
        plan_ms = (time.time() - t0) * 1000.0
        if not result.success.item():
            return None, plan_ms
        traj = result.get_interpolated_plan()
        return traj.position.cpu().numpy(), plan_ms


def fc16_position(pos: int, slave=1) -> list:
    pos = max(0, min(700, int(pos)))
    return _make_frame(bytes([slave, 0x10, 0x01, 0x1A, 0x00, 0x02, 0x04,
                              (pos >> 8) & 0xFF, pos & 0xFF, 0x00, 0x00]))


# ============== 비전 (RealSense + YOLO + 좌표변환) ==============
class VisionDetector:
    def __init__(self):
        if not os.path.exists(CALIB_PATH):
            raise FileNotFoundError(f'캘리브레이션 결과 없음: {CALIB_PATH}')

        d = np.load(CALIB_PATH)
        self.T_cam2base = d['T_cam2base']           # 4x4
        self.calib_err_mm = float(d['pos_err_mean_mm'])
        self.calib_samples = int(d['num_samples']) if 'num_samples' in d.files else 0
        print(f'   캘리브레이션 결과 로드: {CALIB_PATH}')
        print(f'   - T_cam2base shape={self.T_cam2base.shape}, '
              f'translation={[round(v*1000,1) for v in self.T_cam2base[:3,3]]}mm')
        print(f'   - 평균 위치 오차: {self.calib_err_mm:.2f}mm '
              f'(샘플 {self.calib_samples}개)')

        # YOLO-World 모델 로드 (첫 실행 시 ultralytics 자동 다운로드).
        self.model = YOLO(YOLO_WEIGHTS)
        # ultralytics 8.4 의 YOLO-World + CLIP 텍스트 인코더에 device-mismatch 버그가 있어
        # set_classes 호출 시 CPU/CUDA 가 섞여 RuntimeError 발생. 회피책으로 CPU 강제.
        # (yolov8s-world 추론은 CPU 에서도 ~1-2초/frame 이라 강의 데모엔 충분)
        self.model.cpu()
        self.device = 'cpu'
        # World 는 set_classes 로 동적 클래스 지정해야 함. 매 detect() 호출 시 갱신.
        self.classes = {}    # set_classes 호출 후 채워짐

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
        cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()

        # 자동 노출 워밍업
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _yolo_detect(self, bgr):
        """ultralytics YOLO 추론. 반환: [(class_id, conf, cx, cy, w, h), ...] (이미지 좌표계)"""
        results = self.model(bgr, conf=DETECT_CONF_THR, iou=NMS_IOU_THR,
                             verbose=False, device=self.device)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        xywh = boxes.xywh.cpu().numpy()         # cx, cy, w, h (이미 원본 해상도)
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        out = []
        for i in range(len(boxes)):
            out.append((int(cls_ids[i]), float(confs[i]),
                        float(xywh[i, 0]), float(xywh[i, 1]),
                        float(xywh[i, 2]), float(xywh[i, 3])))
        out.sort(key=lambda r: -r[1])           # confidence desc
        return out

    def _depth_at(self, depth_frame, u, v, patch=5):
        """픽셀 (u,v) 주변 depth 평균(m). 0(hole) 제외."""
        z = depth_frame.get_distance(int(u), int(v))
        if z > 0.05:
            return z
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape
        u0, v0 = max(0, int(u) - patch), max(0, int(v) - patch)
        u1, v1 = min(W, int(u) + patch + 1), min(H, int(v) + patch + 1)
        win = arr[v0:v1, u0:u1]
        win = win[win > 0]
        if win.size == 0:
            return 0.0
        return float(win.mean()) * 0.001  # mm → m

    def _top_pixel_and_depth(self, depth_frame, cx, cy, w, h, top_ratio=0.2):
        """
        bbox 상단 영역의 대표 픽셀 (u, v) 와 그 depth(m) 반환.
        - 박스 윗부분 (높이의 top_ratio = 20%) 의 중앙 가로선
        - 그 영역의 depth 중 최솟값(=카메라에서 가장 가까운 = 객체 윗면) 위치
        bbox 중심으로 depth 측정 시 컵·병처럼 측면이 잡혀 윗면 z 가 정확하지 않음 →
        이 함수는 객체 윗면을 더 정확히 짚어준다.
        """
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape

        x1 = max(0, int(cx - w / 2))
        x2 = min(W, int(cx + w / 2))
        y1 = max(0, int(cy - h / 2))
        # 상단 top_ratio 만큼만
        y2 = min(H, int(y1 + max(3, h * top_ratio)))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return int(cx), int(cy), self._depth_at(depth_frame, cx, cy)

        win = arr[y1:y2, x1:x2]
        valid = win[win > 0]
        if valid.size < 10:
            return int(cx), int(cy), self._depth_at(depth_frame, cx, cy)

        # 가장 가까운 점 = 객체 윗면 추정
        min_depth_mm = float(valid.min())
        # 그 픽셀 위치 찾기
        local_yx = np.argwhere(win == min_depth_mm)
        if local_yx.size == 0:
            return int(cx), int(cy), float(min_depth_mm) * 0.001
        ly, lx = local_yx[0]
        u_top = x1 + int(lx)
        v_top = y1 + int(ly)
        return u_top, v_top, float(min_depth_mm) * 0.001

    def detect(self, target_classes, save_debug=None):
        """
        YOLO-World open-vocabulary 검출.
        target_classes: 검출하고 싶은 텍스트 리스트 (예: ['pen', 'wood block']). 필수.
        반환: dict {base_xyz_mm, label, conf, pixel, depth_m} 또는 None.
        """
        if not target_classes:
            return None

        # 안전: 사람·동물 류 단어는 검출 클래스로 자체 등록을 거부 (대소문자 무관)
        safe = [t for t in target_classes if t.strip().lower() not in BLOCKED_CLASSES]
        if not safe:
            print(f'  !! 차단된 클래스만 입력됨 ({target_classes}). BLOCKED_CLASSES 참조.')
            return None

        # World 는 매 호출마다 set_classes 로 클래스를 동적 지정해야 함.
        # set_classes 가 CLIP 인코더를 새로 만들면서 CUDA 텐서를 만들 수 있어,
        # 호출 후 다시 .cpu() 로 device 통일을 강제한다.
        self.model.set_classes(safe)
        self.model.cpu()
        self.classes = {i: n for i, n in enumerate(safe)}

        frames = self.align.process(self.pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return None
        color = np.asanyarray(cf.get_data())
        dets = self._yolo_detect(color)
        if not dets:
            return None

        cls_id, conf, cx, cy, w, h = dets[0]
        u, v = int(cx), int(cy)
        # bbox 상단 영역 depth 로 객체 윗면 z 정확히 측정 (bbox 중심은 측면 잡힘)
        u_top, v_top, depth_m = self._top_pixel_and_depth(df, cx, cy, w, h)
        if depth_m <= 0.05:
            return None

        cam_xyz_m = rs.rs2_deproject_pixel_to_point(self.intr, [u_top, v_top], depth_m)
        cam_h = np.array([cam_xyz_m[0], cam_xyz_m[1], cam_xyz_m[2], 1.0])
        base_h = self.T_cam2base @ cam_h
        base_mm = (base_h[:3] * 1000.0).astype(float)
        label = self.classes.get(cls_id, f'id{cls_id}')

        if save_debug:
            dbg = color.copy()
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(dbg, (u, v), 5, (0, 0, 255), -1)            # bbox 중심
            cv2.circle(dbg, (u_top, v_top), 5, (0, 255, 255), -1)  # 윗면 측정 지점
            cv2.putText(dbg, f'{label} {conf:.2f} d={depth_m*1000:.0f}mm',
                        (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(save_debug, dbg)

        return {
            'base_xyz_mm': (float(base_mm[0]), float(base_mm[1]), float(base_mm[2])),
            'label': label, 'conf': conf,
            'pixel': (u_top, v_top), 'depth_m': depth_m,
        }

    def interactive_preview(self, target_classes, save_debug=None):
        """
        라이브 카메라 창을 띄우고 매 프레임 YOLO-World 검출 결과를 오버레이.
        사용자가 SPACE 또는 Enter 를 누르면 그 시점의 best 검출을 반환,
        q 또는 ESC 면 None 반환. 검출이 없을 때 SPACE/Enter 누르면 None.
        반환 dict 형식은 detect() 와 동일.
        """
        if not target_classes:
            return None
        safe = [t for t in target_classes if t.strip().lower() not in BLOCKED_CLASSES]
        if not safe:
            print(f'  !! 차단된 클래스만 입력됨 ({target_classes}). BLOCKED_CLASSES 참조.')
            return None

        # CLIP 인코딩 (첫 호출은 ~3-5초, 이후는 빠름)
        print('  CLIP 텍스트 인코딩 중...', end='', flush=True)
        self.model.set_classes(safe)
        self.model.cpu()
        self.classes = {i: n for i, n in enumerate(safe)}
        print(' 완료')

        win = 'Detection Preview  [SPACE/Enter] proceed   [q/ESC] cancel'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last_det = None

        try:
            while True:
                frames = self.align.process(self.pipeline.wait_for_frames())
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if not cf or not df:
                    continue
                color = np.asanyarray(cf.get_data())
                disp = color.copy()
                H, W = disp.shape[:2]

                dets = self._yolo_detect(color)

                # 항상 최상단에 캘리브레이션 / 명령 정보 띄움
                cv2.putText(disp, f'targets: {safe}',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(disp,
                            f'T_cam2base loaded  (mean err {self.calib_err_mm:.1f}mm)',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

                last_det = None

                # 모든 후보 약하게, best 는 진하게 그림
                for i, (cls_id, conf, cx, cy, w, h) in enumerate(dets):
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    label = self.classes.get(cls_id, '?')
                    if i == 0:
                        # best — 색깔 진하게 + 좌표 변환
                        u, v = int(cx), int(cy)
                        depth_m = self._depth_at(df, u, v)
                        if depth_m > 0.05:
                            cam_xyz_m = rs.rs2_deproject_pixel_to_point(
                                self.intr, [u, v], depth_m)
                            cam_h = np.array([cam_xyz_m[0], cam_xyz_m[1], cam_xyz_m[2], 1.0])
                            base_h = self.T_cam2base @ cam_h
                            base_mm = (base_h[:3] * 1000.0).astype(float)
                            last_det = {
                                'base_xyz_mm': (float(base_mm[0]), float(base_mm[1]), float(base_mm[2])),
                                'label': label, 'conf': conf,
                                'pixel': (u, v), 'depth_m': depth_m,
                            }
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.circle(disp, (u, v), 6, (0, 0, 255), -1)
                            cv2.putText(disp, f'{label} {conf:.2f}',
                                        (x1, max(20, y1 - 30)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(disp, f'depth={depth_m*1000:.0f}mm',
                                        (x1, max(40, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                            base_str = (f'base xyz=({base_mm[0]:+.0f}, '
                                        f'{base_mm[1]:+.0f}, {base_mm[2]:+.0f})mm')
                            in_work = (WORK_X[0] <= base_mm[0] <= WORK_X[1] and
                                       WORK_Y[0] <= base_mm[1] <= WORK_Y[1] and
                                       WORK_Z[0] <= base_mm[2] <= WORK_Z[1])
                            color_txt = (0, 255, 0) if in_work else (0, 0, 255)
                            cv2.putText(disp, base_str + ('' if in_work else '  [OUT OF WORKAREA]'),
                                        (10, H - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_txt, 2)
                        else:
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(disp, f'{label} {conf:.2f} (depth invalid)',
                                        (x1, max(20, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # 다른 후보 — 옅게
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (180, 180, 180), 1)
                        cv2.putText(disp, f'{label} {conf:.2f}',
                                    (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                if last_det is None:
                    cv2.putText(disp, 'no valid detection',
                                (10, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                cv2.putText(disp, '[SPACE/Enter] proceed   [q/ESC] cancel',
                            (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                cv2.imshow(win, disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(' '), 13, 10):       # SPACE / Enter
                    if last_det is None:
                        print('  유효한 검출이 없어 진행할 수 없습니다.')
                        return None
                    if save_debug:
                        cv2.imwrite(save_debug, disp)
                    return last_det
                if key in (ord('q'), 27):           # q / ESC
                    return None
        finally:
            # 카메라 창은 사이클 끝나도 유지 (마지막 frame). 다음 호출 때 다시 라이브.
            cv2.waitKey(1)

    def interactive_browse(self, candidate_classes=None, save_debug=None):
        """
        라이브 카메라 + 멀티 객체 트래킹.
        - 한 번 검출된 객체는 슬롯 번호([1]~[9])와 첫 측정 좌표를 고정
        - 매 프레임에는 라이브 픽셀로 박스만 따라감 (좌표는 고정)
        - 새 객체가 들어오면 다음 빈 슬롯 부여
        - 일시적으로 사라져도 슬롯·좌표 유지 (잠시 표시 안 함, lock 그대로)
        - 1~9 키: 해당 슬롯 선택 (사라진 상태에서도 저장된 좌표로 잡으러 감)
        - SPACE/Enter: conf 가 가장 높았던 valid 슬롯 자동 선택
        - q/ESC: 취소

        매칭 기준: 같은 label + pixel 거리 < MATCH_PX
        """
        if candidate_classes is None:
            candidate_classes = COMMON_OBJECTS
        safe = [t for t in candidate_classes if t.strip().lower() not in BLOCKED_CLASSES]
        if not safe:
            return None

        print(f'  CLIP 텍스트 인코딩 ({len(safe)} 클래스)...', end='', flush=True)
        self.model.set_classes(safe)
        self.model.cpu()
        self.classes = {i: n for i, n in enumerate(safe)}
        print(' 완료')

        win = 'Object Tracking  [1-9]select  [r]reset  [SPACE]auto  [q/ESC]cancel'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)   # 항상 위
        except Exception:
            pass
        print('  ⓘ 카메라 창에 마우스 클릭으로 포커스 주고 키 입력하세요')
        print('     숫자키 1~9 / r / SPACE / Enter / q / ESC')

        # 슬롯: sid(1..9) -> dict {label, base_xyz_mm(고정), pixel_lock(고정),
        #                          conf_max, in_work, depth_m,
        #                          pixel_live, w_live, h_live, alive_this_frame}
        slots = {}
        MATCH_PX = 100.0
        MAX_SLOTS = 9

        def _next_free_slot():
            for i in range(1, MAX_SLOTS + 1):
                if i not in slots:
                    return i
            return None

        try:
            while True:
                frames = self.align.process(self.pipeline.wait_for_frames())
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if not cf or not df:
                    continue
                color = np.asanyarray(cf.get_data())
                disp = color.copy()
                H, W = disp.shape[:2]

                # 이번 프레임 시작: 모든 슬롯 alive flag 초기화
                for sd in slots.values():
                    sd['alive_this_frame'] = False

                dets = self._yolo_detect(color)
                matched_sids = set()

                for cls_id, conf, cx, cy, w, h in dets:
                    u, v = int(cx), int(cy)
                    label = self.classes.get(cls_id, '?')

                    # 같은 라벨 + 가까운 픽셀의 기존 슬롯 찾기
                    best_sid, best_d = None, float('inf')
                    for sid, sd in slots.items():
                        if sid in matched_sids:
                            continue
                        if sd['label'] != label:
                            continue
                        du = sd['pixel_live'][0] - u
                        dv = sd['pixel_live'][1] - v
                        d = (du*du + dv*dv) ** 0.5
                        if d < MATCH_PX and d < best_d:
                            best_sid, best_d = sid, d

                    if best_sid is not None:
                        # 기존 슬롯 라이브 정보만 업데이트, 좌표·번호 고정
                        sd = slots[best_sid]
                        sd['pixel_live'] = (u, v)
                        sd['w_live'] = w
                        sd['h_live'] = h
                        sd['conf'] = conf
                        if conf > sd.get('conf_max', 0):
                            sd['conf_max'] = conf
                        sd['alive_this_frame'] = True
                        matched_sids.add(best_sid)
                    else:
                        # 새 객체 — bbox 상단 영역 depth 로 객체 윗면 z 정확히 측정
                        u_top, v_top, depth_top_m = self._top_pixel_and_depth(df, cx, cy, w, h)
                        if depth_top_m <= 0.05:
                            continue
                        cam_xyz = rs.rs2_deproject_pixel_to_point(
                            self.intr, [u_top, v_top], depth_top_m)
                        cam_h = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0])
                        base_h = self.T_cam2base @ cam_h
                        base_mm = (base_h[:3] * 1000.0).astype(float)
                        in_work = (WORK_X[0] <= base_mm[0] <= WORK_X[1] and
                                   WORK_Y[0] <= base_mm[1] <= WORK_Y[1] and
                                   WORK_Z[0] <= base_mm[2] <= WORK_Z[1])
                        sid = _next_free_slot()
                        if sid is None:
                            continue
                        slots[sid] = {
                            'label': label, 'conf': conf, 'conf_max': conf,
                            'in_work': in_work,
                            'depth_m': depth_top_m,
                            'base_xyz_mm': (float(base_mm[0]), float(base_mm[1]), float(base_mm[2])),
                            'pixel_lock': (u_top, v_top),   # 윗면 지점
                            'pixel_live': (u, v),           # bbox 중심 (라이브 박스 추적용)
                            'w_live': w, 'h_live': h,
                            'alive_this_frame': True,
                        }
                        matched_sids.add(sid)

                # 그리기: 모든 등록 슬롯 (alive 면 진하게, lost 면 어둡게)
                for sid in sorted(slots.keys()):
                    sd = slots[sid]
                    alive = sd['alive_this_frame']
                    u, v = sd['pixel_live']
                    w_, h_ = sd['w_live'], sd['h_live']
                    x1, y1 = int(u - w_/2), int(v - h_/2)
                    x2, y2 = int(u + w_/2), int(v + h_/2)
                    in_work = sd['in_work']
                    if alive:
                        box_color = (0, 255, 0) if in_work else (0, 140, 255)
                        thickness = 2
                    else:
                        box_color = (90, 90, 90)   # 잃어버린 슬롯 — 짙은 회색
                        thickness = 1
                    cv2.rectangle(disp, (x1, y1), (x2, y2), box_color, thickness)
                    info = f'[{sid}] {sd["label"]} {sd["conf_max"]:.2f}'
                    if not alive:
                        info += ' (lost)'
                    elif not in_work:
                        info += ' OUT'
                    cv2.putText(disp, info, (x1, max(20, y1 - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                    bx, by, bz = sd['base_xyz_mm']
                    sub = f'xyz=({bx:+.0f},{by:+.0f},{bz:+.0f}) [locked]'
                    cv2.putText(disp, sub, (x1, max(40, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

                # 상단·하단 안내
                cv2.putText(disp,
                            f'targets: {len(safe)} classes  |  tracked: {len(slots)}/9  '
                            f'(green=alive, gray=lost)',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                cv2.putText(disp,
                            f'T_cam2base loaded (mean err {self.calib_err_mm:.1f}mm)',
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(disp,
                            '[1-9] select   [r] reset tracker   [SPACE/Enter] auto   [q/ESC] cancel',
                            (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                cv2.imshow(win, disp)
                key = cv2.waitKey(30) & 0xFF   # 30ms 대기 — 키 입력 안정성↑

                if key == 255:   # 키 입력 없음
                    continue

                # 디버그: 어떤 키가 들어왔는지 콘솔에 표시 (포커스/IME 진단)
                key_chr = chr(key) if 32 <= key < 127 else f'0x{key:02x}'
                print(f'  [key] {key} ({key_chr!r})')

                if ord('1') <= key <= ord('9'):
                    sid = key - ord('1') + 1
                    if sid in slots:
                        sd = slots[sid]
                        if save_debug:
                            cv2.imwrite(save_debug, disp)
                        print(f'  → 슬롯 {sid} 선택: {sd["label"]} xyz={sd["base_xyz_mm"]}')
                        return {
                            'base_xyz_mm': sd['base_xyz_mm'],
                            'label': sd['label'],
                            'conf': sd['conf_max'],
                            'pixel': sd['pixel_lock'],
                            'depth_m': sd['depth_m'],
                        }
                    else:
                        print(f'  (슬롯 {sid} 비어있음 — 화면의 표시된 번호만 선택 가능)')
                        continue
                if key in (ord(' '), 13, 10):
                    best = None
                    for sd in slots.values():
                        if not sd['in_work']:
                            continue
                        if best is None or sd['conf_max'] > best['conf_max']:
                            best = sd
                    if best is not None:
                        if save_debug:
                            cv2.imwrite(save_debug, disp)
                        print(f'  → AUTO 선택: {best["label"]} xyz={best["base_xyz_mm"]}')
                        return {
                            'base_xyz_mm': best['base_xyz_mm'],
                            'label': best['label'],
                            'conf': best['conf_max'],
                            'pixel': best['pixel_lock'],
                            'depth_m': best['depth_m'],
                        }
                    print('  (AUTO 선택할 valid(in_work) 슬롯 없음)')
                if key == ord('r'):
                    slots.clear()
                    print('  (트래커 reset — 슬롯 다시 1부터)')
                if key in (ord('q'), 27):
                    return None
        finally:
            # 사용자 요청: 카메라 창은 사이클 동안 계속 보이도록 유지.
            # 마지막 frame 이 화면에 남는다 (라이브 갱신은 다음 scan/<이름> 호출 때 재개).
            cv2.waitKey(1)   # 큐 펌프 — 창이 freeze 됐다고 표시되는 걸 약간 완화


# ============== 로봇 + 그리퍼 컨트롤러 ==============
class PickAndPlace(Node):
    def __init__(self):
        super().__init__('vision_pick_and_place')
        self.cli_mode = self.create_client(SetRobotMode, f'/{NS}/system/set_robot_mode')
        self.cli_ctrl = self.create_client(SetRobotControl, f'/{NS}/system/set_robot_control')
        # 현재 TCP/관절 자세 조회 (자세 캘리브레이션용)
        self.cli_get_pose = self.create_client(GetCurrentPose, f'/{NS}/system/get_current_pose')
        # Safety / 충돌 후 fault recovery
        self.cli_get_state   = self.create_client(GetRobotState,         f'/{NS}/system/get_robot_state')
        self.cli_safety_mode = self.create_client(SetSafetyMode,         f'/{NS}/system/set_safety_mode')
        self.cli_safe_reset  = self.create_client(SetSafeStopResetType,  f'/{NS}/system/set_safe_stop_reset_type')
        # 모션 실행 — MoveIt 컨트롤러의 FollowJointTrajectory action
        # (`/dsr01/motion/move_*` 직접 서비스는 MoveIt mode 에서 거짓 success 반환하므로 X)
        self.traj_action = ActionClient(
            self, FollowJointTrajectory,
            f'/{NS}/{MOVEIT_CONTROLLER}/follow_joint_trajectory'
        )
        # 현재 활성 top-down 자세 (사용자가 cal_top_down 으로 갱신 가능)
        self.top_down_rpy = list(TOP_DOWN_RPY_DEG)
        # 그리퍼 TCP z 오프셋 (런타임에 set_tcp_z 명령으로 갱신 가능)
        self.tcp_z_offset = float(GRIPPER_TCP_OFFSET_Z)
        # cuRobo planner — main 에서 init 후 주입
        self.planner = None
        gp = f'/{NS}/gripper'
        self.cli_g_open = self.create_client(FlangeSerialOpen, f'{gp}/flange_serial_open')
        self.cli_g_close = self.create_client(FlangeSerialClose, f'{gp}/flange_serial_close')
        self.cli_g_write = self.create_client(FlangeSerialWrite, f'{gp}/flange_serial_write')
        self._gripper_serial_open = False

    def _wait(self, cli, name, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'서비스 미응답: {name}')

    def _call(self, cli, req, timeout=30.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    # ---- 시스템 / Safety ----
    _STATE_NAMES = {
        0: 'INITIALIZING', 1: 'STANDBY', 2: 'MOVING',
        3: 'SAFE_OFF', 4: 'TEACHING', 5: 'SAFE_STOP',
        6: 'EMERGENCY_STOP', 7: 'HOMING', 8: 'RECOVERY',
        9: 'SAFE_STOP2', 10: 'SAFE_OFF2', 15: 'NOT_READY',
    }

    def get_robot_state(self):
        """현재 robot_state 정수 반환. 실패 시 -1."""
        if not self.cli_get_state.wait_for_service(timeout_sec=2.0):
            return -1
        r = self._call(self.cli_get_state, GetRobotState.Request(), timeout=3.0)
        return r.robot_state if r else -1

    def recover_safety(self, verbose=True):
        """
        충돌·safe-stop 후 fault 복구.
          - SAFE_STOP / SAFE_STOP2 : RECOVERY 진입 → safe-stop reset → AUTONOMOUS 복귀
          - SAFE_OFF / SAFE_OFF2   : Servo OFF 상태 → SetRobotControl(1) 로 재 ON
          - EMERGENCY_STOP         : 펜던트의 비상정지 버튼 — 코드로 reset 불가, 사용자 안내
          - 그 외(STANDBY 등)      : 그대로 통과
        """
        s = self.get_robot_state()
        name = self._STATE_NAMES.get(s, f'UNKNOWN({s})')
        if verbose:
            print(f'   robot_state={s} ({name})')

        if s in (5, 9):   # SAFE_STOP / SAFE_STOP2 — 충돌·protective stop
            if verbose:
                print('   safe-stop 감지 → recovery 시퀀스 실행')
            for cli, name_ in [(self.cli_safety_mode, 'safety_mode'),
                               (self.cli_safe_reset, 'safe_stop_reset')]:
                if not cli.wait_for_service(timeout_sec=2.0):
                    print(f'   서비스 미응답: {name_}')
                    return False
            # 1) RECOVERY 진입
            m = SetSafetyMode.Request(); m.safety_mode = 2; m.safety_event = 0
            self._call(self.cli_safety_mode, m, timeout=3.0)
            time.sleep(0.5)
            # 2) safe-stop reset (program stop)
            rs = SetSafeStopResetType.Request(); rs.reset_type = 0
            self._call(self.cli_safe_reset, rs, timeout=3.0)
            time.sleep(0.5)
            # 3) AUTONOMOUS 복귀
            m2 = SetSafetyMode.Request(); m2.safety_mode = 1; m2.safety_event = 0
            self._call(self.cli_safety_mode, m2, timeout=3.0)
            time.sleep(0.5)
            new_s = self.get_robot_state()
            if verbose:
                print(f'   recovery 후 robot_state={new_s} ({self._STATE_NAMES.get(new_s, "?")})')
            return new_s in (1, 2)   # STANDBY/MOVING 이면 OK
        if s == 6:    # EMERGENCY_STOP — 펜던트 비상정지
            print('   !! EMERGENCY_STOP 상태 — 펜던트의 비상정지 버튼을 풀고 재시도하세요.')
            return False
        if s in (3, 10):   # SAFE_OFF / SAFE_OFF2
            if verbose:
                print('   Servo OFF 감지 → 재활성 시도')
            return True   # 다음 단계의 SetRobotControl(1) 가 처리
        return True   # 정상 (STANDBY 등)

    def activate_robot(self):
        """
        충돌·safe-stop 후 자동 복구 + AUTONOMOUS + SERVO_ON.
        매 실행 시작 때 호출되며 idempotent (이미 정상이면 빠르게 통과).
        """
        self._wait(self.cli_mode, 'set_robot_mode')
        self._wait(self.cli_ctrl, 'set_robot_control')

        # 0) 충돌·safe-stop 자동 복구
        self.recover_safety(verbose=True)

        # 1) AUTONOMOUS
        m = SetRobotMode.Request()
        m.robot_mode = 1
        r = self._call(self.cli_mode, m)
        if not (r and r.success):
            print('   (이미 AUTONOMOUS 모드일 가능성 — 계속 진행)')

        # 2) SERVO ON
        c = SetRobotControl.Request()
        c.robot_control = 1
        r = self._call(self.cli_ctrl, c)
        if not (r and r.success):
            print('   (이미 Servo ON 상태일 가능성 — 계속 진행)')

        # 3) 활성화 후 상태 한 번 더 확인
        final_s = self.get_robot_state()
        print(f'   활성화 후 robot_state={final_s} ({self._STATE_NAMES.get(final_s,"?")})')

    # ---- 그리퍼 ----
    def gripper_init(self):
        """시리얼을 한 번 open + 토크 enable 하고 살려둔다.
        Trajectory 모션과 섞일 때도 시리얼 핸들이 살아있도록 유지하고,
        gripper_set() 은 매번 토크 enable + 위치 명령만 추가로 보낸다.
        매 cycle close/open 패턴은 일부 환경에서 그리퍼 무응답 유발."""
        for cli, name in [(self.cli_g_open, 'flange_serial_open'),
                          (self.cli_g_close, 'flange_serial_close'),
                          (self.cli_g_write, 'flange_serial_write')]:
            self._wait(cli, name)

        # stale handle 정리
        try:
            pre_close = FlangeSerialClose.Request()
            pre_close.port = GRIPPER_PORT
            self._call(self.cli_g_close, pre_close, timeout=3.0)
        except Exception:
            pass
        time.sleep(0.3)

        # open with retry
        for attempt in range(3):
            req = FlangeSerialOpen.Request()
            req.port = GRIPPER_PORT
            req.baudrate = GRIPPER_BAUD
            req.bytesize = 8
            req.parity = 0
            req.stopbits = 1
            r = self._call(self.cli_g_open, req)
            if r and r.success:
                break
            try:
                rc = FlangeSerialClose.Request(); rc.port = GRIPPER_PORT
                self._call(self.cli_g_close, rc, timeout=2.0)
            except Exception:
                pass
            time.sleep(0.3 + attempt * 0.2)
        else:
            raise RuntimeError('그리퍼 시리얼 open 실패 — 컨트롤러/케이블 점검 필요')
        self._gripper_serial_open = True
        time.sleep(0.1)

        # 토크 enable
        w = FlangeSerialWrite.Request()
        w.port = GRIPPER_PORT
        w.data = fc06_torque_enable()
        self._call(self.cli_g_write, w)
        time.sleep(0.2)
        # 시리얼은 열어둔 채로 유지

    def gripper_set(self, pos: int, settle=1.0, label=''):
        """
        그리퍼 위치 명령. 시리얼은 init 에서 열려있다고 가정 — close/open 사이클 안 함.
        매 호출에서 토크 enable 을 한 번 더 보낸 후 위치 명령. trajectory 가
        다이나믹셀 토크를 풀 수 있어 안전상 매번 다시 enable.
        """
        tag = f'[{label}]' if label else ''
        if not self._gripper_serial_open:
            print(f'  !! 그리퍼{tag} 시리얼이 열리지 않음 — gripper_init 실패 가능')
            return False

        # 1) 토크 enable (매번)
        w_t = FlangeSerialWrite.Request()
        w_t.port = GRIPPER_PORT
        w_t.data = fc06_torque_enable()
        rt = self._call(self.cli_g_write, w_t, timeout=3.0)
        if not (rt and rt.success):
            print(f'  !! 그리퍼{tag} 토크 enable 실패')
        time.sleep(0.15)

        # 2) 위치 명령
        w_p = FlangeSerialWrite.Request()
        w_p.port = GRIPPER_PORT
        w_p.data = fc16_position(pos)
        rp = self._call(self.cli_g_write, w_p, timeout=3.0)
        if not (rp and rp.success):
            print(f'  !! 그리퍼{tag} 위치 명령 실패 (pos={pos})')
            return False
        time.sleep(settle)
        return True

    def gripper_open(self):
        return self.gripper_set(GRIP_OPEN_POS, label='OPEN')

    def gripper_close(self):
        return self.gripper_set(GRIP_CLOSE_POS, label='CLOSE')

    def gripper_shutdown(self):
        if not self._gripper_serial_open:
            return
        req = FlangeSerialClose.Request()
        req.port = GRIPPER_PORT
        self._call(self.cli_g_close, req)
        self._gripper_serial_open = False

    # ---- 진단: 현재 자세 조회 ----
    def get_current_pose(self, space='task'):
        """space='task' → 현재 TCP [x,y,z,rx,ry,rz] 반환, 'joint' → 6 관절각."""
        self._wait(self.cli_get_pose, 'system/get_current_pose')
        req = GetCurrentPose.Request()
        req.space_type = 1 if space == 'task' else 0
        r = self._call(self.cli_get_pose, req, timeout=3.0)
        if not (r and r.success):
            raise RuntimeError('GetCurrentPose 실패')
        return list(r.pos)

    # ---- 모션 (cuRobo plan → MoveSplineJoint 실행) ----
    def _current_joints_rad(self):
        """현재 6 관절각 (rad) — curobo plan_single 의 start_state 입력용."""
        deg = self.get_current_pose('joint')
        return [float(np.deg2rad(a)) for a in deg]

    def _check_joint_limits(self, traj_deg):
        """trajectory 의 어떤 포인트라도 SAFE_JOINT_LIMITS_DEG 밖이면 raise."""
        for k, pt in enumerate(traj_deg):
            for i, j in enumerate(pt):
                lo, hi = SAFE_JOINT_LIMITS_DEG[i]
                if not (lo <= j <= hi):
                    raise RuntimeError(
                        f'관절 한계 초과: point {k} joint_{i+1}={j:.2f}∉[{lo:.0f},{hi:.0f}]. '
                        f'다른 자세/위치로 재시도하세요.'
                    )

    def _send_trajectory_action(self, joints_deg, duration=MOVE_DURATION_SEC):
        """6축 관절 각도(도) 단일 목표를 FollowJointTrajectory 액션으로 전송.
        MoveIt mode 의 dsr_bringup 환경에서는 직접 motion/move_* 서비스가 거짓 success
        반환하므로 trajectory action 만이 실제로 robot 을 움직임."""
        if not self.traj_action.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                'FollowJointTrajectory 액션 서버 미응답 — '
                'dsr_bringup2_moveit.launch.py 가 살아있는지 확인'
            )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = [math.radians(float(a)) for a in joints_deg]
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [point]

        send_fut = self.traj_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            raise RuntimeError('Trajectory 골 거부됨 (Servo OFF / 한계 초과 가능)')
        result_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=duration + 30.0)
        status = result_fut.result().status
        if status != 4:    # GoalStatus.STATUS_SUCCEEDED
            raise RuntimeError(f'Trajectory 실행 실패 (status={status})')

    def _execute_spline(self, traj_rad):
        """
        cuRobo trajectory 의 마지막 점만 FollowJointTrajectory 액션으로 실행.
        cuRobo plan 은 도달가능성·관절한계·충돌 검증 역할.
        실제 모션은 ros2_control + dsr_moveit_controller 가 현재 위치 → 끝점 보간.
        """
        traj_deg = np.rad2deg(np.asarray(traj_rad))
        self._check_joint_limits(traj_deg)
        start_deg = traj_deg[0].tolist()
        final_deg = traj_deg[-1].tolist()
        delta = [round(float(final_deg[i] - start_deg[i]), 2) for i in range(6)]
        max_change = max(abs(d) for d in delta)

        print(f'   trajectory end (deg): {[round(v,2) for v in final_deg]}')
        print(f'   delta (deg)         : {delta}  (max |Δ|={max_change:.2f})')
        if max_change < 0.5:
            print(f'   ⚠️  최대 관절 변화 {max_change:.3f}° < 0.5° — 사실상 같은 자세')

        t0 = time.time()
        self._send_trajectory_action(final_deg)
        print(f'   trajectory 완료, elapsed={(time.time()-t0)*1000:.0f}ms')

    def move_joint_deg(self, joints_deg, duration=MOVE_DURATION_SEC):
        """단일 관절 목표 (HOME 등). FollowJointTrajectory action 사용."""
        self._send_trajectory_action(list(joints_deg), duration)

    def move_line_base(self, xyz_mm, rpy_deg=None):
        """베이스 좌표(mm/deg) → cuRobo plan → MoveSplineJoint 로 실행.
        rpy_deg 미지정 시 self.top_down_rpy 사용. 그리퍼 끝 좌표 기준."""
        if self.planner is None:
            raise RuntimeError('CuroboPlanner 미초기화 — [준비 2/4] 단계 확인')
        if rpy_deg is None:
            rpy_deg = self.top_down_rpy

        x, y, z_user = xyz_mm
        # 그리퍼 끝 → 플랜지 보정 (curobo ee_link 는 플랜지에 직결된 그리퍼 베이스)
        z = z_user + self.tcp_z_offset

        # 작업 영역 가드 (그리퍼 끝 기준)
        if not (WORK_X[0] <= x <= WORK_X[1] and
                WORK_Y[0] <= y <= WORK_Y[1] and
                WORK_Z[0] <= z_user <= WORK_Z[1]):
            raise RuntimeError(
                f'좌표 {[round(v,1) for v in (x,y,z_user)]} mm (그리퍼 끝 기준) 가 작업영역 밖. '
                f'X∈{WORK_X}, Y∈{WORK_Y}, Z∈{WORK_Z}.'
            )

        # 단위 변환: curobo 는 미터 + 쿼터니언(wxyz)
        target_xyz_m = [x * 1e-3, y * 1e-3, z * 1e-3]
        quat_wxyz = _rpy_deg_to_quat_wxyz(rpy_deg)
        start_joints_rad = self._current_joints_rad()

        traj_rad, plan_ms = self.planner.plan(start_joints_rad, target_xyz_m, quat_wxyz)
        if traj_rad is None:
            raise RuntimeError(
                f'cuRobo plan 실패 (target={[round(v,1) for v in (x,y,z)]}mm '
                f'rpy={[round(v,1) for v in rpy_deg]}). '
                '도달 불가/충돌 가능. world_clear 또는 다른 위치/자세 시도.'
            )
        print(f'   cuRobo plan OK ({plan_ms:.0f}ms, {len(traj_rad)} pts)')
        self._execute_spline(traj_rad)


# ============== 드라이버 리셋 + relaunch ==============
def reset_robot_driver():
    """
    매 실행 시작 시 깔끔한 상태로 진입:
      1) 잔존 ROS / RealSense / 본 강의 스크립트 프로세스 모두 정리
      2) 새 gnome-terminal 에서 dsr_bringup2_moveit.launch.py 실행
      3) 안정화 대기 후 반환

    이렇게 하면 두산 컨트롤러 측 시리얼 stale handle, RealSense 점유,
    여러 노드 동시 실행 충돌이 한 번에 해결된다.
    """
    print('\n[준비 0/4] 기존 ROS / 카메라 / 강의 스크립트 정리 + 두산 드라이버 새로 시작')

    cleanup_cmds = [
        'pkill -9 -f dsr',
        'pkill -9 -f rviz2',
        'pkill -9 -f DRCF',
        'pkill -9 -f ros2',
        'pkill -9 -f realsense',
    ]
    for c in cleanup_cmds:
        os.system(c + ' 2>/dev/null')

    # 한글 파일명은 pkill -f 패턴이 매칭 안 되는 경우가 있어 PID 직접 추출
    self_pid = str(os.getpid())
    extract = (
        "ps -ef | grep -E '(08_카메라_핸드아이|09_원샷|10_그리퍼_연결|"
        "11_그리퍼_키보드|13_|14_)' | grep -v grep | "
        f"awk '{{if($2!={self_pid}) print $2}}'"
    )
    pids = subprocess.run(['bash', '-c', extract], capture_output=True, text=True).stdout.strip()
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

    # 드라이버 부팅 후 RViz 실행. GPU 드라이버 SEGFAULT 회피를 위해 software rendering(LIBGL_ALWAYS_SOFTWARE=1)
    # 을 default 로 사용 — hardware accel 보다 약간 느리지만 강의 데모용으로는 안정성이 우선.
    # setsid 로 세션 분리해 RViz 가 죽어도 메인 콘솔이 깨끗하게 유지.
    if os.path.exists(RVIZ_CONFIG):
        rviz_cmd = (
            'source /opt/ros/humble/setup.bash && '
            'source ~/doosan_ws/install/setup.bash && '
            f'LIBGL_ALWAYS_SOFTWARE=1 exec rviz2 -d {RVIZ_CONFIG}'
        )
        subprocess.Popen(
            ['bash', '-lc', rviz_cmd],
            start_new_session=True,
            stdout=open('/tmp/rviz2_dsr.log', 'w'),
            stderr=subprocess.STDOUT,
        )
        print('   RViz 백그라운드 실행 (software rendering, config: dsr_moveit_config_e0509)')
        print('   → RViz 로그: /tmp/rviz2_dsr.log')
    else:
        print(f'   (RViz config 미발견: {RVIZ_CONFIG} — 시각화 생략)')


# ============== 메인 ==============
def banner(text):
    print('\n' + '=' * 70)
    print(f'  {text}')
    print('=' * 70)


def execute_pick_and_place(robot, det, args):
    """검출 결과를 받아 Pick → Place → HOME 까지 한 사이클 실행.

    모드 자동 분기:
      - SIDE   : 컵·병 류 (SIDE_CLASSES) — 그리퍼가 옆으로 회전 + 옆에서 접근
      - FLAT   : 검출 z 가 낮음 (펜·동전 등) — top-down 그리퍼 끝
      - NORMAL : 그 외 — top-down 핑거 가운데, 윗면 아래로 dive
    """
    bx, by, bz = det['base_xyz_mm']
    label = (det.get('label') or '').strip().lower()

    if label in SIDE_CLASSES:
        try:
            _execute_side_grasp(robot, det, args)
            return
        except RuntimeError as e:
            print(f'  !! SIDE-grasp 실패 — TOP-DOWN dive 모드로 fallback')
            print(f'     사유: {e}')
            # 아래 TOP-DOWN 분기로 흘러 다시 시도

    # ---- TOP-DOWN 모드 (HOLLOW / FLAT / NORMAL) ----
    # 우선순위: 클래스 매칭(HOLLOW) > 검출 z(FLAT) > 그 외(NORMAL)
    if label in HOLLOW_CLASSES:
        robot.tcp_z_offset = GRIPPER_TCP_OFFSET_NORMAL
        grasp_dive = GRASP_DIVE_HOLLOW_MM
        print(f'  ⓘ 속 빈 객체 ({label}, z={bz:.1f}mm) '
              f'→ HOLLOW (tcp_z={GRIPPER_TCP_OFFSET_NORMAL:.0f}mm, dive={grasp_dive:.0f}mm — '
              f'입구 가장자리 살짝 잡기)')
    elif bz < FLAT_OBJECT_Z_THRESHOLD:
        robot.tcp_z_offset = GRIPPER_TCP_OFFSET_FLAT
        grasp_dive = GRASP_DIVE_FLAT_MM
        print(f'  ⓘ 바닥 물체 (z={bz:.1f}mm < {FLAT_OBJECT_Z_THRESHOLD:.0f}) '
              f'→ FLAT (tcp_z={GRIPPER_TCP_OFFSET_FLAT:.0f}mm, dive={grasp_dive:.0f}mm)')
    else:
        robot.tcp_z_offset = GRIPPER_TCP_OFFSET_NORMAL
        grasp_dive = GRASP_DIVE_NORMAL_MM
        print(f'  ⓘ 일반 물체 ({label}, z={bz:.1f}mm) '
              f'→ NORMAL (tcp_z={GRIPPER_TCP_OFFSET_NORMAL:.0f}mm, dive={grasp_dive:.0f}mm — '
              f'측면 가운데 잡기)')

    grasp_z = bz - grasp_dive
    pick_xyz = [bx, by, grasp_z]
    approach_xyz = [bx, by, bz + Z_APPROACH]
    lifted_xyz = [bx, by, bz + Z_LIFT]
    place_xyz = [bx + PLACE_OFFSET_XYZ[0],
                 by + PLACE_OFFSET_XYZ[1],
                 grasp_z + PLACE_OFFSET_XYZ[2]]
    place_app_xyz = [place_xyz[0], place_xyz[1], bz + Z_APPROACH + PLACE_OFFSET_XYZ[2]]

    print('\n  -- Pick (TOP-DOWN) --')
    print(f'   1) Pre-grasp → {[round(v,1) for v in approach_xyz]} mm')
    if not args.dry_run: robot.move_line_base(approach_xyz)
    print('   2) 그리퍼 열기')
    if not args.dry_run: robot.gripper_open()
    print(f'   3) 하강 → {[round(v,1) for v in pick_xyz]} mm')
    if not args.dry_run: robot.move_line_base(pick_xyz)
    print('   4) 그리퍼 닫기 (잡기)')
    if not args.dry_run: robot.gripper_close()
    print(f'   5) 들어올리기 → {[round(v,1) for v in lifted_xyz]} mm')
    if not args.dry_run: robot.move_line_base(lifted_xyz)

    if args.no_place:
        print('\n  -- Place 생략 (--no-place) --')
    else:
        print(f'\n  -- Place (offset={PLACE_OFFSET_XYZ}mm) --')
        print(f'   1) Place 위로 → {[round(v,1) for v in place_app_xyz]} mm')
        if not args.dry_run: robot.move_line_base(place_app_xyz)
        print(f'   2) 하강 → {[round(v,1) for v in place_xyz]} mm')
        if not args.dry_run: robot.move_line_base(place_xyz)
        print('   3) 그리퍼 열기 (놓기)')
        if not args.dry_run: robot.gripper_open()
        print(f'   4) 들어올리기 → {[round(v,1) for v in place_app_xyz]} mm')
        if not args.dry_run: robot.move_line_base(place_app_xyz)

    print('\n  -- HOME 복귀 --')
    if not args.dry_run: robot.move_joint_deg(HOME_JOINT_DEG)


def _execute_side_grasp(robot, det, args):
    """
    SIDE-grasp 시퀀스 — 컵·병 같은 직립 원통형 객체용.
    그리퍼 자세를 옆으로 회전(ZYZ [yaw, 90, 0])하고 robot 쪽(base 가까이)에서
    객체 측면으로 접근. 핑거가 객체 옆면을 감쌈.

    좌표·자세 컨벤션:
      - yaw_to_obj  = atan2(by, bx)  (base 원점에서 객체로 향하는 XY 방향)
      - 단위벡터 (ux, uy) = (cos(yaw), sin(yaw))  ← robot에서 객체로 향하는 방향
      - Pre-grasp  = 객체 − (L + SIDE_APPROACH) · (ux, uy, 0)   ← robot 쪽으로 후퇴
      - Grasp(TCP) = 객체 − L · (ux, uy, 0)                      ← 그리퍼 끝이 객체 도달
      - 자세 RPY (deg, ZYZ) = [yaw_deg, 90, 0]
    """
    bx, by, bz = det['base_xyz_mm']
    label = det.get('label', '?')

    yaw_rad = math.atan2(by, bx)
    yaw_deg = math.degrees(yaw_rad)
    ux = math.cos(yaw_rad)
    uy = math.sin(yaw_rad)
    side_rpy = [yaw_deg, 90.0, 0.0]

    # SIDE 모드: 그리퍼 length 가 XY 평면 yaw 방향(객체 향함)으로 작용.
    # robot TCP 는 객체보다 robot 쪽(base 가까이)에 위치해야 그리퍼 끝이 객체에 도달.
    robot.tcp_z_offset = 0.0   # SIDE 에선 XY 방향 보정이라 z 오프셋 무력화
    L = GRIPPER_TCP_OFFSET_NORMAL   # 그리퍼 끝까지 거리 (mm)

    print(f'  ⓘ SIDE-grasp ({label}) → yaw={yaw_deg:+.1f}° '
          f'(base→객체 방향), rpy={side_rpy}, gripper L={L:.0f}mm')

    # Pre-grasp: 객체로부터 robot 쪽으로 (L + approach) 떨어진 위치
    pre_x = bx - ux * (SIDE_APPROACH_DIST_MM + L)
    pre_y = by - uy * (SIDE_APPROACH_DIST_MM + L)
    pre_xyz = [pre_x, pre_y, bz]

    # Grasp: 그리퍼 끝이 (bx, by, bz) 에 도달 → TCP 는 객체에서 L 만큼 robot 쪽
    grasp_x = bx - ux * L
    grasp_y = by - uy * L
    grasp_xyz = [grasp_x, grasp_y, bz]

    # Lift: grasp 위치에서 Z 위로
    lift_xyz = [grasp_x, grasp_y, bz + Z_LIFT]

    # Place — 객체 기준 PLACE_OFFSET 적용. 자세·yaw 동일 (위치별 yaw 보정은 차후).
    place_obj_x = bx + PLACE_OFFSET_XYZ[0]
    place_obj_y = by + PLACE_OFFSET_XYZ[1]
    place_obj_z = bz + PLACE_OFFSET_XYZ[2]
    place_x = place_obj_x - ux * L
    place_y = place_obj_y - uy * L
    place_xyz = [place_x, place_y, place_obj_z]
    place_app_xyz = [place_x - ux * SIDE_APPROACH_DIST_MM,
                     place_y - uy * SIDE_APPROACH_DIST_MM,
                     place_obj_z]
    place_lift_xyz = [place_x, place_y, place_obj_z + Z_LIFT]

    print('\n  -- Pick (SIDE) --')
    print(f'   1) Pre-grasp → {[round(v,1) for v in pre_xyz]} mm  (yaw={yaw_deg:+.1f}°)')
    if not args.dry_run: robot.move_line_base(pre_xyz, rpy_deg=side_rpy)
    print('   2) 그리퍼 열기')
    if not args.dry_run: robot.gripper_open()
    print(f'   3) 측면 접근 → {[round(v,1) for v in grasp_xyz]} mm')
    if not args.dry_run: robot.move_line_base(grasp_xyz, rpy_deg=side_rpy)
    print('   4) 그리퍼 닫기 (잡기)')
    if not args.dry_run: robot.gripper_close()
    print(f'   5) 들어올리기 → {[round(v,1) for v in lift_xyz]} mm')
    if not args.dry_run: robot.move_line_base(lift_xyz, rpy_deg=side_rpy)

    if args.no_place:
        print('\n  -- Place 생략 (--no-place) --')
    else:
        print(f'\n  -- Place (offset={PLACE_OFFSET_XYZ}mm) --')
        print(f'   1) Place 위로 → {[round(v,1) for v in place_lift_xyz]} mm')
        if not args.dry_run: robot.move_line_base(place_lift_xyz, rpy_deg=side_rpy)
        print(f'   2) 접근 → {[round(v,1) for v in place_xyz]} mm')
        if not args.dry_run: robot.move_line_base(place_xyz, rpy_deg=side_rpy)
        print('   3) 그리퍼 열기 (놓기)')
        if not args.dry_run: robot.gripper_open()
        print(f'   4) 뒤로 빠지기 (robot 쪽) → {[round(v,1) for v in place_app_xyz]} mm')
        if not args.dry_run: robot.move_line_base(place_app_xyz, rpy_deg=side_rpy)

    print('\n  -- HOME 복귀 --')
    if not args.dry_run: robot.move_joint_deg(HOME_JOINT_DEG)


def command_loop(robot, vision, args):
    """
    준비가 끝난 뒤 사용자 명령을 기다리는 인터랙티브 루프.
    매 사이클: target 입력 → 검출 → 좌표 확인 → Pick & Place → HOME → 다시 입력.
    """
    print('\n' + '=' * 70)
    print('  [READY] 명령 대기 중 (YOLO-World, open-vocabulary 검출).')
    print('  잡을 물체를 영문 텍스트로 입력하면 카메라 창이 열립니다:')
    print('    pen          / wood block      / wooden ruler')
    print('    cup, bottle  / red apple       / yellow sticky note')
    print('  팁: 한 phrase 는 공백 보존, 여러 후보는 쉼표로 구분')
    print('       예) "wood block"  → 한 phrase')
    print('           "pen, cup"    → 두 후보')
    print('  카메라 창에서:  [SPACE/Enter] 진행   [q/ESC] 취소')
    print(f"  안전: 사람·동물 류 단어는 자동 차단 ({', '.join(sorted(BLOCKED_CLASSES))})")
    print('  ✦ scan        : 카메라 열고 흔한 후보 [1-9] 중 선택 → Pick&Place')
    print('  ✦ <물체이름>  : 입력 텍스트로 직접 검출 + Pick&Place')
    print('  진단:  pose / joint / state / cal_top_down / set_tcp_z / set_work_z / home / recover')
    print('  cuRobo: plan_only <x y z> / world_clear / curobo_warmup     도움말: help')
    print('  종료: 명령창에서 q 또는 Enter (빈 입력)')
    print('=' * 70)

    debug_path = '/tmp/pickplace_detect.png'

    # 진입 직후 자동 scan — 카메라를 먼저 열어 후보를 보여주고 사용자가 선택하게 함.
    # 1~9: 즉시 Pick&Place / SPACE: auto best / q/ESC: 일반 명령 루프 진입.
    if not args.no_autoscan:
        print('\n  [AUTO-SCAN] 카메라 열고 후보 검출 중... (q/ESC 로 건너뛰면 일반 명령 모드)')
        try:
            det = vision.interactive_browse(COMMON_OBJECTS, save_debug=debug_path)
            if det is not None:
                bx, by, bz = det['base_xyz_mm']
                print(f'  선택: {det["label"]} (conf={det["conf"]:.2f})')
                print(f'  베이스 좌표: x={bx:.1f}  y={by:.1f}  z={bz:.1f}  [mm]')
                try:
                    execute_pick_and_place(robot, det, args)
                    print('  ✓ 사이클 완료 — 다음 명령 대기')
                except Exception as e:
                    print(f'  !! 사이클 실패: {e}')
                    try:
                        if not args.dry_run:
                            robot.recover_safety()
                            robot.move_joint_deg(HOME_JOINT_DEG)
                    except Exception:
                        pass
            else:
                print('  (AUTO-SCAN 취소 — 일반 명령 모드 진입)')
        except Exception as e:
            print(f'  !! AUTO-SCAN 실패: {e}')

    while True:
        try:
            line = input('\n[명령] 잡을 물체 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ('q', 'quit', 'exit'):
            print('  종료 요청 — 명령 루프 빠져나갑니다.')
            break

        # === 진단 / 자세 캘리브레이션 명령 ===
        cmd = line.lower()
        if cmd == 'pose':
            try:
                p = robot.get_current_pose('task')
                grip_z = p[2] - robot.tcp_z_offset
                print(f'  현재 TCP : x={p[0]:.1f}  y={p[1]:.1f}  z={p[2]:.1f}  '
                      f'rx={p[3]:.2f}  ry={p[4]:.2f}  rz={p[5]:.2f}  [mm/deg, ZYZ]')
                print(f'  → 그리퍼 끝 z 추정: {grip_z:.1f}mm  '
                      f'(tcp_z_offset={robot.tcp_z_offset:.1f}mm 적용)')
            except Exception as e:
                print(f'  GetCurrentPose 실패: {e}')
            continue
        if cmd.startswith('set_tcp_z'):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    new_off = float(parts[1])
                    old = robot.tcp_z_offset
                    robot.tcp_z_offset = new_off
                    print(f'  ✓ tcp_z_offset 갱신: {old:.1f} → {new_off:.1f} mm')
                except ValueError:
                    print('  사용법: set_tcp_z <mm>   (예: set_tcp_z 130)')
            else:
                print(f'  현재 tcp_z_offset = {robot.tcp_z_offset:.1f} mm '
                      f'(사용법: set_tcp_z <mm>)')
            continue
        if cmd.startswith('set_dive'):
            parts = cmd.split()
            if len(parts) == 4:
                try:
                    new_normal = float(parts[1])
                    new_hollow = float(parts[2])
                    new_flat = float(parts[3])
                    global GRASP_DIVE_NORMAL_MM, GRASP_DIVE_HOLLOW_MM, GRASP_DIVE_FLAT_MM
                    GRASP_DIVE_NORMAL_MM = new_normal
                    GRASP_DIVE_HOLLOW_MM = new_hollow
                    GRASP_DIVE_FLAT_MM = new_flat
                    print(f'  ✓ grasp dive: NORMAL={new_normal:.1f}, '
                          f'HOLLOW={new_hollow:.1f}, FLAT={new_flat:.1f} mm')
                except ValueError:
                    print('  사용법: set_dive <normal_mm> <hollow_mm> <flat_mm>')
                    print('  예) set_dive 40 5 0  (블록 40 깊이 / 컵 5 살짝 / 펜 0 표면)')
            else:
                print(f'  현재 NORMAL={GRASP_DIVE_NORMAL_MM:.1f}mm  '
                      f'HOLLOW={GRASP_DIVE_HOLLOW_MM:.1f}mm  '
                      f'FLAT={GRASP_DIVE_FLAT_MM:.1f}mm')
                print('  사용법: set_dive <normal> <hollow> <flat>')
            continue
        if cmd.startswith('set_work_z'):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    z_min = float(parts[1])
                    z_max = float(parts[2])
                    if z_min >= z_max:
                        print('  실패: 하한이 상한 이상 — z_min < z_max')
                    else:
                        global WORK_Z
                        WORK_Z = (z_min, z_max)
                        print(f'  ✓ WORK_Z 갱신: {WORK_Z}')
                except ValueError:
                    print('  사용법: set_work_z <min_mm> <max_mm>   (예: set_work_z -150 600)')
            else:
                print(f'  현재 WORK_Z = {WORK_Z}  '
                      f'(사용법: set_work_z <min> <max>)')
            continue
        if cmd == 'joint':
            try:
                j = robot.get_current_pose('joint')
                print(f'  현재 관절각: {[round(v,2) for v in j]} [deg]')
            except Exception as e:
                print(f'  GetCurrentPose 실패: {e}')
            continue
        if cmd == 'show_top_down':
            print(f'  현재 등록된 top-down RPY: {[round(v,2) for v in robot.top_down_rpy]} [deg]')
            continue
        if cmd == 'cal_top_down':
            try:
                p = robot.get_current_pose('task')
                rpy = [float(p[3]), float(p[4]), float(p[5])]
                robot.top_down_rpy = rpy
                print(f'  ✓ top-down RPY 갱신: {[round(v,2) for v in rpy]} [deg]')
                print('  (현재 펜던트로 잡기 좋은 자세에 멈춰둔 상태에서 호출했어야 함)')
                print('  (앞으로 모든 Pick/Place 모션이 이 자세를 사용)')
            except Exception as e:
                print(f'  실패: {e}')
            continue
        if cmd == 'home':
            print('  HOME 자세로 이동...')
            try:
                if not args.dry_run:
                    robot.move_joint_deg(HOME_JOINT_DEG)
                print('  ✓ HOME 도착')
            except Exception as e:
                print(f'  HOME 실패: {e}')
            continue
        if cmd in ('recover', 'reset'):
            print('  Safety recovery 실행...')
            try:
                if not args.dry_run:
                    ok = robot.recover_safety()
                    print('  ✓ 복구 완료' if ok else '  복구 실패 — 펜던트 확인')
            except Exception as e:
                print(f'  recovery 실패: {e}')
            continue
        if cmd == 'state':
            try:
                s = robot.get_robot_state()
                print(f'  robot_state = {s} ({robot._STATE_NAMES.get(s, "?")})')
            except Exception as e:
                print(f'  실패: {e}')
            continue
        # ---- cuRobo 전용 명령 ----
        if cmd == 'curobo_warmup':
            if robot.planner is None:
                print('  CuroboPlanner 미초기화')
            else:
                try:
                    robot.planner.warmup()
                except Exception as e:
                    print(f'  warmup 실패: {e}')
            continue
        if cmd == 'world_clear':
            if robot.planner is None:
                print('  CuroboPlanner 미초기화')
            else:
                try:
                    robot.planner.world_clear()
                except Exception as e:
                    print(f'  world_clear 실패: {e}')
            continue
        if cmd.startswith('try_rpy'):
            # try_rpy <x> <y> <z> <α> <β> <γ>  — 임의 RPY 로 plan 가능성 시험
            parts = cmd.split()
            if len(parts) != 7:
                print('  사용법: try_rpy <x_mm> <y_mm> <z_mm> <α_deg> <β_deg> <γ_deg>')
                print('  예) try_rpy 400 100 100   0  90  0   ← horizontal, base +X 향함')
                print('       try_rpy 400 100 100   0 180  0   ← top-down')
                continue
            try:
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                rpy = [float(parts[4]), float(parts[5]), float(parts[6])]
            except ValueError:
                print('  숫자가 아닌 인자')
                continue
            if robot.planner is None:
                print('  CuroboPlanner 미초기화')
                continue
            try:
                z = xyz[2] + robot.tcp_z_offset
                target_xyz_m = [xyz[0]*1e-3, xyz[1]*1e-3, z*1e-3]
                quat = _rpy_deg_to_quat_wxyz(rpy)
                start = robot._current_joints_rad()
                traj, ms = robot.planner.plan(start, target_xyz_m, quat)
                if traj is None:
                    print(f'  plan FAIL ({ms:.0f}ms)  pos={xyz} rpy={rpy}')
                else:
                    print(f'  plan OK   ({ms:.0f}ms, {len(traj)} pts)  '
                          f'end={[round(np.rad2deg(j),1) for j in traj[-1]]}')
            except Exception as e:
                print(f'  try_rpy 실패: {e}')
            continue
        if cmd.startswith('plan_only'):
            parts = cmd.split()
            if len(parts) != 4:
                print('  사용법: plan_only <x_mm> <y_mm> <z_mm>')
                continue
            try:
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                print('  좌표가 숫자가 아님')
                continue
            if robot.planner is None:
                print('  CuroboPlanner 미초기화')
                continue
            try:
                z = xyz[2] + robot.tcp_z_offset
                target_xyz_m = [xyz[0]*1e-3, xyz[1]*1e-3, z*1e-3]
                quat = _rpy_deg_to_quat_wxyz(robot.top_down_rpy)
                start = robot._current_joints_rad()
                traj, ms = robot.planner.plan(start, target_xyz_m, quat)
                if traj is None:
                    print(f'  plan FAIL ({ms:.0f}ms) — 도달 불가/충돌 가능')
                else:
                    print(f'  plan OK  ({ms:.0f}ms, {len(traj)} pts)  '
                          f'start={[round(np.rad2deg(j),1) for j in start]} '
                          f'end={[round(np.rad2deg(j),1) for j in traj[-1]]}')
            except Exception as e:
                print(f'  plan_only 실패: {e}')
            continue
        if cmd in ('scan', 'look', 'browse'):
            # 흔한 물체 후보 전체로 라이브 검출 → 사용자가 1-9 키로 선택
            det = vision.interactive_browse(COMMON_OBJECTS, save_debug=debug_path)
            if det is None:
                print('  취소됨 — 다시 명령 대기로.')
                continue
            bx, by, bz = det['base_xyz_mm']
            print(f'  선택: {det["label"]} (conf={det["conf"]:.2f})')
            print(f'  픽셀={det["pixel"]}, depth={det["depth_m"]*1000:.0f}mm')
            print(f'  베이스 좌표: x={bx:.1f}  y={by:.1f}  z={bz:.1f}  [mm]')
            print(f'  스냅샷: {debug_path}')
            try:
                execute_pick_and_place(robot, det, args)
                print('\n  ✓ 사이클 완료 — 다음 명령 대기')
            except Exception as e:
                print(f'\n  !! 사이클 실패: {e}')
                try:
                    if not args.dry_run:
                        print('  자동 safety recovery 시도...')
                        robot.recover_safety()
                except Exception as e2:
                    print(f'  recovery 시도 중 에러: {e2}')
                print('  HOME 복귀 시도...')
                try:
                    if not args.dry_run:
                        robot.move_joint_deg(HOME_JOINT_DEG)
                except Exception as e2:
                    print(f'  HOME 복귀도 실패: {e2}')
            continue
        if cmd == 'help':
            print('  명령 목록:')
            print('    scan (look)     : 카메라 열고 흔한 물체 후보 [1-9] 선택 → Pick&Place')
            print('    <물체이름>      : YOLO-World 검출 + Pick&Place (cuRobo 사용)')
            print('    pose            : 현재 TCP 자세 [x,y,z,rx,ry,rz] 출력')
            print('    joint           : 현재 6 관절각(deg) 출력')
            print('    cal_top_down    : 현재 자세의 RPY 를 top-down 기준으로 저장')
            print('    show_top_down   : 현재 등록된 top-down RPY 보기')
            print('    set_tcp_z <mm>  : 그리퍼 TCP z 오프셋 설정 (인자 없으면 현재값)')
            print('    set_dive <normal> <hollow> <flat> : grasp dive (mm). 블록·컵·평면 별로')
            print('    set_work_z <min> <max> : 작업영역 z 한계 갱신 (mm)')
            print('    home            : HOME 관절 자세로 이동')
            print('    state           : 현재 robot_state 정수 + 이름')
            print('    recover (reset) : 충돌/safe-stop 후 수동 fault 복구')
            print('    plan_only <x> <y> <z>  : 모션 없이 cuRobo plan 만 시도 (top_down_rpy 사용)')
            print('    try_rpy <x> <y> <z> <α> <β> <γ>  : 임의 RPY 자세로 plan 가능성 시험')
            print('    world_clear     : 검출 객체 obstacle 제거 (책상만 남김)')
            print('    curobo_warmup   : cuRobo 재warmup (디버그)')
            print('    q / Enter       : 종료')
            continue

        # 쉼표가 있으면 쉼표로 클래스 구분 (각 쪼각 안의 공백은 phrase 로 보존),
        # 쉼표가 없으면 입력 전체를 하나의 phrase 로 취급.
        # 예) "wood block"        → ["wood block"]   (한 phrase)
        #     "pen, wood block"   → ["pen", "wood block"]
        if ',' in line:
            targets = [t.strip() for t in line.split(',') if t.strip()]
        else:
            targets = [line.strip()]
        print(f'  → 검출 대상: {targets}')

        # 라이브 카메라 창에서 검출 결과를 실시간 확인 후 SPACE/Enter 로 진행
        det = vision.interactive_preview(target_classes=targets, save_debug=debug_path)
        if det is None:
            print('  취소됨 (또는 유효한 검출 없음) — 다시 명령 대기로.')
            continue

        bx, by, bz = det['base_xyz_mm']
        print(f'  확정: {det["label"]} (conf={det["conf"]:.2f})')
        print(f'  픽셀={det["pixel"]}, depth={det["depth_m"]*1000:.0f}mm')
        print(f'  베이스 좌표: x={bx:.1f}  y={by:.1f}  z={bz:.1f}  [mm]')
        print(f'  스냅샷: {debug_path}')

        try:
            execute_pick_and_place(robot, det, args)
            print('\n  ✓ 사이클 완료 — 다음 명령 대기')
        except Exception as e:
            print(f'\n  !! 사이클 실패: {e}')
            # 충돌·safe-stop 일 가능성 → 자동 recovery 시도
            try:
                if not args.dry_run:
                    print('  자동 safety recovery 시도...')
                    robot.recover_safety()
            except Exception as e2:
                print(f'  recovery 시도 중 에러: {e2}')
            print('  HOME 복귀 시도...')
            try:
                if not args.dry_run:
                    robot.move_joint_deg(HOME_JOINT_DEG)
            except Exception as e2:
                print(f'  HOME 복귀도 실패: {e2}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='로봇/그리퍼 동작 없이 비전+좌표 변환만 수행')
    parser.add_argument('--no-place', action='store_true',
                        help='Pick 까지만 수행 (Place 생략) — 매 사이클 적용')
    parser.add_argument('--quick', action='store_true',
                        help='드라이버 리셋 SKIP — 드라이버가 이미 깨끗히 살아있을 때만 사용')
    parser.add_argument('--yes', action='store_true',
                        help='매 사이클 진행 확인 프롬프트 SKIP (위험 — 무인 데모 전용)')
    parser.add_argument('--no-autoscan', action='store_true',
                        help='[준비 4/4] 명령 루프 진입 시 자동 scan SKIP — 곧장 명령 프롬프트')
    args = parser.parse_args()

    banner('[통합 Pick & Place — 두산 e0509 + RealSense + YOLOv8 + RH-P12-RN-A]')
    if args.dry_run:
        print('  *** DRY-RUN 모드: 실제 모션/그리퍼 명령은 보내지 않음 ***')

    # ===== [A] 준비 단계 (한 번만 자동 진행) =====
    if not args.quick:
        reset_robot_driver()
    else:
        print('\n[준비 0/4] 드라이버 리셋 SKIP (--quick)')

    rclpy.init()
    robot = PickAndPlace()
    vision = None

    try:
        print('\n[준비 1/4] 캘리브레이션 + YOLO + RealSense + 로봇 + 그리퍼 init')
        vision = VisionDetector()
        if not args.dry_run:
            robot.activate_robot()
            robot.gripper_init()
            robot.gripper_open()
        print('   준비 OK')

        print(f'\n[준비 2/4] cuRobo MotionGen 로드 + warmup')
        robot.planner = CuroboPlanner()
        print('   cuRobo 준비 OK')

        print(f'\n[준비 3/4] HOME 자세로 이동 {HOME_JOINT_DEG} deg (FollowJointTrajectory)')
        if not args.dry_run:
            try:
                robot.move_joint_deg(HOME_JOINT_DEG)
                print('   HOME 도착')
            except Exception as e:
                print(f'   !! HOME 이동 실패: {e}')
                print('   → 명령 루프에서 state / recover / home 으로 진단·재시도 가능')
        else:
            print('   (dry-run: HOME 이동 SKIP)')

        print('\n[준비 4/4] 명령 루프 진입')
        # ===== [B] 명령 루프 (사용자가 입력할 때만 움직임) =====
        command_loop(robot, vision, args)

    except Exception as e:
        print(f'\n!! 에러 발생: {e}')
        import traceback
        traceback.print_exc()
    finally:
        try:
            robot.gripper_shutdown()
        except Exception:
            pass
        if vision is not None:
            vision.stop()
        # 종료 시점에 모든 cv2 창 닫음 (사이클 동안에는 유지하다가)
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print('\n[정리 완료, 종료]')


if __name__ == '__main__':
    main()
