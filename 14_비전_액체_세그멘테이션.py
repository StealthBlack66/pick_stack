"""
[액체 세그멘테이션 Pick & Place — 두산 e0509 + RealSense + YOLO26-seg + SAM2 + RH-P12-RN-A]

13번 (yolov8s-world + bbox 만) 을 확장한 액체 처리 데모.
세 가지 기능을 한 사이클에 보여준다:

  1) **수위/잔여량 측정**  — 컵·병의 fill_ratio (0~1) 와 추정 부피 (ml)
  2) **쏟지 않는 그립/이동** — fill_ratio 높으면 그리퍼 수직 유지 + 보수적 trajectory
  3) **액체 회피 (puddle)** — 테이블 위 흘린 액체를 curobo collision world 에 추가

모델 구성:
  - 컨테이너 검출+세그멘테이션 : **yolo26s-seg.pt** (Ultralytics YOLO26, 2026-01-14 출시)
    * NMS-free end-to-end (후처리 불필요)
    * CPU 추론 YOLOv8 대비 약 43% 빠름, mAP 향상
    * COCO 80 클래스 — cup(41) / wine glass(40) / bottle(39) 그대로
    * ultralytics 8.4.48 의 assets v8.4.0 태그에 호스팅 — auto-download 가능
    * 첫 다운로드 실패 시 yolo11s-seg.pt 로 폴백 (구조 호환)
  - 미세 액체 마스크         : sam2.1_b.pt    (Meta SAM2.1 base, ~150MB, ultralytics 내장)

13번과의 차이:
  - YOLO-World 의 임의 영문 텍스트 검출 → 빠짐 (COCO 폐쇄형 어휘 사용).
    데모 시 cup/mug/bottle/wine glass 만 다루므로 영향 없음. 범용 피크앤플레이스는 13번 그대로.
  - GPU 사용 (yolo11-seg 은 CLIP 인코더 없어 13번의 device 버그 무관)
  - 진단 명령(plan_only / try_rpy / set_dive 등) 생략 — 액체 데모 흐름에 집중

실행:
  python3 14_비전_액체_세그멘테이션.py
  python3 14_비전_액체_세그멘테이션.py --quick      # 드라이버 리셋 skip
  python3 14_비전_액체_세그멘테이션.py --dry-run    # 모션 없이 비전+plan 만
  python3 14_비전_액체_세그멘테이션.py --vision-only # 로봇/ROS 전체 SKIP — 비전 단독 테스트
"""

import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print('!! pyrealsense2 미설치. pip install pyrealsense2 후 다시 실행하세요.')
    sys.exit(1)

try:
    from ultralytics import YOLO, SAM
except ImportError:
    print('!! ultralytics 미설치. 다음 순서로 설치 후 다시 실행하세요:')
    print('     source /home/fastcampus/Downloads/test/vision_env/bin/activate')
    print('     pip install ultralytics')
    sys.exit(1)

try:
    from scipy.spatial.transform import Rotation as _ScipyR
except ImportError:
    print('!! scipy 미설치. pip install --user scipy 후 다시 실행하세요.')
    sys.exit(1)

# 로봇/cuRobo 관련은 --vision-only 일 때 import 안 함 (CPU 만 있는 환경에서도 비전 테스트 가능)
_ROBOT_IMPORTS_OK = False
try:
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
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig, JointState as CuroboJointState
    from curobo.types.math import Pose as CuroboPose
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from curobo.geom.types import WorldConfig, Cuboid
    _ROBOT_IMPORTS_OK = True
except ImportError as e:
    # --vision-only 가 아닌데 robot import 가 실패하면 main() 에서 안내 후 종료
    _ROBOT_IMPORT_ERROR = e


# ============== 사용자 조정 가능 상수 ==============
try:
    from doosan_config import (
        NAMESPACE as NS,
        ROBOT_IP, RT_HOST, ROBOT_MODEL,
        BRINGUP_PKG, BRINGUP_LAUNCH, MOVEIT_CONTROLLER,
        DOOSAN_WS, ROS_DISTRO,
    )
except ImportError:
    # --vision-only 에서만 안전
    NS = ROBOT_IP = RT_HOST = ROBOT_MODEL = None
    BRINGUP_PKG = BRINGUP_LAUNCH = MOVEIT_CONTROLLER = DOOSAN_WS = ROS_DISTRO = None

DRIVER_WAIT_SEC = 25
RVIZ_CONFIG = (
    os.path.join(DOOSAN_WS, 'src', 'doosan-robot2', 'dsr_moveit2',
                 f'dsr_moveit_config_{ROBOT_MODEL}', 'launch', 'moveit.rviz')
    if DOOSAN_WS and ROBOT_MODEL else ''
)

# 로봇 자세
HOME_JOINT_DEG   = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
TOP_DOWN_RPY_DEG = [0.0, 180.0, 0.0]
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
MOVE_DURATION_SEC = 5.0
SAFE_JOINT_LIMITS_DEG = [
    (-360.0, 360.0), (-90.0, 90.0), (-150.0, 150.0),
    (-360.0, 360.0), (-135.0, 135.0), (-360.0, 360.0),
]

# cuRobo
CUROBO_URDF = os.environ.get(
    'E0509_CUROBO_URDF',
    os.path.expanduser('~/doosan_ws/src/e0509_gripper_description/config/curobo/e0509_gripper.urdf'),
)
CUROBO_BASE_LINK = 'base_link'
CUROBO_EE_LINK   = 'gripper_rh_p12_rn_base'
WORLD_TABLE = dict(name='table', pose=[0.0, 0.0, -0.02, 1.0, 0.0, 0.0, 0.0], dims=[1.2, 1.2, 0.04])

# Pick & Place 기하 (13번과 동일 — 컵·병 위주이므로 HOLLOW 기준)
Z_APPROACH = 100.0
Z_LIFT     = 100.0
PLACE_OFFSET_XYZ = [0.0, 200.0, 0.0]
GRIPPER_TCP_OFFSET_NORMAL = 160.0
GRASP_DIVE_HOLLOW_MM = 5.0    # 컵 입구 가장자리만 살짝
WORK_X = (-600.0, 600.0); WORK_Y = (-600.0, 600.0); WORK_Z = (-150.0, 600.0)

# 그리퍼
GRIPPER_PORT  = 1
GRIPPER_BAUD  = 57600
GRIP_OPEN_POS  = 0
GRIP_CLOSE_POS = 700

# 비전
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR  = '/home/fastcampus/Downloads/test'
CALIB_PATH     = os.path.join(SCRIPT_DIR, 'calibration_data', 'calibration_result.npz')
YOLO_WEIGHTS         = 'yolo26s-seg.pt'   # 2026-01 출시 ultralytics 공식 최신
YOLO_WEIGHTS_FALLBACK = 'yolo11s-seg.pt'   # YOLO26 다운로드/로드 실패 시 폴백
SAM_WEIGHTS          = 'sam2.1_b.pt'
DETECT_CONF_THR = 0.30
NMS_IOU_THR     = 0.45
RS_W, RS_H, RS_FPS = 640, 480, 30

# 액체 관련 — COCO 클래스 id (yolo11-seg 의 model.names 기준)
#   39: bottle, 40: wine glass, 41: cup
CONTAINER_COCO_IDS  = {39, 40, 41}
CONTAINER_NAME_BY_ID = {39: 'bottle', 40: 'wine glass', 41: 'cup'}

# 수위 임계
FILL_CONSERVATIVE_THR = 0.30   # 이 비율 이상이면 보수적 trajectory + 강제 수직 유지
FILL_VOLUME_DENSITY   = 1.0    # 단순화: mask 면적 × 깊이 = 부피(ml). 보정 계수.

# Puddle 검출
SPILL_PLANE_MARGIN_MM = (5.0, 50.0)   # 테이블 평면 위 5~50mm 범위 검색
SPILL_MIN_AREA_PX     = 400           # 너무 작은 noise 마스크는 무시
SPILL_CUBOID_HEIGHT_M = 0.03          # collision cuboid 높이 30mm


# ============== Modbus RTU (그리퍼) — 13번과 동일 ==============
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

def fc16_position(pos: int, slave=1) -> list:
    pos = max(0, min(700, int(pos)))
    return _make_frame(bytes([slave, 0x10, 0x01, 0x1A, 0x00, 0x02, 0x04,
                              (pos >> 8) & 0xFF, pos & 0xFF, 0x00, 0x00]))

def _rpy_deg_to_quat_wxyz(rpy_deg):
    q = _ScipyR.from_euler('ZYZ', rpy_deg, degrees=True).as_quat()   # [x,y,z,w]
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


# ============== cuRobo Motion Planner (액체 안전 모드 확장) ==============
class LiquidSafePlanner:
    """
    13번 CuroboPlanner 와 동일하지만 fill_ratio 에 따라 다음을 추가:
      - update_world() 로 puddle cuboid 들을 collision world 에 동적으로 추가
      - plan_safe() 에서 fill_ratio 가 임계치 이상이면 자동으로 top-down RPY 강제
        (사용자가 다른 RPY 넘겨도 무시 — 안전 우선)
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
            urdf_path=urdf_path, base_link=base_link, ee_link=ee_link,
            tensor_args=self.tensor_args,
        )

        self._base_world_cuboids = [
            Cuboid(name=WORLD_TABLE['name'],
                   pose=list(WORLD_TABLE['pose']),
                   dims=list(WORLD_TABLE['dims'])),
        ]
        world_cfg = WorldConfig(cuboid=list(self._base_world_cuboids))
        mg_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg, world_cfg, tensor_args=self.tensor_args,
            num_trajopt_seeds=4, num_graph_seeds=4,
            collision_cache={'obb': 30, 'mesh': 10},
        )
        self.motion_gen = MotionGen(mg_cfg)
        print('   cuRobo MotionGen warmup 시작 (30~60초 소요)...')
        t0 = time.time()
        self.motion_gen.warmup(warmup_js_trajopt=False)
        print(f'   cuRobo warmup 완료 ({time.time()-t0:.1f}s)')

        # puddle cuboid (동적). 이름이 중복되지 않도록 자동 increment.
        self._spill_cuboids = []
        self._spill_counter = 0

    def warmup(self):
        t0 = time.time()
        self.motion_gen.warmup(warmup_js_trajopt=False)
        print(f'   재warmup 완료 ({time.time()-t0:.1f}s)')

    def world_clear(self):
        """모든 검출 객체/puddle 제거. 책상만 남김."""
        self._spill_cuboids = []
        world_cfg = WorldConfig(cuboid=list(self._base_world_cuboids))
        self.motion_gen.update_world(world_cfg)
        print('   world cleared (table 만 남음, puddle cuboid 제거)')

    def add_spill_obstacles(self, spill_aabbs_mm):
        """
        spill_aabbs_mm: [(cx_mm, cy_mm, cz_mm, w_mm, h_mm), ...]
        센터 좌표 + 가로/세로 (cuboid 높이는 SPILL_CUBOID_HEIGHT_M 고정).
        """
        new_cubes = []
        for (cx, cy, cz, w, h) in spill_aabbs_mm:
            self._spill_counter += 1
            name = f'spill_{self._spill_counter}'
            new_cubes.append(Cuboid(
                name=name,
                pose=[cx * 1e-3, cy * 1e-3, cz * 1e-3, 1.0, 0.0, 0.0, 0.0],
                dims=[max(w, 30.0) * 1e-3, max(h, 30.0) * 1e-3, SPILL_CUBOID_HEIGHT_M],
            ))
        self._spill_cuboids.extend(new_cubes)
        world_cfg = WorldConfig(cuboid=list(self._base_world_cuboids) + list(self._spill_cuboids))
        self.motion_gen.update_world(world_cfg)
        print(f'   puddle {len(new_cubes)} 개 collision world 추가 '
              f'(누적 {len(self._spill_cuboids)} 개)')

    def plan(self, start_joints_rad, target_xyz_m, target_quat_wxyz):
        """기본 plan — 13번 CuroboPlanner.plan() 과 동일 시그니처."""
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

    def plan_safe(self, start_joints_rad, target_xyz_m, target_quat_wxyz, fill_ratio):
        """
        액체 안전 plan. fill_ratio (0~1) 가 임계 이상이면 RPY 를 강제로 top-down 으로 덮어쓴다.
        (가속/jerk 동적 조정은 ros2_control 컨트롤러 단에서 처리되므로 여기선 자세만 강제)
        """
        if fill_ratio is not None and fill_ratio >= FILL_CONSERVATIVE_THR:
            forced = _rpy_deg_to_quat_wxyz(TOP_DOWN_RPY_DEG)
            if list(target_quat_wxyz) != forced:
                print(f'   [안전] fill_ratio={fill_ratio:.2f} ≥ {FILL_CONSERVATIVE_THR} '
                      f'→ 그리퍼 자세를 top-down 으로 강제')
                target_quat_wxyz = forced
        return self.plan(start_joints_rad, target_xyz_m, target_quat_wxyz)


# ============== 비전: 컨테이너 + 액체 + 스필 ==============
class LiquidVisionSystem:
    """
    구성:
      - yolo11s-seg.pt → 컨테이너(cup/wine glass/bottle) bbox + mask
      - sam2.1_b.pt    → 컨테이너 내부 영역의 정밀 마스크 (bbox prompt)
      - RealSense color+depth aligned
      - T_cam2base 로 base 좌표 변환
    """
    def __init__(self, vision_only=False):
        if not vision_only and not os.path.exists(CALIB_PATH):
            raise FileNotFoundError(
                f'캘리브레이션 결과 없음: {CALIB_PATH}\n'
                '   08_카메라_핸드아이_캘리브레이션.py 를 먼저 실행하거나, '
                '--vision-only 로 실행하세요.'
            )
        if os.path.exists(CALIB_PATH):
            d = np.load(CALIB_PATH)
            self.T_cam2base = d['T_cam2base']
            self.calib_err_mm = float(d['pos_err_mean_mm'])
            self.calib_samples = int(d['num_samples']) if 'num_samples' in d.files else 0
            print(f'   캘리브레이션 로드: {CALIB_PATH}')
            print(f'   - translation={[round(v*1000,1) for v in self.T_cam2base[:3,3]]}mm, '
                  f'mean err {self.calib_err_mm:.2f}mm')
        else:
            self.T_cam2base = np.eye(4)
            self.calib_err_mm = -1.0
            self.calib_samples = 0
            print('   (--vision-only: 캘리브레이션 SKIP → base 좌표는 camera 좌표와 동일)')

        # YOLO26-seg — GPU 사용 (CLIP 없으므로 13번의 device 버그 없음).
        # YOLO26 은 NMS-free end-to-end 라 conf threshold 만 의미 있고 NMS IoU 는 무시됨.
        print(f'   YOLO seg 로딩... ({YOLO_WEIGHTS})')
        try:
            self.yolo = YOLO(YOLO_WEIGHTS)
            self.yolo_weights_used = YOLO_WEIGHTS
        except Exception as e:
            print(f'   !! {YOLO_WEIGHTS} 로드 실패: {e}')
            print(f'   → 폴백: {YOLO_WEIGHTS_FALLBACK}')
            self.yolo = YOLO(YOLO_WEIGHTS_FALLBACK)
            self.yolo_weights_used = YOLO_WEIGHTS_FALLBACK
        self.yolo_names = self.yolo.names   # COCO 80 (YOLO11/26 모두 동일)
        # SAM2.1 base
        print(f'   SAM2 로딩... ({SAM_WEIGHTS})')
        self.sam = SAM(SAM_WEIGHTS)
        print('   SAM2 워밍업 (더미 추론 1회)... ', end='', flush=True)
        _dummy = np.zeros((RS_H, RS_W, 3), dtype=np.uint8)
        try:
            self.sam(_dummy, bboxes=[[10, 10, 100, 100]], verbose=False)
            print('완료')
        except Exception as e:
            print(f'(워밍업 스킵: {e})')

        # RealSense
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
        cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        for _ in range(15):
            self.pipeline.wait_for_frames()

        # 캐시: 테이블 평면 fit 결과 (puddle 검출용)
        self._table_plane = None

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    # ---- 기본 헬퍼 ----
    def _depth_at(self, depth_frame, u, v, patch=5):
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
        return float(win.mean()) * 0.001

    def _grab_frames(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return None, None, None
        color = np.asanyarray(cf.get_data())
        depth_mm = np.asanyarray(df.get_data())   # uint16, mm
        return color, depth_mm, df

    def _cam_to_base_xyz_mm(self, u, v, depth_m):
        cam_xyz_m = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], float(depth_m))
        cam_h = np.array([cam_xyz_m[0], cam_xyz_m[1], cam_xyz_m[2], 1.0])
        base_m = (self.T_cam2base @ cam_h)[:3]
        return tuple(float(v) * 1000.0 for v in base_m)

    # ---- 1) 컨테이너 검출 (YOLO11-seg) ----
    def detect_containers(self, color):
        """
        YOLO11-seg 으로 cup/wine glass/bottle 검출. mask 도 함께.
        반환: [{'class_id', 'class_name', 'conf', 'bbox_xyxy', 'mask', 'cx','cy','w','h'}, ...]
        """
        results = self.yolo(color, conf=DETECT_CONF_THR, iou=NMS_IOU_THR,
                            classes=list(CONTAINER_COCO_IDS),
                            verbose=False)
        out = []
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return out
        r0 = results[0]
        boxes  = r0.boxes
        masks  = r0.masks
        H, W   = color.shape[:2]

        xyxy   = boxes.xyxy.cpu().numpy()
        confs  = boxes.conf.cpu().numpy()
        clsids = boxes.cls.cpu().numpy().astype(int)

        if masks is None:
            # -seg 가 아닌 모델이면 None — 폴백 (bbox 영역 통째로 마스크 처리)
            mask_arrays = [None] * len(boxes)
        else:
            mask_arrays = masks.data.cpu().numpy()   # (N, h, w) — 모델 입력 해상도일 수 있음

        for i in range(len(boxes)):
            cid = int(clsids[i])
            if cid not in CONTAINER_COCO_IDS:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h   = x2 - x1, y2 - y1
            m = mask_arrays[i]
            if m is not None:
                # mask 를 원본 해상도로 리사이즈
                m_resized = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                bool_mask = m_resized.astype(bool)
            else:
                bool_mask = np.zeros((H, W), dtype=bool)
                bool_mask[int(y1):int(y2), int(x1):int(x2)] = True
            out.append(dict(
                class_id=cid,
                class_name=CONTAINER_NAME_BY_ID.get(cid, self.yolo_names.get(cid, '?')),
                conf=float(confs[i]),
                bbox_xyxy=(x1, y1, x2, y2),
                mask=bool_mask,
                cx=cx, cy=cy, w=w, h=h,
            ))
        out.sort(key=lambda d: -d['conf'])
        return out

    # ---- 2) 액체 마스크 (SAM2 + 후처리) ----
    def segment_liquid(self, color, container):
        """
        컨테이너 bbox 내부에서 액체 영역을 추출.
        전략: SAM2 에 bbox prompt → 내부 마스크들 받음 → YOLO 마스크와 IoU 높은 것 = 컨테이너 본체
              본체 마스크 내부에서 saturation 분포로 액체/배경 분리.
        반환: liquid_mask (bool HxW), interior_mask (bool HxW, 컨테이너 안쪽 전체)
        """
        x1, y1, x2, y2 = container['bbox_xyxy']
        H, W = color.shape[:2]
        cont_mask = container['mask']
        try:
            sam_results = self.sam(color, bboxes=[[float(x1), float(y1), float(x2), float(y2)]],
                                   verbose=False)
        except Exception as e:
            print(f'   !! SAM2 추론 실패: {e}')
            sam_results = None

        interior_mask = None
        if sam_results and sam_results[0].masks is not None:
            sam_masks = sam_results[0].masks.data.cpu().numpy()
            # YOLO 마스크와 IoU 가 가장 높은 SAM 마스크 = 컨테이너 본체
            best_iou, best_idx = -1.0, 0
            for i, m in enumerate(sam_masks):
                mr = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                inter = np.logical_and(mr, cont_mask).sum()
                union = np.logical_or(mr, cont_mask).sum()
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            interior_mask = cv2.resize(
                sam_masks[best_idx].astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            interior_mask = cont_mask.copy()

        # 액체/배경 분리: 컨테이너 내부에서 saturation 히스토그램 기반 thresholding
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        # 내부 픽셀의 채도 + 밝기 분포
        interior_pixels_sat = sat[interior_mask]
        interior_pixels_val = val[interior_mask]
        if interior_pixels_sat.size < 50:
            return np.zeros_like(interior_mask), interior_mask

        # 액체는 보통 (a) 채도 높음 (주스·우유) 또는 (b) 채도 낮고 밝기 낮음 (어두운 물).
        # 두 모드를 OR 로 결합.
        sat_thr = max(40, int(np.percentile(interior_pixels_sat, 60)))
        val_thr_lo = int(np.percentile(interior_pixels_val, 35))

        liquid_mask = interior_mask & (
            (sat >= sat_thr) | (val <= val_thr_lo)
        )
        # 노이즈 제거: morphology
        liquid_u8 = (liquid_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        liquid_u8 = cv2.morphologyEx(liquid_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        liquid_u8 = cv2.morphologyEx(liquid_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        liquid_mask = liquid_u8.astype(bool) & interior_mask

        # 액체 마스크가 컨테이너 면적의 5% 미만이면 "거의 비어있음" 으로 간주
        if liquid_mask.sum() < 0.05 * interior_mask.sum():
            liquid_mask = np.zeros_like(interior_mask)
        return liquid_mask, interior_mask

    # ---- 3) 수위 측정 ----
    def estimate_fill_level(self, depth_mm, interior_mask, liquid_mask):
        """
        반환: dict {fill_ratio (0~1), volume_ml, method ('depth'|'pixel'), rim_z_mm, liquid_z_mm}
        - depth 기반: 컨테이너 림 z(min) vs 액체 표면 z(median) → 비율
        - depth 깨지면 픽셀 면적 기반으로 폴백
        """
        H, W = interior_mask.shape
        interior_depth = depth_mm[interior_mask]
        interior_depth = interior_depth[interior_depth > 0]
        liquid_depth   = depth_mm[liquid_mask]
        liquid_depth   = liquid_depth[liquid_depth > 0]

        if interior_depth.size < 30 or liquid_depth.size < 20:
            # 폴백: 픽셀 면적 비율
            interior_area = int(interior_mask.sum())
            liquid_area   = int(liquid_mask.sum())
            ratio = liquid_area / interior_area if interior_area > 0 else 0.0
            return dict(fill_ratio=float(ratio), volume_ml=float(liquid_area * 0.05),
                        method='pixel', rim_z_mm=-1.0, liquid_z_mm=-1.0)

        rim_z   = float(np.percentile(interior_depth, 5))    # 카메라에 가장 가까운 = 림 윗면
        bot_z   = float(np.percentile(interior_depth, 95))   # 가장 먼 = 바닥
        liq_z   = float(np.median(liquid_depth))
        span    = bot_z - rim_z
        if span <= 5:
            ratio = liquid_depth.size / max(1, interior_depth.size)
            method = 'pixel'
        else:
            ratio = (bot_z - liq_z) / span
            ratio = float(np.clip(ratio, 0.0, 1.0))
            method = 'depth'

        # 단순 부피 추정: 액체 마스크 픽셀수 × 평균 cm³ 가정 (강의용 근사)
        vol_ml = float(liquid_mask.sum()) * float(FILL_VOLUME_DENSITY) * 0.05
        return dict(fill_ratio=ratio, volume_ml=vol_ml, method=method,
                    rim_z_mm=rim_z, liquid_z_mm=liq_z)

    # ---- 4) 컨테이너 → base 좌표 (그립용) ----
    def container_base_xyz(self, depth_frame, container):
        """컨테이너 윗면 픽셀 (bbox 상단 영역) 의 base 좌표 mm 반환."""
        cx, cy, w, h = container['cx'], container['cy'], container['w'], container['h']
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape
        x1 = max(0, int(cx - w / 2))
        x2 = min(W, int(cx + w / 2))
        y1 = max(0, int(cy - h / 2))
        y2 = min(H, int(y1 + max(3, h * 0.2)))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        win = arr[y1:y2, x1:x2]
        valid = win[win > 0]
        if valid.size < 10:
            return None
        min_d_mm = float(valid.min())
        loc = np.argwhere(win == min_d_mm)
        ly, lx = loc[0]
        u_top, v_top = x1 + int(lx), y1 + int(ly)
        depth_m = min_d_mm * 1e-3
        if depth_m <= 0.05:
            return None
        bx, by, bz = self._cam_to_base_xyz_mm(u_top, v_top, depth_m)
        return dict(base_xyz_mm=(bx, by, bz), pixel=(u_top, v_top), depth_m=depth_m)

    # ---- 5) Puddle 검출 ----
    def detect_spills(self, color, depth_mm, depth_frame, exclude_mask=None):
        """
        테이블 평면 RANSAC fit 후, 평면 위 SPILL_PLANE_MARGIN_MM 범위의 픽셀에서
        채도 낮은 광택 영역을 SAM2 로 정밀 마스크화 → 외접 박스 + base 좌표 반환.
        exclude_mask: 컨테이너 영역 등 제외할 픽셀.
        반환: [{'mask','base_aabb_mm':(cx,cy,cz,w,h),'pixel_bbox'}, ...]
        """
        H, W = depth_mm.shape
        valid = depth_mm > 0
        if exclude_mask is not None:
            valid = valid & (~exclude_mask)
        if valid.sum() < 5000:
            return []

        # 단순 평면 fit: depth 값의 1st 모드 (테이블) 만 잡기 — 시간 절약을 위해 RANSAC 대신 percentile
        # depth 값 중 가장 큰 60~70 percentile (테이블이 카메라에서 멀리) 가 테이블 평면 근처
        depth_vals = depth_mm[valid]
        table_mid = float(np.percentile(depth_vals, 60))
        table_band_lo = table_mid - SPILL_PLANE_MARGIN_MM[1] * 1.5
        table_band_hi = table_mid + SPILL_PLANE_MARGIN_MM[1] * 1.5

        # puddle 후보: 테이블 평면 근처 + 채도 낮음 (HSV S < 60) + 밝기 보통 이상
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        candidate = (
            (depth_mm >= table_band_lo) &
            (depth_mm <= table_band_hi) &
            (hsv[..., 1] < 60) &
            (hsv[..., 2] > 80) &
            valid
        )
        # 너무 큰 영역(전체 테이블) 은 puddle 이 아니므로 제외 — connected components 로 처리
        cand_u8 = (candidate.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
        spills = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_px = stats[i, cv2.CC_STAT_WIDTH]
            h_px = stats[i, cv2.CC_STAT_HEIGHT]
            # puddle 사이즈: 너무 작거나 거의 화면 전체면 제외
            if area < SPILL_MIN_AREA_PX:
                continue
            if area > 0.4 * H * W:   # 화면의 40% 초과면 테이블 본체로 간주
                continue
            # SAM2 로 정밀 마스크
            bbox_xyxy = [float(x), float(y), float(x + w_px), float(y + h_px)]
            try:
                sam_r = self.sam(color, bboxes=[bbox_xyxy], verbose=False)
                if sam_r and sam_r[0].masks is not None:
                    sm = sam_r[0].masks.data.cpu().numpy()[0]
                    sm = cv2.resize(sm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if exclude_mask is not None:
                        sm = sm & (~exclude_mask)
                else:
                    sm = (labels == i)
            except Exception:
                sm = (labels == i)

            # 중심 픽셀의 base 좌표 + 가로/세로 (mm 환산은 depth 사용해 근사)
            ys, xs = np.where(sm)
            if ys.size < 50:
                continue
            cu, cv_ = float(xs.mean()), float(ys.mean())
            cd_m = self._depth_at(depth_frame, int(cu), int(cv_))
            if cd_m <= 0.05:
                continue
            cx_b, cy_b, cz_b = self._cam_to_base_xyz_mm(cu, cv_, cd_m)
            # 가로/세로 mm 근사: pixel 폭 × (depth / focal)
            fx = self.intr.fx
            fy = self.intr.fy
            w_mm = w_px * cd_m * 1000.0 / fx
            h_mm = h_px * cd_m * 1000.0 / fy
            spills.append(dict(
                mask=sm,
                pixel_bbox=(int(x), int(y), int(x + w_px), int(y + h_px)),
                base_aabb_mm=(cx_b, cy_b, cz_b, w_mm, h_mm),
            ))
        return spills

    # ---- 6) 통합 라이브 미리보기 ----
    def interactive_liquid_preview(self, save_debug=None):
        """
        라이브 카메라 창에서 컨테이너 검출 + 액체 마스크 + puddle 을 실시간 오버레이.
          - SPACE/Enter : 현재 best 컨테이너를 확정 (pick 대상). detect 결과 반환.
          - 'p' 키       : puddle cuboid 들을 결과에 포함 (없으면 자동 검출).
          - q / ESC     : 취소 (None 반환).
        반환: dict { container, liquid_mask, fill_info, spills, container_base, snapshot_path }
        """
        win = 'Liquid Vision  [SPACE/Enter] confirm   [p] include spills   [q/ESC] cancel'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        include_spills = False
        last_result = None
        try:
            while True:
                color, depth_mm, df = self._grab_frames()
                if color is None:
                    continue
                disp = color.copy()
                H, W = disp.shape[:2]

                containers = self.detect_containers(color)
                cv2.putText(disp, f'containers: {[c["class_name"] for c in containers]}',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(disp, f'[p] include puddles = {include_spills}',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cur_result = None
                container_full_mask = np.zeros((H, W), dtype=bool)
                if containers:
                    best = containers[0]
                    liquid_mask, interior_mask = self.segment_liquid(color, best)
                    container_full_mask = interior_mask
                    fill = self.estimate_fill_level(depth_mm, interior_mask, liquid_mask)
                    cbase = self.container_base_xyz(df, best)

                    # 오버레이: 컨테이너 마스크 (녹색 반투명), 액체 (파랑 반투명)
                    overlay = disp.copy()
                    overlay[interior_mask] = (0, 255, 0)
                    overlay[liquid_mask]   = (255, 80, 0)
                    disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)

                    x1, y1, x2, y2 = map(int, best['bbox_xyxy'])
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(disp, f'{best["class_name"]} {best["conf"]:.2f}',
                                (x1, max(20, y1 - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(disp,
                                f'fill={fill["fill_ratio"]*100:5.1f}%  '
                                f'vol~{fill["volume_ml"]:.0f}ml  ({fill["method"]})',
                                (x1, max(40, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 0, 255) if fill['fill_ratio'] > FILL_CONSERVATIVE_THR else (0, 255, 255),
                                2)
                    if cbase is not None:
                        bx, by, bz = cbase['base_xyz_mm']
                        cv2.putText(disp,
                                    f'base xyz=({bx:+.0f},{by:+.0f},{bz:+.0f})mm',
                                    (10, H - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cur_result = dict(container=best, liquid_mask=liquid_mask,
                                      interior_mask=interior_mask,
                                      fill_info=fill, container_base=cbase,
                                      spills=[])

                # Puddle 오버레이 (p 켜져있을 때만 비싼 SAM2 검출 수행)
                if include_spills:
                    spills = self.detect_spills(color, depth_mm, df,
                                                exclude_mask=container_full_mask)
                    for s in spills:
                        ov = disp.copy()
                        ov[s['mask']] = (0, 0, 255)
                        disp = cv2.addWeighted(ov, 0.4, disp, 0.6, 0)
                        x1, y1, x2, y2 = s['pixel_bbox']
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cx, cy, cz, ww, hh = s['base_aabb_mm']
                        cv2.putText(disp, f'spill {ww:.0f}x{hh:.0f}mm',
                                    (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    if cur_result is not None:
                        cur_result['spills'] = spills
                    cv2.putText(disp, f'spills detected: {len(spills)}',
                                (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                cv2.imshow(win, disp)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('p'), ord('P')):
                    include_spills = not include_spills
                    print(f'   [puddle 검출 {"ON" if include_spills else "OFF"}]')
                if key in (ord(' '), 13, 10):   # SPACE / Enter
                    if cur_result is None or cur_result['container_base'] is None:
                        print('   (유효 컨테이너 검출 없음)')
                        continue
                    if save_debug:
                        cv2.imwrite(save_debug, disp)
                    last_result = cur_result
                    last_result['snapshot_path'] = save_debug
                    return last_result
                if key in (ord('q'), 27):
                    return None
        finally:
            cv2.waitKey(1)


# ============== 로봇 컨트롤러 (액체 안전 분기 추가) ==============
class LiquidPickAndPlace(Node if _ROBOT_IMPORTS_OK else object):
    def __init__(self):
        if not _ROBOT_IMPORTS_OK:
            raise RuntimeError('ROS2/curobo import 실패 — --vision-only 로 실행하세요')
        super().__init__('liquid_pick_node')
        self.cli_mode      = self.create_client(SetRobotMode,           f'/{NS}/system/set_robot_mode')
        self.cli_ctrl      = self.create_client(SetRobotControl,        f'/{NS}/system/set_robot_control')
        self.cli_get_pose  = self.create_client(GetCurrentPose,         f'/{NS}/system/get_current_pose')
        self.cli_get_state = self.create_client(GetRobotState,          f'/{NS}/system/get_robot_state')
        self.cli_safety_mode = self.create_client(SetSafetyMode,        f'/{NS}/system/set_safety_mode')
        self.cli_safe_reset  = self.create_client(SetSafeStopResetType, f'/{NS}/system/set_safe_stop_reset_type')
        self.traj_action = ActionClient(
            self, FollowJointTrajectory,
            f'/{NS}/{MOVEIT_CONTROLLER}/follow_joint_trajectory'
        )
        self.top_down_rpy = list(TOP_DOWN_RPY_DEG)
        self.tcp_z_offset = float(GRIPPER_TCP_OFFSET_NORMAL)
        self.planner = None   # LiquidSafePlanner 주입
        gp = f'/{NS}/gripper'
        self.cli_g_open  = self.create_client(FlangeSerialOpen,  f'{gp}/flange_serial_open')
        self.cli_g_close = self.create_client(FlangeSerialClose, f'{gp}/flange_serial_close')
        self.cli_g_write = self.create_client(FlangeSerialWrite, f'{gp}/flange_serial_write')
        self._gripper_serial_open = False

    # ----- ROS 서비스 유틸 (13번과 동일) -----
    def _wait(self, cli, name, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'서비스 미응답: {name}')

    def _call(self, cli, req, timeout=30.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    _STATE_NAMES = {
        0: 'INITIALIZING', 1: 'STANDBY', 2: 'MOVING', 3: 'SAFE_OFF',
        4: 'TEACHING', 5: 'SAFE_STOP', 6: 'EMERGENCY_STOP',
        7: 'HOMING', 8: 'RECOVERY', 9: 'SAFE_STOP2', 10: 'SAFE_OFF2', 15: 'NOT_READY',
    }

    def get_robot_state(self):
        if not self.cli_get_state.wait_for_service(timeout_sec=2.0):
            return -1
        r = self._call(self.cli_get_state, GetRobotState.Request(), timeout=3.0)
        return r.robot_state if r else -1

    def recover_safety(self, verbose=True):
        s = self.get_robot_state()
        name = self._STATE_NAMES.get(s, f'UNKNOWN({s})')
        if verbose:
            print(f'   robot_state={s} ({name})')
        if s in (5, 9):
            print('   safe-stop 감지 → recovery 시퀀스 실행')
            for cli, name_ in [(self.cli_safety_mode, 'safety_mode'),
                               (self.cli_safe_reset, 'safe_stop_reset')]:
                if not cli.wait_for_service(timeout_sec=2.0):
                    print(f'   서비스 미응답: {name_}')
                    return False
            m = SetSafetyMode.Request(); m.safety_mode = 2; m.safety_event = 0
            self._call(self.cli_safety_mode, m, timeout=3.0)
            time.sleep(0.5)
            rs_req = SetSafeStopResetType.Request(); rs_req.reset_type = 0
            self._call(self.cli_safe_reset, rs_req, timeout=3.0)
            time.sleep(0.5)
            m2 = SetSafetyMode.Request(); m2.safety_mode = 1; m2.safety_event = 0
            self._call(self.cli_safety_mode, m2, timeout=3.0)
            time.sleep(0.5)
            return self.get_robot_state() in (1, 2)
        if s == 6:
            print('   !! EMERGENCY_STOP — 펜던트의 비상정지 버튼을 풀고 재시도')
            return False
        return True

    def activate_robot(self):
        self._wait(self.cli_mode, 'set_robot_mode')
        self._wait(self.cli_ctrl, 'set_robot_control')
        self.recover_safety(verbose=True)
        m = SetRobotMode.Request(); m.robot_mode = 1
        self._call(self.cli_mode, m)
        c = SetRobotControl.Request(); c.robot_control = 1
        self._call(self.cli_ctrl, c)
        final_s = self.get_robot_state()
        print(f'   활성화 후 robot_state={final_s} ({self._STATE_NAMES.get(final_s,"?")})')

    # ----- 그리퍼 (13번과 동일) -----
    def gripper_init(self):
        for cli, name in [(self.cli_g_open, 'flange_serial_open'),
                          (self.cli_g_close, 'flange_serial_close'),
                          (self.cli_g_write, 'flange_serial_write')]:
            self._wait(cli, name)
        try:
            pre_close = FlangeSerialClose.Request(); pre_close.port = GRIPPER_PORT
            self._call(self.cli_g_close, pre_close, timeout=3.0)
        except Exception:
            pass
        time.sleep(0.3)
        for attempt in range(3):
            req = FlangeSerialOpen.Request()
            req.port = GRIPPER_PORT; req.baudrate = GRIPPER_BAUD
            req.bytesize = 8; req.parity = 0; req.stopbits = 1
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
            raise RuntimeError('그리퍼 시리얼 open 실패')
        self._gripper_serial_open = True
        time.sleep(0.1)
        w = FlangeSerialWrite.Request()
        w.port = GRIPPER_PORT; w.data = fc06_torque_enable()
        self._call(self.cli_g_write, w)
        time.sleep(0.2)

    def gripper_set(self, pos: int, settle=1.0, label=''):
        tag = f'[{label}]' if label else ''
        if not self._gripper_serial_open:
            print(f'  !! 그리퍼{tag} 시리얼 미오픈'); return False
        w_t = FlangeSerialWrite.Request()
        w_t.port = GRIPPER_PORT; w_t.data = fc06_torque_enable()
        self._call(self.cli_g_write, w_t, timeout=3.0)
        time.sleep(0.15)
        w_p = FlangeSerialWrite.Request()
        w_p.port = GRIPPER_PORT; w_p.data = fc16_position(pos)
        rp = self._call(self.cli_g_write, w_p, timeout=3.0)
        if not (rp and rp.success):
            print(f'  !! 그리퍼{tag} 위치 명령 실패 (pos={pos})'); return False
        time.sleep(settle)
        return True

    def gripper_open(self):  return self.gripper_set(GRIP_OPEN_POS,  label='OPEN')
    def gripper_close(self): return self.gripper_set(GRIP_CLOSE_POS, label='CLOSE')

    def gripper_shutdown(self):
        if not self._gripper_serial_open: return
        req = FlangeSerialClose.Request(); req.port = GRIPPER_PORT
        self._call(self.cli_g_close, req)
        self._gripper_serial_open = False

    # ----- 자세 조회 -----
    def get_current_pose(self, space='task'):
        self._wait(self.cli_get_pose, 'system/get_current_pose')
        req = GetCurrentPose.Request()
        req.space_type = 1 if space == 'task' else 0
        r = self._call(self.cli_get_pose, req, timeout=3.0)
        if not (r and r.success):
            raise RuntimeError('GetCurrentPose 실패')
        return list(r.pos)

    def _current_joints_rad(self):
        deg = self.get_current_pose('joint')
        return [float(math.radians(a)) for a in deg]

    def _check_joint_limits(self, traj_deg):
        for k, pt in enumerate(traj_deg):
            for i, j in enumerate(pt):
                lo, hi = SAFE_JOINT_LIMITS_DEG[i]
                if not (lo <= j <= hi):
                    raise RuntimeError(f'관절 한계 초과 point {k} j{i+1}={j:.2f}')

    def _send_trajectory_action(self, joints_deg, duration=MOVE_DURATION_SEC):
        if not self.traj_action.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('FollowJointTrajectory 액션 서버 미응답')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = [math.radians(float(a)) for a in joints_deg]
        sec = int(duration); nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [point]
        send_fut = self.traj_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            raise RuntimeError('Trajectory 골 거부됨')
        result_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=duration + 30.0)
        if result_fut.result().status != 4:
            raise RuntimeError(f'Trajectory 실행 실패 status={result_fut.result().status}')

    def _execute_spline(self, traj_rad):
        traj_deg = np.rad2deg(np.asarray(traj_rad))
        self._check_joint_limits(traj_deg)
        final_deg = traj_deg[-1].tolist()
        delta = [round(float(final_deg[i] - traj_deg[0][i]), 2) for i in range(6)]
        max_change = max(abs(d) for d in delta)
        print(f'   trajectory end (deg): {[round(v,2) for v in final_deg]}')
        print(f'   delta (deg): {delta}  (max |Δ|={max_change:.2f})')
        t0 = time.time()
        self._send_trajectory_action(final_deg)
        print(f'   trajectory 완료, elapsed={(time.time()-t0)*1000:.0f}ms')

    def move_joint_deg(self, joints_deg, duration=MOVE_DURATION_SEC):
        self._send_trajectory_action(list(joints_deg), duration)

    def move_line_safe(self, xyz_mm, rpy_deg=None, fill_ratio=None):
        """
        액체 안전 버전 move_line. fill_ratio 임계 이상이면 plan_safe 가 rpy 를
        top-down 으로 강제. trajectory 시간(duration) 도 fill_ratio 에 비례해 늘림.
        """
        if self.planner is None:
            raise RuntimeError('LiquidSafePlanner 미초기화')
        if rpy_deg is None:
            rpy_deg = self.top_down_rpy
        x, y, z_user = xyz_mm
        z = z_user + self.tcp_z_offset
        if not (WORK_X[0] <= x <= WORK_X[1] and
                WORK_Y[0] <= y <= WORK_Y[1] and
                WORK_Z[0] <= z_user <= WORK_Z[1]):
            raise RuntimeError(f'좌표 ({x:.1f},{y:.1f},{z_user:.1f}) mm 작업영역 밖')
        target_xyz_m = [x*1e-3, y*1e-3, z*1e-3]
        quat = _rpy_deg_to_quat_wxyz(rpy_deg)
        start = self._current_joints_rad()
        traj, ms = self.planner.plan_safe(start, target_xyz_m, quat, fill_ratio)
        if traj is None:
            raise RuntimeError(f'plan_safe FAIL (target={xyz_mm}mm rpy={rpy_deg})')
        # 천천히 — fill_ratio 가 높을수록 duration 늘림
        dur = MOVE_DURATION_SEC
        if fill_ratio is not None and fill_ratio >= FILL_CONSERVATIVE_THR:
            dur = MOVE_DURATION_SEC * (1.0 + min(1.0, fill_ratio))
            print(f'   [안전] fill_ratio={fill_ratio:.2f} → duration {MOVE_DURATION_SEC:.1f}s '
                  f'→ {dur:.1f}s 로 연장')
        traj_deg = np.rad2deg(np.asarray(traj))
        self._check_joint_limits(traj_deg)
        final_deg = traj_deg[-1].tolist()
        t0 = time.time()
        self._send_trajectory_action(final_deg, duration=dur)
        print(f'   trajectory 완료 ({(time.time()-t0)*1000:.0f}ms, plan {ms:.0f}ms)')


# ============== 드라이버 리셋 (13번과 동일, 14번 PID 추가) ==============
def reset_robot_driver():
    print('\n[준비 0/4] 기존 ROS / 카메라 / 강의 스크립트 정리 + 두산 드라이버 새로 시작')
    for c in ['pkill -9 -f dsr', 'pkill -9 -f rviz2', 'pkill -9 -f DRCF',
              'pkill -9 -f ros2', 'pkill -9 -f realsense']:
        os.system(c + ' 2>/dev/null')

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
    subprocess.Popen(f'gnome-terminal -- bash -c "{ros_cmd}; exec bash"', shell=True)
    print(f'   새 터미널에서 {BRINGUP_LAUNCH} 실행 (host={ROBOT_IP})')
    print(f'   드라이버 안정화 대기 ({DRIVER_WAIT_SEC}초)')
    for i in range(DRIVER_WAIT_SEC, 0, -1):
        sys.stdout.write(f'\r   대기 중... {i:>2}초 '); sys.stdout.flush()
        time.sleep(1)
    print('\n   드라이버 준비 완료.')

    if os.path.exists(RVIZ_CONFIG):
        rviz_cmd = (
            'source /opt/ros/humble/setup.bash && '
            'source ~/doosan_ws/install/setup.bash && '
            f'LIBGL_ALWAYS_SOFTWARE=1 exec rviz2 -d {RVIZ_CONFIG}'
        )
        subprocess.Popen(['bash', '-lc', rviz_cmd], start_new_session=True,
                         stdout=open('/tmp/rviz2_dsr.log', 'w'), stderr=subprocess.STDOUT)
        print('   RViz 백그라운드 실행')


# ============== Pick & Place 사이클 ==============
def execute_liquid_pick_place(robot, det, args):
    """
    액체 인지 pick & place 한 사이클.
      - container 의 base xyz 로 진입 → 그립 → 들어올림 → place → HOME
      - fill_ratio 에 따라 자세/속도 보수화 (LiquidSafePlanner.plan_safe 안에서 처리)
    """
    container_base = det['container_base']
    fill = det['fill_info']
    spills = det.get('spills', [])
    cname = det['container']['class_name']
    fill_ratio = fill['fill_ratio']

    bx, by, bz = container_base['base_xyz_mm']
    print(f'\n  -- 액체 Pick&Place ({cname}, fill={fill_ratio*100:.1f}%, '
          f'vol~{fill["volume_ml"]:.0f}ml, method={fill["method"]}) --')

    # puddle obstacle 추가
    if spills and robot.planner is not None:
        spill_aabbs = [s['base_aabb_mm'] for s in spills]
        robot.planner.add_spill_obstacles(spill_aabbs)

    # 모드: HOLLOW (컵·병 입구 가장자리 살짝 잡기)
    robot.tcp_z_offset = GRIPPER_TCP_OFFSET_NORMAL
    grasp_dive = GRASP_DIVE_HOLLOW_MM
    grasp_z = bz - grasp_dive
    approach_xyz = [bx, by, bz + Z_APPROACH]
    pick_xyz     = [bx, by, grasp_z]
    lifted_xyz   = [bx, by, bz + Z_LIFT]
    place_xyz    = [bx + PLACE_OFFSET_XYZ[0],
                    by + PLACE_OFFSET_XYZ[1],
                    grasp_z + PLACE_OFFSET_XYZ[2]]
    place_app_xyz = [place_xyz[0], place_xyz[1], bz + Z_APPROACH + PLACE_OFFSET_XYZ[2]]

    print(f'   1) Pre-grasp → {[round(v,1) for v in approach_xyz]} mm')
    if not args.dry_run: robot.move_line_safe(approach_xyz, fill_ratio=fill_ratio)
    print('   2) 그리퍼 열기')
    if not args.dry_run: robot.gripper_open()
    print(f'   3) 하강 → {[round(v,1) for v in pick_xyz]} mm')
    if not args.dry_run: robot.move_line_safe(pick_xyz, fill_ratio=fill_ratio)
    print('   4) 그리퍼 닫기 (잡기)')
    if not args.dry_run: robot.gripper_close()
    print(f'   5) 들어올리기 → {[round(v,1) for v in lifted_xyz]} mm  '
          f'(fill 높으면 수직 강제 + duration 연장)')
    if not args.dry_run: robot.move_line_safe(lifted_xyz, fill_ratio=fill_ratio)

    if args.no_place:
        print('\n  -- Place 생략 (--no-place) --')
    else:
        print(f'\n  -- Place (offset={PLACE_OFFSET_XYZ}mm) --')
        if not args.dry_run: robot.move_line_safe(place_app_xyz, fill_ratio=fill_ratio)
        if not args.dry_run: robot.move_line_safe(place_xyz,     fill_ratio=fill_ratio)
        if not args.dry_run: robot.gripper_open()
        if not args.dry_run: robot.move_line_safe(place_app_xyz, fill_ratio=fill_ratio)

    print('\n  -- HOME 복귀 --')
    if not args.dry_run: robot.move_joint_deg(HOME_JOINT_DEG)


# ============== 메인 ==============
def banner(text):
    print('\n' + '=' * 70)
    print(f'  {text}')
    print('=' * 70)


def command_loop(robot, vision, args):
    print('\n명령:')
    print('  scan          : 라이브 카메라 — 컨테이너+액체+puddle 검출 후 pick&place')
    print('  fill          : 비전만 — 현재 fill_ratio·volume 측정 (모션 X)')
    print('  spills        : 비전만 — 현재 puddle 검출 후 cuboid 좌표 출력')
    print('  home          : HOME 자세로 이동')
    print('  world_clear   : 누적 puddle obstacle 제거')
    print('  state         : robot_state 출력')
    print('  recover       : safe-stop 후 복구')
    print('  q             : 종료')

    debug_path = os.path.join(WORKSPACE_DIR, 'last_liquid_debug.jpg')

    while True:
        try:
            line = input('\n> ').strip()
        except EOFError:
            break
        if line in ('', 'q', 'exit'):
            break
        if line == 'home' and robot is not None:
            try:
                if not args.dry_run: robot.move_joint_deg(HOME_JOINT_DEG)
                print('  HOME 완료')
            except Exception as e:
                print(f'  HOME 실패: {e}')
            continue
        if line == 'state' and robot is not None:
            s = robot.get_robot_state()
            print(f'  robot_state={s} ({robot._STATE_NAMES.get(s,"?")})')
            continue
        if line == 'recover' and robot is not None:
            robot.recover_safety(); continue
        if line == 'world_clear' and robot is not None and robot.planner is not None:
            robot.planner.world_clear(); continue

        if line == 'fill':
            # 한 프레임 잡고 best 컨테이너의 fill_ratio 만 출력
            color, depth_mm, df = vision._grab_frames()
            if color is None:
                print('  프레임 없음'); continue
            cs = vision.detect_containers(color)
            if not cs:
                print('  컨테이너 없음'); continue
            best = cs[0]
            liq, inner = vision.segment_liquid(color, best)
            fill = vision.estimate_fill_level(depth_mm, inner, liq)
            print(f'  {best["class_name"]} conf={best["conf"]:.2f}  '
                  f'fill={fill["fill_ratio"]*100:.1f}%  vol~{fill["volume_ml"]:.0f}ml  '
                  f'method={fill["method"]}')
            continue

        if line == 'spills':
            color, depth_mm, df = vision._grab_frames()
            if color is None: print('  프레임 없음'); continue
            cs = vision.detect_containers(color)
            exclude = np.zeros(color.shape[:2], dtype=bool)
            for c in cs:
                exclude |= c['mask']
            spills = vision.detect_spills(color, depth_mm, df, exclude_mask=exclude)
            print(f'  puddle {len(spills)} 개:')
            for i, s in enumerate(spills):
                cx, cy, cz, w, h = s['base_aabb_mm']
                print(f'    [{i}] base=({cx:+.0f},{cy:+.0f},{cz:+.0f})mm  size={w:.0f}x{h:.0f}mm')
            continue

        if line == 'scan':
            det = vision.interactive_liquid_preview(save_debug=debug_path)
            if det is None:
                print('  취소됨'); continue
            cname = det['container']['class_name']
            fill = det['fill_info']
            bx, by, bz = det['container_base']['base_xyz_mm']
            print(f'  확정: {cname}  fill={fill["fill_ratio"]*100:.1f}%  '
                  f'vol~{fill["volume_ml"]:.0f}ml')
            print(f'  base=({bx:+.1f},{by:+.1f},{bz:+.1f})mm  spills={len(det["spills"])}')
            print(f'  스냅샷: {debug_path}')
            if robot is None:
                print('  (--vision-only: pick&place SKIP)')
                continue
            try:
                execute_liquid_pick_place(robot, det, args)
                print('  ✓ 사이클 완료')
            except Exception as e:
                print(f'  !! 사이클 실패: {e}')
                try:
                    if not args.dry_run: robot.recover_safety()
                except Exception:
                    pass
                try:
                    if not args.dry_run: robot.move_joint_deg(HOME_JOINT_DEG)
                except Exception:
                    pass
            continue

        print(f'  알 수 없는 명령: {line!r} — help 는 처음에 출력된 목록 참조')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='로봇 모션 없이 비전+plan 만 수행')
    parser.add_argument('--no-place', action='store_true',
                        help='Pick 까지만, Place 생략')
    parser.add_argument('--quick', action='store_true',
                        help='드라이버 리셋 SKIP')
    parser.add_argument('--vision-only', action='store_true',
                        help='로봇/ROS/cuRobo 전체 SKIP — 비전 단독 테스트')
    args = parser.parse_args()

    banner('[액체 Pick & Place — 두산 e0509 + RealSense + YOLO26-seg + SAM2 + RH-P12-RN-A]')
    if args.dry_run:    print('  *** DRY-RUN: 실제 모션 없음 ***')
    if args.vision_only: print('  *** VISION-ONLY: 로봇/cuRobo 전체 비활성 ***')

    if not args.vision_only and not _ROBOT_IMPORTS_OK:
        print(f'\n!! ROS2/curobo import 실패: {_ROBOT_IMPORT_ERROR}')
        print('   --vision-only 로 실행하면 비전만 테스트 가능')
        sys.exit(1)

    robot = None
    vision = None

    if not args.vision_only and not args.quick:
        reset_robot_driver()

    try:
        if not args.vision_only:
            rclpy.init()
            robot = LiquidPickAndPlace()

        print('\n[준비 1/4] 캘리브레이션 + YOLO26-seg + SAM2 + RealSense')
        vision = LiquidVisionSystem(vision_only=args.vision_only)
        if not args.vision_only and not args.dry_run:
            robot.activate_robot()
            robot.gripper_init()
            robot.gripper_open()
        print('   준비 OK')

        if not args.vision_only:
            print('\n[준비 2/4] cuRobo MotionGen + LiquidSafePlanner 로드 + warmup')
            robot.planner = LiquidSafePlanner()
            print('   cuRobo 준비 OK')

            print(f'\n[준비 3/4] HOME 자세로 이동 {HOME_JOINT_DEG} deg')
            if not args.dry_run:
                try:
                    robot.move_joint_deg(HOME_JOINT_DEG)
                    print('   HOME 도착')
                except Exception as e:
                    print(f'   !! HOME 이동 실패: {e}')

        print('\n[준비 4/4] 명령 루프 진입')
        command_loop(robot, vision, args)

    except Exception as e:
        print(f'\n!! 에러: {e}')
        import traceback
        traceback.print_exc()
    finally:
        try:
            if robot is not None: robot.gripper_shutdown()
        except Exception:
            pass
        if vision is not None:
            vision.stop()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass
        if robot is not None:
            robot.destroy_node()
        if _ROBOT_IMPORTS_OK and 'rclpy' in sys.modules and rclpy.ok():
            rclpy.shutdown()
        print('\n[정리 완료, 종료]')


if __name__ == '__main__':
    main()
