"""
[통합 Pick & Place — 두산 e0509 + RealSense + YOLOv8 + RH-P12-RN-A]

두 단계 흐름:
  [A] 준비 (한 번만) — 실행 직후 자동 진행:
        0. 잔존 ROS/RealSense/강의 스크립트 정리 + dsr_bringup2_moveit 새로 띄움
        1. 캘리브레이션 + YOLO + RealSense + 로봇 Servo ON + 그리퍼 init + 열기
        2. HOME 관절 자세로 이동
  [B] 명령 루프 (사용자 입력에 따라) — 준비 끝난 뒤부터:
        잡을 물체 클래스 입력 (예: "cup", "bottle book")
          → YOLO 검출 → 좌표 변환 → 사용자 확인
          → Pick (pre-grasp → 열기 → 하강 → 닫기 → 들기)
          → Place (Y+200mm → 하강 → 열기 → 들기)
          → HOME 복귀 → 다시 명령 대기
        'q' 또는 빈 입력으로 종료

전제 조건:
  - calibration_data/calibration_result.npz (T_cam2base) 존재
  - RealSense 카메라가 캘리브레이션 시점과 동일하게 고정 설치
  - 그리퍼는 두산 툴 플랜지에 결선
  - 두산 컨트롤러(110.120.1.18)에 PC 가 ping 가능한 네트워크 상태

실행:
  python3 12_비전_피크앤플레이스.py            # 준비 → 명령 루프
  python3 12_비전_피크앤플레이스.py --quick    # 드라이버 리셋 skip
  python3 12_비전_피크앤플레이스.py --dry-run  # 모션 없이 좌표만 보고 흐름 검증
"""

import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from dsr_msgs2.srv import (
    SetRobotMode, SetRobotControl, Ikin, GetCurrentPose,
    GetRobotState, SetSafetyMode, SetSafeStopResetType,
    FlangeSerialOpen, FlangeSerialClose, FlangeSerialWrite,
    ChangeCollisionSensitivity, ChangeOperationSpeed,
    MoveLine,
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

# 모션 시간 (FollowJointTrajectory: time_from_start)
# 1.5 는 ramp 부족으로 일자/거침 — 3.0 으로 복원해서 trapezoidal 가속/감속 시간 확보.
# 이전 5.0 → 단축. status=6 path tolerance abort 자주 발생하면 4.0 / 5.0 으로 복원.
MOVE_DURATION_SEC = 3.0
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# Doosan native motion (move_line) 의 vel/acc — cartesian descend.
# vel=[mm/s, deg/s], acc=[mm/s², deg/s²]. 절반 (75/30, 250/75) 으로 낮췄더니 descend
# 자체가 시작 안 되는 증상 → 원래 값으로 복원. joint motion 부드러움은 MOVE_DURATION_SEC
# 으로 따로 제어됨 (cartesian 에는 영향 없음).
NATIVE_VEL = [150.0, 60.0]
NATIVE_ACC = [500.0, 150.0]
# 시작 시 한 번 세팅
COLLISION_SENSITIVITY = 30   # 0~100. 기본 50 보다 낮춰서 trapezoidal 시작 spike 오인 방지.
OPERATION_SPEED = 100        # 1~100 전역 속도 배율. native vel/acc 와 곱셈으로 적용.

# 두산 e0509 펌웨어가 적용하는 실제 관절 한계 (URDF 보다 빡빡할 수 있음).
# Ikin 결과가 이 안에 들어와야 trajectory action 이 abort 안 됨. 안전 마진 포함.
SAFE_JOINT_LIMITS_DEG = [
    (-360.0, 360.0),    # joint_1
    ( -90.0,  90.0),    # joint_2  (펌웨어 ±95, 마진 5도)
    (-150.0, 150.0),    # joint_3  (URDF ±155, 마진 5도)
    (-360.0, 360.0),    # joint_4
    (-135.0, 135.0),    # joint_5
    (-360.0, 360.0),    # joint_6
]
IK_SOL_SPACE_TRIES = list(range(8))   # Ikin sol_space 후보 (0~7)

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
        depth_m = self._depth_at(df, u, v)
        if depth_m <= 0.05:
            return None

        cam_xyz_m = rs.rs2_deproject_pixel_to_point(self.intr, [u, v], depth_m)
        cam_h = np.array([cam_xyz_m[0], cam_xyz_m[1], cam_xyz_m[2], 1.0])
        base_h = self.T_cam2base @ cam_h
        base_mm = (base_h[:3] * 1000.0).astype(float)
        label = self.classes.get(cls_id, f'id{cls_id}')

        if save_debug:
            dbg = color.copy()
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(dbg, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(dbg, f'{label} {conf:.2f} d={depth_m*1000:.0f}mm',
                        (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(save_debug, dbg)

        return {
            'base_xyz_mm': (float(base_mm[0]), float(base_mm[1]), float(base_mm[2])),
            'label': label, 'conf': conf,
            'pixel': (u, v), 'depth_m': depth_m,
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
            cv2.destroyWindow(win)
            cv2.waitKey(1)   # window 즉시 닫히도록 한 번 더 펌프


# ============== 로봇 + 그리퍼 컨트롤러 ==============
class PickAndPlace(Node):
    def __init__(self):
        super().__init__('vision_pick_and_place')
        self.cli_mode = self.create_client(SetRobotMode, f'/{NS}/system/set_robot_mode')
        self.cli_ctrl = self.create_client(SetRobotControl, f'/{NS}/system/set_robot_control')
        # IK 서비스 (Pose → 관절 각도)
        self.cli_ikin = self.create_client(Ikin, f'/{NS}/motion/ikin')
        # 현재 TCP/관절 자세 조회 (자세 캘리브레이션용)
        self.cli_get_pose = self.create_client(GetCurrentPose, f'/{NS}/system/get_current_pose')
        # Safety / 충돌 후 fault recovery
        self.cli_get_state   = self.create_client(GetRobotState,         f'/{NS}/system/get_robot_state')
        self.cli_safety_mode = self.create_client(SetSafetyMode,         f'/{NS}/system/set_safety_mode')
        self.cli_safe_reset  = self.create_client(SetSafeStopResetType,  f'/{NS}/system/set_safe_stop_reset_type')
        # 실 로봇 매끄러움 튜닝
        self.cli_collision = self.create_client(ChangeCollisionSensitivity,
                                                 f'/{NS}/system/change_collision_sensitivity')
        self.cli_op_speed = self.create_client(ChangeOperationSpeed,
                                                f'/{NS}/motion/change_operation_speed')
        # 카르테시안 직선 단발 (pick / place descend 의 수직 강하 보장용)
        self.cli_move_line = self.create_client(MoveLine, f'/{NS}/motion/move_line')
        # 현재 활성 top-down 자세 (사용자가 cal_top_down 으로 갱신 가능)
        self.top_down_rpy = list(TOP_DOWN_RPY_DEG)
        # 그리퍼 TCP z 오프셋 (런타임에 set_tcp_z 명령으로 갱신 가능)
        self.tcp_z_offset = float(GRIPPER_TCP_OFFSET_Z)
        # 직전 IK 성공 sol_space — 다음 호출에서 우선 시도해서 손목 ±180° flip 방지
        self._last_ik_sol = None
        # MoveIt 컨트롤러 trajectory 액션 (RViz 에 보이는 그 액션)
        self.traj_action = ActionClient(
            self, FollowJointTrajectory,
            f'/{NS}/{MOVEIT_CONTROLLER}/follow_joint_trajectory'
        )
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

        # 0a) 이전 run 잔여 trajectory goal idle 처리
        # rclpy ActionClient 는 cancel_all_goals 미공개 → action 서버 연결만 확인하고
        # 1초 wait 으로 컨트롤러가 stale goal 을 자체 종료하도록 둠.
        # (시작 시 robot_state=2 MOVING 으로 보고되던 증상 회피)
        if self.traj_action.wait_for_server(timeout_sec=1.0):
            time.sleep(1.0)

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

        # 4) 실 하드웨어 매끄러움 튜닝 (시뮬레이션은 무시되거나 무관)
        # 4a) 충돌 감도 — 기본 50 은 trapezoidal 가속 시작 점의 force spike 를
        #     충돌로 오인해서 잠깐 멈춤. 30 정도로 낮추면 안정. 너무 낮추면 실 충돌 위험.
        if self.cli_collision.wait_for_service(timeout_sec=1.0):
            req = ChangeCollisionSensitivity.Request()
            req.sensitivity = COLLISION_SENSITIVITY
            r = self._call(self.cli_collision, req, timeout=3.0)
            if r and r.success:
                print(f'   collision sensitivity = {COLLISION_SENSITIVITY}/100')
            else:
                print('   (collision sensitivity 설정 실패 — 무시하고 계속)')
        # 4b) 전역 속도 배율 — 100 이 native vel/acc 그대로
        if self.cli_op_speed.wait_for_service(timeout_sec=1.0):
            req = ChangeOperationSpeed.Request()
            req.speed = OPERATION_SPEED
            r = self._call(self.cli_op_speed, req, timeout=3.0)
            if r and r.success:
                print(f'   operation speed = {OPERATION_SPEED}/100')

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

    # ---- 모션 (Ikin → FollowJointTrajectory, RViz/MoveIt 경로) ----
    def _send_trajectory_to_joints(self, joints_deg, duration=MOVE_DURATION_SEC):
        """6축 관절 각도(도)로 MoveIt 컨트롤러에 trajectory 골 전송 — 동기 대기.

        목표 자세 한 점(t=duration) 만 보낸다. ros2_control 이 현재 위치에서 보간.
        시작점을 명시하면 timing/단위 미세 차이로 컨트롤러가 받지 않고 hang 할 수 있어 사용 X.
        2-point cubic Hermite 도 시도했으나 Doosan PID 가 trapezoidal 외 프로파일 tracking 실패.
        단발 endpoint + path_tolerance 완화 조합이 가장 robust.
        """
        if not self.traj_action.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('FollowJointTrajectory 액션 서버 미응답 — '
                               'dsr_bringup2_moveit.launch.py 가 살아있는지 확인')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = [math.radians(float(a)) for a in joints_deg]
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [point]

        # Path tolerance 완화 — controller 의 내장 trapezoidal 보간으로도 가끔 lag.
        # 0.25 rad(14°) 안에서 자연스러운 control loop lag 허용. 최종 도달 정확도는 별도.
        for jn in JOINT_NAMES:
            pt_tol = JointTolerance()
            pt_tol.name = jn
            pt_tol.position = 1.0    # 0.5 → 1.0 (J4 가 0.5 rad 도 빈번 위반 — 사실상 endpoint check 모드)
            goal.path_tolerance.append(pt_tol)
            g_tol = JointTolerance()
            g_tol.name = jn
            g_tol.position = 0.05
            goal.goal_tolerance.append(g_tol)
        # 기본값 Duration(0,0) 이면 trajectory 종료 시각에서 단 1ms 라도 늦으면 ABORTED(status=6).
        # +2s 마진 → 컨트롤러 trapezoidal lag 흡수.
        goal.goal_time_tolerance = Duration(sec=2, nanosec=0)

        send_fut = self.traj_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            raise RuntimeError('Trajectory 골이 거부됨 (Servo OFF / 한계 초과 가능)')

        self._current_gh = gh   # 's' 키 cancel 용 공유 핸들
        result_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=duration + 30.0)
        self._current_gh = None
        status = result_fut.result().status
        # 일시적 controller race — sleep 후 재전송. 1회 → 3회 (실 e0509 stutter 회복용).
        for retry in range(3):
            if status != 6:
                break
            time.sleep(0.5)
            send_fut = self.traj_action.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
            gh = send_fut.result()
            if gh is None or not gh.accepted:
                continue
            result_fut = gh.get_result_async()
            rclpy.spin_until_future_complete(self, result_fut, timeout_sec=duration + 30.0)
            status = result_fut.result().status
        if status != 4:   # GoalStatus.STATUS_SUCCEEDED
            raise RuntimeError(f'Trajectory 실행 실패 (status={status})')

    def move_joint_deg(self, joints_deg, duration=MOVE_DURATION_SEC):
        """관절 각도(도) 6개로 직접 이동."""
        self._send_trajectory_to_joints(joints_deg, duration)

    def _send_trajectory_multi(self, waypoints_joints_deg, segment_durations, smooth_steps=1):
        """여러 waypoint 를 한 trajectory action 으로 — segment 사이 정지 없이 연속 모션.
        smooth_steps < 2 (기본 1): waypoint 만 trajectory 에 담아 controller 가 알아서 보간.
        smooth_steps >= 2: 각 segment 안에서 cubic smoothstep sub-waypoint 분포 (PATH_TOL 위험,
          작은 값/긴 duration 에서만 사용).
        첫 waypoint 는 시작점 (현재 robot 위치) 미지라 단일 point.

        waypoints_joints_deg: [[6 joint deg], ...]
        segment_durations: 각 waypoint 까지 segment 시간."""
        if not self.traj_action.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('FollowJointTrajectory 액션 서버 미응답')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        t = 0.0
        for i, (w, dt) in enumerate(zip(waypoints_joints_deg, segment_durations)):
            if i == 0 or smooth_steps < 2:
                # 첫 segment 또는 smooth 안 함 → waypoint 1개만
                t += float(dt)
                pt = JointTrajectoryPoint()
                pt.positions = [math.radians(float(a)) for a in w]
                pt.time_from_start = Duration(
                    sec=int(t), nanosec=int((t - int(t)) * 1e9))
                goal.trajectory.points.append(pt)
            else:
                # prev_w → w 를 smooth_steps 개 sub-waypoint 로 cubic smoothstep 분포
                prev_w = waypoints_joints_deg[i - 1]
                sub_dt = float(dt) / smooth_steps
                for k in range(1, smooth_steps + 1):
                    progress = k / smooth_steps
                    s = progress * progress * (3.0 - 2.0 * progress)  # ease-in-out
                    sub = [prev_w[j] + (w[j] - prev_w[j]) * s for j in range(len(w))]
                    t += sub_dt
                    pt = JointTrajectoryPoint()
                    pt.positions = [math.radians(float(a)) for a in sub]
                    pt.time_from_start = Duration(
                        sec=int(t), nanosec=int((t - int(t)) * 1e9))
                    goal.trajectory.points.append(pt)

        # 중간 waypoint pass-through 매끄럽게 — velocity 미지정 시 controller 가
        # 각 waypoint 에서 v=0 으로 해석해서 멈췄다 가속 → 버벅임. 인접 waypoint 차분으로
        # 추정해 채워주면 spline 이 연속 통과. 마지막 waypoint 만 v=0 (정지).
        pts = goal.trajectory.points
        if len(pts) >= 2:
            times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                     for p in pts]
            njoints = len(pts[0].positions)
            for i in range(len(pts) - 1):   # 마지막은 v=0 유지 (아래에서 명시)
                if i == 0:
                    dt_seg = max(times[1] - times[0], 1e-6)
                    vel = [(pts[1].positions[j] - pts[0].positions[j]) / dt_seg
                           for j in range(njoints)]
                else:
                    dt_total = max(times[i + 1] - times[i - 1], 1e-6)
                    vel = [(pts[i + 1].positions[j] - pts[i - 1].positions[j]) / dt_total
                           for j in range(njoints)]
                pts[i].velocities = vel
            pts[-1].velocities = [0.0] * njoints

        # Path/goal tolerance 완화 (단일 endpoint trajectory 와 동일)
        for jn in JOINT_NAMES:
            pt_tol = JointTolerance()
            pt_tol.name = jn
            pt_tol.position = 1.0    # 0.5 → 1.0 (J4 가 0.5 rad 도 빈번 위반 — 사실상 endpoint check 모드)
            goal.path_tolerance.append(pt_tol)
            g_tol = JointTolerance()
            g_tol.name = jn
            g_tol.position = 0.05
            goal.goal_tolerance.append(g_tol)
        # 기본 0 이면 trajectory 종료 시각 초과 시 즉시 ABORTED — +2s 마진
        goal.goal_time_tolerance = Duration(sec=2, nanosec=0)

        send_fut = self.traj_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            raise RuntimeError('Multi-WP Trajectory 골이 거부됨')

        result_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=t + 30.0)
        status = result_fut.result().status
        # 1회 → 3회 retry (실 e0509 multi-WP stutter 회복용)
        for retry in range(3):
            if status != 6:
                break
            time.sleep(0.5)
            send_fut = self.traj_action.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
            gh = send_fut.result()
            if gh is None or not gh.accepted:
                continue
            result_fut = gh.get_result_async()
            rclpy.spin_until_future_complete(self, result_fut, timeout_sec=t + 30.0)
            status = result_fut.result().status
        if status != 4:
            raise RuntimeError(f'Multi-WP Trajectory 실패 (status={status})')

    def move_line_base_multi(self, waypoints_xyz_rpy, segment_durations):
        """여러 base waypoint (xyz_mm, rpy_deg) 를 한 trajectory 로 — 매끄럽고 빠름.
        waypoints_xyz_rpy: [((x,y,z_mm), (r,p,y_deg)), ...]
        segment_durations: 각 waypoint 까지 시간."""
        joints_list = []
        for xyz_mm, rpy_deg in waypoints_xyz_rpy:
            x, y, z_user = xyz_mm
            z = z_user + self.tcp_z_offset
            if not (WORK_X[0] <= x <= WORK_X[1] and
                    WORK_Y[0] <= y <= WORK_Y[1] and
                    WORK_Z[0] <= z_user <= WORK_Z[1]):
                raise RuntimeError(
                    f'좌표 {[round(v,1) for v in (x,y,z_user)]} mm (그리퍼 끝 기준) 가 작업영역 밖. '
                    f'X∈{WORK_X}, Y∈{WORK_Y}, Z∈{WORK_Z}.')
            self._wait(self.cli_ikin, 'motion/ikin')
            joints_list.append(self._ikin_with_fallback(
                [float(x), float(y), float(z),
                 float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])]))
        self._send_trajectory_multi(joints_list, segment_durations)

    def move_line_base_multi_smooth(self, waypoints_xyz_rpy,
                                     vel=None, acc=None):
        """검증된 ros2_control multi-WP trajectory 한 가지 path 로 통일.

        이전엔 Doosan native motion/move_spline_task 를 사용했으나 실 로봇에서:
          - rpy=[0,180,0] 근처 wrist singularity 에서 spline 보간 fail
          - dt jitter / system 부하 시 controller 가 abort
        등의 이유로 안정성이 떨어져서, 검증된 move_line_base_multi (IK 풀어서
        FollowJointTrajectory 로 보내는 path) 하나로 통일.

        호환성을 위해 동일 signature 유지. vel/acc 인자는 무시됨 (segment 당
        고정 duration 사용 — 부드러움은 _send_trajectory_multi 의 velocity hint
        + goal_time_tolerance 가 보장).
        """
        if not waypoints_xyz_rpy:
            raise ValueError('waypoints 비어있음')
        # vel/acc 인자는 받지만 사용 안 함 (호환성 유지). 신호만 안 보이게 처리.
        _ = vel, acc
        durations = [MOVE_DURATION_SEC] * len(waypoints_xyz_rpy)
        self.move_line_base_multi(waypoints_xyz_rpy, durations)

    def move_line_cartesian(self, xyz_mm, rpy_deg=None,
                             vel=None, acc=None):
        """Doosan native motion/move_line — 단발 카르테시안 직선 보장.

        pick descend / place descend 처럼 수직 강하해야 하는 segment 에 사용.
        중간 경로가 직선이라 cube 옆을 안 치고 정확히 위에서 들어감.

        rpy_deg None → self.top_down_rpy 사용.
        vel None → [150 mm/s, 60 deg/s]. acc None → [500 mm/s², 150 deg/s²].
        e0509 spec 한계의 ~1/4 수준 — 안전하고 충분히 빠름.
        """
        if rpy_deg is None:
            rpy_deg = self.top_down_rpy
        if vel is None:
            vel = list(NATIVE_VEL)
        if acc is None:
            acc = list(NATIVE_ACC)
        x, y, z_user = xyz_mm
        z = z_user + self.tcp_z_offset

        # 작업영역 검증 (사용자 좌표 = 그리퍼 끝 기준)
        if not (WORK_X[0] <= x <= WORK_X[1] and
                WORK_Y[0] <= y <= WORK_Y[1] and
                WORK_Z[0] <= z_user <= WORK_Z[1]):
            raise RuntimeError(
                f'좌표 {[round(v,1) for v in (x,y,z_user)]} mm (그리퍼 끝 기준) 가 작업영역 밖. '
                f'X∈{WORK_X}, Y∈{WORK_Y}, Z∈{WORK_Z}.'
            )

        self._wait(self.cli_move_line, 'motion/move_line')
        req = MoveLine.Request()
        req.pos = [float(x), float(y), float(z),
                    float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])]
        req.vel = [float(vel[0]), float(vel[1])]
        req.acc = [float(acc[0]), float(acc[1])]
        req.time = 0.0       # vel/acc 로 자동 계산
        req.radius = 0.0     # blending off — 끝점에서 정확히 멈춤
        req.ref = 0          # DR_BASE
        req.mode = 0         # ABSOLUTE
        req.blend_type = 0   # BLENDING_SPEED_TYPE_DUPLICATE
        req.sync_type = 0    # SYNC (blocking)
        # sync_type=0 이라 service 는 motion 종료까지 block. 80mm 강하 (vel=150, acc=500)
        # 면 ~1.5s + 펌웨어 오버헤드. timeout=30.0 이었던 게 사용자에게 "강하 안 함 / hang"
        # 으로 보임 → 5.0 으로 단축. 실제 cartesian 이 5초 넘는 경우는 거의 없음.
        r = self._call(self.cli_move_line, req, timeout=5.0)
        if not (r and r.success):
            # cartesian 직선 fail 원인: task-space 직선 보간이 wrist singular / joint
            # limit 통과 시 IK 가 중간점에서 fail. 이 경우 joint trajectory 로 fallback.
            reason = 'no-response/timeout' if r is None else 'success=False'
            print(f'    [motion] move_line cartesian fail ({reason}, '
                  f'yaw={rpy_deg[2]:+.0f}° wrist singular 의심) → joint fallback')
            self.move_line_base(xyz_mm, rpy_deg=rpy_deg,
                                duration=MOVE_DURATION_SEC)

    def move_line_base(self, xyz_mm, rpy_deg=None, duration=MOVE_DURATION_SEC):
        """베이스 좌표 (mm/deg) → Ikin 으로 IK 풀고 trajectory 액션으로 이동.
        rpy_deg 미지정 시 self.top_down_rpy (cal_top_down 으로 갱신 가능) 사용.
        주의: 관절공간 보간이라 카르테시안 직선 경로는 보장되지 않음."""
        if rpy_deg is None:
            rpy_deg = self.top_down_rpy
        x, y, z_user = xyz_mm
        # 사용자/검출 좌표는 "그리퍼 끝 기준"으로 해석. 두산 TCP 가 플랜지에 설정돼있으면
        # tcp_z_offset 만큼 더해서 보내야 그리퍼 끝이 z_user 에 도달한다.
        z = z_user + self.tcp_z_offset

        # 작업 영역은 사용자 좌표(그리퍼 끝 기준)로 검사 — 거부 시 더 직관적
        if not (WORK_X[0] <= x <= WORK_X[1] and
                WORK_Y[0] <= y <= WORK_Y[1] and
                WORK_Z[0] <= z_user <= WORK_Z[1]):
            raise RuntimeError(
                f'좌표 {[round(v,1) for v in (x,y,z_user)]} mm (그리퍼 끝 기준) 가 작업영역 밖. '
                f'X∈{WORK_X}, Y∈{WORK_Y}, Z∈{WORK_Z}. '
                '검출 결과/캘리브레이션을 다시 확인하세요.'
            )

        self._wait(self.cli_ikin, 'motion/ikin')
        joints_deg = self._ikin_with_fallback(
            [float(x), float(y), float(z),
             float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])]
        )
        self._send_trajectory_to_joints(joints_deg, duration)

    def _ikin_with_fallback(self, pos):
        """
        sol_space 0~7 을 순서대로 시도하면서, SAFE_JOINT_LIMITS_DEG 안에 들어가는
        첫 IK 결과를 반환. 모두 실패하면 RuntimeError.

        두산 e0509 펌웨어가 URDF 보다 빡빡한 관절 한계(특히 joint_2/3 ±95°/±155°)
        를 적용하므로, sol_space=0 결과가 한계를 살짝 넘는 경우가 있음 → 다른 분기를 시도.

        직전에 성공한 sol_space 를 우선 시도 (self._last_ik_sol) — pick 과 place 가
        같은 cycle 안에서 일관된 분기를 쓰게 해서 손목 ±180° flip 방지.
        """
        if self._last_ik_sol is not None:
            tries = [self._last_ik_sol] + [s for s in IK_SOL_SPACE_TRIES
                                            if s != self._last_ik_sol]
        else:
            tries = list(IK_SOL_SPACE_TRIES)
        attempts = []
        for sol in tries:
            req = Ikin.Request()
            req.pos = list(pos)
            req.sol_space = int(sol)
            req.ref = 0
            r = self._call(self.cli_ikin, req, timeout=5.0)
            if not (r and r.success):
                attempts.append(f'sol={sol}: Ikin success=False')
                continue
            joints = list(r.conv_posj)
            # 안전 한계 내부 체크
            violated = []
            for i, j in enumerate(joints):
                lo, hi = SAFE_JOINT_LIMITS_DEG[i]
                if not (lo <= j <= hi):
                    violated.append(f'joint_{i+1}={j:.2f}∉[{lo:.0f},{hi:.0f}]')
            if violated:
                attempts.append(f'sol={sol}: {",".join(violated)}')
                continue
            self._last_ik_sol = sol
            print(f'   IK 풀림 (sol_space={sol}): '
                  f'{[round(j, 2) for j in joints]} deg')
            return joints
        raise RuntimeError(
            'Ikin: 어떤 sol_space 도 안전 관절 한계 안에 들어오지 않음. '
            'pose={} 가 도달 불가일 가능성. 시도 결과:\n   '.format(pos) +
            '\n   '.join(attempts)
        )


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
    print('\n[0/9] 기존 ROS / 카메라 / 강의 스크립트 정리 + 두산 드라이버 새로 시작')

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
    """검출 결과를 받아 Pick → Place → HOME 까지 한 사이클 실행."""
    bx, by, bz = det['base_xyz_mm']

    # 검출 z 로 그리퍼 TCP 오프셋 자동 선택
    if bz < FLAT_OBJECT_Z_THRESHOLD:
        robot.tcp_z_offset = GRIPPER_TCP_OFFSET_FLAT
        print(f'  ⓘ 바닥 물체 감지 (z={bz:.1f}mm < {FLAT_OBJECT_Z_THRESHOLD:.0f}) '
              f'→ FLAT 모드 (tcp_z_offset={GRIPPER_TCP_OFFSET_FLAT:.0f}mm, 그리퍼 끝)')
    else:
        robot.tcp_z_offset = GRIPPER_TCP_OFFSET_NORMAL
        print(f'  ⓘ 일반 물체 (z={bz:.1f}mm) '
              f'→ NORMAL 모드 (tcp_z_offset={GRIPPER_TCP_OFFSET_NORMAL:.0f}mm, 핑거 가운데)')

    pick_xyz = [bx, by, bz]
    approach_xyz = [bx, by, bz + Z_APPROACH]
    lifted_xyz = [bx, by, bz + Z_LIFT]
    place_xyz = [bx + PLACE_OFFSET_XYZ[0],
                 by + PLACE_OFFSET_XYZ[1],
                 bz + PLACE_OFFSET_XYZ[2]]
    place_app_xyz = [place_xyz[0], place_xyz[1], place_xyz[2] + Z_APPROACH]

    print('\n  -- Pick --')
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
    print('  진단 명령:  pose / joint / state / cal_top_down / set_tcp_z / home / recover / help')
    print('  종료: 명령창에서 q 또는 Enter (빈 입력)')
    print('=' * 70)

    debug_path = '/tmp/pickplace_detect.png'
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
        if cmd == 'help':
            print('  명령 목록:')
            print('    <물체이름>      : YOLO-World 검출 + Pick&Place')
            print('    pose            : 현재 TCP 자세 [x,y,z,rx,ry,rz] 출력')
            print('    joint           : 현재 6 관절각(deg) 출력')
            print('    cal_top_down    : 현재 자세의 RPY 를 top-down 기준으로 저장')
            print('    show_top_down   : 현재 등록된 top-down RPY 보기')
            print('    set_tcp_z <mm>  : 그리퍼 TCP z 오프셋 설정 (인자 없으면 현재값)')
            print('    set_work_z <min> <max> : 작업영역 z 한계 갱신 (mm)')
            print('    home            : HOME 관절 자세로 이동')
            print('    state           : 현재 robot_state 정수 + 이름')
            print('    recover (reset) : 충돌/safe-stop 후 수동 fault 복구')
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
    args = parser.parse_args()

    banner('[통합 Pick & Place — 두산 e0509 + RealSense + YOLOv8 + RH-P12-RN-A]')
    if args.dry_run:
        print('  *** DRY-RUN 모드: 실제 모션/그리퍼 명령은 보내지 않음 ***')

    # ===== [A] 준비 단계 (한 번만 자동 진행) =====
    if not args.quick:
        reset_robot_driver()
    else:
        print('\n[준비 0/3] 드라이버 리셋 SKIP (--quick)')

    rclpy.init()
    robot = PickAndPlace()
    vision = None

    try:
        print('\n[준비 1/3] 캘리브레이션 + YOLO + RealSense + 로봇 + 그리퍼 init')
        vision = VisionDetector()
        if not args.dry_run:
            robot.activate_robot()
            robot.gripper_init()
            robot.gripper_open()
        print('   준비 OK')

        print(f'\n[준비 2/3] HOME 자세로 이동 {HOME_JOINT_DEG} deg')
        if not args.dry_run:
            try:
                robot.move_joint_deg(HOME_JOINT_DEG)
                print('   HOME 도착')
            except Exception as e:
                # HOME 실패해도 명령 루프는 진입 — 사용자가 pose/state/recover 등으로
                # 진단·복구 후 home 명령으로 다시 시도할 수 있어야 함.
                print(f'   !! HOME 이동 실패: {e}')
                print('   → 명령 루프에서 state / recover / home 으로 진단·재시도 가능')
        else:
            print('   (dry-run: HOME 이동 SKIP)')

        print('\n[준비 3/3] 명령 루프 진입')
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
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print('\n[정리 완료, 종료]')


if __name__ == '__main__':
    main()
