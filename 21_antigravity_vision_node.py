"""
[Antigravity Node] 실시간 비전 추적 및 백그라운드 고속 학습 파이프라인
Task 01: 스트리밍 & 추적 (Vision Tracker)
Task 02: 데이터 셰이핑 (Data Shaping)
Task 03: 백그라운드 학습 (Background Trainer - Multiprocessing)
Task 04: 3D 보정 및 로봇 제어 (Robot Actuator - 25mm 기하학 필터)
"""

import os
import sys
import time
import math
import shutil
import threading
import argparse
import queue as queue_mod
import multiprocessing as mp
import numpy as np
import cv2
import torch

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
)

try:
    import pyrealsense2 as rs
except ImportError:
    print('!! pyrealsense2 미설치. pip install pyrealsense2 후 다시 실행하세요.')
    sys.exit(1)

try:
    from ultralytics import YOLO, SAM
except ImportError:
    print('!! ultralytics 미설치. pip install ultralytics 후 다시 실행하세요.')
    sys.exit(1)

# 환경 변수 및 설정
try:
    from doosan_config import (
        NAMESPACE as NS,
        ROBOT_IP, RT_HOST, ROBOT_MODEL,
        BRINGUP_PKG, BRINGUP_LAUNCH, MOVEIT_CONTROLLER,
        DOOSAN_WS, ROS_DISTRO,
    )
except ImportError:
    # 기본값 설정
    NS = 'dsr01'
    MOVEIT_CONTROLLER = 'dsr_joint_trajectory_controller'

# 상수 정의
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(SCRIPT_DIR, 'calibration_data', 'calibration_result.npz')
BUFFER_DIR = os.path.join(SCRIPT_DIR, 'stream_buffer')
# 이미 cube 에 fine-tune 된 seg 모델 — zero-shot 'yolov8s-world.pt' 보다 1프레임 만에
# 바로 cube 검출 가능. 자가 라벨링 품질이 base model 의 검출 정확도에 의존하므로
# 이미 학습된 모델 사용이 노이즈 라벨 최소화에 유리.
BASE_MODEL_NAME = os.path.join(
    SCRIPT_DIR, 'yolo_dataset', 'runs', 'seg_v6', 'weights', 'best.pt'
)

# 정밀 학습 모드 — CUDA 사용 시 합리적 시간 (RTX 4060 기준 ~5분 내외) 안에 끝남.
# CPU 강제 사용 (CUDA 미사용) 시 30~60분 걸릴 수 있으니 주의.
TARGET_IMAGE_COUNT = 200  # 정밀 학습 목표 데이터 수
BLUR_THRESHOLD = 100.0    # Laplacian variance 임계 — 50 보다 엄격해서 흐린 frame 더 적극 거름

# SAM2 teacher — YOLO 가 검출한 bbox center 를 점 prompt 로 주면 SAM2 가 정확한
# mask 생성. self-distillation (YOLO → 자기 자신) 대신 SAM2 → YOLO 학습이라
# teacher 가 강함 → student 모서리 학습 가능.
SAM_WEIGHTS = '/home/fastcampus/Downloads/test/sam2.1_b.pt'
SAM_ASPECT_REJECT = 2.5   # mask 의 minAreaRect 가로:세로 > 이값이면 cube 아님 (옆면 흘림)
SAM_MIN_AREA_PX = 50      # mask 최소 픽셀 면적 (노이즈 거름)

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
MOVE_DURATION_SEC = 5.0
GRIPPER_TCP_OFFSET_Z = 160.0

# 그리퍼 관련
GRIPPER_PORT = 1
GRIPPER_BAUD = 57600
GRIP_OPEN_POS = 0
GRIP_CLOSE_POS = 700

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

# ==============================================================================
# SAM2 teacher 의 mask → cube top face 4 코너 정사각형 polygon
# (auto_prefill_seg.py 와 동일 로직 — 학습 라벨 품질의 핵심)
# ==============================================================================
def mask_to_square_polygon(mask, W, H,
                             aspect_reject=SAM_ASPECT_REJECT,
                             min_area=SAM_MIN_AREA_PX):
    """SAM2 mask → 짧은 변 정사각형 polygon (4꼭짓점, 픽셀 좌표 int32).
    center = mask centroid (moments), side = minAreaRect 의 짧은 변,
    angle = minAreaRect 의 회전각. cube 옆면이 mask 에 끼어들면 aspect 가 커져서 reject.
    None 반환 = polygon 유효성 fail (호출자가 skip)."""
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    rect = cv2.minAreaRect(cnt)
    (rcx, rcy), (rw, rh), angle = rect
    if min(rw, rh) <= 1:
        return None
    if max(rw, rh) / min(rw, rh) > aspect_reject:
        return None
    M = cv2.moments(cnt)
    if M['m00'] > 0:
        cu = M['m10'] / M['m00']
        cv_ = M['m01'] / M['m00']
    else:
        cu, cv_ = rcx, rcy
    side = float(min(rw, rh))
    box = cv2.boxPoints(((cu, cv_), (side, side), float(angle)))
    box[:, 0] = np.clip(box[:, 0], 0, W - 1)
    box[:, 1] = np.clip(box[:, 1], 0, H - 1)
    return box.astype(np.int32)


# ==============================================================================
# Task 03: Background Trainer Agent (멀티프로세싱 워커)
# ==============================================================================
def background_trainer_agent(trigger_event, status_queue, buffer_dir, base_model,
                              epochs=30, batch=8, imgsz=640):
    """
    메인 로봇 프로세스와 완전히 분리되어 백그라운드에서 YOLO 모델을 파인튜닝합니다.
    (OOM 및 프레임 드랍 방지)
    """
    print("[Task 03: Trainer] 대기 중... (Trigger Event 대기)")
    trigger_event.wait()
    print("[Task 03: Trainer] Trigger 수신! 학습을 시작합니다.")

    # 학습을 위한 data.yaml 경로
    data_yaml = os.path.join(buffer_dir, 'data.yaml')
    project_dir = os.path.join(buffer_dir, 'train_results')
    
    try:
        # 워커 프로세스 내부에서 ultralytics를 새로 불러옵니다.
        from ultralytics import YOLO
        import torch
        
        # GPU 메모리 최적화를 위해 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        model = YOLO(base_model)
        
        print(f"[Task 03: Trainer] model.train() 실행 (epochs={epochs}, batch={batch}, imgsz={imgsz}, lr0=0.001) — 정밀 학습 모드")
        # 정밀 학습 — base weight 위에 적당히 깊이 fine-tune. lr0 낮춰서 catastrophic
        # forgetting 방지. patience 추가로 mAP 정체 시 조기 종료.
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=0.001,
            patience=10,        # mAP 10 epoch 정체 시 조기 종료
            project=project_dir,
            name='finetuned',
            exist_ok=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=False
        )
        
        best_weight = os.path.join(project_dir, 'finetuned', 'weights', 'best.pt')
        target_weight = os.path.join(os.path.dirname(buffer_dir), 'best_finetuned.pt')
        
        if os.path.exists(best_weight):
            shutil.copy(best_weight, target_weight)
            print(f"[Task 03: Trainer] 학습 완료! 가중치 복사 완료: {target_weight}")
            status_queue.put({"status": "SWAP_SUCCESS", "model_path": target_weight})
        else:
            print("[Task 03: Trainer] 학습은 종료되었으나 best.pt를 찾을 수 없습니다.")
            status_queue.put({"status": "SWAP_FAILED"})
            
    except Exception as e:
        print(f"[Task 03: Trainer] 학습 중 오류 발생: {e}")
        status_queue.put({"status": "SWAP_FAILED", "error": str(e)})


# ==============================================================================
# 통합 Agent Node (Task 01, 02, 04 및 로봇 제어)
# ==============================================================================
class AntigravityNode(Node):
    def __init__(self, cli_args=None, use_sim=False):
        super().__init__('antigravity_vision_node')
        # CLI 옵션 (mode, epochs, batch, target, no_gate). None 이면 기본값 채움.
        self.args = cli_args if cli_args is not None else argparse.Namespace(
            mode='demo', epochs=30, batch=8, target=TARGET_IMAGE_COUNT, no_gate=False
        )
        
        # ROS 2 클라이언트 세팅
        self.cli_mode = self.create_client(SetRobotMode, f'/{NS}/system/set_robot_mode')
        self.cli_ctrl = self.create_client(SetRobotControl, f'/{NS}/system/set_robot_control')
        self.cli_ikin = self.create_client(Ikin, f'/{NS}/motion/ikin')
        self.cli_get_pose = self.create_client(GetCurrentPose, f'/{NS}/system/get_current_pose')
        self.cli_get_state = self.create_client(GetRobotState, f'/{NS}/system/get_robot_state')
        self.cli_safety_mode = self.create_client(SetSafetyMode, f'/{NS}/system/set_safety_mode')
        self.cli_safe_reset = self.create_client(SetSafeStopResetType, f'/{NS}/system/set_safe_stop_reset_type')
        
        self.traj_action = ActionClient(
            self, FollowJointTrajectory,
            f'/{NS}/{MOVEIT_CONTROLLER}/follow_joint_trajectory'
        )
        
        gp = f'/{NS}/gripper'
        self.cli_g_open = self.create_client(FlangeSerialOpen, f'{gp}/flange_serial_open')
        self.cli_g_close = self.create_client(FlangeSerialClose, f'{gp}/flange_serial_close')
        self.cli_g_write = self.create_client(FlangeSerialWrite, f'{gp}/flange_serial_write')
        self._gripper_serial_open = False
        self.tcp_z_offset = GRIPPER_TCP_OFFSET_Z
        
        # 캘리브레이션 데이터 로드
        if not os.path.exists(CALIB_PATH):
            raise FileNotFoundError(f'캘리브레이션 결과 없음: {CALIB_PATH}')
        d = np.load(CALIB_PATH)
        self.T_cam2base = d['T_cam2base']

        # Task 01: Vision Tracker 초기화 — train 전용 모드면 RealSense/YOLO base 모델
        # 로드 skip (학습 trainer 가 독립 process 에서 자체 로드)
        self._pipeline_running = False
        if self.args.mode != 'train':
            self.setup_vision()
            self._pipeline_running = True

        # 멀티프로세싱 자원
        self.train_trigger = mp.Event()
        self.train_status_queue = mp.Queue()
        self.trainer_process = None
        self.is_training_active = False

        # 데이터 수집 버퍼 — train 모드 (기존 데이터 재학습) 면 보존, 그 외에는 초기화
        if self.args.mode == 'train':
            if not os.path.isdir(os.path.join(BUFFER_DIR, 'images', 'train')) or \
               not os.listdir(os.path.join(BUFFER_DIR, 'images', 'train')):
                raise FileNotFoundError(
                    f'--mode train 인데 학습 데이터 없음: {BUFFER_DIR}/images/train. '
                    f'먼저 --mode collect 로 데이터 수집하거나 --mode demo 사용.'
                )
            print(f'[System] --mode train: 기존 데이터 사용 ({BUFFER_DIR})')
        else:
            if os.path.exists(BUFFER_DIR):
                shutil.rmtree(BUFFER_DIR)
            os.makedirs(os.path.join(BUFFER_DIR, 'images', 'train'), exist_ok=True)
            os.makedirs(os.path.join(BUFFER_DIR, 'labels', 'train'), exist_ok=True)

        self.collected_frames = 0

        print(f"[System] Antigravity Node 초기화 완료 (mode={self.args.mode}).")

    # --------------------------------------------------------------------------
    # Vision & Task 01, 02
    # --------------------------------------------------------------------------
    def setup_vision(self):
        # 첫 모델 로드 (Base Model)
        self.model = YOLO(BASE_MODEL_NAME)
        # World 모델의 경우 텍스트 프롬프트 설정 (기본값: cube)
        if 'world' in BASE_MODEL_NAME:
            self.model.set_classes(['cube'])
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # SAM2 teacher — YOLO 가 검출한 bbox center 를 점 prompt 로 보내면 cube top
        # 면 polygon 을 정밀하게 생성. self-distill 대체.
        if not os.path.exists(SAM_WEIGHTS):
            print(f'!! SAM2 모델 없음: {SAM_WEIGHTS} — SAM2 teacher 비활성화 (self-distill fallback)')
            self.sam = None
        else:
            print(f'  SAM2 teacher 로딩: {SAM_WEIGHTS}')
            self.sam = SAM(SAM_WEIGHTS)
            # warmup — 첫 호출 latency 줄이기
            try:
                self.sam(np.zeros((480, 640, 3), dtype=np.uint8),
                         points=[[320, 240]], labels=[1], verbose=False)
                print('  SAM2 warmup OK')
            except Exception as e:
                print(f'  SAM2 warmup 경고: {e}')
        
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        
        # 웜업
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def _restart_pipeline(self):
        """학습 중 GPU 양보 위해 닫았던 RealSense pipeline 을 다시 시작 (Task 04 진입 전).
        setup_vision 이 한 번 호출된 적이 있어 self.pipeline / self.intr 가 살아있는
        상태에서만 호출. 모델/카메라 둘 다 그대로 재사용."""
        if self._pipeline_running:
            return
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(cfg)
        # align / intr 는 첫 setup_vision 결과 그대로 사용 (해상도/포맷 동일)
        for _ in range(15):
            self.pipeline.wait_for_frames()
        self._pipeline_running = True

    def check_blur(self, image):
        """이미지의 흐려짐 정도를 판단합니다."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def process_vision_loop(self):
        """
        메인 스레드에서 실행되는 비전 루프입니다.
        데이터 셰이핑(Task 02) 및 스트리밍 처리(Task 01)를 담당합니다.

        Lifecycle (mode-aware):
          - collect: 목표 수집 후 종료 (학습 안 함)
          - demo: 목표 수집 → S1 gate (ENTER) → S2 (pipeline.stop) → 학습 polling → 핫스왑
          - train: 이 메서드 호출 안 됨 (run_train_only 가 별도로 처리)

        수집 정책:
          - 한 frame 안에 검출된 모든 cube polygon 을 **한 .txt 파일** 에 저장
            (이전 버그: cube 별로 새 file 만들어서 다른 cube 가 background 로 학습됨)
          - 시작 전 한 번만 사용자 ENTER 대기 (cube 첫 배치 시간)
          - 그 다음 --save-interval (default 0.1s) 마다 자동으로 frame 한 장씩 저장
          - 사용자는 그 동안 cube 를 자유롭게 움직이면 다양한 frame 이 자동 수집됨
        """
        target_frames = self.args.target
        save_interval = max(0.01, float(self.args.save_interval))
        cv2.namedWindow('Antigravity Vision Stream', cv2.WINDOW_NORMAL)

        target_class_id = 0   # 'cube'

        # 시작 gate — 사용자가 cube 첫 배치할 시간 (한 번만)
        print(f'[Task 01 & 02] 데이터 수집 준비 (목표 {target_frames}장, '
              f'{save_interval*1000:.0f}ms 마다 1장 자동 저장).')
        print('  → cube 를 카메라 시야에 배치하고 윈도우 클릭 + ENTER 로 수집 시작.')
        print('  → ENTER 누르면 자동 수집이 시작되니, 그 다음엔 cube 를 자유롭게 움직이세요.')
        if not self.args.no_gate:
            if not self._wait_for_collection_gate(
                    message='Arrange cubes — ENTER to start auto-collect, q to quit',
                    heading=f'READY (target {target_frames}, auto-save every {save_interval*1000:.0f}ms)'):
                print('[Task 02] q 입력 — 수집 취소.')
                return

        last_save_time = 0.0   # epoch seconds. 0 = 첫 frame 부터 저장

        try:
            while True:
                frames = self.align.process(self.pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                disp_image = color_image.copy()

                # Tracking
                results = self.model.track(color_image, persist=True, verbose=False)

                # 한 frame 안에서 통과한 모든 cube 의 polygon 라벨 모음
                # YOLO = detector (bbox), SAM2 = polygon teacher (정확한 4 코너)
                frame_label_lines = []
                H, W = color_image.shape[:2]
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    yolo_masks = getattr(results[0], 'masks', None)

                    for i in range(len(boxes)):
                        box = boxes[i]
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy()
                        track_id = int(box.id[0].item()) if box.id is not None else -1

                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(disp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(disp_image, f'ID:{track_id} Conf:{conf:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if conf <= 0.75:
                            continue

                        # SAM2 teacher 우선 — bbox center 를 점 prompt 로 → 정확한 mask
                        # → minAreaRect 의 짧은 변 = cube top side → 정사각형 4 코너
                        polygon_norm = None
                        if self.sam is not None:
                            cu = (x1 + x2) / 2.0
                            cv_ = (y1 + y2) / 2.0
                            try:
                                sres = self.sam(color_image,
                                                 points=[[cu, cv_]], labels=[1],
                                                 verbose=False)
                                if (sres and sres[0].masks is not None and
                                        len(sres[0].masks.data) > 0):
                                    mk = sres[0].masks.data[0].cpu().numpy()
                                    box4 = mask_to_square_polygon(mk, W, H)
                                    if box4 is not None:
                                        polygon_norm = box4.astype(np.float32)
                                        polygon_norm[:, 0] /= W
                                        polygon_norm[:, 1] /= H
                                        # 사용자 시각 피드백 — SAM2 가 만든 4 코너
                                        cv2.polylines(disp_image, [box4], True,
                                                      (0, 255, 255), 2)
                                        for pt in box4:
                                            cv2.circle(disp_image,
                                                       (int(pt[0]), int(pt[1])),
                                                       4, (0, 0, 255), -1)
                            except Exception as e:
                                print(f'  SAM2 fail @ ({cu:.0f},{cv_:.0f}): {e}')

                        # SAM2 비활성 또는 실패 시 self-distill fallback (열등)
                        if polygon_norm is None:
                            if yolo_masks is None or i >= len(yolo_masks.xyn):
                                continue
                            polygon_norm = yolo_masks.xyn[i]
                            if polygon_norm is None or len(polygon_norm) < 3:
                                continue

                        parts = [str(target_class_id)]
                        for px, py in polygon_norm:
                            parts.append(f'{float(px):.6f}')
                            parts.append(f'{float(py):.6f}')
                        frame_label_lines.append(' '.join(parts))

                # frame-level 저장 결정 — rate-limit (save_interval) + conf + blur 통과 시
                now = time.time()
                can_save = (not self.is_training_active and
                            self.collected_frames < target_frames and
                            frame_label_lines and
                            (now - last_save_time) >= save_interval)
                if can_save:
                    blur_val = self.check_blur(color_image)
                    if blur_val > BLUR_THRESHOLD:
                        img_name = f"frame_{self.collected_frames:04d}.jpg"
                        label_name = f"frame_{self.collected_frames:04d}.txt"
                        img_path = os.path.join(BUFFER_DIR, 'images', 'train', img_name)
                        label_path = os.path.join(BUFFER_DIR, 'labels', 'train', label_name)

                        cv2.imwrite(img_path, color_image)
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(frame_label_lines) + '\n')

                        self.collected_frames += 1
                        last_save_time = now
                        cv2.putText(disp_image,
                                    f'Collected: {self.collected_frames}/{target_frames} '
                                    f'({len(frame_label_lines)} cubes in this frame)',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.putText(disp_image, f'BLURRED (var={blur_val:.0f} < {BLUR_THRESHOLD:.0f})',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif not frame_label_lines:
                    cv2.putText(disp_image, 'NO CONFIDENT CUBE (conf < 0.75)',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    # rate-limit 대기 중 — 카운터만 계속 표시
                    cv2.putText(disp_image,
                                f'Collected: {self.collected_frames}/{target_frames}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.imshow('Antigravity Vision Stream', disp_image)

                # 데이터가 꽉 찼을 때 mode 별 분기
                if self.collected_frames >= target_frames and not self.is_training_active:
                    print(f"\n[Task 02: Data Shaping] 목표 데이터({target_frames}장) 수집 완료.")
                    self.create_data_yaml()

                    if self.args.mode == 'collect':
                        print('[Task 02] --mode collect 끝 — 학습은 진행하지 않음.')
                        return   # process_vision_loop 종료, main 에서 shutdown 으로 흐름

                    # demo 모드 — S1 gate (--no-gate 면 skip)
                    if not self.args.no_gate:
                        if not self._wait_for_start_gate(disp_image):
                            print('[Gate] 사용자가 q 로 학습 중단 선택 — 종료.')
                            return

                    # S2 GPU 양보 — RealSense + cv2 끄고 학습이 끝날 때까지 대기
                    print("[Task 02] Task 03 백그라운드 학습 트리거 송신...")
                    self.is_training_active = True
                    self.trainer_process = mp.Process(
                        target=background_trainer_agent,
                        args=(self.train_trigger, self.train_status_queue,
                              BUFFER_DIR, BASE_MODEL_NAME,
                              self.args.epochs, self.args.batch)
                    )
                    self.trainer_process.start()
                    self.train_trigger.set()

                    print('[메인] GPU 양보 — RealSense + cv2 정지, 학습이 끝날 때까지 polling...')
                    cv2.destroyAllWindows()
                    try:
                        self.pipeline.stop()
                    except Exception:
                        pass
                    self._pipeline_running = False

                    self._wait_for_training_done()
                    return   # 학습 polling 종료 → main 에서 task04 로 진입

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('[Vision] q 키 입력 — 데이터 수집 중단.')
                    break
        finally:
            cv2.destroyAllWindows()

    def _wait_for_collection_gate(self, message='Press ENTER to continue, q to abort',
                                   heading='READY'):
        """S0 / batch gate: 사용자가 cube 를 배치/재배치할 시간.
        라이브 카메라 프리뷰 띄운 채로 ENTER (계속) / q (취소) 대기.
        return True=ENTER, False=q (취소)."""
        while True:
            frames = self.align.process(self.pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            disp = np.asanyarray(color_frame.get_data()).copy()
            # 실시간 검출 overlay (사용자가 모델이 cube 잡는지 확인)
            try:
                res = self.model.predict(disp, verbose=False, conf=0.5)
                if res and len(res[0].boxes) > 0:
                    for b in res[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                        c = float(b.conf[0].item())
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(disp, f'{c:.2f}', (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            except Exception:
                pass
            cv2.putText(disp, heading, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(disp, message, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('Antigravity Vision Stream', disp)
            k = cv2.waitKey(50) & 0xFF
            if k in (13, 10):     # ENTER / Return
                return True
            if k == ord('q'):
                return False

    def _wait_for_start_gate(self, last_disp_image):
        """S1: 학습 시작 전 ENTER 대기. q 누르면 False 반환 (학습 skip)."""
        print('  → 학습 시작 준비. 윈도우 클릭 후 ENTER = 시작 / q = 종료.')
        while True:
            overlay = last_disp_image.copy()
            cv2.putText(overlay, 'READY TO TRAIN', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(overlay, 'Press ENTER to start training, q to quit',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Antigravity Vision Stream', overlay)
            k = cv2.waitKey(50) & 0xFF
            if k in (13, 10):   # ENTER / Return
                return True
            if k == ord('q'):
                return False

    def _wait_for_training_done(self):
        """S2: 학습 status_queue 폴링. SWAP_SUCCESS 면 핫스왑까지 처리.
        Ctrl+C 면 KeyboardInterrupt 가 main 의 finally → shutdown 으로 흘러 trainer 정리."""
        while True:
            try:
                msg = self.train_status_queue.get(timeout=1.0)
            except queue_mod.Empty:
                continue
            status = msg.get('status')
            if status == 'SWAP_SUCCESS':
                print(f"\n[HOT SWAP] 새로운 가중치로 모델을 교체합니다: {msg['model_path']}")
                # 학습 동안 base 모델 객체는 그대로 살아있음 — 새 가중치로 교체
                self.model = YOLO(msg['model_path'])
                self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
                self.is_training_active = False
                # 핫스왑 카피본 명시적으로 best_finetuned.pt 에 둠
                try:
                    target_copy = os.path.join(SCRIPT_DIR, 'best_finetuned.pt')
                    shutil.copy(msg['model_path'], target_copy)
                    print(f'[HOT SWAP] 카피본 갱신: {target_copy}')
                except Exception as e:
                    print(f'[HOT SWAP] 카피 실패 (무시 가능): {e}')
                print('[HOT SWAP] 교체 완료.')
                return
            elif status == 'SWAP_FAILED':
                err = msg.get('error', '<no error>')
                print(f'[Task 03] 학습 실패: {err}')
                self.is_training_active = False
                return
            
    def create_data_yaml(self):
        """YOLO 학습을 위한 data.yaml 생성"""
        yaml_content = f"""
path: {BUFFER_DIR}
train: images/train
val: images/train
names:
  0: cube
"""
        with open(os.path.join(BUFFER_DIR, 'data.yaml'), 'w') as f:
            f.write(yaml_content)

    # --------------------------------------------------------------------------
    # Task 04: Visual Eval Loop — TCP 타겟 정밀도 평가
    # --------------------------------------------------------------------------
    def task04_execute_robot_action(self):
        """학습 종료 후 실시간 평가 루프 — cube top face 의 4 코너 + 중심점 시각화.

        목적: cube top face 중심점 = TCP 목적지. 정확도가 정밀 탑 쌓기 품질 결정.
        화면 표시:
          - 회색 외곽선 = 원본 segmentation polygon (참고용)
          - 노란 사각형 + 빨간 점 4개 = top face 4 코너 (cv2.minAreaRect 추정)
          - 파란 십자 + 큰 원 = TCP 타겟 중심점
          - 라벨: conf, 회전각, side 길이(px), 픽셀 좌표, base mm 좌표 (depth OK 시)
        """
        print('\n[Task 04: Visual Eval] 새 모델 실시간 검출 — top face 4 코너 + TCP 중심점 평가.')
        print('  - 노란 사각형 = 4 코너, 빨간 점 = 각 코너 위치')
        print('  - 파란 십자 = TCP 목적지 (cube 중심점)')
        print('  - 라벨: conf, rotation°, side px, pixel(u,v), Base(X,Y,Z) mm if depth')
        print('  - 정밀하면 q. 어긋나면 q 후 데이터 다시 + 학습 재실행.')

        if not self._pipeline_running:
            print('[Task 04] RealSense pipeline 재기동...')
            self._restart_pipeline()
        cv2.namedWindow('Antigravity Vision Stream', cv2.WINDOW_NORMAL)

        try:
            while True:
                frames = self.align.process(self.pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                disp = color_image.copy()

                results = self.model.predict(color_image, verbose=False, conf=0.5)
                n_det = 0
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    masks = getattr(results[0], 'masks', None)

                    for i in range(len(boxes)):
                        conf = float(boxes[i].conf[0].item())

                        # polygon → minAreaRect → 4 코너 + 중심
                        rect = None
                        if (masks is not None and hasattr(masks, 'xy') and
                                i < len(masks.xy)):
                            poly = masks.xy[i]
                            if poly is not None and len(poly) >= 4:
                                poly_int = np.array(poly, dtype=np.int32)
                                cv2.polylines(disp, [poly_int], True, (90, 90, 90), 1)
                                rect = cv2.minAreaRect(poly_int)

                        if rect is None:
                            # mask 없음 → bbox center 만
                            xyxy = boxes[i].xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 0), 1)
                            cu, cv_ = (x1 + x2) // 2, (y1 + y2) // 2
                            cv2.drawMarker(disp, (cu, cv_), (255, 100, 0),
                                           cv2.MARKER_CROSS, 18, 2)
                            n_det += 1
                            continue

                        # 노란 4 모서리 + 빨간 점
                        box4 = cv2.boxPoints(rect).astype(np.int32)
                        cv2.drawContours(disp, [box4], 0, (0, 255, 255), 2)
                        for pt in box4:
                            cv2.circle(disp, (int(pt[0]), int(pt[1])), 5,
                                       (0, 0, 255), -1)

                        cu = int(rect[0][0])
                        cv_ = int(rect[0][1])
                        rect_w, rect_h = rect[1]
                        side_px = float(min(rect_w, rect_h))
                        angle_deg = float(rect[2])

                        # TCP 타겟 중심점 — 큰 파란 십자 + 흰 원 (가장 도드라지게)
                        cv2.drawMarker(disp, (cu, cv_), (255, 80, 0),
                                       cv2.MARKER_CROSS, 22, 2)
                        cv2.circle(disp, (cu, cv_), 8, (255, 255, 255), 2)
                        cv2.circle(disp, (cu, cv_), 2, (255, 80, 0), -1)

                        # depth → base mm 변환 시도
                        base_str = ''
                        if depth_frame is not None:
                            try:
                                z_m = depth_frame.get_distance(cu, cv_)
                                if z_m > 0.05:   # 5cm 이상이면 유효
                                    cam_xyz = rs.rs2_deproject_pixel_to_point(
                                        self.intr, [cu, cv_], z_m)
                                    cam_h = np.array([cam_xyz[0], cam_xyz[1],
                                                       cam_xyz[2], 1.0])
                                    base = self.T_cam2base @ cam_h
                                    base_mm = base[:3] * 1000.0
                                    base_str = (f'Base:({base_mm[0]:+.0f},'
                                                f'{base_mm[1]:+.0f},'
                                                f'{base_mm[2]:+.0f})mm')
                            except Exception:
                                pass

                        # 라벨 — 2줄로 (정보량 ↑ 가독성 유지)
                        line1 = (f'#{n_det} conf={conf:.2f} '
                                 f'rot={angle_deg:+.0f}° side={side_px:.0f}px')
                        line2 = f'TCP:({cu},{cv_})px {base_str}'
                        # 박스 아래에 표시 (충돌 적음)
                        ty = max(20, min(int(rect[0][1]) +
                                          int(max(rect_w, rect_h) / 2) + 20,
                                          disp.shape[0] - 30))
                        cv2.putText(disp, line1, (max(5, cu - 100), ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 255, 255), 1)
                        cv2.putText(disp, line2, (max(5, cu - 100), ty + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 200, 0), 1)
                        n_det += 1

                cv2.putText(disp, f'EVAL — {n_det} cubes — q to quit',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
                cv2.imshow('Antigravity Vision Stream', disp)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
        print('[Eval] 평가 종료.')

    def _extract_robust_depth(self, depth_frame, cx, cy, box_w, box_h):
        """
        뭉개진 3D 데이터 위에 '가상의 정육면체 템플릿' 매칭
        박스 중앙 주변의 픽셀들 중 유효한 깊이값의 중간값(Median)을 추출하여 노이즈 제거
        """
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape
        
        # 중심 주변 25% 영역 추출 (가장자리 모서리 노이즈 회피)
        roi_w, roi_h = int(box_w * 0.25), int(box_h * 0.25)
        u0, v0 = max(0, cx - roi_w), max(0, cy - roi_h)
        u1, v1 = min(W, cx + roi_w), min(H, cy + roi_h)
        
        win = arr[v0:v1, u0:u1]
        win = win[win > 0] # hole 제외
        
        if win.size == 0:
            # ROI 내 유효값이 없으면 중앙 단일 픽셀 폴백
            z = depth_frame.get_distance(cx, cy)
            return z if z > 0 else 0.0
            
        # 강건한 중앙값 필터링 적용
        median_depth = np.median(win) * 0.001 # mm -> m
        return median_depth

    # --------------------------------------------------------------------------
    # 로봇 제어 유틸리티 (Pick & Place)
    # --------------------------------------------------------------------------
    def _call(self, cli, req, timeout=30.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def move_to_target_and_grasp(self, x, y, z):
        # 데모용: 실제 로봇 궤적 생성 및 전송 구현
        # 여기서는 로그로 동작을 대신하거나 안전하게 모션을 계획하는 형태로 구성
        print(f" > [Action] 로봇 암이 {x, y, z+100} 위치로 이동 (Approach)")
        time.sleep(1)
        print(f" > [Action] 로봇 암 하강 및 그리퍼 닫힘 (Grasp at Z={z})")
        time.sleep(1)
        print(f" > [Action] 로봇 암 상승 (Lift)")
        print("[System] 워크플로우 1회 사이클 정상 완료.")

    def shutdown(self):
        """깔끔한 종료 — pipeline + trainer + cv2 윈도우. Ctrl+C 로 들어와도 좀비 안 남게.

        - pipeline: _pipeline_running 플래그로 두 번 stop 회피.
        - trainer: terminate → join(5s). 그래도 살아있으면 SIGKILL (spawn 모드의
          tensor op 중 SIGTERM 무시되는 케이스 대비). 이걸 안 두면 다음 실행에서
          좀비 trainer 가 GPU 차지해서 OOM/충돌.
        """
        if getattr(self, '_pipeline_running', False):
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f'[종료] pipeline.stop 무시: {e}')
            self._pipeline_running = False
        if self.trainer_process and self.trainer_process.is_alive():
            print('[종료] 백그라운드 학습 process 정리 중...')
            self.trainer_process.terminate()
            self.trainer_process.join(timeout=5.0)
            if self.trainer_process.is_alive():
                print('[종료] SIGTERM 무시됨 — SIGKILL')
                self.trainer_process.kill()
                self.trainer_process.join(timeout=2.0)
        cv2.destroyAllWindows()

    # --------------------------------------------------------------------------
    # --mode train 전용 진입점 (수집 skip, 기존 데이터로 학습만)
    # --------------------------------------------------------------------------
    def run_train_only(self):
        """이미 모인 데이터로 학습만 — RealSense 안 켜고 GPU 부담 절반."""
        if not os.path.exists(os.path.join(BUFFER_DIR, 'data.yaml')):
            print('[train-only] data.yaml 없음 — 생성')
            self.create_data_yaml()
        n_imgs = len(os.listdir(os.path.join(BUFFER_DIR, 'images', 'train')))
        n_lbls = len(os.listdir(os.path.join(BUFFER_DIR, 'labels', 'train')))
        print(f'[train-only] 입력 데이터: 이미지 {n_imgs}장, 라벨 {n_lbls}개')
        print(f'[train-only] epochs={self.args.epochs}, batch={self.args.batch}, imgsz={self.args.imgsz}')

        self.is_training_active = True
        self.trainer_process = mp.Process(
            target=background_trainer_agent,
            args=(self.train_trigger, self.train_status_queue,
                  BUFFER_DIR, BASE_MODEL_NAME,
                  self.args.epochs, self.args.batch, self.args.imgsz)
        )
        self.trainer_process.start()
        self.train_trigger.set()
        self._wait_for_training_done()

# ==============================================================================
# Main Entry Point
# ==============================================================================
def _parse_cli():
    ap = argparse.ArgumentParser(
        description='Antigravity Vision Node — RealSense + YOLO 자가 라벨링 + 백그라운드 학습. '
                    'mode 로 단계 분리, ENTER gate 로 시작/정지 명시 (과부하 방지).'
    )
    ap.add_argument('--mode', choices=['collect', 'train', 'demo'], default='demo',
                    help='collect=수집만, train=기존 데이터로 학습만, demo=전체 (default)')
    ap.add_argument('--epochs', type=int, default=30,
                    help='학습 epoch 수 (default 30, patience=10 으로 조기 종료)')
    ap.add_argument('--batch', type=int, default=8,
                    help='학습 batch 크기 (default 8). VRAM 부족하면 4 로 낮추기')
    ap.add_argument('--target', type=int, default=TARGET_IMAGE_COUNT,
                    help=f'수집 목표 frame 수 (default {TARGET_IMAGE_COUNT})')
    ap.add_argument('--save-interval', type=float, default=0.1,
                    help='수집 시 frame 저장 간격 (초, default 0.1 = 10fps). 작을수록 빨리 모이고 '
                         '비슷한 frame 이 많이 들어옴. 사용자가 cube 움직이는 속도에 맞춰 조정.')
    ap.add_argument('--imgsz', type=int, default=640,
                    help='학습 이미지 해상도 (default 640). cube 가 화면에서 작을 때 '
                         '1024/1280 으로 키우면 작은 object 디테일 보존 (VRAM 더 씀).')
    ap.add_argument('--no-gate', action='store_true',
                    help='수집/학습 시작 gate 생략 (무인 데모용 — 권장 X)')
    return ap.parse_args()


def main(args=None):
    cli_args = _parse_cli()
    rclpy.init(args=args)

    print(f"Pytorch CUDA: {torch.cuda.is_available()}")
    node = AntigravityNode(cli_args=cli_args)

    try:
        if cli_args.mode == 'train':
            # 기존 stream_buffer 데이터로 학습만 → 학습 끝나면 곧바로 시각 평가
            node.run_train_only()
            if not node.is_training_active:    # 학습 성공한 경우
                node.task04_execute_robot_action()
        else:
            # collect / demo — process_vision_loop 안에서 mode 분기
            node.process_vision_loop()
            # demo 모드에서 학습 완료 (핫스왑 OK) 했으면 평가 루프 실행
            if cli_args.mode == 'demo' and not node.is_training_active:
                node.task04_execute_robot_action()
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.")
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    # 멀티프로세싱 안정성을 위해 spawn 방식 지정
    mp.set_start_method('spawn', force=True)
    main()
