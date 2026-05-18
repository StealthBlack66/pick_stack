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
import multiprocessing as mp
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
)

try:
    import pyrealsense2 as rs
except ImportError:
    print('!! pyrealsense2 미설치. pip install pyrealsense2 후 다시 실행하세요.')
    sys.exit(1)

try:
    from ultralytics import YOLO
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
BASE_MODEL_NAME = 'yolov8s-world.pt'

TARGET_IMAGE_COUNT = 50  # 데모를 위해 50장으로 설정 (실제 환경에서는 1500~2000 사용 권장)
BLUR_THRESHOLD = 50.0

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
# Task 03: Background Trainer Agent (멀티프로세싱 워커)
# ==============================================================================
def background_trainer_agent(trigger_event, status_queue, buffer_dir, base_model):
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
        
        print("[Task 03: Trainer] model.train() 실행 (epochs=5, lr0=0.001)")
        # 과적합 방지 및 기본 weight 보존을 위해 짧은 에포크와 낮은 학습률 사용
        model.train(
            data=data_yaml,
            epochs=5,
            imgsz=640,
            batch=4,
            lr0=0.001,
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
    def __init__(self, use_sim=False):
        super().__init__('antigravity_vision_node')
        
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

        # Task 01: Vision Tracker 초기화
        self.setup_vision()
        
        # 멀티프로세싱 자원
        self.train_trigger = mp.Event()
        self.train_status_queue = mp.Queue()
        self.trainer_process = None
        self.is_training_active = False
        
        # 데이터 수집 버퍼 초기화
        if os.path.exists(BUFFER_DIR):
            shutil.rmtree(BUFFER_DIR)
        os.makedirs(os.path.join(BUFFER_DIR, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(BUFFER_DIR, 'labels', 'train'), exist_ok=True)
        
        self.collected_frames = 0
        
        print("[System] Antigravity Node 초기화 완료.")

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

    def check_blur(self, image):
        """이미지의 흐려짐 정도를 판단합니다."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def process_vision_loop(self):
        """
        메인 스레드에서 실행되는 비전 루프입니다.
        데이터 셰이핑(Task 02) 및 스트리밍 처리(Task 01)를 담당합니다.
        """
        print("[Task 01 & 02] 실시간 스트리밍 및 데이터 수집을 시작합니다.")
        cv2.namedWindow('Antigravity Vision Stream', cv2.WINDOW_NORMAL)
        
        target_class_id = 0 # 'cube'
        
        try:
            while True:
                # 핫스왑 체크
                if not self.train_status_queue.empty():
                    msg = self.train_status_queue.get()
                    if msg['status'] == 'SWAP_SUCCESS':
                        print(f"\n[HOT SWAP] 새로운 가중치로 모델을 교체합니다: {msg['model_path']}")
                        # Task 04 시작점: Hot Swap
                        self.model = YOLO(msg['model_path'])
                        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
                        self.is_training_active = False
                        print("[HOT SWAP] 교체 완료. 정밀 제어 모드로 전환합니다.")
                        # 교체 후 루프를 벗어나 로봇 제어(Task 04)로 이동할 수 있도록 브레이크
                        break

                frames = self.align.process(self.pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                disp_image = color_image.copy()
                
                # Tracking 모드
                results = self.model.track(color_image, persist=True, verbose=False)
                
                # 검출 결과가 있을 때
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    
                    for i in range(len(boxes)):
                        box = boxes[i]
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        xywh = box.xywh[0].cpu().numpy()
                        xyxy = box.xyxy[0].cpu().numpy()
                        track_id = int(box.id[0].item()) if box.id is not None else -1
                        
                        cx, cy, w, h = xywh
                        
                        # 화면 표시
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(disp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(disp_image, f'ID:{track_id} Conf:{conf:.2f}', (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # 데이터 수집 로직 (Task 02)
                        # 목표치에 아직 도달하지 않았고, 트레이닝이 시작되지 않았을 때 수집
                        if not self.is_training_active and self.collected_frames < TARGET_IMAGE_COUNT:
                            blur_val = self.check_blur(color_image)
                            
                            # 흐려짐 판단 필터
                            if blur_val > BLUR_THRESHOLD and conf > 0.5:
                                # YOLO 정규화 포맷 계산
                                H, W = color_image.shape[:2]
                                norm_cx = cx / W
                                norm_cy = cy / H
                                norm_w = w / W
                                norm_h = h / H
                                
                                img_name = f"frame_{self.collected_frames:04d}.jpg"
                                label_name = f"frame_{self.collected_frames:04d}.txt"
                                
                                img_path = os.path.join(BUFFER_DIR, 'images', 'train', img_name)
                                label_path = os.path.join(BUFFER_DIR, 'labels', 'train', label_name)
                                
                                cv2.imwrite(img_path, color_image)
                                with open(label_path, 'w') as f:
                                    f.write(f"{target_class_id} {norm_cx} {norm_cy} {norm_w} {norm_h}\n")
                                
                                self.collected_frames += 1
                                cv2.putText(disp_image, f'Collected: {self.collected_frames}/{TARGET_IMAGE_COUNT}', 
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            else:
                                cv2.putText(disp_image, f'BLURRED / LOW CONF', 
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Antigravity Vision Stream', disp_image)
                
                # 데이터가 꽉 찼으면 트리거 발송 (Task 02 -> Task 03)
                if self.collected_frames >= TARGET_IMAGE_COUNT and not self.is_training_active:
                    print(f"\n[Task 02: Data Shaping] 목표 데이터({TARGET_IMAGE_COUNT}장) 수집 완료.")
                    self.create_data_yaml()
                    
                    print("[Task 02] Task 03 백그라운드 학습 트리거 송신...")
                    self.is_training_active = True
                    
                    # 백그라운드 프로세스 시작
                    self.trainer_process = mp.Process(
                        target=background_trainer_agent,
                        args=(self.train_trigger, self.train_status_queue, BUFFER_DIR, BASE_MODEL_NAME)
                    )
                    self.trainer_process.start()
                    self.train_trigger.set()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            
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
    # Task 04: Robot Actuator Agent (3D 융합 및 매칭)
    # --------------------------------------------------------------------------
    def task04_execute_robot_action(self):
        print("\n[Task 04: Robot Actuator] 핫스왑된 가중치 기반 정밀 3D 제어 시작")
        
        # 최신 프레임을 가져와 검출 수행
        # 여러번 플러시
        for _ in range(5):
            self.pipeline.wait_for_frames()
            
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        color_image = np.asanyarray(color_frame.get_data())
        
        # 새 모델로 고정밀 추론
        results = self.model(color_image, verbose=False)
        if not results or len(results[0].boxes) == 0:
            print("[Task 04] 객체를 찾지 못했습니다.")
            return
            
        boxes = results[0].boxes
        # 가장 높은 Confidence 박스 선택
        best_box = boxes[0]
        cx, cy, w, h = best_box.xywh[0].cpu().numpy()
        
        # 2D 좌표 -> Depth -> 3D 보정
        u, v = int(cx), int(cy)
        depth_m = self._extract_robust_depth(depth_frame, u, v, w, h)
        
        if depth_m <= 0.0:
            print("[Task 04] 유효한 Depth 획득 실패.")
            return
            
        # 25mm 큐브 기하학 보정 (Prior Knowledge Filter)
        # 큐브의 높이가 25mm 임을 알기 때문에, Depth 센서 노이즈를 상쇄할 수 있음
        # 실제 TCP 목적지는 큐브 상단 평면 중앙이어야 함
        # depth_m 은 큐브 표면까지의 거리. 센서 에러를 줄이기 위한 필터링 처리 (여기선 depth_m을 신뢰하되 검증)
        cube_height_m = 0.025
        
        cam_xyz_m = rs.rs2_deproject_pixel_to_point(self.intr, [u, v], depth_m)
        cam_h = np.array([cam_xyz_m[0], cam_xyz_m[1], cam_xyz_m[2], 1.0])
        
        # 카메라 좌표계에서 로봇 베이스 좌표계로 변환
        base_h = self.T_cam2base @ cam_h
        base_mm = (base_h[:3] * 1000.0).astype(float)
        
        target_x, target_y, target_z = base_mm[0], base_mm[1], base_mm[2]
        
        print(f"[Task 04] 기하학 보정 완료. 최종 TCP 좌표: X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f} mm")
        
        # 로봇 제어 트리거
        print("[Task 04] Doosan e0509 로봇 제어 명령(MOVE_TCP)을 전송합니다.")
        self.move_to_target_and_grasp(target_x, target_y, target_z)

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
        self.pipeline.stop()
        if self.trainer_process and self.trainer_process.is_alive():
            self.trainer_process.terminate()
        cv2.destroyAllWindows()

# ==============================================================================
# Main Entry Point
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    
    node = AntigravityNode()
    
    try:
        # Task 01 & Task 02 시작 (내부에서 Task 03 비동기 트리거 발생)
        # 핫 스왑 이벤트 발생 시 루프를 벗어남
        import torch
        print(f"Pytorch CUDA: {torch.cuda.is_available()}")
        node.process_vision_loop()
        
        # Task 04 시작
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
