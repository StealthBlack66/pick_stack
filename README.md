# 두산 e0509 Hand-Eye Calibration (Eye-to-Hand)

원샷 자동 캘리브레이션 스크립트 — RealSense 카메라 + Doosan e0509 + ArUco 마커

## 빠른 시작

```bash
./run_calibration.sh
```

이게 전부예요. 나머지는 자동:
- ROS distro 자동 감지 (humble/iron/jazzy 등)
- doosan_ws 위치 자동 감지 (`~/doosan_ws`, `~/ros2_ws`, `~/dsr_ws`, `/opt/doosan_ws`)
- Python venv 자동 감지 (또는 시스템 python)
- **로봇 IP 자동 탐색** (`110.120.1.1` ~ `110.120.1.100` 범위 port 12345 스캔)
- 첫 실행 후 IP 캐시 → 다음부터 즉시 연결

## 요구사항

### 환경
- Linux (Ubuntu 22.04 권장)
- ROS 2 (humble 등)
- doosan_ws 빌드됨 (`dsr_msgs2`, `dsr_bringup2`, `dsr_moveit_config_e0509` 포함)
- Python 패키지:
  ```bash
  pip install numpy opencv-contrib-python scipy pyrealsense2
  ```
  (반드시 **opencv-contrib-python** — 일반 opencv-python 에는 ArUco 없음)

### 하드웨어
- **두산 e0509** 로봇 + 컨트롤러 (port 12345)
- **Intel RealSense** 카메라 (D455, D435 등)
- **ArUco 마커** DICT_6X6_50, ID=1, 50mm × 50mm
  - 직접 그린 마커 ❌ (정확도 낮음)
  - `generate_aruco_marker.py` 로 생성 후 깨끗하게 인쇄 ✅
  - 평판(아크릴/PVC) 에 단단히 부착, 그리퍼에 견고하게 고정

### 네트워크
- 로봇 컨트롤러 IP: `110.120.1.X` (X = 1~100)
- RT control IP: 기본 `110.120.1.5` (다르면 `--rt-host` 로 지정)

## 사용 방법

### 1. 마커 인쇄 (한 번만)

```bash
python3 generate_aruco_marker.py
# → aruco_DICT_6X6_50_ID0_50mm.png 생성
# → 100% 스케일로 인쇄, 자로 50mm 확인
# → 평판에 부착, 그리퍼에 고정
```

다른 ID/사이즈 원하면:
```bash
python3 generate_aruco_marker.py --id 1 --size 50
```

### 2. 캘리브레이션 실행

```bash
./run_calibration.sh
```

### 3. 진행 (cv2 창에서 키 입력)

| 키 | 동작 |
|---|---|
| `s` | 현재 자세를 base 로 저장 (시작점) |
| Enter | 16 포즈 자동 순회 시작 (~3분) |
| `c` | 캘리브레이션 계산 + 자동 정제 (목표 도달까지 추가 라운드) |
| `h` | base 자세로 복귀 |
| `d` | 디버그: ArUco 사전 자동 탐색 |
| `q` | 종료 |

### 4. 결과 파일

```
calibration_data/
├── calibration_result.npz   # 최종 X (camera → base) + 오차 metric
├── calibration_data.npz     # 수집한 모든 pose 데이터 (재계산 가능)
└── images/                  # 각 pose 의 카메라 이미지 (디버그)
```

`calibration_result.npz` 내용:
- `T_cam2base`: 4×4 카메라→베이스 변환 행렬
- `R_cam2base`, `t_cam2base`: 회전/이동 분리
- `pos_err_mean_mm`, `rot_err_mean_deg`: 오차 metric
- `camera_matrix`, `dist_coeffs`: 카메라 intrinsics

## CLI 옵션

```bash
./run_calibration.sh --help
```

| 옵션 | 의미 |
|---|---|
| `--host 110.120.1.X` | 로봇 IP 직접 지정 (자동탐색 우회) |
| `--rt-host 110.120.1.5` | RT control 호스트 |
| `--no-cleanup` | 기존 ROS 프로세스 정리 생략 |
| `--keep-bringup` | 종료 시 launch 살려두기 (RViz 계속 사용) |
| `--no-moveit` | MoveIt 빼고 RViz-only (디버그용) |

## 환경변수 (CLI 대신)

```bash
DSR_HOST=110.120.1.18 ./run_calibration.sh
DSR_RT_HOST=110.120.1.5 ./run_calibration.sh
DSR_NAME=dsr01 ./run_calibration.sh
DSR_MODEL=e0509 ./run_calibration.sh
DOOSAN_WS=/path/to/doosan_ws ./run_calibration.sh
ROS_DISTRO=humble ./run_calibration.sh
```

## 정확도 가이드

| 등급 | 위치 평균 | 회전 평균 | 의미 |
|---|---|---|---|
| 🟢 매우 우수 | < 3mm | < 0.5° | 산업 등급 정밀 작업 |
| 🟢 우수 | < 8mm | < 1.5° | 일반 정밀 픽앤플레이스 |
| 🟡 양호 | < 15mm | < 3° | 일반 작업용 |
| 🟡 보통 | < 30mm | < 5° | 재시도 권장 |
| 🔴 부정확 | ≥ 30mm | ≥ 5° | 셋업 점검 필수 |

자동 정제는 **🟢 매우 우수 등급 도달까지** 추가 포즈 수집을 반복합니다 (최대 5라운드, 라운드당 8 포즈).

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `Doosan 컨트롤러 응답 없음` | 로봇 전원/네트워크 확인. `--host` 로 IP 직접 지정. |
| `RealSense Device or resource busy` | 다른 프로세스가 카메라 잡고있음. `pkill -9 -f realsense` |
| `doosan_ws 자동 감지 실패` | 환경변수 `DOOSAN_WS=/path/to/ws` 로 지정 |
| 마커 검출 안 됨 | `d` 키 눌러 ArUco 사전 자동 탐색 |
| 축이 떨림/튐 | 마커 인쇄 품질 / 평면성 / 마운팅 강도 점검 |
| `success=True` 인데 robot 안 움직임 | 두산 박스 stuck. 비상정지 한번 → 풀기, 또는 컨트롤러 power cycle |

## 알고리즘 (요약)

```
[1] cv2.calibrateHandEye 5 method 비교 → 가장 낮은 잔차 method 선택
        ↓
[2] Nonlinear refinement (Levenberg-Marquardt) → AX=XB 잔차 직접 minimize
        ↓
[3] Iterative outlier rejection (2σ 기준) → flyer 제외 후 재 refinement
        ↓
[4] Auto-refine 루프 → 매우 우수 등급 도달까지 추가 포즈 수집 반복
```

## 캘리브레이션 결과 사용 예시

```python
import numpy as np

# Calibration 결과 로드
result = np.load("calibration_data/calibration_result.npz")
T_cam2base = result['T_cam2base']  # 4×4

# 이미지에서 검출한 객체 위치 → 로봇 베이스 좌표
def object_in_camera_to_base(t_obj_cam: np.ndarray) -> np.ndarray:
    """t_obj_cam: 카메라 frame 에서 본 객체 위치 (m)"""
    t_obj_cam_h = np.append(t_obj_cam, 1.0)   # homogeneous
    t_obj_base_h = T_cam2base @ t_obj_cam_h
    return t_obj_base_h[:3]                    # base frame (m)
```

## 라이선스 / 출처

- Doosan e0509 driver: [doosan-robotics/doosan-robot2](https://github.com/doosan-robotics/doosan-robot2)
- Hand-Eye math: OpenCV `cv2.calibrateHandEye` (TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS)
- Refinement: scipy.optimize.least_squares (LM)
