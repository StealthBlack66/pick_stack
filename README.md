# Doosan E0509 × RH-P12-RN-A — 비전 기반 큐브 정렬 & 탑 쌓기

> **Intel RealSense + YOLO 세그멘테이션 + 두산 E0509 로봇 + RH-P12-RN-A 그리퍼**로  
> 바닥에 흩어진 나무 큐브를 자동으로 바둑판 배열로 정렬하고, 수직 탑·피라미드를 쌓는 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![ROS2](https://img.shields.io/badge/ROS2-Humble-orange?logo=ros)
![YOLO](https://img.shields.io/badge/YOLOv8-Segmentation-purple?logo=ultralytics)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 데모

| 바둑판 정렬 (15) | 탑 쌓기 (16) |
|:---:|:---:|
| *(영상/GIF 추가 예정)* | *(영상/GIF 추가 예정)* |

---

## 주요 기능

### 15 — 바둑판 정렬 (`15_바둑판_정렬.py`)
- **YOLO 세그멘테이션**으로 실시간 나무 큐브 검출 (RealSense RGB-D)
- **5×5 그리드 (50 mm 간격)** 자동 배치 — 충돌·핑거 공간 자동 회피
- 폴리곤 면적·종횡비 필터로 false positive 억제
- 트래킹 EMA 스무딩으로 떨림 없는 안정적 pick
- `--dry-run` 모드: 실제 모션 없이 전체 시퀀스 콘솔 검증
- `--limit N` 옵션으로 지정 개수만 정렬

### 16 — 탑 쌓기 (`16_탑쌓기.py`)
- 정렬된 큐브를 기반으로 **두 가지 탑 구성**
  - **Tower 1** — 수직 1자 5층 스택
  - **Tower 2** — 피라미드 3-2-1 (바닥 3개 → 중간 2개 → 꼭대기 1개)
- pick yaw + 90° 그립으로 X/Y 양축 friction 정렬 강화
- 탑 높이 이상의 safe-z 경유로 충돌 없는 이동
- `--dry-run` 모드 동일 지원

---

## 하드웨어 구성

| 항목 | 사양 |
|---|---|
| 로봇 암 | Doosan E0509 (6-DOF) |
| 그리퍼 | RH-P12-RN-A (Robotis) |
| 카메라 | Intel RealSense D455 / D435 |
| 마커 | ArUco DICT_6X6_50, ID=0, 50 mm (Hand-Eye 캘리브레이션용) |

---

## 소프트웨어 요구사항

| 항목 | 버전 |
|---|---|
| Ubuntu | 22.04 |
| ROS 2 | Humble (iron / jazzy도 가능) |
| Python | 3.10 |
| doosan_ws | `dsr_msgs2`, `dsr_bringup2`, `dsr_moveit_config_e0509` 포함 빌드 |

```bash
pip install numpy opencv-contrib-python scipy pyrealsense2 ultralytics torch
# (선택) .env 파일 사용 시
pip install python-dotenv
```

> ⚠️ 반드시 **`opencv-contrib-python`** — 일반 `opencv-python`에는 ArUco 모듈 없음

---

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/StealthBlack66/pick_stack.git
cd pick_stack
```

### 2. 환경 설정 (30초 셋업)

모든 스크립트는 `doosan_config.py` 한 곳에서 환경을 읽습니다.  
**IP, namespace, ROS distro 등을 `.env` 파일 하나로 통합 관리**할 수 있습니다.

```bash
# 템플릿 복사
cp .env.example .env

# 자기 환경 값으로 수정 (IP, namespace 등)
nano .env

# 적용 확인
python3 doosan_config.py
```

`NAMESPACE`, `ROBOT_IP` 등이 설정한 값으로 출력되면 완료. 이후 모든 스크립트에 자동 반영됩니다.

#### `.env` 설정 항목

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `DSR_NAME` | `dsr01` | ROS namespace |
| `DSR_MODEL` | `e0509` | 로봇 모델 |
| `DSR_HOST` | `110.120.1.18` | 컨트롤러 IP (미지정 시 서브넷 자동 스캔) |
| `DSR_RT_HOST` | `110.120.1.5` | PC 측 RT 호스트 IP |
| `DSR_SUBNET` | `110.120.1` | IP 자동 탐색 서브넷 (`.1~.100` 스캔) |
| `DSR_BRINGUP_PKG` | `dsr_bringup2` | bringup 패키지 이름 (fork 다른 경우) |
| `DSR_BRINGUP_LAUNCH` | `dsr_bringup2_moveit.launch.py` | launch 파일 이름 |
| `DSR_MOVEIT_CONTROLLER` | `dsr_moveit_controller` | MoveIt 컨트롤러 이름 |
| `DOOSAN_WS` | `~/doosan_ws` | 워크스페이스 경로 |
| `ROS_DISTRO` | `humble` | ROS distro |

**우선순위:** 셸 `export` > `.env` 파일 > `doosan_config.py` 기본값

`.env` 없이 셸 export만으로도 동일하게 동작합니다:

```bash
export DSR_NAME=dsr01e0509
export DSR_HOST=192.168.137.100
python3 15_바둑판_정렬.py
```

> 본인 환경의 namespace를 모른다면:
> ```bash
> ros2 node list | head
> ros2 service list | grep set_robot_mode
> ```

### 3. Hand-Eye 캘리브레이션 (최초 1회)

```bash
./run_calibration.sh
# → calibration_data/calibration_result.npz 생성
```

### 4. 바둑판 정렬 실행

```bash
# Dry-run (모션 없이 시퀀스 검증)
python3 15_바둑판_정렬.py --dry-run

# 실 동작
python3 15_바둑판_정렬.py

# 처음 N개만 정렬
python3 15_바둑판_정렬.py --limit 10
```

### 5. 탑 쌓기 실행

```bash
# Dry-run
python3 16_탑쌓기.py --dry-run

# 실 동작
python3 16_탑쌓기.py
```

---

## 파이프라인 개요

```
[RealSense RGB-D]
       │  RGB 프레임
       ▼
[YOLOv8 Segmentation]  →  큐브 mask polygon
       │  bbox center + depth
       ▼
[Hand-Eye 역변환]  →  base frame XY 좌표 (mm)
       │
       ├─ 15: 그리드 셀 할당 → 충돌 검사 → pick & place
       │
       └─ 16: Tower1 (5층 수직) + Tower2 (피라미드 3-2-1)
                   │
                   ▼
            [Doosan E0509 + RH-P12-RN-A]
```

---

## 🔧 주요 CLI 옵션

### `15_바둑판_정렬.py`

| 옵션 | 설명 |
|---|---|
| `--dry-run` | 모션/그리퍼 명령 없이 콘솔 출력만 |
| `--limit N` | 처음 N개 큐브만 정렬 |
| `--conf 0.4` | YOLO 검출 신뢰도 임계값 |
| `--host IP` | 로봇 IP 직접 지정 |

### `16_탑쌓기.py`

| 옵션 | 설명 |
|---|---|
| `--dry-run` | 모션/그리퍼 명령 없이 콘솔 출력만 |
| `--conf 0.4` | YOLO 검출 신뢰도 임계값 |
| `--host IP` | 로봇 IP 직접 지정 |

---

## 로봇 상태 & 충돌 복구

두산 컨트롤러는 이전 세션의 충돌·protective stop 신호가 남아있으면 모든 명령이 무시됩니다.  
`activate_robot()` 루틴이 **매 실행 시작 시 자동으로 복구**합니다.

### 로봇 상태 코드

| 코드 | 상태 | 처리 방법 |
|---|---|---|
| 1 | STANDBY | 정상 — 명령 가능 |
| 2 | MOVING | 정상 — 이동 중 |
| 3 / 10 | SAFE_OFF / SAFE_OFF2 | `SetRobotControl(1)` 로 Servo 재활성 |
| **5 / 9** | **SAFE_STOP / SAFE_STOP2** | **자동 복구 대상** (아래 시퀀스) |
| **6** | **EMERGENCY_STOP** | **코드로 해제 불가 — 펜던트 비상정지 버튼 수동 해제 필요** |

### 자동 복구 시퀀스 (SAFE_STOP / SAFE_STOP2)

```
1) set_safety_mode(safety_mode=2)         ← RECOVERY 진입
2) set_safe_stop_reset_type(reset_type=0) ← safe-stop 리셋
3) set_safety_mode(safety_mode=1)         ← AUTONOMOUS 복귀
```

각 단계 사이 0.5초 대기 (컨트롤러 내부 상태 전이가 비동기이므로 필요).  
최종 상태가 `STANDBY(1)` 또는 `MOVING(2)` 이면 복구 성공.

> 복구 확인 방법: 로봇을 손으로 막아 충돌 발생 → 스크립트 재실행 →  
> 콘솔에 `robot_state=5 (SAFE_STOP) → recovery 시퀀스 실행 → robot_state=1 (STANDBY)` 확인

자세한 코드 구현은 [`RECOVERY.md`](RECOVERY.md) 참고.

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `FileNotFoundError: best.pt` | `yolo_dataset/runs/seg_v6/weights/best.pt` 경로 확인 |
| `Doosan 컨트롤러 응답 없음` | 로봇 전원/네트워크 확인, `--host` 또는 `DSR_HOST` 로 IP 직접 지정 |
| `doosan_ws 자동 감지 실패` | `DOOSAN_WS=/path/to/ws` 환경변수 지정 |
| `RealSense Device busy` | `pkill -9 -f realsense` 후 재실행 |
| 큐브 검출 안 됨 | `--conf` 낮추거나 조명 개선 |
| 그리퍼가 큐브를 놓침 | 캘리브레이션 재실행 (`./run_calibration.sh`) |
| `success=True` 인데 로봇 미동작 | 로봇 상태 코드 확인 → `RECOVERY.md` 참고 |
| "서비스 미응답" / "namespace 못 찾음" | `python3 doosan_config.py` 로 현재 설정 확인 후 `.env` 의 `DSR_NAME` 수정 |
| `dsr_msgs` import 오류 | 이 프로젝트는 `dsr_msgs2` 기준. 다른 fork 사용 시 import 직접 수정 필요 |

---

## 파일 구조

```
.
├── 15_바둑판_정렬.py              # 메인: 큐브 검출 → 그리드 배치
├── 16_탑쌓기.py                   # 메인: 정렬된 큐브 → 탑 구성
│
├── 12_비전_피크앤플레이스.py       # PickAndPlace / RobotController (공통 모듈)
├── 13_비전_피크앤플레이스_curobo.py
├── 14_비전_액체_세그멘테이션.py
│
├── 08_카메라_핸드아이_캘리브레이션.py
├── 09_원샷_캘리브레이션.py         # 자동 Hand-Eye 캘리브레이션
├── run_calibration.sh              # 캘리브레이션 원클릭 실행
│
├── doosan_config.py                # 환경 설정 허브 (IP, namespace, ROS distro 등)
├── .env.example                    # 환경변수 템플릿 → .env 로 복사해서 사용
├── generate_aruco_marker.py        # ArUco 마커 생성
│
├── PORTABILITY.md                  # 다른 환경에서 실행하기 (상세 가이드)
├── RECOVERY.md                     # 충돌 복구 코드 & 상태 코드 레퍼런스
│
├── yolo_dataset/                   # YOLO 학습 데이터 & 가중치
│   └── runs/seg_v6/weights/best.pt
├── calibration_data/               # Hand-Eye 캘리브레이션 결과
│   └── calibration_result.npz
```

---

## 관련 프로젝트 / 참고

- [doosan-robotics/doosan-robot2](https://github.com/doosan-robotics/doosan-robot2) — ROS 2 드라이버
- [e0509_gripper_description](https://github.com/fhekwn549/e0509_gripper_description) — E0509 + RH-P12-RN-A 결합 URDF 패키지
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 세그멘테이션 모델
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) — 뎁스 카메라
- [Robotis RH-P12-RN-A](https://emanual.robotis.com/docs/en/platform/rh_p12_rna/) — 그리퍼

---

## 라이선스

MIT License — 자유롭게 사용·수정·배포 가능합니다.
