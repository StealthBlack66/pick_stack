# 🧊 Cube Stack Simulator

> **두산 E0509 + RH-P12-RN-A 큐브 적층 시뮬레이터.**
> 3D GUI 에서 큐브 배치를 디자인하고, 가상 그리퍼로 픽앤플레이스를 시뮬레이션 한 뒤,
> 실제 로봇에서 동일한 모양을 재현합니다.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green?logo=qt)
![OpenGL](https://img.shields.io/badge/OpenGL-3.1%2B-orange)

---

## 📦 빠른 시작 (시뮬만 — 다른 컴퓨터 가장 가볍게)

### 1) 다운로드

```bash
git clone https://github.com/StealthBlack66/pick_stack.git
cd pick_stack
```

또는 zip 파일 받아서 압축 해제.

### 2) Python 의존성 설치

```bash
# 권장: 가상환경 만들기
python3 -m venv ~/cube_sim_venv
source ~/cube_sim_venv/bin/activate

# 시뮬·디자인만 사용하는 경우 (가장 가벼움)
pip install -r requirements-sim.txt
```

### 3) 실행

```bash
# 방법 1 — 직접 실행
python 20_큐브_시뮬레이터.py

# 방법 2 — 모듈로 실행 (한글 파일명 안 됨 환경)
python -m cube_simulator

# 방법 3 — pip install 후 명령으로
pip install .
cube-simulator
```

GUI 가 뜨면 끝. 큐브 드래그·프리셋·시뮬레이션 모두 동작.

---

## 🎯 모드별 기능

| 모드 | 동작 | 의존성 |
|------|------|--------|
| **디자인 only** | 3D 큐브 배치 디자인, JSON 저장/로드, 모션 계획 표 편집 | `requirements-sim.txt` |
| **가상 시뮬레이션** | 위 + 가상 그리퍼 픽앤플레이스 애니메이션, 충돌 검사, yaw 자동 보정 | `requirements-sim.txt` |
| **정렬 (15번)** | 위 + RealSense 카메라로 큐브 검출 → 5×5 그리드 자동 정렬 | `+ requirements-vision.txt` + ROS2 |
| **실 로봇 실행** | 위 + 두산 e0509 로 실제 픽앤플레이스 수행 | `+ requirements-robot.txt` + ROS2 + dsr_bringup2 |

---

## 🖥️ Ubuntu 22.04 자동 설치 (시뮬 only)

```bash
# 시스템 의존성 (PyQt5 OpenGL 등이 X11 / libGL 필요)
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libxcb-xinerama0 libxkbcommon-x11-0

# 본 프로젝트 설치
git clone https://github.com/StealthBlack66/pick_stack.git
cd pick_stack
./install_simulator_deps.sh   # pip install + 데스크탑 런처 등록

# 실행
./run_simulator.sh
```

설치 후 앱 메뉴에서 **"Cube Stack Simulator"** 검색해서 클릭으로도 실행 가능.

---

## 🐧 다른 Linux / macOS

### macOS

```bash
brew install python@3.10
python3 -m venv ~/cube_sim_venv
source ~/cube_sim_venv/bin/activate
pip install -r requirements-sim.txt
python 20_큐브_시뮬레이터.py
```

### 다른 Linux

`apt`/`dnf`/`pacman` 으로 `python3-pyqt5`, `python3-opengl` 시스템 설치 후 venv 의존성 설치.

### Windows (시뮬 only)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements-sim.txt
python 20_큐브_시뮬레이터.py
```

실 로봇 모드는 ROS2 Windows 지원이 제한적이라 Linux 환경 권장.

---

## 🤖 실 로봇 모드 추가 설치

### 1) ROS2 Humble + 두산 ROS2 패키지

```bash
# ROS2 Humble (Ubuntu 22.04 기준)
sudo apt install ros-humble-desktop ros-humble-rmw-fastrtps-cpp
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

# 두산 ROS2 패키지 빌드
mkdir -p ~/doosan_ws/src && cd ~/doosan_ws/src
git clone https://github.com/DoosanRobotics/doosan-robot2.git
cd .. && rosdep install --from-paths src --ignore-src -r -y
colcon build && source install/setup.bash
```

### 2) Python 비전 + 로봇 의존성

```bash
source ~/cube_sim_venv/bin/activate
pip install -r requirements-robot.txt
```

### 3) `.env` 로 네트워크/네임스페이스 설정

```bash
cp .env.example .env
$EDITOR .env   # ROBOT_IP, NAMESPACE, DOOSAN_WS 수정
```

자세한 내용은 [PORTABILITY.md](PORTABILITY.md) 참고.

### 4) 실 로봇 실행

```bash
./run_simulator.sh   # ROS distro / doosan_ws 자동 감지
```

GUI 에서 **"Dry-run"** 체크 해제 → ▶ 바둑판 정렬 / ▶ 로봇 실행.

---

## 🎮 사용법 요약

### 화면 구성 (3-pane)

```
[ 디자인 패널 ][   3D 뷰    ][ 실행·계획 패널 ]
   왼쪽         가운데        오른쪽
```

- **왼쪽 컬럼** — 도구 모드 (추가/선택/지우개), 선택 큐브 편집, 프리셋, JSON 저장/로드
- **가운데** — 3D 뷰. 마우스로 회전/줌, 클릭으로 큐브 배치, 드래그로 이동
- **오른쪽 컬럼** — 모션 계획 표, 정렬·시뮬·실행 버튼, z/yaw 옵션, 로그

### 권장 동작 순서

1. **디자인** — 프리셋 (예: "탑 5층") 로드 또는 직접 드래그
2. **시뮬레이션** ▶ — 충돌 다이얼로그가 뜨면 yaw 회전 "예" 응답
3. **(실 로봇 모드)** ▶ 바둑판 정렬 — 카메라로 큐브 검출 후 5×5 그리드에 배치
4. **(선택)** 플랜 표에서 미세 조정 — `src x/y/yaw`, `tgt x/y/z/yaw`
5. **`Dry-run` 체크 해제** → ▶ 로봇 실행 → 실제 동작
6. **🏠 원점복귀** — 비상정지·SAFE_OFF·trajectory 실패 등 어떤 상황이든 안전하게 복구

### 키보드 단축키

| 키 | 동작 |
|---|---|
| `A` / `S` / `E` | 추가 / 선택 / 지우개 모드 |
| `T` / `I` | Top view / Isometric view |
| `Delete` | 선택 큐브 삭제 |
| `Esc` | 비상정지 |
| `Ctrl+S` / `Ctrl+O` | JSON 저장 / 불러오기 |

---

## ⚙️ 안전 옵션 (실 로봇 모드)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| **table z 수동** | ✓ ON, -30 mm | 실측 테이블 표면 z. 비전 z 가 불안정해도 항상 이 값 기준 |
| **pick 안전 마진** | 5 mm | finger tip 이 테이블 위로 유지할 최소 여유 — 자동 clamp |
| **pick z offset** | 0 mm | cube 잡는 깊이 보정. 음수 = 더 깊이 (-3 ~ -5 권장) |
| **placement z offset** | -10 mm | cube 놓을 때 z 보정. 음수 = 더 깊이 |
| **비전 conf 임계값** | 0.40 | YOLO 검출 임계. 검출 안 되면 낮춤 (0.20), 과검이면 올림 (0.60) |
| **시뮬 시 yaw 자동 보정** | ✓ ON | 시뮬 시작 전 인접 큐브 finger 충돌 회피 |

---

## 🔧 트러블슈팅

### GUI 가 즉시 꺼짐 / 로그 안 보임

`~/.cache/cube_simulator/crash.log` 에 traceback 누적. 가장 최근 항목 확인:

```bash
tail -50 ~/.cache/cube_simulator/crash.log
```

### "서비스 미응답: set_robot_mode"

bringup 은 살아 있지만 새 Python 노드의 DDS service discovery 가 늦음. 보통 워밍업 30초로 해결. 30초 지나도 안 풀리면:
- 펜던트가 AUTO 모드인지 확인
- 🏠 원점복귀 → dsr_bringup2 통째 재시작

### SAFE_OFF (state=3) 가 안 풀림

🏠 원점복귀 클릭 → `_hard_reset_driver` 가 dsr/DRCF 프로세스 통째 kill + bringup 재시작 (~30초). 펜던트 안 만져도 SAFE_OFF 풀림.

### "Trajectory 실행 실패 (status=6 ABORTED)"

trajectory duration 이 짧아서 컨트롤러 추종 실패. 자동으로 12초 재시도하고 시작 자세 멀면 🏠 원점복귀로 HOME 부터 시작.

### 그리퍼가 바닥 부딪침

**오른쪽 컬럼 > "table z 수동"** 체크박스 ON 인지 확인, 값이 실측 테이블 z (예: -30mm) 인지 확인. 자동 floor 가 작동해 finger tip 이 table 위 5mm 유지.

### 시뮬과 실 로봇 큐브 회전이 다름

자동 처리됨 — worker 가 Doosan RPY 컨벤션 (pitch=180 으로 yaw 반전) 에 맞춰 부호 flip 해서 송신. 그래도 차이 나면 yaw 자동 보정 체크박스 해제 후 plan 표에서 수동 입력.

---

## 📁 디렉토리 구조

```
02_Doosan_Robot_제어/
├── 20_큐브_시뮬레이터.py        # 진입점 (한글 파일명)
├── cube_simulator/              # Python 패키지 (시뮬 코어)
│   ├── __init__.py              # MODULE_PATHS — 외부 스크립트 경로 dict
│   ├── __main__.py              # `python -m cube_simulator` 진입
│   ├── main_window.py           # 3-pane GUI
│   ├── controller.py            # MVC controller — 모든 비즈니스 로직
│   ├── model.py                 # CubeModel (큐브 배치 데이터)
│   ├── gl_view.py               # 3D OpenGL view
│   ├── gl_primitives.py         # 큐브/그리퍼 메쉬
│   ├── sim_animator.py          # 가상 시뮬 (phase machine, SAT 충돌)
│   ├── motion_plan.py           # PlanItem + build_plan
│   ├── robot_worker.py          # 실 로봇 QThread (정렬·실행·복귀)
│   ├── shapes_io.py             # 프리셋 JSON
│   └── widgets/                 # 분리된 패널들
│       ├── selected_cube_panel.py
│       ├── plan_table_panel.py
│       └── run_control_panel.py
├── 12_비전_피크앤플레이스.py     # 두산 PickAndPlace 클래스 (p12)
├── 15_바둑판_정렬.py             # 5×5 정렬 (p15)
├── 16_탑쌓기.py                  # pick&place 시퀀스 (p16)
├── 17_미술쌓기.py                # 추가 패턴 (p17)
├── doosan_config.py             # 네임스페이스/IP/WS 설정
├── pyproject.toml               # Python 패키지 메타
├── requirements-sim.txt         # 시뮬 only 의존성
├── requirements-vision.txt      # 비전 의존성
├── requirements-robot.txt       # 실 로봇 의존성
├── install_simulator_deps.sh    # Ubuntu 자동 설치 (시뮬)
├── run_simulator.sh             # 실행 래퍼 (ROS 환경 자동 감지)
├── .env.example                 # 환경 변수 템플릿
├── SIMULATOR_README.md          # 본 문서
└── PORTABILITY.md               # 환경 이전 가이드
```

---

## 🐛 작업했던 문제들 (참고)

지금까지 해결한 주요 이슈는 [`SIMULATOR_CHANGELOG.md`](SIMULATOR_CHANGELOG.md) 또는 git log 참고.

핵심 교훈:
- **두산 RPY `[0, 180, Y]`** = TCP pitch 180° → world 에서 yaw 부호 반전. worker 가 `-yaw` 송신.
- **15번 `reset_robot_driver()`** = SAFE_OFF 포함 모든 stale 상태의 만능 복구. 🏠 원점복귀에 인라인.
- **DDS discovery 는 5초로 부족** — 새 노드 만든 후 30초 워밍업 필수.
- **비전 sample_z 불안정** — 사용자 실측 `z_table_top` override + safety floor 필수.

---

## 📄 라이선스

MIT. 자유롭게 사용·수정·재배포 가능. 출처 표기 권장.

---

## 🙏 기여

이슈/PR 환영. 강의 자료로 활용하는 경우 사용 사례 공유해 주시면 감사.
