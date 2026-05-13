#!/bin/bash
# ============================================================================
# Hand-Eye Calibration 원샷 launcher
#
# 자동 감지:
#   - ROS distro (humble, iron, jazzy, rolling)
#   - Doosan workspace 위치
#   - Python 가상환경 (vision_env, venv, 시스템 python 순)
#   - Doosan 컨트롤러 IP (110.120.1.1~100 자동 스캔, port 12345)
#
# 사용:
#   ./run_calibration.sh             # 모두 자동 감지
#   ./run_calibration.sh --host 110.120.1.18    # 로봇 IP 직접 지정
#   DSR_HOST=110.120.1.5 ./run_calibration.sh   # 환경변수로 지정
#
# 요구사항:
#   - ROS 2 (humble 권장) + dsr_msgs2 + dsr_bringup2 패키지 빌드된 doosan_ws
#   - Python: numpy, opencv-contrib-python (4.7+), pyrealsense2, scipy
#   - Intel RealSense 카메라 (Eye-to-Hand)
#   - DICT_6X6_50 ArUco 마커 (50mm) 그리퍼 부착
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  🤖 Doosan Hand-Eye Calibration Launcher"
echo "═══════════════════════════════════════════════════════════════════════"

# 1) ROS distro 자동 감지
ROS_DISTRO_AUTO=""
for d in humble iron jazzy rolling foxy; do
    if [ -d "/opt/ros/$d" ]; then
        ROS_DISTRO_AUTO="$d"
        break
    fi
done
if [ -z "$ROS_DISTRO_AUTO" ]; then
    echo "❌ ROS 가 /opt/ros 에 없습니다. ROS 2 설치 필요." >&2
    exit 1
fi
export ROS_DISTRO="${ROS_DISTRO:-$ROS_DISTRO_AUTO}"
echo "  ✓ ROS distro: $ROS_DISTRO"
source "/opt/ros/$ROS_DISTRO/setup.bash"

# 2) doosan_ws 자동 감지
DOOSAN_WS_AUTO=""
for d in "$DOOSAN_WS" "$HOME/doosan_ws" "$HOME/ros2_ws" "$HOME/dsr_ws" "$HOME/ws" "/opt/doosan_ws"; do
    if [ -n "$d" ] && [ -d "$d/install" ]; then
        DOOSAN_WS_AUTO="$d"
        break
    fi
done
if [ -z "$DOOSAN_WS_AUTO" ]; then
    echo "❌ doosan_ws 자동 감지 실패. 다음 위치 중 하나에 install/ 있어야:" >&2
    echo "   ~/doosan_ws  ~/ros2_ws  ~/dsr_ws  /opt/doosan_ws" >&2
    echo "   또는 DOOSAN_WS 환경변수로 직접 지정" >&2
    exit 1
fi
export DOOSAN_WS="$DOOSAN_WS_AUTO"
echo "  ✓ Doosan WS: $DOOSAN_WS"
source "$DOOSAN_WS/install/setup.bash"

# 3) Python venv 자동 감지 (현재 dir → ~/vision_env → 시스템 python 순)
PYTHON_BIN=""
for v in \
    "$SCRIPT_DIR/../../vision_env/bin/python" \
    "$SCRIPT_DIR/../../venv/bin/python" \
    "$HOME/vision_env/bin/python" \
    "$HOME/venv/bin/python" \
    "/usr/bin/python3"; do
    if [ -x "$v" ]; then
        # 필수 패키지 import 가능 한지 빠르게 확인
        if "$v" -c "import cv2, numpy, scipy, pyrealsense2" 2>/dev/null; then
            PYTHON_BIN="$v"
            break
        fi
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Python 환경 못 찾음. 다음 패키지 모두 가진 venv 또는 시스템 python 필요:" >&2
    echo "   numpy  opencv-contrib-python (4.7+)  scipy  pyrealsense2" >&2
    echo "   설치: pip install numpy opencv-contrib-python scipy pyrealsense2" >&2
    exit 1
fi
echo "  ✓ Python: $PYTHON_BIN"

# 4) 09 실행 (CLI 인자 그대로 전달)
echo "═══════════════════════════════════════════════════════════════════════"
exec "$PYTHON_BIN" "$SCRIPT_DIR/09_원샷_캘리브레이션.py" "$@"
