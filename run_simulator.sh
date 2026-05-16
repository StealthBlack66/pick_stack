#!/bin/bash
# ============================================================================
# 큐브 적층 시뮬레이터 launcher (20_큐브_시뮬레이터.py)
#
# 자동 감지:
#   - ROS distro (humble, iron, jazzy, rolling)
#   - Doosan workspace 위치
#   - Python 가상환경 (vision_env 우선, system python fallback)
#
# 사용:
#   ./run_simulator.sh                 # GUI 띄움. 기본 --dry-run.
#   ./run_simulator.sh --run           # 실 로봇 모션
#   ./run_simulator.sh --no-vision     # 비전 없이 fake dets
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  🧊 Cube Stack Simulator Launcher"
echo "═══════════════════════════════════════════════════════════════════════"

# 1) ROS distro
ROS_DISTRO_AUTO=""
for d in humble iron jazzy rolling foxy; do
    if [ -d "/opt/ros/$d" ]; then
        ROS_DISTRO_AUTO="$d"
        break
    fi
done
if [ -z "$ROS_DISTRO_AUTO" ]; then
    echo "❌ ROS 2 가 /opt/ros 에 없습니다." >&2
    exit 1
fi
export ROS_DISTRO="${ROS_DISTRO:-$ROS_DISTRO_AUTO}"
echo "  ✓ ROS distro: $ROS_DISTRO"
source "/opt/ros/$ROS_DISTRO/setup.bash"

# 2) doosan_ws
DOOSAN_WS_AUTO=""
for d in "$DOOSAN_WS" "$HOME/doosan_ws" "$HOME/ros2_ws" "$HOME/dsr_ws" "$HOME/ws" "/opt/doosan_ws"; do
    if [ -n "$d" ] && [ -d "$d/install" ]; then
        DOOSAN_WS_AUTO="$d"
        break
    fi
done
if [ -z "$DOOSAN_WS_AUTO" ]; then
    echo "⚠ doosan_ws 자동 감지 실패 — 로봇 통신 안 됨 (dry-run 만 가능)"
else
    export DOOSAN_WS="$DOOSAN_WS_AUTO"
    echo "  ✓ Doosan WS: $DOOSAN_WS"
    source "$DOOSAN_WS/install/setup.bash"
fi

# 3) Python venv 자동 감지
PYTHON_BIN=""
for v in \
    "$SCRIPT_DIR/../../vision_env/bin/python" \
    "$SCRIPT_DIR/../../venv/bin/python" \
    "$HOME/vision_env/bin/python" \
    "$HOME/venv/bin/python" \
    "/usr/bin/python3"; do
    if [ -x "$v" ]; then
        if "$v" -c "import numpy, PyQt5, OpenGL.GL" 2>/dev/null; then
            PYTHON_BIN="$v"
            break
        fi
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Python 환경 못 찾음. 필수 패키지: numpy, PyQt5, PyOpenGL" >&2
    echo "   설치: $SCRIPT_DIR/install_simulator_deps.sh" >&2
    exit 1
fi
echo "  ✓ Python: $PYTHON_BIN"

# 4) 실행
echo "═══════════════════════════════════════════════════════════════════════"
exec "$PYTHON_BIN" "$SCRIPT_DIR/20_큐브_시뮬레이터.py" "$@"
