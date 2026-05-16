#!/bin/bash
# ============================================================================
# 큐브 시뮬레이터 의존성 설치 + .desktop 런처 등록
#
# 사용:
#   ./install_simulator_deps.sh             # 패키지 설치 + 런처 등록
#   ./install_simulator_deps.sh --no-launcher  # 패키지만
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NO_LAUNCHER=0
for arg in "$@"; do
    case "$arg" in
        --no-launcher) NO_LAUNCHER=1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Cube Simulator — 의존성 설치"
echo "═══════════════════════════════════════════════════════════════════════"

# 1) Python venv 자동 감지
PYTHON_BIN=""
for v in \
    "$SCRIPT_DIR/../../vision_env/bin/python" \
    "$SCRIPT_DIR/../../venv/bin/python" \
    "$HOME/vision_env/bin/python" \
    "$HOME/venv/bin/python" \
    "/usr/bin/python3"; do
    if [ -x "$v" ]; then
        PYTHON_BIN="$v"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Python 인터프리터 못 찾음" >&2
    exit 1
fi
echo "  ✓ Python: $PYTHON_BIN"

# 2) pip install — venv 면 --user 없이, 시스템 python 이면 --user 로
PIP_USER_FLAG=""
if "$PYTHON_BIN" -c "import sys; sys.exit(0 if sys.prefix == sys.base_prefix else 1)"; then
    PIP_USER_FLAG="--user"
fi
echo "  → PyQt5 + PyOpenGL 설치 ($PIP_USER_FLAG)"
"$PYTHON_BIN" -m pip install $PIP_USER_FLAG --upgrade PyQt5 PyOpenGL PyOpenGL_accelerate

# 3) import 확인
"$PYTHON_BIN" - <<'PY'
import importlib
for m in ['PyQt5.QtWidgets', 'OpenGL.GL', 'numpy']:
    importlib.import_module(m)
print('  ✓ import OK')
PY

# 4) .desktop 런처
if [ "$NO_LAUNCHER" = "0" ]; then
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    DESKTOP_FILE="$DESKTOP_DIR/cube_stack_simulator.desktop"
    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Cube Stack Simulator
Name[ko]=큐브 적층 시뮬레이터
Comment=Doosan e0509 cube stacking GUI
Exec=$SCRIPT_DIR/run_simulator.sh
Path=$SCRIPT_DIR
Icon=applications-engineering
Terminal=true
Categories=Development;Robotics;Science;
StartupNotify=true
EOF
    chmod +x "$DESKTOP_FILE"
    echo "  ✓ 런처 등록: $DESKTOP_FILE"
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
    fi
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "  설치 완료. 실행:"
echo "      $SCRIPT_DIR/run_simulator.sh"
echo "  또는 앱 메뉴에서 'Cube Stack Simulator' 검색"
echo "═══════════════════════════════════════════════════════════════════════"
