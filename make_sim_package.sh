#!/bin/bash
# ============================================================================
# 큐브 시뮬레이터 배포 패키지 생성
#
# 사용:
#   ./make_sim_package.sh              # 시뮬 + 강의 모듈 묶음 zip
#   ./make_sim_package.sh --sim-only   # 시뮬 코어만 (가벼움)
#   ./make_sim_package.sh --full       # 모든 파일 (yolo*.pt 등 포함)
#
# 출력: dist/cube_simulator_<mode>_<version>_<date>.zip
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 모드 선택
MODE="standard"
INCLUDE_SAM=0
for arg in "$@"; do
    case "$arg" in
        --sim-only) MODE="sim-only" ;;
        --full)     MODE="full" ;;
        --include-sam) INCLUDE_SAM=1 ;;
        --help|-h)
            grep -E '^# ' "$0" | sed 's/^# //'
            exit 0 ;;
    esac
done

VERSION="$(grep '^version' pyproject.toml | head -1 | cut -d'"' -f2)"
DATE="$(date +%Y%m%d)"
OUT_NAME="cube_simulator_${MODE}_v${VERSION}_${DATE}"
OUT_DIR="dist"
STAGE="$OUT_DIR/staging_${OUT_NAME}"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Cube Simulator — 패키지 생성 (mode=$MODE, version=$VERSION)"
echo "═══════════════════════════════════════════════════════════════════════"

# 1) 기존 staging 정리
rm -rf "$STAGE"
mkdir -p "$STAGE"

# 2) 필수 파일 (모든 모드 공통)
echo "  → 코어 파일 복사"
cp -r cube_simulator "$STAGE/"
cp pyproject.toml "$STAGE/"
cp requirements-sim.txt "$STAGE/"
cp SIMULATOR_README.md "$STAGE/README.md"   # 배포 zip 안에선 README.md 가 기본
cp 20_큐브_시뮬레이터.py "$STAGE/"
cp install_simulator_deps.sh "$STAGE/"
cp run_simulator.sh "$STAGE/"
cp .env.example "$STAGE/"

# 3) 모드별 추가
case "$MODE" in
    sim-only)
        # 시뮬·디자인 only — 강의 모듈 (12·15·16·17) 없음. 정렬·실 로봇 불가.
        # cube_simulator 가 정렬 모드 시도하면 ModuleNotFoundError 로 깨끗히 실패.
        echo "  → sim-only 모드 — 강의 모듈 미포함 (정렬·실 로봇 불가)"
        # 빈 placeholder 만 만들어 import 에러 시 명확한 메시지
        cat > "$STAGE/SIMULATOR_MODE.txt" <<'EOF'
sim-only 패키지 — 디자인 + 시뮬레이션 only.
정렬·실 로봇 모드는 동작하지 않습니다 (강의 모듈 미포함).
전체 패키지가 필요하면 standard 또는 full 모드로 재패키징하세요.
EOF
        ;;
    standard|full)
        # 강의 모듈 포함
        echo "  → 강의 모듈 (00·06·07·08·09·10·11·12·15·16·17) 포함"
        # 로봇 연결/활성화
        cp 00_두산로봇_리눅스_실물_연결.py "$STAGE/" 2>/dev/null || true
        cp 00_로봇_활성화하기.py "$STAGE/" 2>/dev/null || true
        # 관절 제어 / 캘리브레이션 순회
        cp 06_로봇_관절_제어_보정형.py "$STAGE/" 2>/dev/null || true
        cp 07_캘리브레이션_순회_모션.py "$STAGE/" 2>/dev/null || true
        # Hand-Eye 캘리브레이션
        cp 08_카메라_핸드아이_캘리브레이션.py "$STAGE/" 2>/dev/null || true
        cp 09_원샷_캘리브레이션.py "$STAGE/" 2>/dev/null || true
        cp run_calibration.sh "$STAGE/" 2>/dev/null || true
        # 그리퍼
        cp 10_그리퍼_연결.py "$STAGE/" 2>/dev/null || true
        cp 11_그리퍼_키보드_제어.py "$STAGE/" 2>/dev/null || true
        cp 11_2_그리퍼_물성측정_그래프.py "$STAGE/" 2>/dev/null || true
        # 비전 + 시퀀스 (시뮬레이터가 import 하는 것들)
        cp 12_비전_피크앤플레이스.py "$STAGE/"
        cp 15_바둑판_정렬.py "$STAGE/"
        cp 16_탑쌓기.py "$STAGE/"
        cp 17_미술쌓기.py "$STAGE/"
        # 설정/마커/문서
        cp doosan_config.py "$STAGE/"
        cp requirements-vision.txt "$STAGE/"
        cp requirements-robot.txt "$STAGE/"
        cp requirements-llm.txt "$STAGE/" 2>/dev/null || true
        cp PORTABILITY.md "$STAGE/" 2>/dev/null || true
        cp RECOVERY.md "$STAGE/" 2>/dev/null || true
        cp aruco_DICT_6X6_50_ID0_50mm.png "$STAGE/" 2>/dev/null || true
        cp generate_aruco_marker.py "$STAGE/" 2>/dev/null || true

        if [ "$MODE" = "full" ]; then
            echo "  → YOLO 모델 + calibration data 포함 (시뮬레이터에 필요)"
            cp yolo*.pt "$STAGE/" 2>/dev/null || true
            if [ -d calibration_data ]; then
                cp -r calibration_data "$STAGE/"
            fi
            if [ "$INCLUDE_SAM" = "1" ]; then
                echo "  → SAM 모델도 포함 (14번 액체 세그용, 큐브 시뮬엔 불필요)"
                cp sam*.pt "$STAGE/" 2>/dev/null || true
            else
                echo "  ↷ SAM 모델 제외 (--include-sam 으로 명시 포함 가능)"
            fi
        fi
        ;;
esac

# 4) __pycache__ / .pyc 정리
find "$STAGE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete 2>/dev/null || true

# 5) 파일 크기 출력
SIZE_HUMAN="$(du -sh "$STAGE" | cut -f1)"
FILE_COUNT="$(find "$STAGE" -type f | wc -l)"
echo "  → 스테이징 완료: $FILE_COUNT 파일, $SIZE_HUMAN"

# 6) zip 묶기
ZIP_PATH="$OUT_DIR/$OUT_NAME.zip"
rm -f "$ZIP_PATH"
(cd "$OUT_DIR" && zip -r -q "$OUT_NAME.zip" "staging_${OUT_NAME}/" && \
    mv "$OUT_NAME.zip" "$OUT_NAME.zip.tmp" && \
    cd "staging_${OUT_NAME}" && \
    zip -r -q "../$OUT_NAME.zip" . && \
    cd .. && rm -f "$OUT_NAME.zip.tmp")
# 위 dance 는 staging 디렉토리 prefix 없이 zip 만듦

# 정리: staging 삭제 (zip 만 남김)
rm -rf "$STAGE"

ZIP_SIZE="$(du -sh "$ZIP_PATH" | cut -f1)"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  ✓ 생성 완료"
echo "      $ZIP_PATH"
echo "      $ZIP_SIZE"
echo ""
echo "  다른 컴퓨터에서 사용:"
echo "      unzip $OUT_NAME.zip"
echo "      cd $OUT_NAME"
echo "      ./install_simulator_deps.sh"
echo "      ./run_simulator.sh"
echo "═══════════════════════════════════════════════════════════════════════"
