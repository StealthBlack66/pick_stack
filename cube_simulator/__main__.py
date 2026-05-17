"""`python -m cube_simulator` 또는 `cube-simulator` console script 진입점.

기존 `20_큐브_시뮬레이터.py` 와 동일 동작 — 한글 파일명 import 불가능한
환경에서 사용. 명령행 옵션도 동일.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='cube-simulator',
        description='큐브 적층 시뮬레이터 — 두산 E0509 + RH-P12-RN-A',
    )
    p.add_argument('--dry-run', dest='dry_run', action='store_true',
                   help='로봇 모션 SKIP — UI/로직 검증용 (기본 ON)')
    p.add_argument('--run', dest='dry_run', action='store_false',
                   help='실제 로봇 모션 (실 동작)')
    p.set_defaults(dry_run=True)
    p.add_argument('--no-vision', action='store_true',
                   help='비전 호출 막고 fake dets 사용')
    return p.parse_args(argv)


def _install_crash_logger(log_path: Path) -> None:
    """Qt 슬롯 안에서 던진 unhandled exception 을 파일로 흘림.

    Qt5.5+ 는 슬롯 예외가 앱 종료를 유발 — GUI 만 보던 사용자는 traceback 을
    볼 길이 없어 디버깅 불가. log_path 에 누적 기록해 사후 확인 가능.
    """
    import traceback as _tb
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prev = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                import datetime as _dt
                f.write('\n' + '=' * 60 + '\n')
                f.write(f'crash @ {_dt.datetime.now().isoformat(timespec="seconds")}\n')
                _tb.print_exception(exc_type, exc_value, exc_tb, file=f)
            sys.stderr.write(f'\n[CRASH → {log_path}]\n')
            _tb.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
            sys.stderr.flush()
        except Exception:
            pass
        if prev is not None:
            prev(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def main(argv=None) -> int:
    args = parse_args(argv)

    crash_log = Path.home() / '.cache' / 'cube_simulator' / 'crash.log'
    _install_crash_logger(crash_log)

    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError as e:
        sys.stderr.write(
            f'\n❌ PyQt5 import 실패: {e}\n'
            '   해결: pip install -r requirements-sim.txt\n'
        )
        return 2

    from .main_window import CubeSimulatorMainWindow

    app = QApplication(sys.argv)

    # 세션 자동 정리 — 앱 종료 시 ROS 노드/bringup 깔끔히 폐기 (실 로봇 모드일 때만)
    try:
        from .robot_worker import teardown_session, teardown_bringup
        app.aboutToQuit.connect(teardown_session)
        app.aboutToQuit.connect(teardown_bringup)
    except Exception:
        pass

    win = CubeSimulatorMainWindow(
        dry_run_default=args.dry_run,
        no_vision=args.no_vision,
    )
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
