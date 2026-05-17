#!/usr/bin/env python3
"""두산 e0509 + RH-P12-RN-A 큐브 적층 시뮬레이터.

3D GUI 에서 큐브를 마우스로 적층 설계한 뒤 [로봇 실행] 으로 실제 로봇에
신호를 보내 동일한 모양을 재현합니다.

사용법:
    python3 20_큐브_시뮬레이터.py [--dry-run] [--no-vision]

옵션:
    --dry-run    기본값. 로봇 모션을 SKIP 하고 흐름만 검증.
    --no-vision  비전 호출을 막고, fake dets (grid_cells) 로 동작.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
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
                f.write('\n' + '=' * 60 + '\n')
                import datetime as _dt
                f.write(f'crash @ {_dt.datetime.now().isoformat(timespec="seconds")}\n')
                _tb.print_exception(exc_type, exc_value, exc_tb, file=f)
            # stderr 에도 (터미널 실행 시 즉시 보이게)
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

    # 부모 디렉토리를 path 에 추가 (cube_simulator 패키지 import)
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    # 크래시 로그 — 윈도우가 닫혀도 ~/.cache/cube_simulator/crash.log 에 traceback 남음
    crash_log = Path.home() / '.cache' / 'cube_simulator' / 'crash.log'
    _install_crash_logger(crash_log)

    from PyQt5.QtWidgets import QApplication

    from cube_simulator.main_window import CubeSimulatorMainWindow
    from cube_simulator.robot_worker import teardown_session, teardown_bringup

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(teardown_session)   # 노드 + rclpy 정리
    app.aboutToQuit.connect(teardown_bringup)   # 자동으로 띄운 dsr_bringup2 정리
    win = CubeSimulatorMainWindow(
        dry_run_default=args.dry_run,
        no_vision=args.no_vision,
    )
    win.show()
    rc = app.exec_()
    return rc


if __name__ == '__main__':
    sys.exit(main())
