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


def main(argv=None) -> int:
    args = parse_args(argv)

    # 부모 디렉토리를 path 에 추가 (cube_simulator 패키지 import)
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from PyQt5.QtWidgets import QApplication

    from cube_simulator.main_window import CubeSimulatorMainWindow
    from cube_simulator.robot_worker import teardown_session

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(teardown_session)  # closeEvent 안 거치는 경로 안전망
    win = CubeSimulatorMainWindow(
        dry_run_default=args.dry_run,
        no_vision=args.no_vision,
    )
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
