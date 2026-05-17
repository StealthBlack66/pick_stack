"""큐브 적층 시뮬레이터 패키지.

3D GUI에서 큐브 배치를 설계하고, 두산 e0509 + RH-P12-RN-A 로 동일하게 재현.
"""

import os
from pathlib import Path

__all__ = [
    'PlacedCube',
    'CubeModel',
    'PickPlaceTask',
    'MODULE_DIR',
    'MODULE_PATHS',
]

from .model import PlacedCube, CubeModel, PickPlaceTask


# 두산 강의 폴더 = 패키지 부모 디렉토리. 환경변수 CUBE_SIM_MODULE_DIR 로 override
# 가능 (테스트 시 모킹 / 다른 경로의 12·15·16·17.py 사용 시).
_DEFAULT_MODULE_DIR = Path(__file__).resolve().parent.parent
MODULE_DIR: Path = Path(os.environ.get('CUBE_SIM_MODULE_DIR') or _DEFAULT_MODULE_DIR)

# RobotWorker / 시뮬레이터 가 동적 import 하는 외부 스크립트 경로.
# key 는 _load_module(key, ...) 가 sys.modules 캐시에 쓰는 이름.
MODULE_PATHS: dict[str, Path] = {
    'p08': MODULE_DIR / '08_카메라_핸드아이_캘리브레이션.py',
    'p09': MODULE_DIR / '09_원샷_캘리브레이션.py',
    'p12': MODULE_DIR / '12_비전_피크앤플레이스.py',
    'p15': MODULE_DIR / '15_바둑판_정렬.py',
    'p16': MODULE_DIR / '16_탑쌓기.py',
    'p17': MODULE_DIR / '17_미술쌓기.py',
    'doosan_config': MODULE_DIR / 'doosan_config.py',
}
