"""모양 프리셋 변환 + JSON I/O.

p17.SHAPES 가 (gx, gy, layer) 또는 (gx, gy, layer, yaw) 형태 튜플이고
SHAPE_PITCH_MM 으로 모양별 pitch override 가 있음.
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Iterable

from .model import CubeModel, PlacedCube


_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent  # 02_Doosan_Robot_제어/


def _load_p17():
    """17_미술쌓기.py 를 importlib 로 로드 (한글/숫자 시작 파일명 우회)."""
    p17_path = _PARENT_DIR / '17_미술쌓기.py'
    spec = importlib.util.spec_from_file_location('p17_art', str(p17_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_preset_names() -> list[str]:
    p17 = _load_p17()
    return list(p17.SHAPES.keys())


def cubes_from_preset(name: str) -> tuple[list[PlacedCube], float]:
    """프리셋 이름 → PlacedCube 리스트 + pitch_mm.

    pitch 는 모양별 override 가 있을 수 있어 함께 반환.
    """
    p17 = _load_p17()
    items = p17.SHAPES.get(name)
    if items is None:
        raise KeyError(f'Unknown shape: {name}')
    pitch = p17.SHAPE_PITCH_MM.get(name, p17.ART_PITCH_MM)
    default_yaw = float(p17.ART_PLACE_YAW)
    cubes: list[PlacedCube] = []
    for it in items:
        gx = float(it[0])
        gy = float(it[1])
        layer = int(it[2])
        yaw = float(it[3]) if len(it) > 3 else default_yaw
        cubes.append(PlacedCube(gx=gx, gy=gy, layer=layer, yaw_deg=yaw))
    return cubes, float(pitch)


def apply_preset(model: CubeModel, name: str) -> None:
    cubes, pitch = cubes_from_preset(name)
    model.pitch_mm = pitch
    model.replace_all(cubes)


# JSON ----------------------------------------------------------------------

def save_json(model: CubeModel, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(model.to_dict(), f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> CubeModel:
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return CubeModel.from_dict(d)
