"""그리드 ↔ 로봇 base 좌표 변환.

좌표 수식은 17_미술쌓기.py 의 기존 로직과 동일하게 유지.
- target_x = base_xy[0] + gx * pitch
- target_y = base_xy[1] + gy * pitch
- target_z = sample_z + layer * cube_width    (z_table_top 기준이 아님!)
"""
from __future__ import annotations

from .model import CubeModel, PlacedCube


def cell_to_mm(model: CubeModel,
               gx: float,
               gy: float,
               layer: int,
               sample_z: float) -> tuple[float, float, float]:
    """그리드 셀 (gx, gy, layer) → 로봇 base mm 좌표.

    sample_z 는 검출된 큐브들의 base_z 중간값 (17_미술쌓기.py:240 과 동일 기준).
    """
    x = model.base_xy[0] + gx * model.pitch_mm
    y = model.base_xy[1] + gy * model.pitch_mm
    z = sample_z + layer * model.cube_width_mm
    return (x, y, z)


def cell_to_view_mm(model: CubeModel,
                    gx: float,
                    gy: float,
                    layer: int,
                    z_table_top: float = 0.0) -> tuple[float, float, float]:
    """뷰 렌더용 좌표 (sample_z 가 없을 때 z_table_top 기준).

    sample_z = z_table_top + cube_width 가 첫 층 중심 z 이므로
    layer k 의 중심 z = z_table_top + cube_width * (k + 1).
    """
    x = model.base_xy[0] + gx * model.pitch_mm
    y = model.base_xy[1] + gy * model.pitch_mm
    z = z_table_top + model.cube_width_mm * (layer + 1)
    return (x, y, z)


def layer_z_plane(model: CubeModel, layer: int, z_table_top: float = 0.0) -> float:
    """raycast 시 사용할 해당 layer 의 큐브 중심 z (뷰 좌표계)."""
    return z_table_top + model.cube_width_mm * (layer + 1)


def mm_to_cell(model: CubeModel, x_mm: float, y_mm: float) -> tuple[float, float]:
    """월드 mm → 가장 가까운 정수 그리드 셀.

    valley 안착(0.5 단위) 은 controller 가 별도 모드로 처리.
    """
    gx = round((x_mm - model.base_xy[0]) / model.pitch_mm)
    gy = round((y_mm - model.base_xy[1]) / model.pitch_mm)
    return (float(gx), float(gy))


def mm_to_cell_snap_half(model: CubeModel, x_mm: float, y_mm: float) -> tuple[float, float]:
    """0.5 단위로 스냅 (valley snap mode)."""
    gx = round(2 * (x_mm - model.base_xy[0]) / model.pitch_mm) / 2.0
    gy = round(2 * (y_mm - model.base_xy[1]) / model.pitch_mm) / 2.0
    return (gx, gy)
