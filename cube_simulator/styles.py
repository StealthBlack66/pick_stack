"""색상 팔레트 (RGB 0~1 float)."""
from __future__ import annotations


# 배경 / 그리드
BG_COLOR = (0.12, 0.13, 0.16)
GRID_COLOR = (0.45, 0.45, 0.50)
TABLE_COLOR = (0.20, 0.22, 0.26)

# 좌표축
AXIS_X = (1.0, 0.3, 0.3)
AXIS_Y = (0.3, 1.0, 0.3)
AXIS_Z = (0.3, 0.4, 1.0)

# layer 별 색 (0 → 진한 파랑, 위로 갈수록 청록 → 노랑)
LAYER_COLORS = [
    (0.20, 0.45, 0.95),  # 0
    (0.20, 0.75, 0.85),  # 1
    (0.30, 0.85, 0.55),  # 2
    (0.85, 0.85, 0.30),  # 3
    (0.95, 0.55, 0.25),  # 4
    (0.90, 0.30, 0.30),  # 5+
]

INVALID_COLOR = (0.95, 0.18, 0.18)
ACTIVE_COLOR = (1.0, 0.95, 0.30)
SOURCE_COLOR = (0.60, 0.60, 0.62)
SELECTED_COLOR = (1.0, 0.55, 0.10)
GHOST_ADD_COLOR = (0.30, 0.85, 0.55)   # 호버 시 추가될 위치
GHOST_ERASE_COLOR = (0.95, 0.18, 0.18) # 호버 시 제거될 큐브
GRIPPER_COLOR = (0.85, 0.85, 0.90)
GRIPPER_FINGER_COLOR = (0.30, 0.35, 0.45)


def layer_color(layer: int) -> tuple[float, float, float]:
    if layer < 0:
        layer = 0
    if layer >= len(LAYER_COLORS):
        return LAYER_COLORS[-1]
    return LAYER_COLORS[layer]
