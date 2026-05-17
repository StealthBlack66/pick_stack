"""모션 계획 (시뮬 / 실행 공용 데이터).

큐브 배치 모델 → 순서대로 pick&place 할 PlanItem 리스트.
사용자가 표에서 좌표/각도를 편집한 결과도 같은 PlanItem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import CubeModel


# 적층 시 받침 cube top 과 자기 cube bottom 사이에 두는 공중 gap (mm).
# z touch 경계에서 SAT 가 미세하게 침범으로 잡는 false positive 를 막기 위해
# layer 마다 누적 적용 — L0: +0, L1: +0.1, L2: +0.2, ...
Z_AIR_GAP_MM = 0.1


def cube_center_z(layer: int, cube_width_mm: float = 25.0,
                  z_table_top: float = 0.0) -> float:
    """layer 별 cube 중심 z (air gap 누적)."""
    return z_table_top + cube_width_mm * (layer + 0.5) + Z_AIR_GAP_MM * layer


@dataclass
class PlanItem:
    """한 큐브에 대한 픽앤플레이스 작업.

    좌표는 모두 mm (로봇 base frame). src 는 source grid (왼쪽 영역).
    """
    idx: int
    # 큐브의 그리드 좌표 (참고용, 모델 동기화에 사용)
    cube_gx: float
    cube_gy: float
    cube_layer: int
    # source (집기) — mm
    src_x: float
    src_y: float
    src_z: float
    src_yaw: float
    # target (놓기) — mm
    tgt_x: float
    tgt_y: float
    tgt_z: float
    tgt_yaw: float

    def src_xyz(self) -> tuple[float, float, float]:
        return (self.src_x, self.src_y, self.src_z)

    def tgt_xyz(self) -> tuple[float, float, float]:
        return (self.tgt_x, self.tgt_y, self.tgt_z)


def build_plan(model: CubeModel,
               z_table_top: float = 0.0,
               source_positions: Optional[list[tuple[float, float, float]]] = None,
               ) -> list[PlanItem]:
    """모델 → 정렬된 PlanItem 리스트.

    정렬 순서: (layer, gy, gx) — 17_미술쌓기.build_art 와 동일.

    source_positions: (x, y, z) 리스트 (보통 15번 정렬 후 캡처된 dets 좌표).
        주어지면 cube 순서대로 매핑해 src 좌표로 사용 — "이미 정렬된 5×5 큐브
        에서 픽". 부족하면 남은 큐브는 fake 그리드로 폴백.
        None 이면 전부 fake 그리드 (base 왼쪽 5×N) 사용.
        z 는 시각화 프레임 통일을 위해 z_table_top + cw/2 로 강제 (15번이
        캡처한 robot-base sz 는 view 와 좌표계 다름).
    """
    cubes = sorted(model.cubes, key=lambda c: (c.layer, c.gy, c.gx))
    base = model.base_xy
    cw = model.cube_width_mm
    pitch = model.pitch_mm
    src_origin_x = base[0] - 8.0 * pitch
    layer0_z = z_table_top + cw * 0.5
    sp = list(source_positions) if source_positions else []
    plan: list[PlanItem] = []
    for i, c in enumerate(cubes):
        if i < len(sp):
            sx, sy, _ = sp[i]
            sz = layer0_z
        else:
            sx = src_origin_x + (i % 5) * pitch
            sy = base[1] + (i // 5) * pitch
            sz = layer0_z
        tx = base[0] + c.gx * pitch
        ty = base[1] + c.gy * pitch
        tz = cube_center_z(c.layer, cw, z_table_top)
        plan.append(PlanItem(
            idx=i,
            cube_gx=float(c.gx),
            cube_gy=float(c.gy),
            cube_layer=int(c.layer),
            src_x=sx, src_y=sy, src_z=sz, src_yaw=90.0,
            tgt_x=tx, tgt_y=ty, tgt_z=tz, tgt_yaw=float(c.yaw_deg),
        ))
    return plan


def update_plan_field(plan: list[PlanItem],
                      row: int,
                      field_name: str,
                      value: float) -> Optional[PlanItem]:
    """plan[row] 의 field 1 개를 value 로 갱신. 갱신된 항목 반환."""
    if row < 0 or row >= len(plan):
        return None
    if not hasattr(plan[row], field_name):
        return None
    setattr(plan[row], field_name, float(value))
    return plan[row]
