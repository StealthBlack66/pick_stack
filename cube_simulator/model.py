"""큐브 데이터 모델.

View / GL 의존성 없음 — 단위 테스트 가능.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PlacedCube:
    """그리드 셀 단위 큐브 한 개.

    gx, gy: 그리드 좌표 (정수 또는 0.5 단위 — valley 안착).
    layer:  0 = 테이블 위 첫 층.
    yaw_deg: 놓을 때 그리퍼 yaw (기본 90°, 17_미술쌓기.ART_PLACE_YAW 호환).
    """
    gx: float
    gy: float
    layer: int
    yaw_deg: float = 90.0

    def key(self) -> tuple[float, float, int]:
        return (self.gx, self.gy, self.layer)


@dataclass
class PickPlaceTask:
    """로봇이 실행할 한 번의 pick&place 작업.

    좌표는 모두 로봇 base frame 의 mm 단위, 그리퍼 끝 기준.
    move_line_base 가 내부에서 TCP z offset 을 더하므로 절대 직접 더하지 말 것.
    """
    src_xyz: tuple[float, float, float]
    src_yaw: float
    tgt_xyz: tuple[float, float, float]
    tgt_yaw: float
    z_table_top: float
    pre_open_mm: float = 40.0


class CubeModel:
    """배치된 큐브 집합. (gx, gy, layer) 키로 점유 검사."""

    def __init__(self,
                 base_xy: tuple[float, float] = (450.0, 200.0),
                 pitch_mm: float = 27.0,
                 cube_width_mm: float = 25.0):
        self.cubes: list[PlacedCube] = []
        self.base_xy = base_xy
        self.pitch_mm = pitch_mm
        self.cube_width_mm = cube_width_mm

    # --- 조회 ----------------------------------------------------------------
    def find(self, gx: float, gy: float, layer: int) -> Optional[PlacedCube]:
        for c in self.cubes:
            if c.gx == gx and c.gy == gy and c.layer == layer:
                return c
        return None

    def top_layer_at(self, gx: float, gy: float) -> int:
        """해당 (gx, gy) 위치의 다음으로 비어 있는 layer 를 반환.

        0.5 단위 valley 셀은 같은 (gx, gy) 만 검사 (gx-0.5/gx+0.5 의 받침은
        is_supported 에서 별도 검증).
        """
        max_layer = -1
        for c in self.cubes:
            if c.gx == gx and c.gy == gy:
                if c.layer > max_layer:
                    max_layer = c.layer
        return max_layer + 1

    def is_supported(self, gx: float, gy: float, layer: int) -> bool:
        """이 셀에 큐브를 놓을 때 받침이 있는지.

        - layer 0: 항상 True (테이블이 받쳐줌).
        - 그 외: layer-1 에 mm 영역이 겹치는 점유 (|dx|<w AND |dy|<w) 가
          하나라도 있으면 supported.
          직립 적층 / valley 안착 / 대각선(diamond) 안착 모두 통과.
        """
        if layer <= 0:
            return True
        w = self.cube_width_mm
        for c in self.cubes:
            if c.layer != layer - 1:
                continue
            dx = (c.gx - gx) * self.pitch_mm
            dy = (c.gy - gy) * self.pitch_mm
            if abs(dx) < w and abs(dy) < w:
                return True
        return False

    def is_floating(self, c: PlacedCube) -> bool:
        return not self.is_supported(c.gx, c.gy, c.layer)

    def would_collide(self, gx: float, gy: float, layer: int) -> bool:
        """동일 layer 내 다른 큐브와 mm 거리가 cube_width_mm 미만이면 True.

        valley(0.5 단위) 와 정수 셀이 같은 layer 에서 겹치는 케이스를 잡음.
        예: pitch=27mm, cube_width=25mm 일 때
            (0,0) ↔ (1,0)   : 27mm  → OK
            (0,0) ↔ (0.5,0) : 13.5mm → 충돌
        """
        threshold = self.cube_width_mm
        for c in self.cubes:
            if c.layer != layer:
                continue
            if c.gx == gx and c.gy == gy:
                return True  # 동일 셀 — find() 와 동등
            dx = (c.gx - gx) * self.pitch_mm
            dy = (c.gy - gy) * self.pitch_mm
            if (dx * dx + dy * dy) < threshold * threshold:
                return True
        return False

    # --- 편집 ----------------------------------------------------------------
    def add(self, c: PlacedCube) -> None:
        if self.find(c.gx, c.gy, c.layer) is not None:
            return  # 중복 무시
        self.cubes.append(c)
        self._sort()

    def add_auto_layer(self,
                       gx: float,
                       gy: float,
                       yaw_deg: float = 90.0,
                       max_layer: int = 15) -> Optional[PlacedCube]:
        """해당 (gx, gy) 위치에 자동 적층. 같은 layer 내 충돌은 회피하고 위로.

        충돌·중복이 모든 layer 에서 발생하면 None 반환.
        """
        for layer in range(0, max_layer + 1):
            if self.would_collide(gx, gy, layer):
                continue
            c = PlacedCube(gx=gx, gy=gy, layer=layer, yaw_deg=yaw_deg)
            self.add(c)
            return c
        return None

    def remove(self, gx: float, gy: float, layer: int) -> bool:
        for i, c in enumerate(self.cubes):
            if c.gx == gx and c.gy == gy and c.layer == layer:
                del self.cubes[i]
                return True
        return False

    def update_cube(self,
                    gx: float,
                    gy: float,
                    layer: int,
                    new_gx: Optional[float] = None,
                    new_gy: Optional[float] = None,
                    new_layer: Optional[int] = None,
                    new_yaw: Optional[float] = None) -> Optional[PlacedCube]:
        """기존 큐브 (gx, gy, layer) 의 값을 부분 업데이트.

        위치 변경 시 충돌 검사. 충돌이면 원위치 복귀 + None.
        성공 시 갱신된 PlacedCube 반환.
        """
        src = self.find(gx, gy, layer)
        if src is None:
            return None
        tgt_gx = src.gx if new_gx is None else new_gx
        tgt_gy = src.gy if new_gy is None else new_gy
        tgt_layer = src.layer if new_layer is None else new_layer
        tgt_yaw = src.yaw_deg if new_yaw is None else new_yaw
        position_changed = (
            tgt_gx != src.gx or tgt_gy != src.gy or tgt_layer != src.layer
        )
        if position_changed:
            # 잠시 제거 후 충돌 검사 (자기 자신 제외)
            self.cubes.remove(src)
            if self.would_collide(tgt_gx, tgt_gy, tgt_layer):
                self.cubes.append(src)
                self._sort()
                return None
            src.gx = tgt_gx
            src.gy = tgt_gy
            src.layer = tgt_layer
            src.yaw_deg = tgt_yaw
            self.cubes.append(src)
            self._sort()
        else:
            src.yaw_deg = tgt_yaw
        return src

    def remove_top_at(self, gx: float, gy: float) -> Optional[PlacedCube]:
        top = -1
        idx = -1
        for i, c in enumerate(self.cubes):
            if c.gx == gx and c.gy == gy and c.layer > top:
                top = c.layer
                idx = i
        if idx < 0:
            return None
        return self.cubes.pop(idx)

    def clear(self) -> None:
        self.cubes.clear()

    def replace_all(self, cubes: list[PlacedCube]) -> None:
        self.cubes = list(cubes)
        self._sort()

    def _sort(self) -> None:
        self.cubes.sort(key=lambda c: (c.layer, c.gy, c.gx))

    # --- 직렬화 ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            'base_xy': list(self.base_xy),
            'pitch_mm': self.pitch_mm,
            'cube_width_mm': self.cube_width_mm,
            'cubes': [asdict(c) for c in self.cubes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CubeModel':
        m = cls(
            base_xy=tuple(d.get('base_xy', (450.0, 200.0))),
            pitch_mm=float(d.get('pitch_mm', 27.0)),
            cube_width_mm=float(d.get('cube_width_mm', 25.0)),
        )
        m.cubes = [PlacedCube(**c) for c in d.get('cubes', [])]
        m._sort()
        return m
