"""가상 픽앤플레이스 시뮬레이션 (로봇 없이 GUI 안에서 보여주기).

각 큐브를 'source 영역' (그리드 왼쪽) 에서 집어 → safe-z 위로 → target 위치로
이동 → 하강 → 릴리즈 시퀀스를 QTimer 보간으로 표시.

좌표 텍스트(step.message) 를 phase 마다 emit 해서 로그/HUD 에 띄움.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from .gl_view import Cube3DView
from .model import CubeModel, PlacedCube


SAFE_TRAVEL_Z_ABOVE_TABLE = 250.0
GRIPPER_FINGER_HEIGHT = 60.0
TICK_MS = 16   # ~60 fps

# phase 별 지속 시간 (ms)
PHASE_DURATIONS = {
    'approach_src': 600,
    'descend_src': 400,
    'grab':         300,
    'lift_src':     400,
    'travel':       800,
    'descend_tgt':  400,
    'release':      300,
    'lift_tgt':     400,
}


@dataclass
class SimStep:
    cube_index: int
    total: int
    phase: str
    message: str = ''
    gripper_xyz: tuple = (0.0, 0.0, 0.0)
    gripper_yaw: float = 0.0
    held_xyz: Optional[tuple] = None
    held_yaw: float = 0.0


class SimAnimator(QObject):
    """단계별 픽앤플레이스 시각 시뮬레이터."""

    step = pyqtSignal(object)       # SimStep — phase 진입 시
    finished = pyqtSignal(bool)     # ok

    def __init__(self,
                 model: CubeModel,
                 view: Cube3DView,
                 parent=None):
        super().__init__(parent)
        self._model = model
        self._view = view
        self._cubes_sorted: list[PlacedCube] = []
        self._cube_index = 0
        self._phase_index = 0
        self._phase_list = list(PHASE_DURATIONS.keys())
        self._phase_t0 = 0.0
        self._phase_dur = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._running = False
        self._cancel = False
        self._cw = model.cube_width_mm
        self._z_table_top = view._z_table_top

        # 현재 phase 시작/끝 그리퍼 pose
        self._pose_start = (0.0, 0.0, 0.0, 0.0)
        self._pose_end = (0.0, 0.0, 0.0, 0.0)
        self._gripper_open_start = 35.0
        self._gripper_open_end = 35.0
        self._holding = False  # 큐브 잡고 있는가
        self._current_src = None  # (x, y, yaw)
        self._current_tgt = None  # (x, y, z_center, yaw)

    # --- public ---------------------------------------------------------------
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._cancel = True

    def start(self) -> None:
        self._cubes_sorted = sorted(
            self._model.cubes,
            key=lambda c: (c.layer, c.gy, c.gx),
        )
        if not self._cubes_sorted:
            self.finished.emit(False)
            return
        # 모든 target 큐브를 일단 hide → release 후 보임
        all_keys = {(c.gx, c.gy, c.layer) for c in self._cubes_sorted}
        self._view.set_hidden_cube_keys(all_keys)
        self._view.set_held_cube(None)
        self._cube_index = 0
        self._phase_index = 0
        self._holding = False
        self._running = True
        self._cancel = False
        self._begin_phase()
        self._timer.start(TICK_MS)

    # --- internal -------------------------------------------------------------
    def _src_position_for(self, idx: int) -> tuple[float, float, float, float]:
        """idx 번째 큐브의 source 위치 (gl_view 의 source grid 와 동일)."""
        base = self._model.base_xy
        src_origin_x = base[0] - 8.0 * self._model.pitch_mm
        sx = src_origin_x + (idx % 5) * self._model.pitch_mm
        sy = base[1] + (idx // 5) * self._model.pitch_mm
        sz = self._z_table_top + self._cw * 0.5  # 큐브 중심 z
        yaw = 0.0
        return sx, sy, sz, yaw

    def _tgt_position(self, c: PlacedCube) -> tuple[float, float, float, float]:
        base = self._model.base_xy
        tx = base[0] + c.gx * self._model.pitch_mm
        ty = base[1] + c.gy * self._model.pitch_mm
        tz_center = self._z_table_top + self._cw * (c.layer + 0.5)
        return tx, ty, tz_center, c.yaw_deg

    def _tip_z_above(self, cube_center_z: float) -> float:
        """큐브를 잡는 그리퍼 끝 (tip) z = 큐브 윗면 = center + cube_w/2."""
        return cube_center_z + self._cw * 0.5

    def _begin_phase(self) -> None:
        if self._cube_index >= len(self._cubes_sorted):
            self._stop_done(True)
            return
        phase = self._phase_list[self._phase_index]
        c = self._cubes_sorted[self._cube_index]
        src = self._src_position_for(self._cube_index)
        tgt = self._tgt_position(c)
        self._current_src = src
        self._current_tgt = tgt
        # source/target 의 tip z (그리퍼 끝)
        src_tip_z = self._tip_z_above(src[2])
        tgt_tip_z = self._tip_z_above(tgt[2])
        safe_z = self._z_table_top + SAFE_TRAVEL_Z_ABOVE_TABLE

        # 현재 그리퍼 pose 시작점 = 직전 phase 끝점 (없으면 첫 cube 의 safe_z 위)
        if self._phase_index == 0 and self._cube_index == 0:
            self._pose_start = (src[0], src[1], safe_z + 80.0, src[3])
            self._gripper_open_start = 35.0
        else:
            self._pose_start = self._pose_end
            self._gripper_open_start = self._gripper_open_end

        if phase == 'approach_src':
            self._pose_end = (src[0], src[1], safe_z, src[3])
            self._gripper_open_end = 40.0
        elif phase == 'descend_src':
            self._pose_end = (src[0], src[1], src_tip_z, src[3])
            self._gripper_open_end = 40.0
        elif phase == 'grab':
            self._pose_end = (src[0], src[1], src_tip_z, src[3])
            self._gripper_open_end = self._cw - 4.0  # 큐브 폭에 맞춰 닫음
            self._holding = True
        elif phase == 'lift_src':
            self._pose_end = (src[0], src[1], safe_z, src[3])
            self._gripper_open_end = self._gripper_open_start
        elif phase == 'travel':
            self._pose_end = (tgt[0], tgt[1], safe_z, tgt[3])
            self._gripper_open_end = self._gripper_open_start
        elif phase == 'descend_tgt':
            self._pose_end = (tgt[0], tgt[1], tgt_tip_z, tgt[3])
            self._gripper_open_end = self._gripper_open_start
        elif phase == 'release':
            self._pose_end = (tgt[0], tgt[1], tgt_tip_z, tgt[3])
            self._gripper_open_end = 40.0
            self._holding = False
        elif phase == 'lift_tgt':
            self._pose_end = (tgt[0], tgt[1], safe_z, tgt[3])
            self._gripper_open_end = 40.0

        self._phase_dur = PHASE_DURATIONS[phase] / 1000.0
        self._phase_t0 = time.monotonic()

        # 좌표 텍스트 emit
        msg = self._format_message(c, phase, src, tgt)
        s = SimStep(
            cube_index=self._cube_index,
            total=len(self._cubes_sorted),
            phase=phase,
            message=msg,
            gripper_xyz=(self._pose_end[0], self._pose_end[1], self._pose_end[2]),
            gripper_yaw=self._pose_end[3],
        )
        self.step.emit(s)
        self._view.set_active_index(-1)

    def _format_message(self,
                        cube: PlacedCube,
                        phase: str,
                        src: tuple,
                        tgt: tuple) -> str:
        sx, sy, sz, syaw = src
        tx, ty, tz, tyaw = tgt
        label = {
            'approach_src': f'[{self._cube_index+1}] APPROACH src ({sx:.0f},{sy:.0f},safe_z) yaw={syaw:+.0f}°',
            'descend_src':  f'[{self._cube_index+1}] DESCEND  src ({sx:.0f},{sy:.0f},{sz:.0f})',
            'grab':         f'[{self._cube_index+1}] GRAB     src ({sx:.0f},{sy:.0f},{sz:.0f})',
            'lift_src':     f'[{self._cube_index+1}] LIFT     ({sx:.0f},{sy:.0f},safe_z)',
            'travel':       f'[{self._cube_index+1}] TRAVEL   → tgt ({tx:.0f},{ty:.0f}) yaw={tyaw:+.0f}°',
            'descend_tgt':  f'[{self._cube_index+1}] DESCEND  tgt ({tx:.0f},{ty:.0f},{tz:.0f}) L{cube.layer}',
            'release':      f'[{self._cube_index+1}] RELEASE  ({tx:.0f},{ty:.0f},{tz:.0f}) grid=({cube.gx:+.1f},{cube.gy:+.1f})',
            'lift_tgt':     f'[{self._cube_index+1}] LIFT_TGT ({tx:.0f},{ty:.0f},safe_z)',
        }
        return label.get(phase, phase)

    def _on_tick(self) -> None:
        if self._cancel:
            self._stop_done(False)
            return
        t = time.monotonic() - self._phase_t0
        u = 1.0 if self._phase_dur <= 0 else min(1.0, t / self._phase_dur)
        # ease-in-out
        eu = 0.5 - 0.5 * math.cos(math.pi * u)
        gx = self._pose_start[0] + (self._pose_end[0] - self._pose_start[0]) * eu
        gy = self._pose_start[1] + (self._pose_end[1] - self._pose_start[1]) * eu
        gz = self._pose_start[2] + (self._pose_end[2] - self._pose_start[2]) * eu
        # yaw 보간 (단순 선형, 360 wrap 안 함)
        gyaw = self._pose_start[3] + (self._pose_end[3] - self._pose_start[3]) * eu
        gopen = (self._gripper_open_start
                 + (self._gripper_open_end - self._gripper_open_start) * eu)
        self._view.set_gripper_pose((gx, gy, gz, gyaw), open_mm=gopen)
        if self._holding:
            # 잡힌 큐브는 그리퍼 끝 바로 아래
            hz = gz - self._cw * 0.5
            self._view.set_held_cube((gx, gy, hz, gyaw))
        else:
            self._view.set_held_cube(None)

        if u >= 1.0:
            self._advance_phase()

    def _advance_phase(self) -> None:
        # phase 끝남 → 다음 phase
        prev_phase = self._phase_list[self._phase_index]
        if prev_phase == 'release':
            # 모델의 target 큐브를 보이게 (hidden 에서 제거)
            c = self._cubes_sorted[self._cube_index]
            keys = set(getattr(self._view, '_hidden_cube_keys', set()))
            keys.discard((c.gx, c.gy, c.layer))
            self._view.set_hidden_cube_keys(keys)
            self._view.set_held_cube(None)

        self._phase_index += 1
        if self._phase_index >= len(self._phase_list):
            self._phase_index = 0
            self._cube_index += 1
            if self._cube_index >= len(self._cubes_sorted):
                self._stop_done(True)
                return
        self._begin_phase()

    def _stop_done(self, ok: bool) -> None:
        self._timer.stop()
        self._running = False
        self._view.set_gripper_pose(None)
        self._view.set_held_cube(None)
        if ok:
            self._view.set_hidden_cube_keys(set())
        self.finished.emit(ok)


# field 가 어디서도 안 쓰여 import 누락 시 silent 하므로 의도 표명
_ = field  # noqa
