"""가상 픽앤플레이스 시뮬레이션 (로봇 없이 GUI 안에서 보여주기).

각 큐브를 'source 영역' (그리드 왼쪽) 에서 집어 → safe-z 위로 → target 위치로
이동 → 하강 → 릴리즈 시퀀스를 QTimer 보간으로 표시.

좌표 텍스트(step.message) 를 phase 마다 emit 해서 로그/HUD 에 띄움.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from .gl_view import Cube3DView
from .model import CubeModel
from .motion_plan import PlanItem, cube_center_z


SAFE_TRAVEL_Z_ABOVE_TABLE = 250.0
GRIPPER_FINGER_HEIGHT = 60.0
TICK_MS = 16   # ~60 fps
# 그리퍼가 cube 를 잡을 때 finger tip 이 cube top 보다 얼마나 더 깊이 들어가는지
# (cube center z 까지 내려와야 시각적으로 finger 가 cube 의 중간을 잡음)
GRIP_DEPTH_MM = 12.5


def _ease(name: str, u: float) -> float:
    """u∈[0,1] → 보간 계수. name='linear'|'ease_in_out'|'ease_out'|'ease_in'."""
    if name == 'linear':
        return u
    if name == 'ease_out':
        # 초반 빠르고 끝에서 천천히 — release/cube_drop 느낌
        return 1.0 - (1.0 - u) * (1.0 - u)
    if name == 'ease_in':
        return u * u
    # default: ease_in_out (cosine)
    return 0.5 - 0.5 * math.cos(math.pi * u)


@dataclass
class PhaseSpec:
    """애니메이션 phase 한 개의 시간·곡선 사양.

    pose / gripper open 계산 자체는 _begin_phase 의 if-elif 에 남아 있고,
    이 spec 은 phase 길이 + 보간 곡선 + 시작 시 1회 훅만 담당.
    """
    name: str
    duration_ms: int = 400
    ease: str = 'ease_in_out'
    # pre_hook(animator, item) — phase 시작 시 1회 호출. None 이면 미실행.
    pre_hook: Optional[Callable] = None


def _grab_pre_widen(anim, _item) -> None:
    """grab 시작 시 finger 를 살짝 더 벌렸다 닫음 — '윈드업' 모션."""
    anim._gripper_open_start = max(anim._gripper_open_start, anim._cw + 8.0)


PHASE_SPECS: list[PhaseSpec] = [
    PhaseSpec('approach_src', 600, 'ease_in_out'),
    PhaseSpec('descend_src',  400, 'ease_in_out'),
    PhaseSpec('grab',         450, 'ease_out', pre_hook=_grab_pre_widen),
    PhaseSpec('lift_src',     400, 'ease_in_out'),
    PhaseSpec('travel',       800, 'ease_in_out'),
    PhaseSpec('descend_tgt',  400, 'ease_in_out'),
    PhaseSpec('release',      500, 'ease_out'),
    PhaseSpec('lift_tgt',     400, 'ease_in_out'),
]
PHASE_SPEC_BY_NAME: dict[str, PhaseSpec] = {ps.name: ps for ps in PHASE_SPECS}

# 하위 호환 — 기존 코드 참조용 (duration 만)
PHASE_DURATIONS = {ps.name: ps.duration_ms for ps in PHASE_SPECS}


@dataclass
class SimulationConfig:
    """SimAnimator 튜닝 가능 파라미터 묶음.

    None 으로 두면 모듈 default 상수 사용. 실 로봇 / 가상 시뮬 간 차이를
    이 한 곳에서 조정 가능 (e.g. 더 안전한 travel z, 더 깊은 grip).
    """
    safe_travel_z_above_table: float = SAFE_TRAVEL_Z_ABOVE_TABLE
    gripper_finger_height: float = GRIPPER_FINGER_HEIGHT
    grip_depth_mm: float = GRIP_DEPTH_MM
    tick_ms: int = TICK_MS
    phase_specs: list[PhaseSpec] = field(default_factory=lambda: list(PHASE_SPECS))
    # finger 사양 — gl_primitives.draw_gripper 와 맞춰야 함
    finger_thickness_mm: float = 10.0
    finger_width_mm: float = 25.0
    finger_clearance_mm: float = 2.0
    finger_safety_inflation_mm: float = 3.0   # 사전 yaw 보정용
    realtime_finger_inflation_mm: float = 0.0  # 실시간 빨강 표시용
    realtime_min_penetration_mm: float = 2.0
    body_height_mm: float = 40.0
    z_touch_tolerance_mm: float = 0.5


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

    step = pyqtSignal(object)              # SimStep — phase 진입 시
    finished = pyqtSignal(bool)            # ok
    plan_yaw_corrected = pyqtSignal()      # 실시간 충돌 보정 → plan 갱신 됨
    decision_needed = pyqtSignal(int, str) # 충돌 시 사용자 확인 요청 (cube_idx, msg)
    # 시뮬 진행률 (overall_u 는 전체 0.0~1.0) — main_window 의 progress bar 용
    progress = pyqtSignal(int, int, str, float)  # cube_idx, total, phase, overall_u

    def __init__(self,
                 plan: list[PlanItem],
                 view: Cube3DView,
                 model: CubeModel,
                 config: Optional[SimulationConfig] = None,
                 parent=None):
        super().__init__(parent)
        self._plan = list(plan)
        self._model = model
        self._view = view
        self._cfg = config if config is not None else SimulationConfig()
        self._cube_index = 0
        self._phase_index = 0
        self._phase_specs = list(self._cfg.phase_specs)
        self._phase_list = [ps.name for ps in self._phase_specs]
        self._phase_t0 = 0.0
        self._phase_dur = 0.0
        self._phase_ease = 'ease_in_out'
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
        self._holding = False
        # 현재 큐브의 finger 충돌 보정 yaw (큐브 시작 시 1회 계산)
        self._corrected_src_yaw = 0.0
        self._corrected_tgt_yaw = 0.0
        # 실시간 충돌 상태 — 매 tick 갱신
        self._collision_active = False
        # 충돌 컴포넌트 — 'none'|'gripper'|'held'|'both'.
        # __init__ 에서 set 안 하면 PyQt5 QObject 가 getattr-fallback 시
        # 'super-class __init__() was never called' RuntimeError 던짐.
        self._collision_kind = 'none'
        self._collision_msgs: list[str] = []
        # 충돌 시 자동 보정 — cube 당 src/tgt 각각 1회만 시도 (무한 루프 방지)
        self._src_corrected_this_cube = False
        self._tgt_corrected_this_cube = False
        # 다이얼로그 응답 대기 중 보정 정보 (is_src, old_yaw, new_yaw, tag)
        self._pending_correction: Optional[tuple] = None
        # 다이얼로그로 일시정지된 phase 의 경과 시간 (s). _consume_pause_offset()
        # 으로 1회 소비하면 0.0 으로 reset — 두 번 적용되지 않게.
        self._phase_elapsed_at_pause: float = 0.0

    # --- public ---------------------------------------------------------------
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._cancel = True

    def start(self) -> None:
        if not self._plan:
            self.finished.emit(False)
            return
        # 모든 target 큐브 hide → release 후 보임
        all_keys = {
            (p.cube_gx, p.cube_gy, p.cube_layer) for p in self._plan
        }
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
    def _tip_z_above(self, cube_center_z: float) -> float:
        """cube 를 잡는 finger tip 의 z.
        cube top (= center + cw/2) 에서 grip_depth_mm 만큼 더 내려간 위치.
        grip_depth=cw/2 면 tip 이 정확히 cube center z 와 같음.
        """
        return cube_center_z + self._cw * 0.5 - self._cfg.grip_depth_mm

    def _begin_phase(self) -> None:
        if self._cube_index >= len(self._plan):
            self._stop_done(True)
            return
        phase = self._phase_list[self._phase_index]
        spec = self._phase_specs[self._phase_index]
        item = self._plan[self._cube_index]

        # 새 cube 시작 시 보정 카운터 reset
        if self._phase_index == 0:
            self._src_corrected_this_cube = False
            self._tgt_corrected_this_cube = False

        # plan 의 yaw 그대로 사용 (실시간 충돌 시 _check_realtime_collision 가
        # 즉석에서 plan yaw 를 +90° 갱신 → 다음 phase 부터 새 yaw 적용)
        src = (item.src_x, item.src_y, item.src_z, item.src_yaw)
        tgt = (item.tgt_x, item.tgt_y, item.tgt_z, item.tgt_yaw)
        # source/target 의 tip z (그리퍼 끝)
        src_tip_z = self._tip_z_above(src[2])
        tgt_tip_z = self._tip_z_above(tgt[2])
        safe_z = self._z_table_top + self._cfg.safe_travel_z_above_table

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
            # release 시작과 동시에 자기 cube 를 placed 로 즉시 visible
            # (held cube 그림은 사라지지만 같은 자리에 placed cube 가 나타남
            #  → 사라졌다 나타나는 끊김 없음)
            keys = set(getattr(self._view, '_hidden_cube_keys', set()))
            keys.discard((item.cube_gx, item.cube_gy, item.cube_layer))
            self._view.set_hidden_cube_keys(keys)
            self._view.set_held_cube(None)
        elif phase == 'lift_tgt':
            self._pose_end = (tgt[0], tgt[1], safe_z, tgt[3])
            self._gripper_open_end = 40.0

        # phase 시작 시 1회 훅 — grab 의 finger pre-widen 등.
        # _gripper_open_start 가 hook 안에서 변경될 수 있으므로 후 dur/ease 계산.
        if spec.pre_hook is not None:
            spec.pre_hook(self, item)

        self._phase_dur = spec.duration_ms / 1000.0
        self._phase_ease = spec.ease
        self._phase_t0 = time.monotonic()

        msg = self._format_message(item, phase, src, tgt)
        s = SimStep(
            cube_index=self._cube_index,
            total=len(self._plan),
            phase=phase,
            message=msg,
            gripper_xyz=(self._pose_end[0], self._pose_end[1], self._pose_end[2]),
            gripper_yaw=self._pose_end[3],
        )
        self.step.emit(s)
        self._view.set_active_index(-1)

    def _format_message(self,
                        item: PlanItem,
                        phase: str,
                        src: tuple,
                        tgt: tuple) -> str:
        sx, sy, sz, syaw = src
        tx, ty, tz, tyaw = tgt
        i = self._cube_index + 1
        label = {
            'approach_src': f'[{i}] APPROACH src ({sx:.0f},{sy:.0f},safe_z) yaw={syaw:+.0f}°',
            'descend_src':  f'[{i}] DESCEND  src ({sx:.0f},{sy:.0f},{sz:.0f})',
            'grab':         f'[{i}] GRAB     src ({sx:.0f},{sy:.0f},{sz:.0f})',
            'lift_src':     f'[{i}] LIFT     ({sx:.0f},{sy:.0f},safe_z)',
            'travel':       f'[{i}] TRAVEL   → tgt ({tx:.0f},{ty:.0f}) yaw={tyaw:+.0f}°',
            'descend_tgt':  f'[{i}] DESCEND  tgt ({tx:.0f},{ty:.0f},{tz:.0f}) L{item.cube_layer}',
            'release':      f'[{i}] RELEASE  ({tx:.0f},{ty:.0f},{tz:.0f}) grid=({item.cube_gx:+.1f},{item.cube_gy:+.1f})',
            'lift_tgt':     f'[{i}] LIFT_TGT ({tx:.0f},{ty:.0f},safe_z)',
        }
        return label.get(phase, phase)

    def _on_tick(self) -> None:
        if self._cancel:
            self._stop_done(False)
            return
        t = time.monotonic() - self._phase_t0
        u = 1.0 if self._phase_dur <= 0 else min(1.0, t / self._phase_dur)
        eu = _ease(self._phase_ease, u)
        # progress bar / HUD 용 — 매 tick (62 Hz). QProgressBar 갱신 비용 충분히 가벼움.
        total = max(1, len(self._plan))
        n_phases = max(1, len(self._phase_list))
        overall_u = (self._cube_index * n_phases
                     + self._phase_index + u) / (total * n_phases)
        overall_u = max(0.0, min(1.0, overall_u))
        self.progress.emit(
            self._cube_index, total,
            self._phase_list[self._phase_index], overall_u,
        )
        gx = self._pose_start[0] + (self._pose_end[0] - self._pose_start[0]) * eu
        gy = self._pose_start[1] + (self._pose_end[1] - self._pose_start[1]) * eu
        gz = self._pose_start[2] + (self._pose_end[2] - self._pose_start[2]) * eu
        # yaw 보간 (단순 선형, 360 wrap 안 함)
        gyaw = self._pose_start[3] + (self._pose_end[3] - self._pose_start[3]) * eu
        gopen = (self._gripper_open_start
                 + (self._gripper_open_end - self._gripper_open_start) * eu)
        self._view.set_gripper_pose((gx, gy, gz, gyaw), open_mm=gopen)
        held_xyz = None
        if self._holding:
            hz = gz + self._cw * 0.5 - GRIP_DEPTH_MM
            held_xyz = (gx, gy, hz)
            self._view.set_held_cube((gx, gy, hz, gyaw))
        else:
            self._view.set_held_cube(None)

        # 실시간 충돌 검사 — 매 tick
        self._check_realtime_collision(gx, gy, gz, gyaw, gopen, held_xyz)

        if u >= 1.0:
            self._advance_phase()

    def _advance_phase(self) -> None:
        prev_phase = self._phase_list[self._phase_index]
        # release/lift_tgt 진입 시점에 이미 hidden 해제 + held_cube=None 처리됨.
        # lift_tgt 끝 시점엔 충돌 상태 reset (cube 가 그리퍼에서 완전히 빠져나감).
        if prev_phase == 'lift_tgt':
            self._collision_active = False
            try:
                self._view.set_collision_state(False)
            except AttributeError:
                pass

        self._phase_index += 1
        if self._phase_index >= len(self._phase_list):
            self._phase_index = 0
            self._cube_index += 1
            if self._cube_index >= len(self._plan):
                self._stop_done(True)
                return
        self._begin_phase()

    # --- finger 충돌 검사 + yaw 보정 -----------------------------------------
    # SAT (Separating Axis Theorem) 기반 정확한 OBB ↔ AABB 검사.
    # finger 사양은 SimulationConfig 에서 주입 — 두께(yaw 축) × 폭(yaw 직각).
    # 인접 cube 는 axis-aligned (회전 안 함) 가정 — AABB.
    # 아래 클래스 상수는 apply_yaw_corrections 의 helper 객체 (__new__ 로
    # 만들어 __init__ 우회) 가 self._cfg 없이도 동작하게 하기 위한 fallback.

    FINGER_THICKNESS_MM = 10.0
    FINGER_WIDTH_MM = 25.0
    FINGER_CLEARANCE_MM = 2.0
    FINGER_SAFETY_INFLATION_MM = 3.0
    REALTIME_FINGER_INFLATION_MM = 0.0
    REALTIME_MIN_PENETRATION_MM = 2.0
    BODY_HEIGHT_MM = 40.0
    Z_TOUCH_TOLERANCE_MM = 0.5

    def _cf(self, attr: str, fallback) -> float:
        """SimulationConfig 값 lookup with fallback.

        __dict__.get 사용 — PyQt5 QObject 의 __new__ 우회 인스턴스에서
        `getattr(self, attr, default)` 가 RuntimeError 던지는 걸 회피.
        """
        cfg = self.__dict__.get('_cfg')
        if cfg is None:
            return fallback
        return getattr(cfg, attr, fallback)

    def _finger_collides(self,
                         center_xy: tuple[float, float],
                         yaw_deg: float,
                         others_xy: list[tuple[float, float]]) -> bool:
        """yaw 회전된 finger 2 개 (OBB) 와 인접 큐브들 (AABB) 충돌 검사.

        finger 의 두께 방향 = yaw 축. 폭 방향 = yaw 직각.
        finger 중심 = 큐브 중심 ± yaw 축 방향 × (cw/2 + clearance + thickness/2).
        """
        cw = self._cw
        ft = self._cf('finger_thickness_mm', self.FINGER_THICKNESS_MM)
        fw = self._cf('finger_width_mm', self.FINGER_WIDTH_MM)
        clearance = self._cf('finger_clearance_mm', self.FINGER_CLEARANCE_MM)
        finger_offset = cw / 2.0 + clearance + ft / 2.0
        yaw = math.radians(yaw_deg)
        ax, ay = math.cos(yaw), math.sin(yaw)   # yaw 축 (두께 축)
        bx, by = -ay, ax                         # yaw 직각 (폭 축)
        cx, cy = center_xy
        fingers = (
            (cx + ax * finger_offset, cy + ay * finger_offset),
            (cx - ax * finger_offset, cy - ay * finger_offset),
        )
        cube_half = cw / 2.0
        # SAT 에 적용할 inflated half — finger 외곽에 안전 여유 추가
        infl = self._cf('finger_safety_inflation_mm',
                        self.FINGER_SAFETY_INFLATION_MM)
        half_u = ft / 2.0 + infl
        half_v = fw / 2.0 + infl
        # 빠른 거리 컷오프
        max_reach = finger_offset + half_u + cube_half + 1.0
        for ox, oy in others_xy:
            if abs(ox - cx) < 0.5 and abs(oy - cy) < 0.5:
                continue  # 자기 자신
            if math.hypot(ox - cx, oy - cy) > max_reach + cube_half:
                continue
            for fx, fy in fingers:
                if self._obb_aabb_overlap(
                    (fx, fy), (ax, ay), (bx, by),
                    half_u, half_v,
                    (ox, oy), cube_half, cube_half,
                ):
                    return True
        return False

    @staticmethod
    def _obb_aabb_overlap(obb_c: tuple[float, float],
                          axis_u: tuple[float, float],
                          axis_v: tuple[float, float],
                          half_u: float,
                          half_v: float,
                          aabb_c: tuple[float, float],
                          aabb_hx: float,
                          aabb_hy: float,
                          min_penetration: float = 0.0) -> bool:
        """2D SAT — OBB 와 AABB 가 침범해 있으면 True.

        4 분리축 (axis_u, axis_v, (1,0), (0,1)) 의 최소 침범 깊이가
        min_penetration 보다 클 때만 충돌로 인정.
        min_penetration=0 → 모든 침범 인정 (기존 동작).
        min_penetration=2 → 2mm 이하 가벼운 침범은 분리로 봄.
        """
        dx = aabb_c[0] - obb_c[0]
        dy = aabb_c[1] - obb_c[1]
        min_pen = float('inf')
        for ax_, ay_ in (axis_u, axis_v, (1.0, 0.0), (0.0, 1.0)):
            obb_proj = (abs(half_u * (axis_u[0] * ax_ + axis_u[1] * ay_))
                        + abs(half_v * (axis_v[0] * ax_ + axis_v[1] * ay_)))
            aabb_proj = abs(aabb_hx * ax_) + abs(aabb_hy * ay_)
            center_proj = abs(dx * ax_ + dy * ay_)
            if center_proj >= obb_proj + aabb_proj:
                return False  # 완전 분리축 발견
            pen = obb_proj + aabb_proj - center_proj
            if pen < min_pen:
                min_pen = pen
        return min_pen > min_penetration

    @staticmethod
    def _wrap_yaw(yaw: float) -> float:
        """(-180, 180] 으로 wrap."""
        wrapped = ((yaw + 180.0) % 360.0) - 180.0
        return wrapped

    def _compute_corrected_yaws(self, item) -> tuple[float, float, str]:
        """현재 cube_index 시점의 pick/place yaw 를 인접 큐브 검사로 ±90° 보정.

        검사 대상:
            pick  → 아직 안 픽한 source 큐브들 (모두 같은 평면 가정)
            place → 이미 놓은 placed 큐브들 중 finger 와 z 영역이 겹치는 것
                    즉 같은 layer (=자기 layer) 큐브들. layer 차이 큰 cube 는
                    z 차이 만큼 finger 가 안 닿으므로 제외.
        """
        notes: list[str] = []

        # 1) pick yaw — 아직 안 픽한 source 큐브들 (전부 검사)
        others_pick = [
            (p.src_x, p.src_y)
            for i, p in enumerate(self._plan) if i > self._cube_index
        ]
        pick_yaw = self._try_correct_yaw(
            (item.src_x, item.src_y), item.src_yaw, others_pick,
            label=f'[{self._cube_index+1}] pick',
            adjacent_label='옆 source 큐브',
            notes=notes,
        )

        # 2) place yaw — 이미 놓은 placed 큐브 중 같은 layer 큐브만.
        # 다른 layer cube 는 z 가 달라 finger 가 안 닿음 (false positive 차단).
        my_layer = item.cube_layer
        others_place = [
            (p.tgt_x, p.tgt_y)
            for j, p in enumerate(self._plan)
            if j < self._cube_index and p.cube_layer == my_layer
        ]
        place_yaw = self._try_correct_yaw(
            (item.tgt_x, item.tgt_y), item.tgt_yaw, others_place,
            label=f'[{self._cube_index+1}] place L{my_layer}',
            adjacent_label=f'같은 layer{my_layer} placed 큐브',
            notes=notes,
        )

        return pick_yaw, place_yaw, ' / '.join(notes)

    def _try_correct_yaw(self,
                         center_xy: tuple[float, float],
                         yaw: float,
                         others_xy: list[tuple[float, float]],
                         label: str,
                         adjacent_label: str,
                         notes: list) -> float:
        """yaw 가 충돌이면 +90°, 그것도 충돌이면 -90°, 그래도 충돌이면 원래 값."""
        if not self._finger_collides(center_xy, yaw, others_xy):
            return yaw
        for delta in (90.0, -90.0):
            new_yaw = self._wrap_yaw(yaw + delta)
            if not self._finger_collides(center_xy, new_yaw, others_xy):
                notes.append(
                    f'{label} yaw 보정 {yaw:+.0f}°→{new_yaw:+.0f}° '
                    f'({adjacent_label} 회피)'
                )
                return new_yaw
        notes.append(
            f'{label} yaw {yaw:+.0f}° 충돌 — ±90° 모두 회피 불가, 그대로 진행'
        )
        return yaw

    # --- 실시간 충돌 검사 (매 tick) ------------------------------------------
    # z 분리 허용 오차 — held cube 바닥과 받침 cube 윗면이 정확히 touch 일 때 분리로 봄
    # (적층 시 자연스러운 touch 를 false positive 로 잡지 않게)

    def _check_realtime_collision(self,
                                  gx: float, gy: float, gz: float,
                                  gyaw: float, gopen: float,
                                  held_xyz: Optional[tuple[float, float, float]]):
        """그리퍼 (finger×2 + body) + held cube 가 환경 cube 와 침범하는지.

        충돌 시 self._collision_active=True, view 에 표시. 매 frame 발화.
        """
        cw = self._cw
        ft = self._cfg.finger_thickness_mm
        fw = self._cfg.finger_width_mm
        fh = self._cfg.gripper_finger_height
        bh = self._cfg.body_height_mm

        yaw = math.radians(gyaw)
        ax, ay = math.cos(yaw), math.sin(yaw)
        bx, by = -ay, ax

        # finger 두 개 — OBB (yaw 회전), z 영역 [gz, gz+fh]
        # 실시간 검사는 정확한 finger 크기 (rt 인플레이션 0 디폴트)
        rt_infl = self._cfg.realtime_finger_inflation_mm
        finger_offset = cw / 2.0 + self._cfg.finger_clearance_mm + ft / 2.0
        finger_centers = (
            (gx + ax * finger_offset, gy + ay * finger_offset),
            (gx - ax * finger_offset, gy - ay * finger_offset),
        )
        finger_z = (gz, gz + fh)
        finger_half_u = ft / 2.0 + rt_infl
        finger_half_v = fw / 2.0 + rt_infl

        # body — OBB (yaw 회전), z 영역 [gz+fh, gz+fh+bh]
        body_half_u = (gopen + ft * 2.0 + 12.0) / 2.0  # gl_primitives.draw_gripper 와 동일
        body_half_v = (fw + 10.0) / 2.0
        body_z = (gz + fh, gz + fh + bh)

        # held cube — AABB (yaw 무시: cube 정사각형이라 SAT 결과 같지만 일관 위해 OBB)
        held_z = None
        if held_xyz is not None:
            hx, hy, hz = held_xyz
            held_center_xy = (hx, hy)
            held_z = (hz - cw / 2.0, hz + cw / 2.0)
            held_half_u = cw / 2.0
            held_half_v = cw / 2.0

        env_cubes = self._collect_env_cubes()
        msgs: list[str] = []

        eps = self._cfg.z_touch_tolerance_mm
        min_pen = self._cfg.realtime_min_penetration_mm
        for cube_xy, cube_z in env_cubes:
            cube_half = cw / 2.0
            # z 분리 빠른 컷오프 — touch (gap ≤ eps) 는 분리로 봄 (false positive 방지)
            for label, comp in (
                ('finger0', (finger_centers[0], (ax, ay), (bx, by),
                              finger_half_u, finger_half_v, finger_z)),
                ('finger1', (finger_centers[1], (ax, ay), (bx, by),
                              finger_half_u, finger_half_v, finger_z)),
                ('body',    ((gx, gy), (ax, ay), (bx, by),
                              body_half_u, body_half_v, body_z)),
            ):
                c, u, v, hu, hv, zr = comp
                if zr[1] - cube_z[0] <= eps or cube_z[1] - zr[0] <= eps:
                    continue
                if self._obb_aabb_overlap(c, u, v, hu, hv,
                                          cube_xy, cube_half, cube_half,
                                          min_penetration=min_pen):
                    msgs.append(f'{label} ↔ cube({cube_xy[0]:.0f},{cube_xy[1]:.0f})')

            if held_z is not None:
                if (held_z[1] - cube_z[0] <= eps
                        or cube_z[1] - held_z[0] <= eps):
                    continue
                if (abs(cube_xy[0] - held_center_xy[0]) < 0.5 and
                        abs(cube_xy[1] - held_center_xy[1]) < 0.5):
                    continue
                if self._obb_aabb_overlap(
                        held_center_xy, (ax, ay), (bx, by),
                        held_half_u, held_half_v,
                        cube_xy, cube_half, cube_half,
                        min_penetration=min_pen):
                    msgs.append(
                        f'held ↔ cube({cube_xy[0]:.0f},{cube_xy[1]:.0f})'
                    )

        # 어떤 컴포넌트가 충돌인지 — view 의 시각화 분기에 사용
        has_held = any(m.startswith('held') for m in msgs)
        has_gripper = any(
            m.startswith(('finger', 'body')) for m in msgs
        )
        if has_held and has_gripper:
            kind = 'both'
        elif has_held:
            kind = 'held'
        elif has_gripper:
            kind = 'gripper'
        else:
            kind = 'none'

        was = self._collision_active
        now = bool(msgs)
        prev_kind = self._collision_kind
        view_kind = kind if now else 'none'
        # view 는 kind 가 변할 때마다 갱신 (예: gripper → both 전환 시 held cube 도 빨강)
        if now != was or view_kind != prev_kind:
            self._collision_kind = view_kind
            try:
                self._view.set_collision_state(now, view_kind)
            except TypeError:
                # 구버전 view (kind 인자 모름) fallback
                self._view.set_collision_state(now)
            except AttributeError:
                pass
        # step.emit 은 충돌 시작/종료 transition 시에만 (로그 폭주 방지)
        if now != was:
            self._collision_active = now
            if now:
                self.step.emit(SimStep(
                    cube_index=self._cube_index,
                    total=len(self._plan),
                    phase='collision',
                    message='⚠ 충돌: ' + ', '.join(sorted(set(msgs))[:4]),
                ))
                # 자동 yaw +90° 보정 시도 (cube 당 src/tgt 각 1회)
                self._auto_correct_on_collision()
            else:
                self.step.emit(SimStep(
                    cube_index=self._cube_index,
                    total=len(self._plan),
                    phase='collision_clear',
                    message='충돌 해소',
                ))
        self._collision_msgs = msgs

    def _auto_correct_on_collision(self) -> None:
        """실시간 충돌 감지 시 사용자에게 yaw +90° 보정 여부 확인 요청.

        - 시뮬 타이머 일시정지 + decision_needed emit (main_window 가 다이얼로그)
        - 사용자 응답 후 apply_decision(accept) 호출 → 시뮬 재개
        - cube 당 src/tgt 각 1회만 묻기 (무한 루프 방지)
        """
        phase = self._phase_list[self._phase_index]
        src_phases = ('approach_src', 'descend_src', 'grab', 'lift_src')
        is_src = phase in src_phases
        item = self._plan[self._cube_index]

        if is_src:
            if self._src_corrected_this_cube:
                return
            old = item.src_yaw
            tag = 'src (집기)'
        else:
            if self._tgt_corrected_this_cube:
                return
            old = item.tgt_yaw
            tag = 'tgt (놓기)'
        new = self._wrap_yaw(old + 90.0)

        # 응답 대기 중 — 시뮬 일시정지 (보간 진행도 보존)
        self._timer.stop()
        self._phase_elapsed_at_pause = time.monotonic() - self._phase_t0
        self._pending_correction = (is_src, old, new, tag)
        msg = (
            f'[#{self._cube_index + 1}] {tag} yaw 충돌 감지\n\n'
            f'  현재 yaw: {old:+.0f}°\n'
            f'  보정 yaw: {new:+.0f}° (+90° 회전)\n\n'
            f'그리퍼를 90° 회전해서 충돌을 회피할까요?\n'
            f'(아니오 = 현재 yaw 유지하고 그대로 진행)'
        )
        # 다이얼로그는 main event loop 의 다음 iteration 에 emit —
        # _on_tick (timer slot) 콜스택 안에서 modal exec_() 가 떠 nested
        # event loop 이 생기는 reentrant 위험 차단.
        idx = self._cube_index
        QTimer.singleShot(0, lambda: self.decision_needed.emit(idx, msg))

    def _consume_pause_offset(self) -> float:
        """일시정지로 흘렀던 phase 경과 시간을 1회 꺼내고 reset.

        보간 진행도 보존을 위해 apply_decision 에서 호출. 두 번 적용되면
        같은 phase 가 중복 보정되므로 명시적으로 0.0 으로 reset.
        """
        elapsed = self._phase_elapsed_at_pause
        self._phase_elapsed_at_pause = 0.0
        return float(elapsed)

    def apply_decision(self, accept: bool) -> None:
        """controller 가 사용자 다이얼로그 응답 후 호출.

        accept=True  → plan yaw +90° 적용
        accept=False → 그대로
        어느 쪽이든 같은 cube 에는 다시 묻지 않고 시뮬 재개.
        """
        # 다이얼로그 modal 동안 Esc 등으로 잘못 set 됐을 수 있는 _cancel 을
        # 사용자가 명시 응답했으니 reset (사용자 의도는 "계속 진행")
        if getattr(self, '_cancel', False):
            self.step.emit(SimStep(
                cube_index=self._cube_index, total=len(self._plan),
                phase='cancel_reset',
                message='다이얼로그 응답 — 비상정지 플래그 reset',
            ))
            self._cancel = False

        if self._pending_correction is None:
            # 잘못된 호출 — silent 하게 무시 (단 시뮬은 살아있도록 timer 재시작)
            if self._running and not self._timer.isActive():
                self._phase_t0 = time.monotonic()
                QTimer.singleShot(0, lambda: self._timer.start(TICK_MS))
            return

        is_src, old, new, tag = self._pending_correction
        self._pending_correction = None  # 먼저 비워둠 (재진입 안전)

        try:
            item = self._plan[self._cube_index]
        except IndexError:
            # cube_index 범위 벗어남 — 시뮬 이미 끝났을 수도
            return

        if accept:
            if is_src:
                item.src_yaw = new
                self._src_corrected_this_cube = True
            else:
                item.tgt_yaw = new
                self._tgt_corrected_this_cube = True
                # tgt_yaw 변경은 model.c.yaw_deg 로 즉시 반영 — 시뮬 placed cube
                # 시각과 실 로봇 placement 가 같은 yaw 를 쓰도록.
                c = self._model.find(
                    item.cube_gx, item.cube_gy, item.cube_layer
                )
                if c is not None:
                    c.yaw_deg = float(new)
            # pose_end yaw 즉시 갱신
            px, py, pz, _ = self._pose_end
            self._pose_end = (px, py, pz, new)
            self.step.emit(SimStep(
                cube_index=self._cube_index, total=len(self._plan),
                phase='yaw_corrected',
                message=f'⟲ {tag} yaw {old:+.0f}° → {new:+.0f}° (사용자 승인)',
            ))
            self.plan_yaw_corrected.emit()
        else:
            if is_src:
                self._src_corrected_this_cube = True
            else:
                self._tgt_corrected_this_cube = True
            self.step.emit(SimStep(
                cube_index=self._cube_index, total=len(self._plan),
                phase='collision_skip',
                message=f'사용자 선택: {tag} yaw {old:+.0f}° 유지 — 그대로 진행',
            ))

        # 시뮬 재개 — 다이얼로그 동안 흐른 시간 제외하고 같은 진행도부터
        if not self._running:
            self.step.emit(SimStep(
                cube_index=self._cube_index, total=len(self._plan),
                phase='resume_blocked',
                message='재개 차단: 시뮬 not running (이미 종료됨?)',
            ))
            return
        if self._cancel:
            self.step.emit(SimStep(
                cube_index=self._cube_index, total=len(self._plan),
                phase='resume_blocked',
                message='재개 차단: 비상정지 요청 상태',
            ))
            return
        elapsed = self._consume_pause_offset()
        dur = max(0.001, float(self._phase_dur or 0.001))
        elapsed = max(0.0, min(elapsed, dur))
        self._phase_t0 = time.monotonic() - elapsed

        # timer 재시작도 deferred — apply_decision 콜스택 (다이얼로그 응답 직후)
        # 안에서 timer 가 곧장 _on_tick 호출하는 nested 위험 차단.
        def _resume_timer():
            if (self._running and not self._cancel
                    and not self._timer.isActive()):
                self._timer.start(TICK_MS)
        QTimer.singleShot(0, _resume_timer)

        self.step.emit(SimStep(
            cube_index=self._cube_index, total=len(self._plan),
            phase='resumed',
            message=f'시뮬 재개 예약 (elapsed={elapsed:.2f}/{dur:.2f}s)',
        ))

    def _collect_env_cubes(self) -> list[tuple[tuple[float, float],
                                                tuple[float, float]]]:
        """현재 시뮬 상태 환경 cube 위치 + z 영역 (cube AABB).

        검사 대상 = model 의 placed cube 중 시각적으로 보이는 (hidden 아닌) 것.
        plan 의 source cube 는 dry-run 에서 보이지 않거나 정렬 후 충분히 떨어져
        있다고 가정 — 검사 제외 (사용자 시각엔 충돌 없는데 false positive 방지).
        """
        cubes = []
        cw = self._cw
        hidden = getattr(self._view, '_hidden_cube_keys', set())
        for c in self._model.cubes:
            key = (c.gx, c.gy, c.layer)
            if key in hidden:
                continue
            cx = self._model.base_xy[0] + c.gx * self._model.pitch_mm
            cy = self._model.base_xy[1] + c.gy * self._model.pitch_mm
            cz = cube_center_z(c.layer, cw, self._z_table_top)
            cubes.append(((cx, cy), (cz - cw / 2.0, cz + cw / 2.0)))
        return cubes

    def _stop_done(self, ok: bool) -> None:
        self._timer.stop()
        self._running = False
        self._view.set_gripper_pose(None)
        self._view.set_held_cube(None)
        try:
            self._view.set_collision_state(False)
        except AttributeError:
            pass
        self._collision_active = False
        if ok:
            self._view.set_hidden_cube_keys(set())
        self.finished.emit(ok)
