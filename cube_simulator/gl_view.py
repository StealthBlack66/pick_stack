"""3D 시뮬레이터 뷰포트.

QOpenGLWidget 서브클래스 — PyOpenGL immediate mode 로 렌더.

마우스:
- 좌클릭     : 도구(add/select/erase) 에 따라 처리. press→release 거리 4px 미만이면 클릭.
- 좌 드래그  : (Add/Select 모드) 큐브를 잡고 옮기기.
- 우드래그   : 카메라 회전. 거리 4px 미만이면 우클릭(컨텍스트) 으로 인정.
- 휠        : 줌.
- 중드래그   : 팬.
호버: 버튼 미클릭 시 마우스 따라 고스트 큐브 표시.
"""
from __future__ import annotations

import math
import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QOpenGLWidget

from . import gl_primitives as glp
from . import raycast as rc
from . import styles
from .geometry import mm_to_cell, mm_to_cell_snap_half
from .model import CubeModel, PlacedCube


class Cube3DView(QOpenGLWidget):
    """3D 뷰. controller 가 시그널로 모델/모드 변경."""

    # 단순 좌클릭. (gx, gy, layer, button) — layer=-1 빈평면.
    cell_clicked = pyqtSignal(float, float, int, int)
    cell_right_clicked = pyqtSignal(float, float, int)
    # 큐브 드래그 종료. (src_gx, src_gy, src_layer, new_gx, new_gy)
    cube_dragged = pyqtSignal(float, float, int, float, float)

    def __init__(self, model: CubeModel, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._model = model

        # 카메라
        self._cam_target = np.array(
            [model.base_xy[0], model.base_xy[1], 50.0], dtype=np.float64
        )
        self._cam_distance = 500.0
        self._cam_yaw_deg = 45.0
        self._cam_pitch_deg = 55.0
        self._proj = np.eye(4)
        self._view = np.eye(4)

        # 마우스
        self._last_pos = None
        self._press_pos = None
        self._press_button = None
        self._rotating = False
        self._panning = False
        self._dragged = False
        self._click_pixel_tolerance = 4

        # 시뮬레이션 환경
        self._z_table_top = 0.0
        self._grid_radius = 4
        self._current_layer = 0
        self._snap_half = False

        # 모드 / 선택 / 호버 / 드래그
        self._tool_mode = 'add'   # 'add' | 'select' | 'erase'
        self._active_index = -1
        self._selected_key: tuple | None = None  # (gx, gy, layer)
        self._hover_pick: tuple | None = None    # (gx, gy, hit_layer)
        self._drag_cube: PlacedCube | None = None
        self._source_cubes: list[PlacedCube] = []
        self._source_offset_x = -8.0
        # 정렬 후 실제 mm 좌표 그대로 source 큐브 표시 (set_source_positions_mm)
        self._source_positions_mm: list[tuple[float, float, float]] = []
        # 정렬에 사용한 그리드 영역 (center_xy, spacing, rows, cols)
        self._align_grid: tuple[tuple[float, float], float, int, int] | None = None

        # 로봇 시뮬레이션 (가상 그리퍼)
        self._gripper_pose: tuple[float, float, float, float] | None = None
        # (gx_mm, gy_mm, gz_mm_tip, yaw_deg)
        self._gripper_open_mm = 35.0
        self._held_cube_view: tuple[float, float, float, float] | None = None
        # 잡혀서 그리퍼 따라 움직이는 큐브 시각 (cx, cy, cz, yaw_deg)
        self._hidden_cube_keys: set = set()  # 시뮬 중 모델에서 일시 hide
        self._sim_status_text: str = ''       # 시뮬 진행 상태 화면 표시
        self._collision_state: bool = False   # 실시간 충돌 시 그리퍼 빨강
        # 충돌 컴포넌트 — 'none'|'gripper'|'held'|'both'
        # 'held' / 'both' 일 때 held cube 도 빨간 외곽선
        self._collision_kind: str = 'none'

    # --- 외부 API -------------------------------------------------------------
    def set_model(self, model: CubeModel) -> None:
        self._model = model
        self.update()

    def set_current_layer(self, layer: int) -> None:
        self._current_layer = max(0, layer)
        self.update()

    def set_snap_half(self, on: bool) -> None:
        self._snap_half = on
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        if mode not in ('add', 'select', 'erase'):
            return
        self._tool_mode = mode
        self.update()

    def set_active_index(self, idx: int) -> None:
        self._active_index = idx
        self.update()

    def set_selected_key(self, key: tuple | None) -> None:
        self._selected_key = key
        self.update()

    def set_source_cubes(self, cubes: list[PlacedCube]) -> None:
        self._source_cubes = cubes
        self.update()

    def set_source_positions_mm(self,
                                positions: list[tuple[float, float, float]]) -> None:
        """정렬된 큐브의 실제 mm 좌표 그대로 source 로 표시."""
        self._source_positions_mm = list(positions)
        self.update()

    def set_align_grid(self,
                       center: tuple[float, float] | None,
                       spacing: float | None,
                       rows: int | None,
                       cols: int | None) -> None:
        """정렬에 사용한 그리드 영역을 화면 바닥에 표시.

        center=None / spacing=None 이면 영역 표시 해제.
        rows, cols 둘 다 None 이면 spacing 만 알고 정사각형(5×5) 가정.
        """
        if center is None or spacing is None:
            self._align_grid = None
        else:
            r = int(rows) if rows is not None else 5
            c = int(cols) if cols is not None else 5
            self._align_grid = (center, float(spacing), r, c)
        # 정렬 그리드를 카메라가 한눈에 보이도록 살짝 줌아웃 + 중심 이동
        if self._align_grid is not None:
            (cx, cy), sp, r_, c_ = self._align_grid
            extent = max(r_, c_) * sp
            base_x, base_y = self._model.base_xy
            mid_x = (cx + base_x) / 2.0
            mid_y = (cy + base_y) / 2.0
            self._cam_target = np.array([mid_x, mid_y, 50.0], dtype=np.float64)
            target_dist = max(extent * 2.4, 500.0)
            self._cam_distance = max(self._cam_distance, target_dist)
        self.update()

    def set_gripper_pose(self,
                        pose: tuple[float, float, float, float] | None,
                        open_mm: float = 35.0) -> None:
        self._gripper_pose = pose
        self._gripper_open_mm = open_mm
        self.update()

    def set_held_cube(self,
                      pose: tuple[float, float, float, float] | None) -> None:
        self._held_cube_view = pose
        self.update()

    def set_hidden_cube_keys(self, keys: set) -> None:
        self._hidden_cube_keys = set(keys)
        self.update()

    def set_sim_status(self, text: str) -> None:
        self._sim_status_text = text
        # 상태 텍스트는 GL에 직접 안 그림; main_window 상태바가 이미 표시.
        # 이 값은 outside polling 용.

    def set_collision_state(self, active: bool, kind: str = 'gripper') -> None:
        """실시간 충돌 상태.

        active=True 면 kind 에 따라 그리퍼/held cube 를 빨강 강조.
        kind: 'gripper'(default) | 'held' | 'both' | 'none'.
        """
        new_kind = kind if active else 'none'
        if self._collision_state != active or self._collision_kind != new_kind:
            self._collision_state = active
            self._collision_kind = new_kind
            self.update()

    def top_view(self) -> None:
        self._cam_yaw_deg = 0.0
        self._cam_pitch_deg = 88.0
        self.update()

    def iso_view(self) -> None:
        self._cam_yaw_deg = 45.0
        self._cam_pitch_deg = 55.0
        self.update()

    # --- 자동 회전 (360° 뷰) -------------------------------------------------
    # 카메라 yaw 만 일정 속도로 증가 → 워크피스 주위를 한 바퀴 돌면서 관찰.
    # 마우스로 카메라 조작은 그대로 동작 (드래그하면 회전 잠시 멈춤 효과 없이
    # 사용자 입력이 그대로 반영됨 — 다음 timer tick 부터 다시 yaw 증가).
    def _ensure_rotate_timer(self) -> None:
        if getattr(self, '_rotate_timer', None) is None:
            self._rotate_timer = QTimer(self)
            self._rotate_timer.timeout.connect(self._on_rotate_tick)
            self._rotate_deg_per_sec = 30.0    # 12 초에 한 바퀴
            self._rotate_interval_ms = 33      # ≈30 FPS

    def _on_rotate_tick(self) -> None:
        step = self._rotate_deg_per_sec * (self._rotate_interval_ms / 1000.0)
        self._cam_yaw_deg = (self._cam_yaw_deg + step) % 360.0
        self.update()

    def start_auto_rotate(self, deg_per_sec: float = 30.0) -> None:
        """카메라를 yaw 방향으로 일정 속도 회전 시작 (기본 30°/sec = 12s/회전)."""
        self._ensure_rotate_timer()
        self._rotate_deg_per_sec = float(deg_per_sec)
        if not self._rotate_timer.isActive():
            self._rotate_timer.start(self._rotate_interval_ms)

    def stop_auto_rotate(self) -> None:
        if getattr(self, '_rotate_timer', None) is not None and self._rotate_timer.isActive():
            self._rotate_timer.stop()

    def toggle_auto_rotate(self, on: bool) -> None:
        """QPushButton(checkable=True).toggled 신호에 직접 연결 가능."""
        if on:
            self.start_auto_rotate()
        else:
            self.stop_auto_rotate()

    def set_auto_rotate_speed(self, deg_per_sec: float) -> None:
        self._ensure_rotate_timer()
        self._rotate_deg_per_sec = float(deg_per_sec)

    # --- QOpenGLWidget --------------------------------------------------------
    def initializeGL(self) -> None:
        from OpenGL import GL
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(*styles.BG_COLOR, 1.0)

    def resizeGL(self, w: int, h: int) -> None:
        from OpenGL import GL
        GL.glViewport(0, 0, w, h)
        aspect = w / max(1, h)
        self._proj = glp.perspective(45.0, aspect, 5.0, 10000.0)

    def paintGL(self) -> None:
        from OpenGL import GL
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        eye = glp.spherical_camera(
            self._cam_target,
            self._cam_distance,
            self._cam_yaw_deg,
            self._cam_pitch_deg,
        )
        self._view = glp.look_at(eye, self._cam_target, np.array([0, 0, 1.0]))

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixd(self._proj.T.astype(np.float64).flatten())
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixd(self._view.T.astype(np.float64).flatten())

        glp.draw_table(self._model.base_xy, self._model.pitch_mm,
                       self._grid_radius, self._z_table_top - 0.5)
        glp.draw_grid(self._model.base_xy, self._model.pitch_mm,
                      self._grid_radius, self._z_table_top + 0.1)
        glp.draw_axes(
            (self._model.base_xy[0], self._model.base_xy[1], self._z_table_top + 0.1),
            length=self._model.pitch_mm * 1.5,
        )

        # 0.5 스냅 모드 안내: 보조 그리드 (반 단위 가로/세로 라인)
        if self._snap_half:
            self._draw_half_grid()

        # 정렬 그리드 영역 (테이블 + 라인) — 일반 그리드 위에 같이 그림
        self._draw_align_grid_area()

        # source 큐브 — 실제 mm 좌표 (set_source_positions_mm) 우선,
        # 없으면 기존 set_source_cubes 의 fake 위치 사용
        if self._source_positions_mm:
            # 시각화 좌표계는 z_table_top=0 기준이지만 15번이 캡처한 sample_z 는
            # 두산 로봇 base 프레임 mm 라 좌표계가 달라 그대로 박으면 underground.
            # 정렬된 큐브는 항상 layer 0 (테이블 위 1층) 으로 시각화 통일.
            cube_top_z = self._z_table_top + self._model.cube_width_mm * 0.5
            for (sx, sy, _sz) in self._source_positions_mm:
                glp.draw_cube(sx, sy, cube_top_z,
                              size=self._model.cube_width_mm, yaw_deg=0.0,
                              color=styles.SOURCE_COLOR)
        elif self._source_cubes:
            src_origin_x = (
                self._model.base_xy[0] + self._source_offset_x * self._model.pitch_mm
            )
            for i, c in enumerate(self._source_cubes):
                x = src_origin_x + (i % 5) * self._model.pitch_mm
                y = self._model.base_xy[1] + (i // 5) * self._model.pitch_mm
                z = self._z_table_top + self._model.cube_width_mm * 0.5
                glp.draw_cube(x, y, z,
                              size=self._model.cube_width_mm, yaw_deg=0.0,
                              color=styles.SOURCE_COLOR)

        # 배치된 큐브
        for i, c in enumerate(self._model.cubes):
            key = (c.gx, c.gy, c.layer)
            if key in self._hidden_cube_keys:
                continue
            if self._drag_cube is not None and c is self._drag_cube:
                continue  # 드래그 중 — 별도 위치로
            x = self._model.base_xy[0] + c.gx * self._model.pitch_mm
            y = self._model.base_xy[1] + c.gy * self._model.pitch_mm
            from .motion_plan import cube_center_z as _ccz
            z = _ccz(c.layer, self._model.cube_width_mm, self._z_table_top)
            if i == self._active_index:
                color = styles.ACTIVE_COLOR
            elif self._model.is_floating(c):
                color = styles.INVALID_COLOR
            else:
                color = styles.layer_color(c.layer)
            glp.draw_cube(x, y, z,
                          size=self._model.cube_width_mm,
                          yaw_deg=c.yaw_deg - 90.0, color=color)
            if self._selected_key == key:
                glp.draw_cube_outline(
                    x, y, z,
                    size=self._model.cube_width_mm * 1.04,
                    yaw_deg=c.yaw_deg - 90.0,
                    color=styles.SELECTED_COLOR, line_width=3.0,
                )

        # 잡힌 큐브 (시뮬레이션)
        if self._held_cube_view is not None:
            hx, hy, hz, hyaw = self._held_cube_view
            held_collide = (self._collision_state
                            and self._collision_kind in ('held', 'both'))
            held_color = ((1.0, 0.30, 0.30) if held_collide
                          else styles.ACTIVE_COLOR)
            glp.draw_cube(hx, hy, hz,
                          size=self._model.cube_width_mm,
                          yaw_deg=hyaw - 90.0,
                          color=held_color)
            if held_collide:
                # 빨간 외곽선 강조 — 사용자 시각 인지 강화
                glp.draw_cube_outline(
                    hx, hy, hz,
                    size=self._model.cube_width_mm * 1.04,
                    yaw_deg=hyaw - 90.0,
                    color=(0.95, 0.10, 0.10), line_width=3.0,
                )

        # 드래그 중 큐브 (반투명 호버 위치)
        if self._drag_cube is not None:
            target = self._compute_drop_target()
            if target is not None:
                tg_gx, tg_gy, tg_layer, ok = target
                x = self._model.base_xy[0] + tg_gx * self._model.pitch_mm
                y = self._model.base_xy[1] + tg_gy * self._model.pitch_mm
                z = self._z_table_top + self._model.cube_width_mm * (tg_layer + 0.5)
                color = styles.GHOST_ADD_COLOR if ok else styles.GHOST_ERASE_COLOR
                glp.draw_cube_ghost(x, y, z,
                                    size=self._model.cube_width_mm,
                                    yaw_deg=self._drag_cube.yaw_deg - 90.0,
                                    color=color, alpha=0.45)

        # 호버 고스트 (드래그 중 아닐 때)
        elif self._hover_pick is not None and not self._is_button_down():
            self._draw_hover_ghost()

        # 그리퍼 — 충돌 컴포넌트가 gripper/both 일 때만 finger·body 빨강 강조
        if self._gripper_pose is not None:
            gx, gy, gz, gyaw = self._gripper_pose
            gripper_collide = (self._collision_state
                               and self._collision_kind in ('gripper', 'both'))
            if gripper_collide:
                glp.draw_gripper(
                    gx, gy, gz,
                    yaw_deg=gyaw,
                    open_mm=self._gripper_open_mm,
                    body_color=(1.0, 0.30, 0.30),
                    finger_color=(0.95, 0.10, 0.10),
                )
            else:
                glp.draw_gripper(
                    gx, gy, gz,
                    yaw_deg=gyaw,
                    open_mm=self._gripper_open_mm,
                )

    # --- 마우스/휠 ------------------------------------------------------------
    def _is_button_down(self) -> bool:
        return self._press_button is not None

    def mousePressEvent(self, ev) -> None:
        self._last_pos = ev.pos()
        self._press_pos = ev.pos()
        self._press_button = ev.button()
        self._dragged = False
        if ev.button() == Qt.RightButton:
            self._rotating = True
        elif ev.button() == Qt.MiddleButton:
            self._panning = True
        elif ev.button() == Qt.LeftButton:
            # 큐브 위에 press 면 드래그 후보로 기억 (Add/Select 모드에 한함)
            if self._tool_mode in ('add', 'select'):
                pick = self._pick_cube_or_cell(ev.x(), ev.y(), allow_obb=True)
                if pick is not None and pick[2] >= 0:
                    cube = self._model.find(pick[0], pick[1], pick[2])
                    if cube is not None:
                        self._drag_cube = cube
                        self.update()

    def mouseMoveEvent(self, ev) -> None:
        moved = False
        if self._last_pos is None:
            self._last_pos = ev.pos()
        else:
            dx = ev.x() - self._last_pos.x()
            dy = ev.y() - self._last_pos.y()
            if self._press_pos is not None and not self._dragged:
                if (abs(ev.x() - self._press_pos.x()) > self._click_pixel_tolerance
                        or abs(ev.y() - self._press_pos.y()) > self._click_pixel_tolerance):
                    self._dragged = True
            if self._rotating:
                self._cam_yaw_deg += dx * 0.4
                self._cam_pitch_deg = max(5.0, min(89.0, self._cam_pitch_deg - dy * 0.4))
                moved = True
            elif self._panning:
                scale = self._cam_distance * 0.0018
                yaw = math.radians(self._cam_yaw_deg)
                right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
                forward_xy = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                self._cam_target -= right * dx * scale
                self._cam_target -= forward_xy * (-dy) * scale
                moved = True
            self._last_pos = ev.pos()

        # 호버 / 드래그 위치 갱신
        if self._drag_cube is not None:
            pick = self._pick_cube_or_cell(ev.x(), ev.y(),
                                           allow_obb=False, exclude_cube=self._drag_cube)
            self._hover_pick = pick
            self.update()
        elif not self._is_button_down():
            pick = self._pick_cube_or_cell(ev.x(), ev.y(), allow_obb=True)
            if pick != self._hover_pick:
                self._hover_pick = pick
                self.update()
        elif moved:
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        was_rotating = self._rotating
        was_dragged = self._dragged
        if ev.button() == Qt.RightButton:
            self._rotating = False
        if ev.button() == Qt.MiddleButton:
            self._panning = False

        try:
            if ev.button() == Qt.MiddleButton:
                return
            if ev.button() == Qt.RightButton:
                if not (was_rotating and was_dragged):
                    pick = self._pick_cube_or_cell(ev.x(), ev.y(), allow_obb=True)
                    if pick is not None:
                        gx, gy, layer = pick
                        self.cell_right_clicked.emit(gx, gy, layer)
                return
            if ev.button() == Qt.LeftButton:
                if self._drag_cube is not None and was_dragged:
                    pick = self._pick_cube_or_cell(
                        ev.x(), ev.y(),
                        allow_obb=False, exclude_cube=self._drag_cube,
                    )
                    if pick is not None:
                        new_gx, new_gy, _ = pick
                        if (new_gx, new_gy) != (self._drag_cube.gx, self._drag_cube.gy):
                            self.cube_dragged.emit(
                                self._drag_cube.gx, self._drag_cube.gy,
                                self._drag_cube.layer,
                                new_gx, new_gy,
                            )
                    return
                if not was_dragged:
                    pick = self._pick_cube_or_cell(ev.x(), ev.y(), allow_obb=True)
                    if pick is not None:
                        gx, gy, layer = pick
                        self.cell_clicked.emit(gx, gy, layer, 0)
        finally:
            self._press_button = None
            self._press_pos = None
            self._drag_cube = None
            self.update()

    def leaveEvent(self, ev) -> None:
        if self._hover_pick is not None:
            self._hover_pick = None
            self.update()
        super().leaveEvent(ev)

    def wheelEvent(self, ev) -> None:
        steps = ev.angleDelta().y() / 120.0
        factor = 0.9 ** steps
        self._cam_distance = max(80.0, min(3000.0, self._cam_distance * factor))
        self.update()

    # --- 픽업 -----------------------------------------------------------------
    def _pick_cube_or_cell(self,
                           mouse_x: int,
                           mouse_y: int,
                           allow_obb: bool = True,
                           exclude_cube: PlacedCube | None = None
                           ) -> tuple[float, float, int] | None:
        """클릭 위치를 (gx, gy, layer) 로 해석.

        - allow_obb=True 그리고 snap_half=False 이면 ray-OBB 우선.
        - snap_half=True 이면 OBB 건너뛰고 평면 hit + 0.5 스냅 (valley 만들기 쉽게).
        - exclude_cube 는 OBB 검사에서 제외 (드래그 중 자기 자신 무시).
        """
        w = max(1, self.width())
        h = max(1, self.height())
        origin, direction = rc.screen_to_world_ray(
            mouse_x, mouse_y, w, h, self._proj, self._view
        )

        use_obb = allow_obb and not self._snap_half
        if use_obb:
            best_t = None
            best_cube = None
            cw = self._model.cube_width_mm
            hw = cw * 0.5
            for c in self._model.cubes:
                if exclude_cube is not None and c is exclude_cube:
                    continue
                cx = self._model.base_xy[0] + c.gx * self._model.pitch_mm
                cy = self._model.base_xy[1] + c.gy * self._model.pitch_mm
                cz = self._z_table_top + cw * (c.layer + 0.5)
                t = rc.ray_obb_z(
                    origin, direction,
                    (cx, cy, cz), (hw, hw, hw),
                    yaw_deg=c.yaw_deg - 90.0,
                )
                if t is None:
                    continue
                if best_t is None or t < best_t:
                    best_t = t
                    best_cube = c
            if best_cube is not None:
                return (best_cube.gx, best_cube.gy, best_cube.layer)

        # 평면 hit — 가장 위 큐브 윗면 또는 테이블면 중 가까운 쪽
        candidates = [self._z_table_top]
        for c in self._model.cubes:
            if exclude_cube is not None and c is exclude_cube:
                continue
            top_z = self._z_table_top + self._model.cube_width_mm * (c.layer + 1)
            candidates.append(top_z)
        best = None
        for z in candidates:
            pt = rc.ray_plane_z(origin, direction, z)
            if pt is None:
                continue
            # 거리: origin → pt
            t = np.linalg.norm(pt - origin)
            if best is None or t < best[0]:
                best = (t, pt)
        if best is None:
            return None
        hit = best[1]
        if self._snap_half:
            gx, gy = mm_to_cell_snap_half(self._model, float(hit[0]), float(hit[1]))
        else:
            gx, gy = mm_to_cell(self._model, float(hit[0]), float(hit[1]))
        if abs(gx) > self._grid_radius + 0.5 or abs(gy) > self._grid_radius + 0.5:
            return None
        return (gx, gy, -1)

    # --- 호버/드래그 고스트 계산 ---------------------------------------------
    def _compute_drop_target(self) -> tuple[float, float, int, bool] | None:
        """드래그 중 새 위치 후보 (gx, gy, layer, ok)."""
        if self._drag_cube is None or self._hover_pick is None:
            return None
        gx, gy, _hit_layer = self._hover_pick
        # 자기 자신을 제외한 충돌 검사 — exclude_cube 가 이미 모델에 있으니
        # would_collide 는 자기 위치를 충돌로 잡을 수 있다.
        # 따라서 드래그 큐브를 임시로 제거하고 평가:
        src = self._drag_cube
        try:
            self._model.cubes.remove(src)
            for layer in range(0, 16):
                if not self._model.would_collide(gx, gy, layer):
                    return (gx, gy, layer, True)
            return (gx, gy, 0, False)
        finally:
            self._model.cubes.append(src)
            self._model._sort()

    def _draw_hover_ghost(self) -> None:
        if self._hover_pick is None:
            return
        gx, gy, hit_layer = self._hover_pick
        if self._tool_mode == 'erase':
            target_layer = hit_layer
            if target_layer < 0:
                target_layer = self._model.top_layer_at(gx, gy) - 1
            if target_layer < 0:
                return
            x = self._model.base_xy[0] + gx * self._model.pitch_mm
            y = self._model.base_xy[1] + gy * self._model.pitch_mm
            z = self._z_table_top + self._model.cube_width_mm * (target_layer + 0.5)
            glp.draw_cube_ghost(x, y, z,
                                size=self._model.cube_width_mm * 1.02,
                                yaw_deg=0.0,
                                color=styles.GHOST_ERASE_COLOR, alpha=0.35)
        elif self._tool_mode == 'select':
            if hit_layer < 0:
                return
            x = self._model.base_xy[0] + gx * self._model.pitch_mm
            y = self._model.base_xy[1] + gy * self._model.pitch_mm
            z = self._z_table_top + self._model.cube_width_mm * (hit_layer + 0.5)
            glp.draw_cube_outline(x, y, z,
                                  size=self._model.cube_width_mm * 1.04,
                                  yaw_deg=0.0,
                                  color=styles.SELECTED_COLOR, line_width=2.5)
        else:  # add
            # 자동 적층 layer 시뮬레이션
            for layer in range(0, 16):
                if not self._model.would_collide(gx, gy, layer):
                    x = self._model.base_xy[0] + gx * self._model.pitch_mm
                    y = self._model.base_xy[1] + gy * self._model.pitch_mm
                    z = self._z_table_top + self._model.cube_width_mm * (layer + 0.5)
                    glp.draw_cube_ghost(x, y, z,
                                        size=self._model.cube_width_mm,
                                        yaw_deg=0.0,
                                        color=styles.GHOST_ADD_COLOR, alpha=0.40)
                    return

    def _draw_align_grid_area(self) -> None:
        """정렬에 사용한 그리드 영역 (테이블 면 + 라인) — 직사각형 지원."""
        if self._align_grid is None:
            return
        from OpenGL import GL
        (cx, cy), sp, rows, cols = self._align_grid
        z_floor = self._z_table_top - 0.3
        z_lines = self._z_table_top + 0.12
        half_w = (cols * sp) / 2.0
        half_h = (rows * sp) / 2.0
        pad = sp * 0.5
        x_min = cx - half_w - pad
        x_max = cx + half_w + pad
        y_min = cy - half_h - pad
        y_max = cy + half_h + pad
        # 테이블 면 (살짝 다른 색)
        GL.glColor3f(0.18, 0.24, 0.30)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(x_min, y_min, z_floor); GL.glVertex3f(x_max, y_min, z_floor)
        GL.glVertex3f(x_max, y_max, z_floor); GL.glVertex3f(x_min, y_max, z_floor)
        GL.glEnd()
        # 그리드 라인 (cell 경계)
        GL.glColor3f(0.55, 0.70, 0.85)
        GL.glLineWidth(1.5)
        GL.glBegin(GL.GL_LINES)
        for i in range(cols + 1):
            x = cx - half_w + i * sp
            GL.glVertex3f(x, cy - half_h, z_lines)
            GL.glVertex3f(x, cy + half_h, z_lines)
        for j in range(rows + 1):
            y = cy - half_h + j * sp
            GL.glVertex3f(cx - half_w, y, z_lines)
            GL.glVertex3f(cx + half_w, y, z_lines)
        GL.glEnd()
        # 셀 중심에 작은 십자 (각 셀 좌표 표시)
        GL.glColor3f(0.40, 0.55, 0.70)
        GL.glLineWidth(1.0)
        GL.glBegin(GL.GL_LINES)
        tick = sp * 0.12
        x0 = cx - half_w + sp * 0.5
        y0 = cy - half_h + sp * 0.5
        for r in range(rows):
            for c in range(cols):
                cxk = x0 + c * sp
                cyk = y0 + r * sp
                GL.glVertex3f(cxk - tick, cyk, z_lines)
                GL.glVertex3f(cxk + tick, cyk, z_lines)
                GL.glVertex3f(cxk, cyk - tick, z_lines)
                GL.glVertex3f(cxk, cyk + tick, z_lines)
        GL.glEnd()
        GL.glLineWidth(1.0)

    def _draw_half_grid(self) -> None:
        """0.5 스냅 모드에서 valley 위치를 알려주는 보조 라인."""
        from OpenGL import GL
        n = self._grid_radius
        pitch = self._model.pitch_mm * 0.5
        cx, cy = self._model.base_xy
        z = self._z_table_top + 0.05
        GL.glColor3f(0.32, 0.32, 0.40)
        GL.glLineWidth(1.0)
        GL.glBegin(GL.GL_LINES)
        for i in range(-2 * n, 2 * n + 1):
            if i % 2 == 0:
                continue  # 정수 라인은 main grid 가 이미 그림
            x = cx + i * pitch
            GL.glVertex3f(x, cy - (n + 0.5) * self._model.pitch_mm, z)
            GL.glVertex3f(x, cy + (n + 0.5) * self._model.pitch_mm, z)
        for j in range(-2 * n, 2 * n + 1):
            if j % 2 == 0:
                continue
            y = cy + j * pitch
            GL.glVertex3f(cx - (n + 0.5) * self._model.pitch_mm, y, z)
            GL.glVertex3f(cx + (n + 0.5) * self._model.pitch_mm, y, z)
        GL.glEnd()
