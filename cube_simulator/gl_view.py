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

from PyQt5.QtCore import Qt, pyqtSignal
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

        # 로봇 시뮬레이션 (가상 그리퍼)
        self._gripper_pose: tuple[float, float, float, float] | None = None
        # (gx_mm, gy_mm, gz_mm_tip, yaw_deg)
        self._gripper_open_mm = 35.0
        self._held_cube_view: tuple[float, float, float, float] | None = None
        # 잡혀서 그리퍼 따라 움직이는 큐브 시각 (cx, cy, cz, yaw_deg)
        self._hidden_cube_keys: set = set()  # 시뮬 중 모델에서 일시 hide
        self._sim_status_text: str = ''       # 시뮬 진행 상태 화면 표시

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

    def top_view(self) -> None:
        self._cam_yaw_deg = 0.0
        self._cam_pitch_deg = 88.0
        self.update()

    def iso_view(self) -> None:
        self._cam_yaw_deg = 45.0
        self._cam_pitch_deg = 55.0
        self.update()

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

        # source 큐브 (왼쪽 영역, 회색)
        if self._source_cubes:
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
            z = self._z_table_top + self._model.cube_width_mm * (c.layer + 0.5)
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
            glp.draw_cube(hx, hy, hz,
                          size=self._model.cube_width_mm,
                          yaw_deg=hyaw - 90.0,
                          color=styles.ACTIVE_COLOR)

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

        # 그리퍼
        if self._gripper_pose is not None:
            gx, gy, gz, gyaw = self._gripper_pose
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
