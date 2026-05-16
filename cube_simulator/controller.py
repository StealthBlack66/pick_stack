"""SimulatorController — model⇄view⇄worker 중재 + 도구 모드 + 시뮬레이터.

핵심:
- 도구 모드 (add / select / erase) 변경
- 큐브 추가/제거/이동/편집
- 프리셋·JSON I/O
- [로봇 실행] (RobotWorker) 및 [▶ 시뮬레이션] (SimAnimator) 트리거
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from . import shapes_io
from .gl_view import Cube3DView
from .model import CubeModel, PlacedCube
from .robot_worker import RobotWorker, RunRequest
from .sim_animator import SimAnimator, SimStep


class SimulatorController(QObject):

    log = pyqtSignal(str)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    run_finished = pyqtSignal(bool)
    model_changed = pyqtSignal()
    # 큐브 선택 (None 이면 선택 해제). PlacedCube 값 + 키 모두 전달.
    selection_changed = pyqtSignal(object)
    # 시뮬 단계 진행. (step_dict)
    sim_step = pyqtSignal(object)
    sim_finished = pyqtSignal(bool)

    def __init__(self, model: CubeModel, view: Cube3DView, parent=None):
        super().__init__(parent)
        self._model = model
        self._view = view
        self._worker: Optional[RobotWorker] = None
        self._sim: Optional[SimAnimator] = None
        self._tool_mode = 'add'
        self._next_yaw_deg = 90.0
        self._dets_cache: Optional[list[dict]] = None
        self._selected_key: Optional[tuple] = None

        view.cell_clicked.connect(self._on_cell_clicked)
        view.cell_right_clicked.connect(self._on_cell_right_clicked)
        view.cube_dragged.connect(self._on_cube_dragged)

    # --- 모델 / 모드 ---------------------------------------------------------
    @property
    def model(self) -> CubeModel:
        return self._model

    def set_tool_mode(self, mode: str) -> None:
        if mode not in ('add', 'select', 'erase'):
            return
        self._tool_mode = mode
        self._view.set_tool_mode(mode)
        self.status.emit(f'도구: {mode}')

    def get_tool_mode(self) -> str:
        return self._tool_mode

    def set_next_yaw(self, yaw_deg: float) -> None:
        self._next_yaw_deg = float(yaw_deg)

    # --- 클릭 / 드래그 -------------------------------------------------------
    def _on_cell_clicked(self, gx: float, gy: float, layer: int, _button: int) -> None:
        if self._tool_mode == 'erase':
            self._erase_at(gx, gy, layer)
        elif self._tool_mode == 'select':
            if layer >= 0:
                self._select_cube((gx, gy, layer))
            else:
                self._select_cube(None)
        else:  # add
            c = self._model.add_auto_layer(gx, gy, yaw_deg=self._next_yaw_deg)
            if c is None:
                self.status.emit(f'추가 실패: ({gx:+.1f},{gy:+.1f}) 충돌')
            else:
                self.status.emit(f'추가: ({gx:+.1f},{gy:+.1f}) L{c.layer}')
        self._view.update()
        self.model_changed.emit()

    def _on_cell_right_clicked(self, gx: float, gy: float, layer: int) -> None:
        self._erase_at(gx, gy, layer)
        self._view.update()
        self.model_changed.emit()

    def _on_cube_dragged(self,
                         src_gx: float, src_gy: float, src_layer: int,
                         new_gx: float, new_gy: float) -> None:
        """드래그 종료 — 자동 layer 로 이동."""
        src = self._model.find(src_gx, src_gy, src_layer)
        if src is None:
            return
        # 자기 자신 제외하고 충돌 없는 첫 layer
        self._model.cubes.remove(src)
        new_layer = None
        for ly in range(0, 16):
            if not self._model.would_collide(new_gx, new_gy, ly):
                new_layer = ly
                break
        if new_layer is None:
            # 복귀
            self._model.cubes.append(src)
            self._model._sort()
            self.status.emit('이동 실패: 충돌')
            return
        src.gx = new_gx
        src.gy = new_gy
        src.layer = new_layer
        self._model.cubes.append(src)
        self._model._sort()
        self._select_cube((src.gx, src.gy, src.layer))
        self.status.emit(
            f'이동: ({src_gx:+.1f},{src_gy:+.1f})L{src_layer} → '
            f'({new_gx:+.1f},{new_gy:+.1f})L{new_layer}'
        )
        self._view.update()
        self.model_changed.emit()

    def _erase_at(self, gx: float, gy: float, layer: int) -> None:
        if layer >= 0:
            ok = self._model.remove(gx, gy, layer)
            if ok:
                self.status.emit(f'삭제: ({gx:+.1f},{gy:+.1f}) L{layer}')
                if self._selected_key == (gx, gy, layer):
                    self._select_cube(None)
            else:
                self.status.emit('큐브 없음')
            return
        removed = self._model.remove_top_at(gx, gy)
        if removed is not None:
            self.status.emit(f'삭제: ({gx:+.1f},{gy:+.1f}) L{removed.layer}')
            if self._selected_key == (gx, gy, removed.layer):
                self._select_cube(None)
        else:
            self.status.emit('해당 위치에 큐브 없음')

    # --- 선택 -----------------------------------------------------------------
    def _select_cube(self, key: Optional[tuple]) -> None:
        if key is None:
            self._selected_key = None
            self._view.set_selected_key(None)
            self.selection_changed.emit(None)
            return
        c = self._model.find(*key)
        if c is None:
            self._selected_key = None
            self._view.set_selected_key(None)
            self.selection_changed.emit(None)
            return
        self._selected_key = key
        self._view.set_selected_key(key)
        self.selection_changed.emit(c)

    def select_at(self, gx: float, gy: float, layer: int) -> None:
        self._select_cube((gx, gy, layer))

    def clear_selection(self) -> None:
        self._select_cube(None)

    def get_selected(self) -> Optional[PlacedCube]:
        if self._selected_key is None:
            return None
        return self._model.find(*self._selected_key)

    def update_selected(self,
                        new_gx: Optional[float] = None,
                        new_gy: Optional[float] = None,
                        new_layer: Optional[int] = None,
                        new_yaw: Optional[float] = None) -> None:
        if self._selected_key is None:
            return
        gx, gy, layer = self._selected_key
        c = self._model.update_cube(gx, gy, layer,
                                    new_gx=new_gx, new_gy=new_gy,
                                    new_layer=new_layer, new_yaw=new_yaw)
        if c is None:
            self.error.emit('편집 실패: 충돌')
            return
        new_key = (c.gx, c.gy, c.layer)
        self._selected_key = new_key
        self._view.set_selected_key(new_key)
        self.selection_changed.emit(c)
        self._view.update()
        self.model_changed.emit()

    # --- 모델 조작 ------------------------------------------------------------
    def clear_model(self) -> None:
        self._model.clear()
        self._view.set_active_index(-1)
        self._select_cube(None)
        self._view.update()
        self.model_changed.emit()
        self.status.emit('모델 초기화')

    def load_preset(self, name: str) -> None:
        try:
            shapes_io.apply_preset(self._model, name)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'프리셋 로드 실패: {e!r}')
            return
        self._select_cube(None)
        self._view.update()
        self.model_changed.emit()
        self.status.emit(f'프리셋 로드: {name} ({len(self._model.cubes)}개)')

    def list_preset_names(self) -> list[str]:
        try:
            return shapes_io.list_preset_names()
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'프리셋 목록 로드 실패: {e!r}')
            return []

    def save_json(self, path: str) -> None:
        try:
            shapes_io.save_json(self._model, path)
            self.status.emit(f'저장: {path}')
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'저장 실패: {e!r}')

    def load_json(self, path: str) -> None:
        try:
            m = shapes_io.load_json(path)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'불러오기 실패: {e!r}')
            return
        self._model.base_xy = m.base_xy
        self._model.pitch_mm = m.pitch_mm
        self._model.cube_width_mm = m.cube_width_mm
        self._model.replace_all(m.cubes)
        self._select_cube(None)
        self._view.update()
        self.model_changed.emit()
        self.status.emit(f'불러오기: {path} ({len(self._model.cubes)}개)')

    # --- 실 로봇 실행 ---------------------------------------------------------
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def is_simulating(self) -> bool:
        return self._sim is not None and self._sim.is_running()

    def start_run(self,
                  dry_run: bool,
                  use_cached_dets: bool = False) -> bool:
        if self.is_running():
            self.error.emit('이미 실행 중')
            return False
        if not self._model.cubes:
            self.error.emit('배치된 큐브가 없습니다')
            return False
        floating = [c for c in self._model.cubes if self._model.is_floating(c)]
        if floating:
            cells = ', '.join(
                f'({c.gx:+.1f},{c.gy:+.1f})L{c.layer}' for c in floating
            )
            self.error.emit(f'받침 없는 큐브 {len(floating)}개: {cells}')
            return False

        req = RunRequest(
            layout=list(self._model.cubes),
            base_xy=self._model.base_xy,
            pitch_mm=self._model.pitch_mm,
            cube_width_mm=self._model.cube_width_mm,
            dry_run=dry_run,
            dets=self._dets_cache if use_cached_dets else None,
        )
        worker = RobotWorker(req)
        worker.log.connect(self.log)
        worker.error.connect(self.error)
        worker.progress.connect(self._on_progress)
        worker.finished_run.connect(self._on_finished_run)
        self._worker = worker
        worker.start()
        self.status.emit('워커 시작')
        return True

    def request_estop(self) -> None:
        if self._worker is not None:
            self._worker.request_estop()
            self.status.emit('비상정지 요청')
        if self._sim is not None and self._sim.is_running():
            self._sim.stop()
            self.status.emit('시뮬레이션 정지')

    def _on_progress(self, idx: int, total: int, msg: str) -> None:
        self._view.set_active_index(idx)
        self.progress.emit(idx, total, msg)

    def _on_finished_run(self, ok: bool) -> None:
        self._view.set_active_index(-1)
        self.run_finished.emit(ok)
        self.status.emit('완료' if ok else '중단/오류')
        self._worker = None

    # --- 가상 시뮬레이션 ------------------------------------------------------
    def start_simulation(self) -> bool:
        if self.is_simulating():
            self.error.emit('이미 시뮬레이션 중')
            return False
        if not self._model.cubes:
            self.error.emit('배치된 큐브가 없습니다')
            return False
        floating = [c for c in self._model.cubes if self._model.is_floating(c)]
        if floating:
            self.error.emit(f'받침 없는 큐브 {len(floating)}개')
            return False
        sim = SimAnimator(self._model, self._view)
        sim.step.connect(self._on_sim_step)
        sim.finished.connect(self._on_sim_finished)
        self._sim = sim
        sim.start()
        self.status.emit('시뮬레이션 시작')
        return True

    def _on_sim_step(self, step: SimStep) -> None:
        # 진행 상황은 view 가 set_gripper_pose / set_held_cube 로 이미 갱신
        idx = step.cube_index
        total = step.total
        msg = step.message
        if msg:
            self.log.emit(msg)
        # 상태바에는 짧게
        self.status.emit(f'시뮬 [{idx+1}/{total}] {step.phase}')
        self.progress.emit(idx, total, msg)
        self.sim_step.emit(step)

    def _on_sim_finished(self, ok: bool) -> None:
        self._view.set_gripper_pose(None)
        self._view.set_held_cube(None)
        self._view.set_hidden_cube_keys(set())
        self._view.set_active_index(-1)
        self.status.emit('시뮬 완료' if ok else '시뮬 정지')
        self.sim_finished.emit(ok)
        self._sim = None

    # --- 비전 dets 캐시 -------------------------------------------------------
    def cache_dets(self, dets: list[dict]) -> None:
        self._dets_cache = dets
        cubes = [PlacedCube(gx=0, gy=0, layer=0) for _ in dets]
        self._view.set_source_cubes(cubes)

    def clear_dets_cache(self) -> None:
        self._dets_cache = None
        self._view.set_source_cubes([])
