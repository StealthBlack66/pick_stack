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
from .motion_plan import PlanItem, build_plan, update_plan_field
from .robot_worker import RobotWorker, RunRequest
from .sim_animator import SimAnimator, SimStep


class SimulatorController(QObject):

    log = pyqtSignal(str)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    run_finished = pyqtSignal(bool)
    model_changed = pyqtSignal()
    selection_changed = pyqtSignal(object)
    sim_step = pyqtSignal(object)
    sim_finished = pyqtSignal(bool)
    sim_progress = pyqtSignal(int, int, str, float)  # cube_idx, total, phase, u
    plan_changed = pyqtSignal(object)  # list[PlanItem]
    align_finished = pyqtSignal(bool, object)  # ok, dets
    collision_decision_needed = pyqtSignal(int, str)  # cube_idx, message
    # LLM 응답 (raw_text, reasoning) — GUI 의 응답 영역 표시용
    llm_response = pyqtSignal(str, str)
    # LLM 액션 실행 시작/끝 (UI 잠금용)
    llm_started = pyqtSignal()
    llm_finished = pyqtSignal()

    def __init__(self, model: CubeModel, view: Cube3DView, parent=None):
        super().__init__(parent)
        self._model = model
        self._view = view
        self._worker: Optional[RobotWorker] = None
        self._sim: Optional[SimAnimator] = None
        self._llm_worker = None
        self._tool_mode = 'add'
        self._next_yaw_deg = 90.0
        self._dets_cache: Optional[list[dict]] = None
        self._selected_key: Optional[tuple] = None
        self._plan: list[PlanItem] = []
        self._plan_dirty = True  # 모델 변경 시 True, rebuild_plan 시 False

        view.cell_clicked.connect(self._on_cell_clicked)
        view.cell_right_clicked.connect(self._on_cell_right_clicked)
        view.cube_dragged.connect(self._on_cube_dragged)
        # 모델이 바뀌면 plan 을 dirty 표시 (사용자 편집을 함부로 덮어쓰지 않게,
        # rebuild 는 명시적 호출에서만 수행)
        self.model_changed.connect(self._mark_plan_dirty)

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
        new_layer = self._model.move_cube(src, new_gx, new_gy)
        if new_layer is None:
            self.status.emit('이동 실패: 충돌')
            return
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

    def delete_selected(self) -> bool:
        """선택된 큐브 삭제 + view/refresh. 성공 시 True."""
        c = self.get_selected()
        if c is None:
            return False
        if not self._model.remove(c.gx, c.gy, c.layer):
            return False
        self._select_cube(None)
        self._view.update()
        self.status.emit(f'삭제: ({c.gx:+.1f},{c.gy:+.1f}) L{c.layer}')
        self.model_changed.emit()
        return True

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

    def start_alignment(self,
                        dry_run: bool,
                        limit: Optional[int] = None,
                        quick: bool = True) -> bool:
        """[▶ 바둑판 정렬] — 15_바둑판_정렬.py 를 subprocess 로 통째 실행.

        dry_run=False + quick 적절히 + dsr_bringup2 살아있는 상태여야 실 모션.
        """
        if self.is_running():
            self.error.emit('이미 실행 중')
            return False
        req = RunRequest(
            layout=[],
            base_xy=self._model.base_xy,
            pitch_mm=self._model.pitch_mm,
            cube_width_mm=self._model.cube_width_mm,
            dry_run=dry_run,
            mode='align',
            align_limit=limit,
            align_quick=quick,
        )
        worker = RobotWorker(req)
        worker.log.connect(self.log)
        worker.error.connect(self.error)
        worker.progress.connect(self._on_progress)
        worker.finished_run.connect(self._on_finished_run)
        worker.finished_align.connect(self._on_finished_align)
        self._worker = worker
        worker.start()
        self.status.emit('정렬 워커 시작')
        return True

    def _on_finished_align(self, ok: bool, payload) -> None:
        dets = None
        if ok and isinstance(payload, dict):
            dets = payload.get('dets')
            if dets:
                self.cache_dets(dets)
                self.log.emit(
                    f'  → dets {len(dets)}개를 캐시에 보관 (sample_z='
                    f'{payload.get("sample_z", 0.0):.1f}mm) — '
                    f'쌓기 단계 [이전 비전 결과 재사용] 시 사용'
                )
            # 정렬에 사용한 5×5 그리드 영역을 화면 바닥에 표시
            try:
                self._view.set_align_grid(
                    center=payload['grid_center'],
                    spacing=payload['grid_spacing'],
                    rows=payload['grid_rows'],
                    cols=payload['grid_cols'],
                )
            except KeyError:
                pass
        elif ok and isinstance(payload, list):
            # 구버전 호환 — dets 만 들어온 경우
            dets = payload
            if dets:
                self.cache_dets(dets)
        # 정렬 성공 + model 에 cube 있으면 plan 자동 rebuild — src 가 정렬된
        # 위치로 갱신돼 사용자가 표/시뮬에서 바로 확인 가능.
        if ok and dets and self._model.cubes:
            self.rebuild_plan()
            self.log.emit(
                '  → plan src 좌표를 정렬된 위치로 갱신 — 표 확인 후 '
                '[▶ 시뮬레이션] 또는 [▶ 로봇 실행]'
            )
        self.align_finished.emit(ok, dets)

    def start_run(self,
                  dry_run: bool,
                  use_cached_dets: bool = False,
                  vision_conf: float = 0.40,
                  place_z_offset_mm: float = -10.0,
                  pick_z_offset_mm: float = 0.0,
                  z_table_top_override_mm: Optional[float] = None,
                  pick_min_clearance_mm: float = 5.0) -> bool:
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

        # 현재 plan (사용자 표 편집 반영) 을 그대로 전달 — worker 가 src/tgt
        # x/y/yaw 를 plan 그대로 사용 (z 만 sample_z 기반 robot frame).
        plan_for_run = self.ensure_plan() if self._model.cubes else None
        req = RunRequest(
            layout=list(self._model.cubes),
            base_xy=self._model.base_xy,
            pitch_mm=self._model.pitch_mm,
            cube_width_mm=self._model.cube_width_mm,
            dry_run=dry_run,
            dets=self._dets_cache if use_cached_dets else None,
            vision_conf_thr=float(vision_conf),
            plan=plan_for_run,
            place_z_offset_mm=float(place_z_offset_mm),
            pick_z_offset_mm=float(pick_z_offset_mm),
            z_table_top_override_mm=(
                float(z_table_top_override_mm)
                if z_table_top_override_mm is not None else None
            ),
            pick_min_clearance_mm=float(pick_min_clearance_mm),
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

    def start_oneshot_calibration(self,
                                  skip_launch: bool = False,
                                  keep_bringup: bool = True) -> bool:
        """[🎯 원샷 캘리브레이션] — 09_원샷_캘리브레이션.py subprocess 실행.

        09 가 자체 bringup + servo on + 08 호출. 우리 세션은 충돌 방지로
        미리 teardown. 진행 로그는 GUI 에 흘림. cv2 창은 별도로 뜸 (사용자가
        ArUco 마커 보면서 's' 키로 base 자세 저장).
        """
        if self.is_running():
            self.error.emit('이미 실행 중')
            return False
        # 시뮬 시각 리셋 + 세션 통째 teardown — 09 가 깨끗한 상태에서 시작하도록
        self.reset_simulator(reset_model=False)
        try:
            from .robot_worker import default_manager
            default_manager().teardown_session()
        except Exception as e:  # noqa: BLE001
            self.log.emit(f'[캘리브] 세션 teardown 무시: {e!r}')

        req = RunRequest(
            layout=[],
            base_xy=self._model.base_xy,
            pitch_mm=self._model.pitch_mm,
            cube_width_mm=self._model.cube_width_mm,
            dry_run=False,
            mode='calibrate',
            cal_skip_launch=bool(skip_launch),
            cal_keep_bringup=bool(keep_bringup),
        )
        worker = RobotWorker(req)
        worker.log.connect(self.log)
        worker.error.connect(self.error)
        worker.finished_run.connect(self._on_finished_run)
        self._worker = worker
        worker.start()
        self.status.emit('원샷 캘리브레이션 시작 — cv2 창 확인')
        return True

    def start_recover_home(self, dry_run: bool = False) -> bool:
        """[원점복귀] — 시뮬 + ROS 세션 모두 리셋 후 activate + safe HOME.

        물리 비상정지 해제 후 등 stale 상태에서는 기존 세션의 service client 가
        끊긴 컨트롤러를 가리켜 살아나지 못함. 15번 main 처럼 노드를 새로
        만들어야 확실히 복구되므로 **항상 세션 부수고 새로 생성**.
        + 시뮬레이터의 모든 시각 상태도 리셋 — 그리퍼/held cube/hidden/충돌상태.
        """
        if self.is_running():
            self.error.emit('이미 실행 중')
            return False
        # 1) 시뮬 통째 리셋 (UI 즉시 반응)
        self.reset_simulator(reset_model=False)
        # 2) ROS 세션 강제 teardown — 다음 워커가 새 노드 만들 수 있도록
        try:
            from .robot_worker import default_manager
            default_manager().teardown_session()
            self.log.emit('[원점복귀] 기존 ROS 세션 강제 종료 — 새 노드로 재초기화 예정')
        except Exception as e:  # noqa: BLE001
            self.log.emit(f'[원점복귀] 세션 teardown 무시: {e!r}')

        req = RunRequest(
            layout=[],
            base_xy=self._model.base_xy,
            pitch_mm=self._model.pitch_mm,
            cube_width_mm=self._model.cube_width_mm,
            dry_run=dry_run,
            mode='recover',
        )
        worker = RobotWorker(req)
        worker.log.connect(self.log)
        worker.error.connect(self.error)
        worker.finished_run.connect(self._on_finished_run)
        self._worker = worker
        worker.start()
        self.status.emit('원점복귀 시작...')
        return True

    def reset_simulator(self, reset_model: bool = False) -> None:
        """시뮬레이터의 시각·내부 상태 리셋.

        reset_model=False: 사용자가 디자인한 큐브 배치(model.cubes)는 보존.
        reset_model=True: 모델 큐브 전체 clear (사용자 데이터 손실 — 주의).

        - 진행 중인 SimAnimator 정지 + cleanup
        - GL view: gripper, held cube, hidden, collision, active idx 모두 리셋
        - 진행률 0, 충돌 다이얼로그 대기 상태 cancel
        """
        # 시뮬 정지
        if self._sim is not None:
            try:
                self._sim.stop()
            except Exception:
                pass
            self._sim = None

        # GL view 시각 상태 모두 클리어
        try:
            self._view.set_gripper_pose(None)
            self._view.set_held_cube(None)
            self._view.set_hidden_cube_keys(set())
            self._view.set_active_index(-1)
            try:
                self._view.set_collision_state(False, 'none')
            except TypeError:
                self._view.set_collision_state(False)
        except Exception as e:  # noqa: BLE001
            self.log.emit(f'[리셋] view 일부 실패 (무시): {e!r}')

        if reset_model:
            self._model.clear()
            self._select_cube(None)
            self._plan = []
            self._plan_dirty = True
            self.model_changed.emit()
            self.plan_changed.emit(self._plan)

        self.status.emit('시뮬 리셋 완료')
        self.log.emit('=== 시뮬레이터 시각 상태 리셋 ===')

    # ============================================================
    # LLM 자연어 인터페이스 (Mode A/B/C/D)
    # ============================================================
    def start_llm_command(self,
                          mode: str,
                          user_text: str,
                          image_bytes=None,
                          image_mime: str = 'image/png') -> bool:
        """LLM 백그라운드 worker 시작. mode = 'A' | 'B' | 'C' | 'D'.

        worker.finished_ok → _on_llm_response → dispatch.
        worker.failed → error.emit.
        """
        from .llm import AnthropicClient, LlmRequest, LlmWorker

        if self._llm_worker is not None and self._llm_worker.isRunning():
            self.error.emit('이미 LLM 작업 중')
            return False

        client = AnthropicClient()
        if not client.has_api_key:
            self.error.emit(
                'ANTHROPIC_API_KEY 미설정. .env 또는 셸 env 에 키 추가 후 재시작.'
            )
            return False

        req = LlmRequest(
            mode=mode,
            user_text=user_text,
            image_bytes=image_bytes,
            image_mime=image_mime,
            model_snapshot=self._model.to_dict(),
            dets=self._dets_cache,
            plan_summary=self._plan_summary_for_llm(),
        )
        worker = LlmWorker(req, client=client, parent=self)
        worker.finished_ok.connect(self._on_llm_response)
        worker.failed.connect(self._on_llm_failed)
        worker.finished.connect(self._on_llm_thread_finished)
        self._llm_worker = worker
        self.llm_started.emit()
        self.status.emit(f'🤖 LLM (mode {mode}) 요청 전송...')
        worker.start()
        return True

    def _plan_summary_for_llm(self) -> str:
        """plan 요약 한 줄 — LLM 컨텍스트용."""
        n = len(self._plan)
        if n == 0:
            return 'plan 비어있음 (rebuild_plan 으로 생성 필요)'
        dirty = ' (dirty — 모델 변경 후 rebuild 권장)' if self._plan_dirty else ''
        return f'plan {n} 단계{dirty}'

    def _on_llm_response(self, raw_text: str, cmd, _raw_reasoning: str) -> None:
        """LLM 응답 도착 — 응답 emit + 액션 dispatch.

        worker 가 보낸 _raw_reasoning 은 LLM 응답 전체 텍스트 (디버그용). 우리는
        cmd.reasoning (LLM 이 JSON 안에 넣은 짧은 한 줄) 만 사용자에게 표시.
        """
        short = (cmd.reasoning or '').strip() or '(reasoning 없음)'
        self.log.emit(f'🤖 LLM [{cmd.action}]: {short}')
        self.llm_response.emit(raw_text, short)
        self._dispatch_llm_command(cmd, depth=0)

    def _on_llm_failed(self, msg: str) -> None:
        self.error.emit(f'🤖 LLM 실패: {msg}')
        self.llm_response.emit('', f'❌ {msg}')

    def _on_llm_thread_finished(self) -> None:
        self._llm_worker = None
        self.llm_finished.emit()

    # 액션 → 메서드 매핑. 안전상 화이트리스트로 dispatch.
    def _dispatch_llm_command(self, cmd, depth: int = 0) -> None:
        """LLM 명령 실행. next_action 체인은 최대 3 깊이까지."""
        if cmd is None or not cmd.is_valid():
            return
        if depth > 3:
            self.log.emit('🤖 LLM 체인 깊이 초과 — 추가 액션 무시')
            return

        action = cmd.action
        args = dict(cmd.args or {})

        try:
            if action == 'load_preset':
                name = args.get('name', '')
                if name:
                    self.load_preset(str(name))
            elif action == 'add_cube':
                gx = float(args.get('gx', 0.0))
                gy = float(args.get('gy', 0.0))
                yaw = float(args.get('yaw', self._next_yaw_deg))
                c = self._model.add_auto_layer(gx, gy, yaw_deg=yaw)
                self._view.update()
                self.model_changed.emit()
                if c is None:
                    self.status.emit(f'🤖 add_cube ({gx:+.1f},{gy:+.1f}) 충돌')
            elif action == 'remove_cube':
                gx = float(args.get('gx', 0.0))
                gy = float(args.get('gy', 0.0))
                layer = int(args.get('layer', 0))
                self._model.remove(gx, gy, layer)
                self._view.update()
                self.model_changed.emit()
            elif action == 'move_cube':
                src = args.get('from') or {}
                dst = args.get('to') or {}
                cube = self._model.find(
                    float(src.get('gx', 0.0)),
                    float(src.get('gy', 0.0)),
                    int(src.get('layer', 0)),
                )
                if cube is not None:
                    self._model.move_cube(cube,
                                          float(dst.get('gx', 0.0)),
                                          float(dst.get('gy', 0.0)))
                    self._view.update()
                    self.model_changed.emit()
            elif action == 'clear_model':
                self.clear_model()
            elif action == 'update_cube_yaw':
                gx = float(args.get('gx', 0.0))
                gy = float(args.get('gy', 0.0))
                layer = int(args.get('layer', 0))
                yaw = float(args.get('yaw', 90.0))
                self._model.update_cube(gx, gy, layer, new_yaw=yaw)
                self._view.update()
                self.model_changed.emit()
            elif action == 'rebuild_plan':
                self.rebuild_plan()
            elif action == 'apply_yaw_corrections':
                self.apply_yaw_corrections()
            elif action == 'start_simulation':
                auto = bool(args.get('auto_correct', True))
                self.start_simulation(auto_correct=auto)
            elif action == 'start_alignment':
                dry = bool(args.get('dry_run', True))
                quick = bool(args.get('quick', True))
                self.start_alignment(dry_run=dry, quick=quick)
            elif action == 'start_run':
                dry = bool(args.get('dry_run', True))
                self.start_run(dry_run=dry)
            elif action == 'start_oneshot_calibration':
                skip = bool(args.get('skip_launch', False))
                keep = bool(args.get('keep_bringup', True))
                self.start_oneshot_calibration(skip_launch=skip, keep_bringup=keep)
            elif action == 'start_recover_home':
                dry = bool(args.get('dry_run', False))
                self.start_recover_home(dry_run=dry)
            elif action == 'reset_simulator':
                reset_model = bool(args.get('reset_model', False))
                self.reset_simulator(reset_model=reset_model)
            elif action in ('explain', 'noop'):
                pass   # 응답만 (reasoning 은 이미 emit 됨)
            else:
                self.log.emit(f'🤖 미허용 action: {action}')
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'🤖 액션 {action} 실행 실패: {e!r}')

        # 체인 다음 액션
        if cmd.next_action is not None:
            self._dispatch_llm_command(cmd.next_action, depth=depth + 1)

    def request_estop(self) -> None:
        """비상정지 — bringup/DRCF 프로세스 통째 kill = "터미널 끈 효과".

        ROS 통신이 끊어져 trajectory 가 즉시 멈춤. 우리 세션도 stale 이라
        같이 teardown. 복구는 [🏠 원점복귀] 로 — bringup 자동 재시작 + HOME.

        정식 ServoOff service 안 거치는 이유:
          - service 호출에 시간 걸리고 컨트롤러 상태에 따라 실패 가능
          - 어차피 비상정지 후엔 다시 연결·HOME 해야 하니 hard kill 이 단순·확실
          - "내가 띄운 터미널 ctrl+C / kill 했을 때와 동일 효과"
        """
        # 1) 시뮬 정지 (UI 즉시 반응)
        if self._sim is not None and self._sim.is_running():
            self._sim.stop()
        # 2) 워커 estop 플래그 — 진행 중 코드가 ROS 끊김 감지 후 정상 종료 경로로
        if self._worker is not None:
            self._worker.request_estop()
        # 3) ★ bringup/DRCF/dsr_controller 통째 kill — trajectory 중이라도 즉시 멈춤
        self._kill_bringup_for_estop()
        # 4) 우리 ROS 세션도 teardown — 어차피 죽은 노드 가리키니
        try:
            from .robot_worker import default_manager
            default_manager().teardown_session()
        except Exception:
            pass
        self.status.emit('🛑 비상정지 — 로봇 연결 끊음. 복구는 [🏠 원점복귀]')

    def _kill_bringup_for_estop(self) -> None:
        """dsr_bringup / DRCF / dsr_controller 강제 kill.

        터미널 ctrl+C 와 동일 효과. ROS publisher/service 모두 끊겨 컨트롤러
        쪽 trajectory action 이 즉시 ABORTED.
        """
        import os
        from .robot_worker import default_manager

        self.log.emit('🛑 [estop] bringup/DRCF 프로세스 강제 종료')
        # 우리 GUI 자체는 안 죽이려고 범위 좁힘
        for pat in [
            'dsr_bringup',
            'DRCF',
            'dsr_controller',
            'dsr_hardware',
            'dsr_dual_controller',
        ]:
            os.system(f'pkill -9 -f {pat} 2>/dev/null')

        # 우리 SessionManager 의 bringup_proc 핸들도 정리 — 이미 죽었지만
        sm = default_manager()
        with sm.bringup_lock:
            if sm.bringup_proc is not None:
                try:
                    sm.bringup_proc.terminate()
                except Exception:
                    pass
                sm.bringup_proc = None
        self.log.emit('🛑 [estop] 완료 — 복구: [🏠 원점복귀] (bringup 자동 재시작)')

    def _on_progress(self, idx: int, total: int, msg: str) -> None:
        self._view.set_active_index(idx)
        self.progress.emit(idx, total, msg)

    def _on_finished_run(self, ok: bool) -> None:
        self._view.set_active_index(-1)
        self.run_finished.emit(ok)
        self.status.emit('완료' if ok else '중단/오류')
        self._worker = None

    # --- 모션 계획 (plan) -----------------------------------------------------
    def get_plan(self) -> list[PlanItem]:
        return self._plan

    def is_plan_dirty(self) -> bool:
        return self._plan_dirty

    def _mark_plan_dirty(self) -> None:
        self._plan_dirty = True

    def rebuild_plan(self) -> list[PlanItem]:
        """모델 → plan 새로 빌드 (사용자 편집 내용 버림).

        정렬 후 dets 가 캐시돼 있으면 src 좌표를 그 위치로 — 시뮬에서 이미
        '정렬된 큐브'를 픽하는 모습으로 보이도록.
        """
        source_positions = None
        if self._dets_cache:
            source_positions = [
                (float(d['base_xyz_mm'][0]),
                 float(d['base_xyz_mm'][1]),
                 float(d['base_xyz_mm'][2]))
                for d in self._dets_cache
                if d.get('base_xyz_mm') is not None
            ]
        self._plan = build_plan(
            self._model,
            z_table_top=0.0,
            source_positions=source_positions,
        )
        self._plan_dirty = False
        self.plan_changed.emit(self._plan)
        src_note = (f' (정렬 dets {len(source_positions)}개 src 사용)'
                    if source_positions else '')
        self.status.emit(f'계획 갱신: {len(self._plan)}개 단계' + src_note)
        return self._plan

    def ensure_plan(self) -> list[PlanItem]:
        """plan 이 없거나 dirty 면 자동 rebuild. 있으면 그대로 사용."""
        if not self._plan or self._plan_dirty:
            return self.rebuild_plan()
        return self._plan

    def update_plan_cell(self, row: int, field_name: str, value: float) -> bool:
        """표 셀 편집 콜백. 성공 시 True. (plan_changed emit 하지 않음 —
        UI 가 이미 그 셀을 표시 중이라 row 전체 rebuild 시 RuntimeError)."""
        item = update_plan_field(self._plan, row, field_name, value)
        if item is None:
            return False
        # tgt_yaw 편집은 model.yaw_deg 로 즉시 동기화 — 시뮬 placed cube 시각과
        # 로봇 실행이 같은 yaw 를 사용하도록 (사용자가 표에서 회전 보정).
        if field_name == 'tgt_yaw':
            self._sync_plan_yaw_to_model()
            self._view.update()
        self.status.emit(f'계획[{row}] {field_name}={value}')
        return True

    def _sync_plan_yaw_to_model(self) -> None:
        """plan.tgt_yaw 변경 → model.c.yaw_deg 반영.

        충돌 회피 보정(apply_yaw_corrections/sim apply_decision)이나 사용자 표
        편집으로 plan 의 tgt_yaw 가 바뀌면 model 도 같이 갱신해야 시뮬의
        '놓인 cube' 시각과 실 로봇 동작이 같은 yaw 를 사용한다.
        """
        for item in self._plan:
            c = self._model.find(item.cube_gx, item.cube_gy, item.cube_layer)
            if c is not None and abs(c.yaw_deg - item.tgt_yaw) > 1e-3:
                c.yaw_deg = float(item.tgt_yaw)

    # --- 가상 시뮬레이션 ------------------------------------------------------
    def start_simulation(self, auto_correct: bool = False) -> bool:
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
        plan = self.ensure_plan()
        if not plan:
            self.error.emit('계획이 비어있습니다')
            return False
        # auto_correct=True 면 시뮬 직전 yaw 충돌 보정 후 plan_changed 발화.
        # 사용자가 표에서 편집한 값을 보존하려면 OFF.
        if auto_correct:
            before = [(p.src_yaw, p.tgt_yaw) for p in plan]
            changed = self.apply_yaw_corrections()
            after = [(p.src_yaw, p.tgt_yaw) for p in plan]
            if changed == 0:
                self.log.emit('[자동 보정] 충돌 없음 또는 ±90° 회피 불가 — plan 변화 없음')
            else:
                diffs = [
                    f'[{i+1}] src{b[0]:+.0f}→{a[0]:+.0f}° tgt{b[1]:+.0f}→{a[1]:+.0f}°'
                    for i, (b, a) in enumerate(zip(before, after)) if b != a
                ]
                self.log.emit(f'[자동 보정] {changed}건 변경: ' + ' / '.join(diffs[:4])
                              + (' …' if len(diffs) > 4 else ''))

        # 실제 시뮬에 들어갈 plan yaw 요약 (사용자가 즉시 확인 가능)
        summary = ', '.join(
            f'[{i+1}]src{p.src_yaw:+.0f}/tgt{p.tgt_yaw:+.0f}°'
            for i, p in enumerate(plan[:6])
        )
        self.log.emit('[시뮬 plan yaw] ' + summary
                      + (' …' if len(plan) > 6 else ''))

        sim = SimAnimator(plan, self._view, self._model)
        sim.step.connect(self._on_sim_step)
        sim.finished.connect(self._on_sim_finished)
        sim.progress.connect(self.sim_progress)  # 매 tick 진행률 forwarding
        # 실시간 충돌 자동 보정 → plan 갱신 → 표 갱신 + model yaw 동기화 + 뷰 refresh
        sim.plan_yaw_corrected.connect(self._on_plan_yaw_corrected_sync)
        # 충돌 시 사용자 결정 요청 → main_window 가 다이얼로그
        sim.decision_needed.connect(self.collision_decision_needed)
        self._sim = sim
        sim.start()

    def respond_collision_decision(self, accept: bool) -> None:
        """main_window 가 사용자 다이얼로그 응답 후 호출."""
        if self._sim is not None:
            self._sim.apply_decision(accept)

    def apply_yaw_corrections(self) -> int:
        """[yaw 충돌 보정] 버튼 — plan 의 각 항목 yaw 를 finger 충돌 회피로 보정.

        사용자가 명시 호출. plan 을 in-place 수정 + plan_changed emit → 표 갱신.
        보정이 발생한 항목 수 반환.
        """
        from .sim_animator import SimulationConfig

        plan = self.ensure_plan()
        if not plan:
            self.error.emit('계획이 비어있습니다')
            return 0
        # SimAnimator 의 yaw 보정 로직만 빌려 쓰기 위해 __init__ 우회 helper 생성.
        # PyQt5 QObject 는 set 안 된 속성에 getattr-fallback 접근하면
        # 'super-class __init__() was never called' RuntimeError 를 던지므로
        # _compute_corrected_yaws 가 참조하는 모든 속성을 명시적으로 set.
        helper = SimAnimator.__new__(SimAnimator)
        helper._plan = plan
        helper._cw = self._model.cube_width_mm
        helper._cfg = SimulationConfig()
        changed = 0
        for idx, item in enumerate(plan):
            helper._cube_index = idx
            new_src, new_tgt, note = helper._compute_corrected_yaws(item)
            if (abs(new_src - item.src_yaw) > 1e-3
                    or abs(new_tgt - item.tgt_yaw) > 1e-3):
                item.src_yaw = new_src
                item.tgt_yaw = new_tgt
                changed += 1
            if note:
                self.log.emit('[yaw 보정] ' + note)
        if changed:
            # tgt_yaw 변경분을 model 로 동기화 (시뮬 placed cube 시각 + 로봇 실행 일관성)
            self._sync_plan_yaw_to_model()
            self._view.update()
            self.plan_changed.emit(plan)
            self.status.emit(f'yaw 보정 {changed}건 적용 — 표 갱신')
        else:
            self.status.emit('보정할 yaw 없음 (모든 yaw 가 충돌 회피 기준 통과)')
        return changed

    def _on_plan_yaw_corrected_sync(self) -> None:
        """시뮬 중 충돌 보정으로 plan.tgt_yaw 변경 시 — model yaw + 뷰 + 표 갱신."""
        self._sync_plan_yaw_to_model()
        self._view.update()
        self.plan_changed.emit(self._plan)

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
        positions = []
        for d in dets:
            xyz = d.get('base_xyz_mm')
            if xyz is None:
                continue
            positions.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        self._view.set_source_positions_mm(positions)

    def clear_dets_cache(self) -> None:
        self._dets_cache = None
        self._view.set_source_positions_mm([])
        self._view.set_align_grid(None, None, None, None)
