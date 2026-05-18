"""메인 윈도우 — 3D 뷰 + 컨트롤 패널 + 상태바.

UI 의 가장 응집도 높은 부분은 `widgets/` 패키지의 패널 클래스로 분리됨:
- `SelectedCubePanel`  — 선택된 큐브 속성 편집
- `PlanTablePanel`     — 모션 계획 표 + 갱신/yaw보정 버튼
- `RunControlPanel`    — 정렬·시뮬·실행·비상정지 트리거 + dry-run/yaw옵션

main_window 는 레이아웃 조립 + 패널-컨트롤러 신호 라우팅 + 도구모드/프리셋/IO/
충돌다이얼로그/로그·상태바 담당.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .controller import SimulatorController
from .gl_view import Cube3DView
from .model import CubeModel
from .widgets import (
    LlmPanel,
    PlanTablePanel,
    PreparePanel,
    RunControlPanel,
    SelectedCubePanel,
)


class CubeSimulatorMainWindow(QMainWindow):

    def __init__(self,
                 dry_run_default: bool = True,
                 no_vision: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle('큐브 적층 시뮬레이터 — 두산 e0509')
        self.resize(1640, 820)

        self._model = CubeModel()
        self._view = Cube3DView(self._model)
        self._controller = SimulatorController(self._model, self._view, parent=self)
        self._no_vision = no_vision
        self._dry_run_default = dry_run_default

        # 컨트롤 패널을 2 컬럼으로 분할 — 디자인 = 화면 왼쪽, 실행 = 화면 오른쪽,
        # 3D 뷰는 가운데. 옵션이 많아 한 컬럼이 너무 길어지는 것 방지.
        self._design_panel = self._build_design_column()
        self._exec_panel = self._build_execution_column(dry_run_default)
        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        splitter.addWidget(self._design_panel)   # 왼쪽
        splitter.addWidget(self._view)            # 가운데
        splitter.addWidget(self._exec_panel)      # 오른쪽
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([340, 880, 420])
        self.setCentralWidget(splitter)

        self._status = QStatusBar()
        self._status_label = QLabel('준비')
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(220)
        self._status.addWidget(self._status_label, 1)
        self._status.addPermanentWidget(self._progress)
        self.setStatusBar(self._status)

        self._wire_signals()
        self._build_menus()
        self._populate_presets()
        self._refresh_count()
        self._selected_panel.show_cube(None)

    # --- 시그널 라우팅 --------------------------------------------------------
    def _wire_signals(self) -> None:
        c = self._controller
        c.status.connect(self._status_label.setText)
        c.error.connect(self._on_error)
        c.log.connect(self._log)
        c.progress.connect(self._on_progress)
        c.run_finished.connect(self._on_run_finished)
        c.sim_finished.connect(self._on_sim_finished)
        c.sim_progress.connect(self._on_sim_progress)
        c.align_finished.connect(self._on_align_finished)
        c.collision_decision_needed.connect(self._on_collision_decision)
        c.model_changed.connect(self._refresh_count)
        c.model_changed.connect(self._plan_panel.mark_dirty)
        c.selection_changed.connect(self._selected_panel.show_cube)
        c.plan_changed.connect(self._plan_panel.refresh)
        # cross-panel: 어느 한쪽에서 실행 시작하면 양쪽 패널 다 잠금 + progress 리셋
        self._prepare_panel.running_started.connect(self._on_any_run_started)
        self._run_panel.running_started.connect(self._on_any_run_started)

    def _on_any_run_started(self) -> None:
        """좌·우 패널 어느 쪽이 실행 시작해도 양쪽 다 잠금 + progress 리셋."""
        self._progress.setValue(0)
        self._prepare_panel.set_run_enabled(False)
        self._run_panel.set_run_enabled(False)

    # --- 컨트롤 패널 (2 컬럼 분할) ------------------------------------------
    def _build_design_column(self) -> QWidget:
        """왼쪽 컬럼 — 편집·디자인 (도구, 선택, 프리셋, IO, 큐브 카운트)."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 4, 8)
        v.setSpacing(8)

        # 도구 모드
        v.addWidget(self._section_label('도구'))
        h_tool = QHBoxLayout()
        self._tool_group = QButtonGroup(self)
        self._tool_add = QRadioButton('추가 (A)')
        self._tool_select = QRadioButton('선택 (S)')
        self._tool_erase = QRadioButton('지우개 (E)')
        self._tool_add.setChecked(True)
        for btn, mode in (
            (self._tool_add, 'add'),
            (self._tool_select, 'select'),
            (self._tool_erase, 'erase'),
        ):
            self._tool_group.addButton(btn)
            btn.toggled.connect(
                lambda checked, m=mode: checked and self._controller.set_tool_mode(m)
            )
            h_tool.addWidget(btn)
        v.addLayout(h_tool)

        snap_chk = QCheckBox('0.5 단위 스냅 (valley)')
        snap_chk.toggled.connect(self._view.set_snap_half)
        v.addWidget(snap_chk)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel('현재 layer:'))
        layer_spin = QSpinBox()
        layer_spin.setRange(0, 10)
        layer_spin.setValue(0)
        layer_spin.valueChanged.connect(self._view.set_current_layer)
        h2.addWidget(layer_spin)
        h2.addWidget(QLabel('  place yaw °:'))
        yaw_spin = QDoubleSpinBox()
        yaw_spin.setRange(-180.0, 180.0)
        yaw_spin.setSingleStep(15.0)
        yaw_spin.setValue(90.0)
        yaw_spin.valueChanged.connect(self._controller.set_next_yaw)
        h2.addWidget(yaw_spin)
        h2.addStretch(1)
        v.addLayout(h2)

        # 선택된 큐브 편집 패널
        self._selected_panel = SelectedCubePanel(self._controller, self._model)
        v.addWidget(self._selected_panel)

        # 프리셋 / 뷰 / IO
        v.addWidget(self._section_label('모양 프리셋'))
        h = QHBoxLayout()
        self._preset_combo = QComboBox()
        h.addWidget(self._preset_combo, 1)
        btn_load = QPushButton('불러오기')
        btn_load.clicked.connect(self._on_load_preset)
        h.addWidget(btn_load)
        v.addLayout(h)

        h4 = QHBoxLayout()
        btn_clear = QPushButton('전체 지우기')
        btn_clear.clicked.connect(self._controller.clear_model)
        h4.addWidget(btn_clear)
        btn_top = QPushButton('Top')
        btn_top.clicked.connect(self._view.top_view)
        h4.addWidget(btn_top)
        btn_iso = QPushButton('Iso')
        btn_iso.clicked.connect(self._view.iso_view)
        h4.addWidget(btn_iso)
        v.addLayout(h4)

        h5 = QHBoxLayout()
        btn_save = QPushButton('JSON 저장')
        btn_save.clicked.connect(self._on_save)
        h5.addWidget(btn_save)
        btn_open = QPushButton('JSON 불러오기')
        btn_open.clicked.connect(self._on_open)
        h5.addWidget(btn_open)
        v.addLayout(h5)

        self._count_label = QLabel('큐브 0개')
        self._count_label.setStyleSheet('color: #888; padding-top: 2px;')
        v.addWidget(self._count_label)

        # 캘리브레이션 + 바둑판 정렬 — 실 로봇 실행 전 '준비' 단계로 왼쪽 컬럼에 배치
        self._prepare_panel = PreparePanel(
            self._controller,
            dry_run_default=self._dry_run_default,
        )
        # 시작 신호는 main_window 가 통합 처리 (다른 컬럼의 패널도 같이 잠금).
        # 양쪽 패널이 모두 만들어진 후 _wire_signals 에서 connect.
        v.addWidget(self._prepare_panel)

        # 🤖 LLM 자연어 인터페이스 — 디자인 컬럼 맨 아래
        self._llm_panel = LlmPanel(self._controller)
        v.addWidget(self._llm_panel)

        v.addStretch(1)  # 위쪽 컴팩트 — 아래 빈공간
        return w

    def _build_execution_column(self, dry_run_default: bool) -> QWidget:
        """오른쪽 컬럼 — 계획·실행·로그 (plan 표, 정렬·시뮬·실행 옵션, 로그)."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 8, 8, 8)
        v.setSpacing(8)

        # 모션 계획 표 패널
        v.addWidget(self._section_label('모션 계획 (시뮬 전 좌표/각도 편집 가능)'))
        self._plan_panel = PlanTablePanel(self._controller, log_fn=self._log)
        v.addWidget(self._plan_panel)

        # 실행 / 시뮬 / 정렬 패널
        self._run_panel = RunControlPanel(
            self._controller,
            cube_count_fn=lambda: len(self._model.cubes),
            dry_run_default=dry_run_default,
        )
        # running_started 는 main_window._wire_signals 에서 통합 처리.
        v.addWidget(self._run_panel)

        # 로그
        v.addWidget(self._section_label('로그 / 좌표'))
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        self._log_view.setStyleSheet('font-family: monospace; font-size: 11px;')
        v.addWidget(self._log_view, 1)
        return w

    @staticmethod
    def _section_label(txt: str) -> QLabel:
        lbl = QLabel(txt)
        lbl.setStyleSheet('font-weight: bold; color: #ddd; padding-top: 4px;')
        return lbl

    # --- 메뉴 ----------------------------------------------------------------
    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu('&파일')
        act_open = QAction('JSON 불러오기...', self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._on_open)
        file_menu.addAction(act_open)
        act_save = QAction('JSON 저장...', self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._on_save)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction('종료', self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = bar.addMenu('&뷰')
        act_top = QAction('Top View', self)
        act_top.setShortcut('T')
        act_top.triggered.connect(self._view.top_view)
        view_menu.addAction(act_top)
        act_iso = QAction('Iso View', self)
        act_iso.setShortcut('I')
        act_iso.triggered.connect(self._view.iso_view)
        view_menu.addAction(act_iso)

        for key, btn in (('A', self._tool_add),
                         ('S', self._tool_select),
                         ('E', self._tool_erase)):
            act = QAction(f'모드 {key}', self)
            act.setShortcut(key)
            act.triggered.connect(lambda _=False, b=btn: b.setChecked(True))
            self.addAction(act)

        act_estop = QAction('비상정지', self)
        act_estop.setShortcut('Esc')
        act_estop.triggered.connect(self._controller.request_estop)
        self.addAction(act_estop)

        act_del = QAction('선택 큐브 삭제', self)
        act_del.setShortcut(QKeySequence.Delete)
        act_del.triggered.connect(self._controller.delete_selected)
        self.addAction(act_del)

    # --- 콜백 ----------------------------------------------------------------
    def _populate_presets(self) -> None:
        self._preset_combo.clear()
        names = self._controller.list_preset_names()
        if not names:
            self._preset_combo.addItem('(프리셋 로드 실패)')
            self._preset_combo.setEnabled(False)
            return
        for n in names:
            self._preset_combo.addItem(n)

    def _on_load_preset(self) -> None:
        name = self._preset_combo.currentText()
        if name:
            self._controller.load_preset(name)

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'JSON 저장', '', 'JSON (*.json)')
        if path:
            if not path.endswith('.json'):
                path += '.json'
            self._controller.save_json(path)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'JSON 불러오기', '', 'JSON (*.json)')
        if path:
            self._controller.load_json(path)

    def _on_collision_decision(self, cube_idx: int, msg: str) -> None:
        """SimAnimator 가 충돌 감지 → 사용자에게 보정 여부 묻기.

        다이얼로그 modal 동안 Esc 가 QMessageBox 의 reject 와 main_window 의
        비상정지 단축키 양쪽을 트리거할 수 있어 — 비상정지 액션을 잠깐 비활성화.
        """
        estop_action = None
        for act in self.actions():
            if act.shortcut().toString() == 'Esc':
                estop_action = act
                break
        prev_enabled = True
        if estop_action is not None:
            prev_enabled = estop_action.isEnabled()
            estop_action.setEnabled(False)
        try:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(f'⚠ 충돌 감지 — cube #{cube_idx + 1}')
            box.setText(msg)
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.Yes)
            box.setEscapeButton(QMessageBox.No)
            box.button(QMessageBox.Yes).setText('예 (90° 회전)')
            box.button(QMessageBox.No).setText('아니오 (그대로)')
            ans = box.exec_()
        finally:
            if estop_action is not None:
                estop_action.setEnabled(prev_enabled)
        self._controller.respond_collision_decision(ans == QMessageBox.Yes)

    def _on_align_finished(self, ok: bool, dets) -> None:
        # 정렬·캘리브 버튼은 왼쪽 PreparePanel 측. 시뮬/실행 잠금은 RunControlPanel 측.
        self._prepare_panel.set_run_enabled(True)
        if ok and dets:
            self._run_panel.check_cache_dets_after_align()
            self._log(
                f'=== 정렬 완료 ({len(dets)}개) — '
                f'[이전 비전 결과 재사용] 자동 체크 ==='
            )
        else:
            self._log('=== 정렬 중단/실패 ===')

    def _on_progress(self, idx: int, total: int, msg: str) -> None:
        pct = int(100 * (idx + 1) / max(1, total))
        self._progress.setValue(pct)
        self._status_label.setText(msg)

    def _on_run_finished(self, ok: bool) -> None:
        self._run_panel.set_run_enabled(True)
        self._prepare_panel.set_run_enabled(True)
        if ok:
            self._progress.setValue(100)
        self._log('--- 완료 ---' if ok else '--- 중단/오류 ---')

    def _on_sim_progress(self,
                         cube_idx: int, total: int,
                         phase: str, overall_u: float) -> None:
        pct = int(round(overall_u * 100))
        self._progress.setValue(max(0, min(100, pct)))
        self._status_label.setText(
            f'시뮬 [{cube_idx + 1}/{total}] {phase} ({pct}%)'
        )

    def _on_sim_finished(self, ok: bool) -> None:
        self._run_panel.set_run_enabled(True)
        self._prepare_panel.set_run_enabled(True)
        if ok:
            self._progress.setValue(100)
        self._log('=== 시뮬 완료 ===' if ok else '=== 시뮬 정지 ===')

    def _on_error(self, msg: str) -> None:
        self._log(f'[ERR] {msg}')
        self._status_label.setText(msg)

    def _log(self, msg: str) -> None:
        self._log_view.appendPlainText(msg)

    def _refresh_count(self) -> None:
        n = len(self._model.cubes)
        floating = sum(1 for c in self._model.cubes if self._model.is_floating(c))
        if floating:
            self._count_label.setText(f'큐브 {n}개 (받침 없음 {floating}개 ⚠)')
        else:
            self._count_label.setText(f'큐브 {n}개')

    # --- 종료 ----------------------------------------------------------------
    def closeEvent(self, ev) -> None:
        try:
            from .robot_worker import teardown_session, teardown_bringup
            teardown_session()
            teardown_bringup()
        except Exception:
            pass
        super().closeEvent(ev)
