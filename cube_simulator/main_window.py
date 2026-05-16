"""메인 윈도우 — 3D 뷰 + 컨트롤 패널 + 상태바."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
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
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .controller import SimulatorController
from .gl_view import Cube3DView
from .model import CubeModel


class CubeSimulatorMainWindow(QMainWindow):

    def __init__(self,
                 dry_run_default: bool = True,
                 no_vision: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle('큐브 적층 시뮬레이터 — 두산 e0509')
        self.resize(1280, 760)

        # 모델 / 뷰 / 컨트롤러
        self._model = CubeModel()
        self._view = Cube3DView(self._model)
        self._controller = SimulatorController(self._model, self._view, parent=self)
        self._no_vision = no_vision

        # 컨트롤 패널
        self._panel = self._build_control_panel(dry_run_default)
        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        splitter.addWidget(self._view)
        splitter.addWidget(self._panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([920, 360])
        self.setCentralWidget(splitter)

        # 상태바 + 로그
        self._status = QStatusBar()
        self._status_label = QLabel('준비')
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(220)
        self._status.addWidget(self._status_label, 1)
        self._status.addPermanentWidget(self._progress)
        self.setStatusBar(self._status)

        # 시그널 연결
        self._controller.status.connect(self._status_label.setText)
        self._controller.error.connect(self._on_error)
        self._controller.log.connect(self._log)
        self._controller.progress.connect(self._on_progress)
        self._controller.run_finished.connect(self._on_run_finished)
        self._controller.model_changed.connect(self._refresh_count)

        # 메뉴 / 단축키
        self._build_menus()

        # 프리셋 채우기
        self._populate_presets()
        self._refresh_count()

    # --- 컨트롤 패널 ----------------------------------------------------------
    def _build_control_panel(self, dry_run_default: bool) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        # 모양 프리셋
        v.addWidget(self._section_label('1) 모양 프리셋'))
        h = QHBoxLayout()
        self._preset_combo = QComboBox()
        h.addWidget(self._preset_combo, 1)
        btn_load = QPushButton('불러오기')
        btn_load.clicked.connect(self._on_load_preset)
        h.addWidget(btn_load)
        v.addLayout(h)

        # 편집 도구
        v.addWidget(self._section_label('2) 편집'))
        self._erase_chk = QCheckBox('지우개 모드 (E)')
        self._erase_chk.toggled.connect(self._controller.set_erase_mode)
        v.addWidget(self._erase_chk)

        self._snap_chk = QCheckBox('0.5 단위 스냅 (valley)')
        self._snap_chk.toggled.connect(self._view.set_snap_half)
        v.addWidget(self._snap_chk)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel('현재 layer:'))
        self._layer_spin = QSpinBox()
        self._layer_spin.setRange(0, 10)
        self._layer_spin.setValue(0)
        self._layer_spin.valueChanged.connect(self._view.set_current_layer)
        h2.addWidget(self._layer_spin)
        h2.addStretch(1)
        v.addLayout(h2)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel('place yaw °:'))
        self._yaw_spin = QDoubleSpinBox()
        self._yaw_spin.setRange(-180.0, 180.0)
        self._yaw_spin.setSingleStep(15.0)
        self._yaw_spin.setValue(90.0)
        self._yaw_spin.valueChanged.connect(self._controller.set_next_yaw)
        h3.addWidget(self._yaw_spin)
        h3.addStretch(1)
        v.addLayout(h3)

        h4 = QHBoxLayout()
        btn_clear = QPushButton('전체 지우기')
        btn_clear.clicked.connect(self._controller.clear_model)
        h4.addWidget(btn_clear)
        btn_top = QPushButton('Top View')
        btn_top.clicked.connect(self._view.top_view)
        h4.addWidget(btn_top)
        btn_iso = QPushButton('Iso View')
        btn_iso.clicked.connect(self._view.iso_view)
        h4.addWidget(btn_iso)
        v.addLayout(h4)

        # 저장 / 불러오기
        v.addWidget(self._section_label('3) 저장 / 불러오기'))
        h5 = QHBoxLayout()
        btn_save = QPushButton('JSON 저장')
        btn_save.clicked.connect(self._on_save)
        h5.addWidget(btn_save)
        btn_open = QPushButton('JSON 불러오기')
        btn_open.clicked.connect(self._on_open)
        h5.addWidget(btn_open)
        v.addLayout(h5)

        # 실행
        v.addWidget(self._section_label('4) 로봇 실행'))
        self._dry_chk = QCheckBox('Dry-run (모션 SKIP)')
        self._dry_chk.setChecked(dry_run_default)
        v.addWidget(self._dry_chk)

        self._cache_dets_chk = QCheckBox('이전 비전 결과 재사용')
        self._cache_dets_chk.setChecked(False)
        v.addWidget(self._cache_dets_chk)

        h6 = QHBoxLayout()
        btn_vision = QPushButton('비전 캡처')
        btn_vision.clicked.connect(self._on_capture_vision)
        if self._no_vision:
            btn_vision.setEnabled(False)
            btn_vision.setToolTip('--no-vision 모드')
        h6.addWidget(btn_vision)
        btn_clear_dets = QPushButton('검출 비우기')
        btn_clear_dets.clicked.connect(self._controller.clear_dets_cache)
        h6.addWidget(btn_clear_dets)
        v.addLayout(h6)

        self._run_btn = QPushButton('▶ 로봇 실행')
        self._run_btn.setStyleSheet('font-weight: bold; padding: 8px;')
        self._run_btn.clicked.connect(self._on_run)
        v.addWidget(self._run_btn)

        self._estop_btn = QPushButton('■ 비상정지 (Esc)')
        self._estop_btn.setStyleSheet('color: white; background: #b03030; padding: 6px;')
        self._estop_btn.clicked.connect(self._controller.request_estop)
        v.addWidget(self._estop_btn)

        # 카운트
        self._count_label = QLabel('큐브 0개')
        self._count_label.setStyleSheet('color: #888; padding-top: 4px;')
        v.addWidget(self._count_label)

        # 로그
        v.addWidget(self._section_label('5) 로그'))
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        self._log_view.setStyleSheet('font-family: monospace; font-size: 11px;')
        v.addWidget(self._log_view, 1)
        return w

    def _section_label(self, txt: str) -> QLabel:
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

        # Esc = 비상정지
        act_estop = QAction('비상정지', self)
        act_estop.setShortcut('Esc')
        act_estop.triggered.connect(self._controller.request_estop)
        self.addAction(act_estop)

        # E = 지우개 토글
        act_erase = QAction('지우개 토글', self)
        act_erase.setShortcut('E')
        act_erase.triggered.connect(
            lambda: self._erase_chk.setChecked(not self._erase_chk.isChecked())
        )
        self.addAction(act_erase)

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
        if not name:
            return
        self._controller.load_preset(name)

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, 'JSON 저장', '', 'JSON (*.json)'
        )
        if path:
            if not path.endswith('.json'):
                path += '.json'
            self._controller.save_json(path)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'JSON 불러오기', '', 'JSON (*.json)'
        )
        if path:
            self._controller.load_json(path)

    def _on_capture_vision(self) -> None:
        if self._controller.is_running():
            QMessageBox.warning(self, '비전 캡처', '워커 실행 중에는 캡처 불가')
            return
        # 일회성 캡처: dry-run + use_cached_dets=False, 워커 시작 후 finished 시 dets 만 가져오기
        # — 단순화: 별도 캡처 전용 함수가 없으므로, MVP 에서는 운영 안내만 남기고
        #   실제 실행 시 --dry-run 으로 비전+로봇 시뮬레이션 흐름 자체로 검증
        self._log('비전 캡처는 [로봇 실행] 시 자동 수행 (재사용 체크 시 캐시 사용)')

    def _on_run(self) -> None:
        if self._controller.is_running():
            QMessageBox.information(self, '실행', '이미 실행 중')
            return
        dry = self._dry_chk.isChecked()
        use_cached = self._cache_dets_chk.isChecked()
        if not dry:
            ans = QMessageBox.question(
                self, '실 로봇 실행',
                f'실 로봇 동작 — 큐브 {len(self._model.cubes)}개 배치합니다.\n진행할까요?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        self._progress.setValue(0)
        ok = self._controller.start_run(dry_run=dry, use_cached_dets=use_cached)
        if ok:
            self._run_btn.setEnabled(False)

    def _on_progress(self, idx: int, total: int, msg: str) -> None:
        pct = int(100 * (idx + 1) / max(1, total))
        self._progress.setValue(pct)
        self._status_label.setText(msg)

    def _on_run_finished(self, ok: bool) -> None:
        self._run_btn.setEnabled(True)
        if ok:
            self._progress.setValue(100)
        self._log('--- 완료 ---' if ok else '--- 중단/오류 ---')

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
        # 세션 (PickAndPlace 노드 + rclpy) 1회 정리
        try:
            from .robot_worker import teardown_session
            teardown_session()
        except Exception:
            pass
        super().closeEvent(ev)
