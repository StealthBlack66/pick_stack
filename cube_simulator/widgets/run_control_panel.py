"""실행/시뮬레이션/비상정지 + 정렬 트리거 컨트롤 패널.

- 1) 바둑판 정렬 — 15_바둑판_정렬.py subprocess 실행
- 2) 시뮬 / 실 로봇 실행 + dry-run / cached dets / auto-yaw 옵션
- 비상정지 버튼

`SimulatorController` 와만 통신. 다이얼로그(Yes/No 확인) 는 panel 안에서 처리.
실행 시작 시 _running_started signal emit → main_window 가 다른 UI 잠금에 사용.
"""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..controller import SimulatorController


class RunControlPanel(QWidget):
    """정렬·시뮬·실행·비상정지 통합 컨트롤.

    Signals:
        running_started(): 정렬/실행/시뮬 중 하나가 시작됐을 때
        running_finished(): 동일 작업 종료 시 (run_finished/sim_finished/align_finished
                            를 main_window 가 라우팅)
    """

    running_started = pyqtSignal()
    running_finished = pyqtSignal()

    def __init__(self,
                 controller: SimulatorController,
                 cube_count_fn,
                 dry_run_default: bool = True,
                 parent=None):
        """cube_count_fn: 실 로봇 실행 확인 다이얼로그에 표시할 큐브 개수 콜백."""
        super().__init__(parent)
        self._controller = controller
        self._cube_count_fn = cube_count_fn

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        # 캘리브레이션 + 정렬은 왼쪽 디자인 컬럼의 PreparePanel 로 이동했음.
        # 여기는 시뮬·실행·복귀 + 옵션만.

        # 2) 시뮬 / 실행 -----------------------------------------------------
        v.addWidget(self._section_label('2) 실행 / 시뮬레이션'))
        self._dry_chk = QCheckBox('Dry-run (실 로봇 모션 SKIP)')
        self._dry_chk.setChecked(dry_run_default)
        v.addWidget(self._dry_chk)

        self._cache_dets_chk = QCheckBox('이전 비전 결과 재사용 (정렬 후 추천)')
        v.addWidget(self._cache_dets_chk)

        # 비전 conf 임계값 — 검출이 안 되면 낮추고 (예: 0.20), 과검이면 올림 (0.60)
        h_conf = QHBoxLayout()
        h_conf.addWidget(QLabel('비전 conf 임계값:'))
        self._vision_conf = QDoubleSpinBox()
        self._vision_conf.setRange(0.05, 0.95)
        self._vision_conf.setSingleStep(0.05)
        self._vision_conf.setDecimals(2)
        self._vision_conf.setValue(0.40)
        self._vision_conf.setToolTip(
            'YOLO 세그멘테이션 검출 conf threshold. 검출 안 되면 낮춤 (0.20), '
            '과검이면 올림 (0.60). 이전 비전 결과 재사용 시엔 영향 없음.'
        )
        h_conf.addWidget(self._vision_conf)
        h_conf.addStretch(1)
        v.addLayout(h_conf)

        # placement z offset — 큐브 내려놓을 때 sample_z 기반 target_z 에 더할 mm.
        # 음수 = 더 깊이. 큐브가 공중에 떠 있거나 너무 박혀 들어가면 여기서 조정.
        h_z = QHBoxLayout()
        h_z.addWidget(QLabel('placement z offset (mm):'))
        self._place_z_offset = QDoubleSpinBox()
        self._place_z_offset.setRange(-30.0, 30.0)
        self._place_z_offset.setSingleStep(1.0)
        self._place_z_offset.setDecimals(1)
        self._place_z_offset.setValue(-10.0)
        self._place_z_offset.setToolTip(
            '실 로봇이 큐브를 놓을 때 target_z 에 더할 보정 (mm).\n'
            '음수 = 더 깊이 내려감. 큐브가 공중에 떠 있으면 더 음수 (-15 등),\n'
            '큐브가 너무 박혀들면 0 근처 또는 양수.'
        )
        h_z.addWidget(self._place_z_offset)
        h_z.addStretch(1)
        v.addLayout(h_z)

        # pick z offset — 큐브 잡을 때 sample_z 기반 src_z 에 더할 mm.
        # 기본 0 = p16 의 원래 동작 (cube 중심까지 = 12.5mm 깊이로 클램핑).
        h_pick_z = QHBoxLayout()
        h_pick_z.addWidget(QLabel('pick z offset (mm):'))
        self._pick_z_offset = QDoubleSpinBox()
        self._pick_z_offset.setRange(-20.0, 10.0)
        self._pick_z_offset.setSingleStep(1.0)
        self._pick_z_offset.setDecimals(1)
        self._pick_z_offset.setValue(0.0)
        self._pick_z_offset.setToolTip(
            '실 로봇이 큐브를 집을 때 src_z 에 더할 보정 (mm).\n'
            '기본 0 = cube 중심까지만 (12.5mm 깊이) — 안전한 원래 동작.\n'
            '큐브가 미끄러지면 음수로 (-3 ~ -5),\n'
            '비전 z 가 낮게 잡혀 너무 깊으면 양수로 (+2 ~ +5).\n'
            '너무 깊으면 finger 가 테이블에 충돌 위험 — 자동 clamp 됨.'
        )
        h_pick_z.addWidget(self._pick_z_offset)
        h_pick_z.addStretch(1)
        v.addLayout(h_pick_z)

        # z_table_top override — 비전 sample_z 가 불안정하면 실측치 입력해 충돌 방지.
        # 체크박스 ON: spinbox 값 사용, OFF: 자동 (sample_z - cube_w)
        h_table = QHBoxLayout()
        self._table_z_override_chk = QCheckBox('table z 수동:')
        self._table_z_override_chk.setToolTip(
            '체크 시 실 로봇이 사용할 테이블 표면 z (mm, robot base frame) 를 \n'
            '아래 값으로 강제. 비전 sample_z 가 매 정렬마다 크게 변하거나 \n'
            'finger 가 바닥 부딪칠 때 캘리브된 실측치 입력.\n'
            '체크 해제 시 sample_z - cube_w 로 자동 추정.'
        )
        h_table.addWidget(self._table_z_override_chk)
        self._table_z_value = QDoubleSpinBox()
        self._table_z_value.setRange(-300.0, 100.0)
        self._table_z_value.setSingleStep(1.0)
        self._table_z_value.setDecimals(1)
        self._table_z_value.setValue(-30.0)  # 사용자 실측 — 두산 base frame 의 테이블 표면
        self._table_z_override_chk.setChecked(True)  # 기본 ON — 비전 sample_z 변동 회피
        self._table_z_value.setEnabled(True)
        self._table_z_override_chk.toggled.connect(self._table_z_value.setEnabled)
        h_table.addWidget(self._table_z_value)
        h_table.addWidget(QLabel('mm'))
        h_table.addStretch(1)
        v.addLayout(h_table)

        # pick 안전 마진 — finger tip 이 추정 테이블 표면 위로 이만큼 이상 유지.
        # 위반 시 자동 clamp + 경고 로그.
        h_clr = QHBoxLayout()
        h_clr.addWidget(QLabel('pick 안전 마진 (mm):'))
        self._pick_min_clearance = QDoubleSpinBox()
        self._pick_min_clearance.setRange(0.0, 30.0)
        self._pick_min_clearance.setSingleStep(1.0)
        self._pick_min_clearance.setDecimals(1)
        self._pick_min_clearance.setValue(5.0)
        self._pick_min_clearance.setToolTip(
            'finger tip 이 추정 테이블 표면 위로 최소 이만큼 유지.\n'
            'pick_z 가 (table_z + clearance) 보다 깊으면 자동으로 그 floor 로 clamp.\n'
            '5mm 가 안전한 기본값. cube 가 살짝 떠 있어 안 잡히면 0~2 로 낮춤.'
        )
        h_clr.addWidget(self._pick_min_clearance)
        h_clr.addStretch(1)
        v.addLayout(h_clr)

        self._sim_auto_yaw_chk = QCheckBox('시뮬 시 yaw 자동 보정 (충돌 회피)')
        self._sim_auto_yaw_chk.setToolTip(
            '체크 시 시뮬 시작 직전에 finger 충돌 회피로 plan 의 yaw 를 ±90° 보정.\n'
            '체크 해제 시 표의 yaw 그대로 사용 (사용자 편집 보존).'
        )
        self._sim_auto_yaw_chk.setChecked(True)
        v.addWidget(self._sim_auto_yaw_chk)

        self._sim_btn = QPushButton('▶ 시뮬레이션 (가상 그리퍼)')
        self._sim_btn.setStyleSheet('padding: 6px; font-weight: bold;')
        self._sim_btn.setToolTip('표의 좌표/각도 그대로 가상 그리퍼가 픽앤플레이스')
        self._sim_btn.clicked.connect(self._on_sim)
        v.addWidget(self._sim_btn)

        self._run_btn = QPushButton('▶ 로봇 실행')
        self._run_btn.setStyleSheet('padding: 6px; font-weight: bold;')
        self._run_btn.clicked.connect(self._on_run)
        v.addWidget(self._run_btn)

        self._estop_btn = QPushButton('■ 비상정지 (Esc)')
        self._estop_btn.setStyleSheet(
            'color: white; background: #b03030; padding: 5px;'
        )
        self._estop_btn.clicked.connect(self._controller.request_estop)
        v.addWidget(self._estop_btn)

        # 원점복귀 — trajectory 실패/safe-off/어디 박혔는지 모를 때 명시 복구
        self._recover_btn = QPushButton('🏠 원점복귀 (recovery)')
        self._recover_btn.setStyleSheet(
            'color: white; background: #2a6e8a; padding: 5px;'
        )
        self._recover_btn.setToolTip(
            'recover_safety + activate + HOME 자세 이동.\n'
            'trajectory 실패 / SAFE_OFF / 로봇이 중간에 멈췄을 때 사용.'
        )
        self._recover_btn.clicked.connect(self._on_recover)
        v.addWidget(self._recover_btn)

    # --- 외부에서 호출하는 슬롯 -------------------------------------------------
    def set_run_enabled(self, enabled: bool) -> None:
        """실행 중일 때 run/sim/recover 버튼 잠금 (estop 은 항상 활성)."""
        self._run_btn.setEnabled(enabled)
        self._sim_btn.setEnabled(enabled)
        self._recover_btn.setEnabled(enabled)

    def get_pick_z_offset_mm(self) -> float:
        return self._pick_z_offset.value()

    def get_place_z_offset_mm(self) -> float:
        return self._place_z_offset.value()

    def check_cache_dets_after_align(self) -> None:
        """정렬 성공 후 자동으로 [이전 비전 결과 재사용] 체크."""
        self._cache_dets_chk.setChecked(True)

    # --- 콜백 -----------------------------------------------------------------
    def _on_run(self) -> None:
        if self._controller.is_running():
            QMessageBox.information(self, '실행', '이미 실행 중')
            return
        dry = self._dry_chk.isChecked()
        use_cached = self._cache_dets_chk.isChecked()
        if not dry:
            ans = QMessageBox.question(
                self, '실 로봇 실행',
                f'실 로봇 동작 — 큐브 {self._cube_count_fn()}개 배치합니다.\n'
                '진행할까요?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        z_table = (self._table_z_value.value()
                   if self._table_z_override_chk.isChecked() else None)
        ok = self._controller.start_run(
            dry_run=dry,
            use_cached_dets=use_cached,
            vision_conf=self._vision_conf.value(),
            place_z_offset_mm=self._place_z_offset.value(),
            pick_z_offset_mm=self._pick_z_offset.value(),
            z_table_top_override_mm=z_table,
            pick_min_clearance_mm=self._pick_min_clearance.value(),
        )
        if ok:
            self.set_run_enabled(False)
            self.running_started.emit()

    def _on_sim(self) -> None:
        if self._controller.is_simulating():
            QMessageBox.information(self, '시뮬', '이미 시뮬레이션 중')
            return
        auto = self._sim_auto_yaw_chk.isChecked()
        ok = self._controller.start_simulation(auto_correct=auto)
        if ok:
            self.set_run_enabled(False)
            self.running_started.emit()

    def _on_recover(self) -> None:
        if self._controller.is_running():
            QMessageBox.information(self, '원점복귀', '이미 실행 중')
            return
        dry = self._dry_chk.isChecked()
        if not dry:
            ans = QMessageBox.question(
                self, '원점복귀',
                '로봇을 HOME 자세로 이동시킵니다.\n'
                'recover_safety + activate + HOME 순으로 진행.\n'
                '진행할까요?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if ans != QMessageBox.Yes:
                return
        ok = self._controller.start_recover_home(dry_run=dry)
        if ok:
            self.set_run_enabled(False)
            self._recover_btn.setEnabled(False)
            self.running_started.emit()

    @staticmethod
    def _section_label(txt: str) -> QLabel:
        lbl = QLabel(txt)
        lbl.setStyleSheet('font-weight: bold; color: #ddd; padding-top: 4px;')
        return lbl
