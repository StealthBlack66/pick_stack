"""캘리브레이션 + 바둑판 정렬 — 실 로봇 실행 전 '준비' 단계.

화면 왼쪽 (디자인 컬럼) 에 배치. 사용자가 디자인 → 캘리브 → 정렬 → 오른쪽의
시뮬·실행으로 흐르도록 좌→우 순서로 진행.
"""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..controller import SimulatorController


class PreparePanel(QWidget):
    """캘리브 + 정렬 트리거 패널. 실행 시작·종료를 main_window 에 알림."""

    running_started = pyqtSignal()

    def __init__(self,
                 controller: SimulatorController,
                 dry_run_default: bool = True,
                 parent=None):
        super().__init__(parent)
        self._controller = controller

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        # 0) Hand-Eye 캘리브레이션 -------------------------------------------
        v.addWidget(self._section_label('0) Hand-Eye 캘리브레이션 (필요 시)'))
        cal_help = QLabel(
            '09_원샷_캘리브레이션.py 를 그대로 실행 — bringup + servo + 08 자동.\n'
            '※ ArUco 마커 (DICT_6X6_50, 50mm) 그리퍼 부착 필수.\n'
            "※ cv2 창에서 마커 잘 보이는 자세로 이동 후 's' 키.\n"
            '결과: calibration_data/ 폴더에 자동 저장.'
        )
        cal_help.setStyleSheet('color: #999; font-size: 10px;')
        cal_help.setWordWrap(True)
        v.addWidget(cal_help)

        self._cal_skip_launch_chk = QCheckBox('bringup launch SKIP (--skip-launch)')
        self._cal_skip_launch_chk.setToolTip(
            '체크 시 09 가 자체 bringup launch 안 함 — 이미 떠있는 경우.\n'
            '첫 캘리브엔 OFF, 재 캘리브 시 ON.'
        )
        v.addWidget(self._cal_skip_launch_chk)

        self._cal_keep_bringup_chk = QCheckBox('종료 시 bringup 살려둠 (--keep-bringup)')
        self._cal_keep_bringup_chk.setChecked(True)
        self._cal_keep_bringup_chk.setToolTip(
            '체크 시 캘리브 끝나도 bringup 안 죽임 — 이어서 정렬·실행 가능.'
        )
        v.addWidget(self._cal_keep_bringup_chk)

        self._cal_btn = QPushButton('🎯 원샷 캘리브레이션 (09번 실행)')
        self._cal_btn.setStyleSheet('padding: 5px;')
        self._cal_btn.clicked.connect(self._on_calibrate)
        v.addWidget(self._cal_btn)

        # 1) 바둑판 정렬 -----------------------------------------------------
        v.addWidget(self._section_label('1) 바둑판 정렬'))
        align_help = QLabel(
            '15_바둑판_정렬.py 를 subprocess 로 실행.\n'
            '※ dsr_bringup2 는 자동으로 띄움.\n'
            '실 모션 조건: Dry-run 해제 + 컨트롤러/네트워크 OK.'
        )
        align_help.setStyleSheet('color: #999; font-size: 10px;')
        align_help.setWordWrap(True)
        v.addWidget(align_help)

        self._align_dry_chk = QCheckBox('Dry-run (정렬 — 모션/그리퍼 SKIP)')
        self._align_dry_chk.setChecked(dry_run_default)
        self._align_dry_chk.setToolTip(
            '체크 시 15번 --dry-run — 비전·노드·HOME 은 시도하지만\n'
            '실제 모션·gripper init·activate 는 SKIP.'
        )
        v.addWidget(self._align_dry_chk)

        self._align_quick_chk = QCheckBox('드라이버 리셋 SKIP (--quick)')
        self._align_quick_chk.setChecked(True)
        self._align_quick_chk.setToolTip(
            '체크 시 reset_robot_driver() 안 함 — bringup 이 이미 정상 동작 중.\n'
            '첫 연결이거나 driver stuck 시 해제 (약 25초 reset wait).'
        )
        v.addWidget(self._align_quick_chk)

        self._align_btn = QPushButton('▶ 바둑판 정렬 (15번 main 실행)')
        self._align_btn.setStyleSheet('padding: 5px;')
        self._align_btn.clicked.connect(self._on_align)
        v.addWidget(self._align_btn)

    # --- 외부 호출 ---------------------------------------------------------
    def set_run_enabled(self, enabled: bool) -> None:
        """실행 중일 때 cal/align 버튼 잠금."""
        self._cal_btn.setEnabled(enabled)
        self._align_btn.setEnabled(enabled)

    def check_cache_dets_after_align_called_by_main_window(self) -> None:
        """정렬 성공 후 main_window 가 RunControlPanel.check_cache_dets...
        호출 — PreparePanel 자체는 cache_dets 체크박스 없음 (오른쪽 RunControlPanel 측에)."""
        pass

    # --- 콜백 -------------------------------------------------------------
    def _on_calibrate(self) -> None:
        if self._controller.is_running():
            QMessageBox.information(self, '캘리브레이션', '이미 실행 중')
            return
        ans = QMessageBox.question(
            self, 'Hand-Eye 캘리브레이션',
            '실 로봇으로 자동 캘리브레이션을 시작합니다.\n\n'
            '준비:\n'
            '  ① ArUco 마커 (DICT_6X6_50, 50mm) 그리퍼 부착\n'
            '  ② 마커 보이는 안전 자세로 미리 이동\n'
            '  ③ 작업 공간 ±200mm 여유\n'
            '  ④ 비상정지 손에 닿는 위치\n\n'
            '진행할까요?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        ok = self._controller.start_oneshot_calibration(
            skip_launch=self._cal_skip_launch_chk.isChecked(),
            keep_bringup=self._cal_keep_bringup_chk.isChecked(),
        )
        if ok:
            self.set_run_enabled(False)
            self.running_started.emit()

    def _on_align(self) -> None:
        if self._controller.is_running():
            QMessageBox.information(self, '정렬', '이미 실행 중')
            return
        dry = self._align_dry_chk.isChecked()
        quick = self._align_quick_chk.isChecked()
        if not dry:
            ans = QMessageBox.question(
                self, '바둑판 정렬',
                '실 로봇으로 비전 검출 → 큐브 정렬을 수행합니다.\n'
                'dsr_bringup2 가 떠 있어야 합니다. 진행할까요?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        ok = self._controller.start_alignment(dry_run=dry, quick=quick)
        if ok:
            self.set_run_enabled(False)
            self.running_started.emit()

    @staticmethod
    def _section_label(txt: str) -> QLabel:
        lbl = QLabel(txt)
        lbl.setStyleSheet('font-weight: bold; color: #ddd; padding-top: 4px;')
        return lbl
