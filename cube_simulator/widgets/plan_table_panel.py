"""모션 계획 (plan) 표시·편집 패널.

- [계획 갱신] 버튼 → controller.rebuild_plan
- [yaw 충돌 보정] 버튼 → controller.apply_yaw_corrections
- 표 셀 직접 편집 → controller.update_plan_cell
- controller.plan_changed → refresh()
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..controller import SimulatorController


# 표 열 → PlanItem 의 어떤 필드를 편집하는지. None = 편집 불가.
PLAN_COL_FIELDS = (
    None, None,  # 0: #, 1: 큐브 (gx,gy,L)
    'src_x', 'src_y', 'src_yaw',
    'tgt_x', 'tgt_y', 'tgt_z', 'tgt_yaw',
)


class PlanTablePanel(QWidget):
    """모션 계획 표 + 갱신/보정 버튼 + dirty 표시."""

    def __init__(self,
                 controller: SimulatorController,
                 log_fn=None,
                 parent=None):
        """log_fn: 표 입력 오류 등을 로그 패널에 보낼 콜백 (str)."""
        super().__init__(parent)
        self._controller = controller
        self._log = log_fn or (lambda _msg: None)
        self._updating = False  # plan_changed rebuild 중 cellChanged 차단

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        h_btn = QHBoxLayout()
        self._refresh_btn = QPushButton('계획 갱신 (모델→표)')
        self._refresh_btn.setToolTip(
            '현재 모델 상태로 계획을 다시 빌드. 표에 직접 입력한 값은 덮어씁니다.'
        )
        self._refresh_btn.clicked.connect(self._controller.rebuild_plan)
        h_btn.addWidget(self._refresh_btn)
        self._yaw_btn = QPushButton('yaw 충돌 보정')
        self._yaw_btn.setToolTip(
            '인접 큐브와 finger 충돌이 예상되는 yaw 를 ±90° 회전.\n'
            '명시 호출 — 시뮬은 보정 안 함. 사용자가 원할 때만 적용.'
        )
        self._yaw_btn.clicked.connect(self._controller.apply_yaw_corrections)
        h_btn.addWidget(self._yaw_btn)
        v.addLayout(h_btn)

        self._dirty_label = QLabel('')
        self._dirty_label.setStyleSheet('color: #d8a040; padding-left: 2px;')
        v.addWidget(self._dirty_label)

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels([
            '#', '큐브 (gx,gy,L)',
            'src x', 'src y', 'src yaw°',
            'tgt x', 'tgt y', 'tgt z', 'tgt yaw°',
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet('font-family: monospace; font-size: 11px;')
        hdr = self._table.horizontalHeader()
        for col in range(self._table.columnCount()):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._table.setMinimumHeight(180)
        self._table.cellChanged.connect(self._on_cell_changed)
        v.addWidget(self._table)

    # --- 외부에서 호출하는 슬롯 -------------------------------------------------
    def mark_dirty(self) -> None:
        """controller.model_changed 시 — 다음 시뮬 전 갱신 권장 알림."""
        self._dirty_label.setText(
            '⚠ 모델이 바뀜 — [계획 갱신] 또는 [시뮬레이션] 시 자동 갱신'
        )

    def refresh(self, plan) -> None:
        """controller.plan_changed signal handler — 표 통째 재구성."""
        self._dirty_label.setText('')
        self._updating = True
        try:
            self._table.setRowCount(len(plan))
            for r, item in enumerate(plan):
                cells = [
                    f'{r+1}',
                    f'({item.cube_gx:+.1f},{item.cube_gy:+.1f}) L{item.cube_layer}',
                    f'{item.src_x:.1f}', f'{item.src_y:.1f}', f'{item.src_yaw:+.1f}',
                    f'{item.tgt_x:.1f}', f'{item.tgt_y:.1f}', f'{item.tgt_z:.1f}',
                    f'{item.tgt_yaw:+.1f}',
                ]
                for c, text in enumerate(cells):
                    cell = QTableWidgetItem(text)
                    if PLAN_COL_FIELDS[c] is None:
                        cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                        cell.setForeground(Qt.gray)
                    else:
                        cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self._table.setItem(r, c, cell)
        finally:
            self._updating = False

    # --- 내부 ------------------------------------------------------------------
    def _on_cell_changed(self, row: int, col: int) -> None:
        if self._updating:
            return
        field_name = PLAN_COL_FIELDS[col] if col < len(PLAN_COL_FIELDS) else None
        if field_name is None:
            return
        item = self._table.item(row, col)
        if item is None:
            return
        text = item.text().strip()
        try:
            v = float(text)
        except ValueError:
            self._log(f'[표] 잘못된 숫자: {text!r}')
            # 원래 값 복구
            self.refresh(self._controller.get_plan())
            return
        self._controller.update_plan_cell(row, field_name, v)
        # 셀 텍스트는 사용자가 친 그대로 둠 (재포맷 setText 는 row rebuild 와
        # 경합해 dangling C++ ref 가 될 수 있음)
