"""선택된 큐브 속성 편집 패널 (gx/gy/layer/yaw + 삭제/해제 버튼).

`SimulatorController` 와만 통신 (view 직접 의존 없음).
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..controller import SimulatorController
from ..model import CubeModel, PlacedCube


class SelectedCubePanel(QGroupBox):
    """선택된 큐브의 좌표/yaw 를 표시·편집.

    controller.selection_changed → show_cube() 로 동기화.
    spinbox 값 변경 → controller.update_selected() 호출.
    """

    def __init__(self,
                 controller: SimulatorController,
                 model: CubeModel,
                 parent=None):
        super().__init__('선택된 큐브', parent)
        self._controller = controller
        self._model = model
        self._updating = False  # selection_changed 갱신 중 spinbox callback 차단

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(4)

        self._info_label = QLabel('— 없음')
        self._info_label.setStyleSheet('color: #aaa;')
        layout.addWidget(self._info_label)

        h_pos = QHBoxLayout()
        h_pos.addWidget(QLabel('gx:'))
        self._gx = QDoubleSpinBox()
        self._gx.setRange(-10.0, 10.0)
        self._gx.setSingleStep(0.5)
        self._gx.setDecimals(2)
        self._gx.valueChanged.connect(self._on_gx)
        h_pos.addWidget(self._gx)
        h_pos.addWidget(QLabel('gy:'))
        self._gy = QDoubleSpinBox()
        self._gy.setRange(-10.0, 10.0)
        self._gy.setSingleStep(0.5)
        self._gy.setDecimals(2)
        self._gy.valueChanged.connect(self._on_gy)
        h_pos.addWidget(self._gy)
        layout.addLayout(h_pos)

        h_lyy = QHBoxLayout()
        h_lyy.addWidget(QLabel('layer:'))
        self._layer = QSpinBox()
        self._layer.setRange(0, 15)
        self._layer.valueChanged.connect(self._on_layer)
        h_lyy.addWidget(self._layer)
        h_lyy.addWidget(QLabel('yaw °:'))
        self._yaw = QDoubleSpinBox()
        self._yaw.setRange(-180.0, 180.0)
        self._yaw.setSingleStep(15.0)
        self._yaw.setDecimals(1)
        self._yaw.valueChanged.connect(self._on_yaw)
        h_lyy.addWidget(self._yaw)
        layout.addLayout(h_lyy)

        self._mm_label = QLabel('mm: —')
        self._mm_label.setStyleSheet('color: #bbb; font-family: monospace;')
        layout.addWidget(self._mm_label)

        h_btn = QHBoxLayout()
        self._clear_btn = QPushButton('선택 해제')
        self._clear_btn.clicked.connect(self._controller.clear_selection)
        h_btn.addWidget(self._clear_btn)
        self._delete_btn = QPushButton('삭제')
        self._delete_btn.clicked.connect(self._on_delete)
        h_btn.addWidget(self._delete_btn)
        layout.addLayout(h_btn)

        for wgt in (self._gx, self._gy, self._layer, self._yaw,
                    self._clear_btn, self._delete_btn):
            wgt.setEnabled(False)

    def show_cube(self, cube) -> None:
        """controller.selection_changed signal handler."""
        self._updating = True
        try:
            enabled = isinstance(cube, PlacedCube)
            for wgt in (self._gx, self._gy, self._layer, self._yaw,
                        self._clear_btn, self._delete_btn):
                wgt.setEnabled(enabled)
            if not enabled:
                self._info_label.setText('— 없음')
                self._mm_label.setText('mm: —')
                return
            self._info_label.setText(
                f'({cube.gx:+.1f}, {cube.gy:+.1f}) L{cube.layer}'
            )
            self._gx.setValue(float(cube.gx))
            self._gy.setValue(float(cube.gy))
            self._layer.setValue(int(cube.layer))
            self._yaw.setValue(float(cube.yaw_deg))
            mx = self._model.base_xy[0] + cube.gx * self._model.pitch_mm
            my = self._model.base_xy[1] + cube.gy * self._model.pitch_mm
            self._mm_label.setText(
                f'mm: ({mx:.1f}, {my:.1f})  pitch={self._model.pitch_mm:.1f} '
                f'cube={self._model.cube_width_mm:.1f}'
            )
        finally:
            self._updating = False

    def _on_gx(self, v: float) -> None:
        if not self._updating:
            self._controller.update_selected(new_gx=v)

    def _on_gy(self, v: float) -> None:
        if not self._updating:
            self._controller.update_selected(new_gy=v)

    def _on_layer(self, v: int) -> None:
        if not self._updating:
            self._controller.update_selected(new_layer=int(v))

    def _on_yaw(self, v: float) -> None:
        if not self._updating:
            self._controller.update_selected(new_yaw=v)

    def _on_delete(self) -> None:
        self._controller.delete_selected()
