"""🤖 LLM 어시스턴트 패널 — 자연어 / 분석 / vision / (음성) 4 모드.

왼쪽 디자인 컬럼 맨 아래에 배치. controller.start_llm_command 호출.
응답은 controller.llm_response 시그널 → 본 패널의 응답 영역.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..controller import SimulatorController
from ..llm import LlmMode


_MODE_BY_KEY: dict[str, LlmMode] = {
    'A': LlmMode.NL_TO_ACTION,
    'B': LlmMode.EXPLAIN,
    'C': LlmMode.VISION,
    'D': LlmMode.VOICE,
}

_PLACEHOLDERS: dict[str, str] = {
    'A': '예: "탑 5층 만들어줘", "정렬 후 시뮬 시작"',
    'B': '예: "지금 상황 알려줘", "다음에 뭐 할 수 있어?"',
    'C': '예: "가장 왼쪽 노란 큐브 좌표 알려줘"  (이미지 첨부 필요)',
    'D': '🎤 녹음 버튼 클릭 후 발화 (별도 마이크 셋업 필요)',
}


class LlmPanel(QWidget):
    """LLM 자연어 인터페이스. controller 와만 통신."""

    def __init__(self, controller: SimulatorController, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._image_bytes: Optional[bytes] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        v.addWidget(self._section_label('🤖 LLM 어시스턴트 (Claude)'))

        # 모드 라디오 — toggled.connect 는 input/img 위젯 만든 뒤에 (콜백이 참조)
        h_mode = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._mode_buttons: dict[str, QRadioButton] = {}
        for key, label in [('A', 'A 명령'), ('B', 'B 분석'),
                           ('C', 'C 비전'), ('D', 'D 음성')]:
            btn = QRadioButton(label)
            btn.setToolTip(_MODE_TOOLTIPS[key])
            self._mode_group.addButton(btn)
            self._mode_buttons[key] = btn
            h_mode.addWidget(btn)
        v.addLayout(h_mode)

        # 입력 (text)
        self._input = QLineEdit()
        self._input.setPlaceholderText(_PLACEHOLDERS['A'])
        self._input.returnPressed.connect(self._on_send)
        v.addWidget(self._input)

        # 이미지 첨부 (Mode C) — 디폴트 숨김
        h_img = QHBoxLayout()
        self._img_btn = QPushButton('📷 이미지 첨부')
        self._img_btn.clicked.connect(self._on_attach_image)
        h_img.addWidget(self._img_btn)
        self._img_label = QLabel('첨부 없음')
        self._img_label.setStyleSheet('color: #888; font-size: 10px;')
        h_img.addWidget(self._img_label, 1)
        self._img_widget = QWidget()
        self._img_widget.setLayout(h_img)
        self._img_widget.setVisible(False)
        v.addWidget(self._img_widget)

        # 녹음 (Mode D) — 디폴트 숨김. arecord + Google Web Speech.
        h_mic = QHBoxLayout()
        self._mic_btn = QPushButton('🎤 5초 녹음 후 자동 전송')
        self._mic_btn.clicked.connect(self._on_record)
        h_mic.addWidget(self._mic_btn, 1)
        self._mic_dur = QLineEdit('5')
        self._mic_dur.setFixedWidth(40)
        self._mic_dur.setToolTip('녹음 길이 (초)')
        h_mic.addWidget(self._mic_dur)
        h_mic.addWidget(QLabel('s'))
        self._mic_widget = QWidget()
        self._mic_widget.setLayout(h_mic)
        self._mic_widget.setVisible(False)
        v.addWidget(self._mic_widget)

        # 실행 + 클리어
        h_run = QHBoxLayout()
        self._send_btn = QPushButton('▶ LLM 에 보내기')
        self._send_btn.setStyleSheet('padding: 4px; font-weight: bold;')
        self._send_btn.clicked.connect(self._on_send)
        h_run.addWidget(self._send_btn, 1)
        self._clear_btn = QPushButton('지움')
        self._clear_btn.clicked.connect(self._on_clear)
        h_run.addWidget(self._clear_btn)
        v.addLayout(h_run)

        # 응답 표시
        self._response = QPlainTextEdit()
        self._response.setReadOnly(True)
        self._response.setMaximumBlockCount(500)
        self._response.setMinimumHeight(80)
        self._response.setMaximumHeight(150)
        self._response.setStyleSheet(
            'font-family: monospace; font-size: 10px; color: #cce;'
        )
        self._response.setPlaceholderText('LLM 응답이 여기 표시됩니다.')
        v.addWidget(self._response)

        # controller 시그널 연결
        self._controller.llm_response.connect(self._on_llm_response)
        self._controller.llm_started.connect(self._on_llm_started)
        self._controller.llm_finished.connect(self._on_llm_finished)

        # 모든 위젯 만든 후 라디오 toggled 연결 + default A 선택 (콜백 안전)
        for btn in self._mode_buttons.values():
            btn.toggled.connect(self._on_mode_changed)
        self._mode_buttons['A'].setChecked(True)

    # --- 외부 ---------------------------------------------------------------
    def set_enabled_all(self, enabled: bool) -> None:
        for btn in self._mode_buttons.values():
            btn.setEnabled(enabled)
        self._input.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        self._img_btn.setEnabled(enabled)

    # --- 콜백 ---------------------------------------------------------------
    def _current_mode_key(self) -> str:
        for key, btn in self._mode_buttons.items():
            if btn.isChecked():
                return key
        return 'A'

    def _on_mode_changed(self, _checked: bool) -> None:
        key = self._current_mode_key()
        self._input.setPlaceholderText(_PLACEHOLDERS.get(key, ''))
        self._img_widget.setVisible(key == 'C')
        self._mic_widget.setVisible(key == 'D')

    def _on_record(self) -> None:
        """🎤 클릭 → arecord 5초 → Google STT → Mode A 자연어 명령으로 자동 전송."""
        try:
            duration = float(self._mic_dur.text().strip() or '5')
        except ValueError:
            duration = 5.0
        duration = max(1.0, min(30.0, duration))

        # blocking 이지만 GUI 응답성 위해 별도 thread 로
        self._mic_btn.setEnabled(False)
        self._mic_btn.setText(f'🎤 {int(duration)}초 녹음 중...')
        self._response.appendPlainText(f'🎤 {int(duration)}초 녹음 시작 — 말해주세요')

        from PyQt5.QtCore import QThread, pyqtSignal as _sig

        class _SttThread(QThread):
            done = _sig(str)
            failed = _sig(str)

            def __init__(self, dur):
                super().__init__()
                self._dur = dur

            def run(self):
                try:
                    from ..llm.stt import record_and_transcribe
                    text = record_and_transcribe(duration_sec=self._dur,
                                                  language='ko-KR')
                    self.done.emit(text)
                except Exception as e:  # noqa: BLE001
                    self.failed.emit(str(e))

        self._stt_thread = _SttThread(duration)
        self._stt_thread.done.connect(self._on_stt_done)
        self._stt_thread.failed.connect(self._on_stt_failed)
        self._stt_thread.finished.connect(self._on_stt_finished)
        self._stt_thread.start()

    def _on_stt_done(self, text: str) -> None:
        self._response.appendPlainText(f'🎤 인식: "{text}"')
        # Mode A 흐름으로 자동 전환해 LLM 에 보냄
        self._mode_buttons['A'].setChecked(True)
        self._input.setText(text)
        self._on_send()

    def _on_stt_failed(self, msg: str) -> None:
        self._response.appendPlainText(f'❌ STT 실패: {msg}')

    def _on_stt_finished(self) -> None:
        self._mic_btn.setEnabled(True)
        self._mic_btn.setText('🎤 5초 녹음 후 자동 전송')

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if not text and self._current_mode_key() != 'C':
            return
        if not text and self._image_bytes is None:
            return
        key = self._current_mode_key()
        ok = self._controller.start_llm_command(
            mode=key,
            user_text=text,
            image_bytes=self._image_bytes if key == 'C' else None,
        )
        if ok:
            self._response.appendPlainText(f'> [{key}] {text}')

    def _on_clear(self) -> None:
        self._input.clear()
        self._response.clear()
        self._image_bytes = None
        self._img_label.setText('첨부 없음')

    def _on_attach_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, '이미지 첨부', '',
            'Image (*.png *.jpg *.jpeg *.bmp)'
        )
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                self._image_bytes = f.read()
            from pathlib import Path
            self._img_label.setText(
                f'{Path(path).name}  ({len(self._image_bytes) // 1024} KB)'
            )
        except Exception as e:  # noqa: BLE001
            self._img_label.setText(f'로드 실패: {e!r}')
            self._image_bytes = None

    def _on_llm_response(self, raw_text: str, reasoning: str) -> None:
        # Mode B (분석/설명) 는 raw 안의 풍부한 reasoning 이 핵심 — 길이 무관 전체 표시.
        # 다른 모드는 짧은 reasoning 만 (raw 는 dispatch 결과로 충분).
        if reasoning:
            self._response.appendPlainText(f'🤖 {reasoning}')
        mode = self._current_mode_key()
        if mode == 'B' and raw_text and raw_text != reasoning:
            # ```json {...} ``` 블록 안의 reasoning 만 발췌
            extracted = _extract_reasoning_text(raw_text)
            if extracted and extracted != reasoning:
                self._response.appendPlainText(extracted)

    def _on_llm_started(self) -> None:
        self._send_btn.setEnabled(False)
        self._send_btn.setText('⏳ 응답 대기...')

    def _on_llm_finished(self) -> None:
        self._send_btn.setEnabled(True)
        self._send_btn.setText('▶ LLM 에 보내기')

    @staticmethod
    def _section_label(txt: str) -> QLabel:
        lbl = QLabel(txt)
        lbl.setStyleSheet('font-weight: bold; color: #ddd; padding-top: 4px;')
        return lbl


def _extract_reasoning_text(raw: str) -> str:
    """raw 응답에서 사람이 읽을 텍스트 필드 모두 추출.

    LLM 이 분석 모드에서 reasoning/analysis/details 등 임의 필드에 답을 넣을
    수 있어 — 모든 문자열 값(긴 것 우선)을 합쳐 반환. JSON 못 찾으면 raw 그대로.
    """
    import json
    import re

    # ```json ... ``` 블록 추출
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    candidate = m.group(1) if m else raw.strip()

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return raw.strip()

    if not isinstance(obj, dict):
        return raw.strip()

    # action / args / next_action 은 메타 — 제외
    skip_keys = {'action', 'args', 'next_action'}
    parts: list[str] = []
    for k, v in obj.items():
        if k in skip_keys:
            continue
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return '\n'.join(parts)


_MODE_TOOLTIPS = {
    'A': '자연어 → 액션. "탑 5층 만들어줘" 같은 명령을 실행 가능한 동작으로 변환.',
    'B': '분석/설명. 현재 모델·dets·plan 을 LLM 이 한국어로 요약·다음 동작 제안.',
    'C': 'Vision. 첨부 이미지를 멀티모달로 분석 → 시맨틱 좌표 추출. Claude Sonnet 사용.',
    'D': '음성 입력. 마이크 → STT → A 흐름. (현재 Phase E 에서 추후 구현)',
}
