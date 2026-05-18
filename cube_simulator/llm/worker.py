"""LLM 호출 QThread — UI 안 막히게 백그라운드에서 Claude API 호출.

controller.start_llm_command → LlmWorker(req).start() → finished 시그널.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from .anthropic_client import AnthropicClient, AnthropicError
from .commands import LlmCommand, LlmMode, parse_llm_response
from .prompts import build_system_prompt, build_user_message


@dataclass
class LlmRequest:
    mode: str                                    # 'A' | 'B' | 'C' | 'D'
    user_text: str
    image_bytes: Optional[bytes] = None
    image_mime: str = 'image/png'
    # 컨텍스트 스냅샷 — controller 가 현재 상태 직렬화해서 전달
    model_snapshot: Optional[dict] = None
    dets: Optional[list] = None
    plan_summary: Optional[str] = None
    # 옵션
    max_tokens: int = 1024
    timeout_sec: float = 30.0


class LlmWorker(QThread):
    """1회 LLM 호출. 결과는 시그널로."""

    # (raw_response_text, parsed_command, reasoning)
    finished_ok = pyqtSignal(str, object, str)
    # 에러 메시지
    failed = pyqtSignal(str)

    def __init__(self,
                 req: LlmRequest,
                 client: Optional[AnthropicClient] = None,
                 parent=None):
        super().__init__(parent)
        self._req = req
        self._client = client if client is not None else AnthropicClient(
            max_tokens=req.max_tokens,
        )

    def run(self) -> None:
        try:
            system = build_system_prompt(
                self._req.mode,
                model_snapshot=self._req.model_snapshot,
                dets=self._req.dets,
                plan_summary=self._req.plan_summary,
            )
            user_msg = build_user_message(self._req.mode, self._req.user_text)
            raw = self._client.send(
                system=system,
                user_text=user_msg,
                image_bytes=self._req.image_bytes,
                image_mime=self._req.image_mime,
                timeout_sec=self._req.timeout_sec,
            )
        except AnthropicError as e:
            self.failed.emit(str(e))
            return
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f'LLM 예외: {e!r}')
            return

        cmd, reasoning = parse_llm_response(raw)
        self.finished_ok.emit(raw, cmd, reasoning)
