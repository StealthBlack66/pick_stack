"""LLM 통합 — Claude API 기반 자연어 인터페이스.

4 모드 지원:
- A: 자연어 → 로봇/시뮬 액션
- B: dets/model → 자연어 분석·제안
- C: 카메라 한 프레임 → vision-language 시맨틱 추출
- D: 음성 → STT → A 흐름 (선택)

진입점: `controller.start_llm_command(mode, text, image=None)` (controller.py 참조)
"""
from __future__ import annotations

from .anthropic_client import AnthropicClient, AnthropicError, load_api_key
from .commands import (
    ALLOWED_ACTIONS,
    LlmCommand,
    LlmMode,
    parse_llm_response,
)
from .prompts import build_system_prompt, build_user_message
from .worker import LlmRequest, LlmWorker

__all__ = [
    'AnthropicClient',
    'AnthropicError',
    'load_api_key',
    'LlmCommand',
    'LlmMode',
    'ALLOWED_ACTIONS',
    'parse_llm_response',
    'build_system_prompt',
    'build_user_message',
    'LlmRequest',
    'LlmWorker',
]
