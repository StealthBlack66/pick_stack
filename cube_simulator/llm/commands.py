"""LLM 응답 → controller 액션 매핑 + JSON 파싱.

LLM 은 system prompt 가 정한 화이트리스트 action 만 출력하도록 instruct.
controller 가 dispatch 시 enum 검증 → 미허용 action 은 silent drop.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class LlmMode(str, Enum):
    """사용자가 GUI 에서 고르는 모드."""
    OFF = 'off'                  # LLM 비활성 — 기본
    NL_TO_ACTION = 'A'           # 자연어 → 액션
    EXPLAIN = 'B'                # 검출/모델 → 설명·제안
    VISION = 'C'                 # 카메라 frame → 시맨틱
    VOICE = 'D'                  # 음성 → A (STT 경유)


# controller 의 메서드와 1:1 매핑되는 허용 액션. LLM 출력의 action 필드를
# 이 enum 으로 검증해 임의 코드 실행 방지.
ALLOWED_ACTIONS: tuple[str, ...] = (
    # 모델 조작
    'load_preset',
    'add_cube',
    'remove_cube',
    'move_cube',
    'clear_model',
    'update_cube_yaw',
    # 계획
    'rebuild_plan',
    'apply_yaw_corrections',
    # 실행 트리거
    'start_simulation',
    'start_alignment',
    'start_run',
    'start_oneshot_calibration',
    'start_recover_home',
    'reset_simulator',
    # 대화형 (no-op, 응답만)
    'explain',
    'noop',
)


@dataclass
class LlmCommand:
    """파싱된 LLM 단일 액션."""
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ''
    # 체인 액션 — LLM 이 여러 단계 계획 시. controller 가 순차 dispatch.
    next_action: Optional['LlmCommand'] = None

    def is_valid(self) -> bool:
        return self.action in ALLOWED_ACTIONS


def parse_llm_response(text: str) -> tuple[LlmCommand, str]:
    """Claude 응답 (전체 텍스트) → (LlmCommand, raw_reasoning).

    LLM 출력 형식:
      1) ```json {...} ``` 블록을 포함하거나
      2) 그냥 JSON 객체로 시작하거나
      3) 자연어 설명 안에 JSON 객체 포함

    JSON 못 찾으면 action='explain', reasoning=전체 텍스트.
    """
    raw = (text or '').strip()
    json_obj = _extract_json_object(raw)
    if json_obj is None:
        # 자연어 응답으로 간주 — explain 액션 + 텍스트 그대로
        return LlmCommand(action='explain', reasoning=raw), raw
    cmd = _build_command(json_obj)
    return cmd, raw


def _extract_json_object(text: str) -> Optional[dict]:
    """텍스트에서 첫 번째 유효 JSON 객체 추출."""
    # 1) ```json {...} ``` 블록
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) 텍스트 시작이 { 면 통째로
    if text.lstrip().startswith('{'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # 3) 본문 안의 첫 { ... } 매칭 (greedy balanced)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                snippet = text[start:i + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


def _build_command(obj: dict) -> LlmCommand:
    """dict → LlmCommand. action 화이트리스트 검증."""
    action = str(obj.get('action', 'noop')).strip()
    if action not in ALLOWED_ACTIONS:
        # 미허용 action 은 explain 으로 강등 (reasoning 보존)
        return LlmCommand(
            action='explain',
            reasoning=f'unsupported action {action!r} — {obj.get("reasoning", "")}',
        )
    args = obj.get('args') or {}
    if not isinstance(args, dict):
        args = {}
    reasoning = str(obj.get('reasoning', ''))
    next_obj = obj.get('next_action')
    nxt: Optional[LlmCommand] = None
    if isinstance(next_obj, dict):
        nxt = _build_command(next_obj)
    return LlmCommand(
        action=action,
        args=args,
        reasoning=reasoning,
        next_action=nxt,
    )
