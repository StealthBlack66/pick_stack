"""4 모드별 system prompt + user message builder.

LLM 에 항상 현재 상태 스냅샷을 같이 보내 컨텍스트 인식. 출력 포맷 강제.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .commands import ALLOWED_ACTIONS, LlmMode


_COORDINATE_GUIDE = """\
좌표계:
- gx, gy: 그리드 좌표 (정수 또는 0.5 단위 valley). 양수 = +X/+Y 방향.
- layer: 0 = 테이블 위 첫 층. 위로 올라갈수록 +1.
- yaw: 그리퍼 회전 각도 (도). 90° 가 큐브 기본 자세. ±90° 단위 권장.
- 큐브 크기 = 25mm. 그리드 pitch = 27mm (모델별 다를 수 있음).
"""

_BASE_INSTRUCTION = f"""\
당신은 두산 e0509 로봇과 RH-P12-RN-A 그리퍼로 큐브를 적층하는 시뮬레이터의
어시스턴트입니다. 사용자의 자연어 요청을 시스템이 실행 가능한 단일 JSON 액션
으로 변환하세요. 안전이 최우선 — 모호하면 묻지 말고 explain 액션으로 명확화.

{_COORDINATE_GUIDE}

허용 액션 (이 외엔 모두 거부됨):
{', '.join(ALLOWED_ACTIONS)}

출력 형식 — JSON 객체 1개 (```json ... ``` 블록 또는 그냥 JSON):
{{
  "action": "<위 목록 중 하나>",
  "args": {{...액션별 파라미터...}},
  "reasoning": "사용자에게 보일 한 줄 설명 (한국어, 50자 이내)",
  "next_action": null   // 또는 같은 형식의 체인 액션
}}

액션별 args 스펙:
- load_preset:        {{"name": "staircase"|"pyramid_1d"|"cross_3d"|...}}
- add_cube:           {{"gx": float, "gy": float, "yaw": float}}
- remove_cube:        {{"gx": float, "gy": float, "layer": int}}
- move_cube:          {{"from": {{"gx","gy","layer"}}, "to": {{"gx","gy"}}}}
- clear_model:        {{}}
- update_cube_yaw:    {{"gx": float, "gy": float, "layer": int, "yaw": float}}
- rebuild_plan:       {{}}
- apply_yaw_corrections: {{}}
- start_simulation:   {{"auto_correct": bool}}
- start_alignment:    {{"dry_run": bool, "quick": bool}}
- start_run:          {{"dry_run": bool}}
- start_oneshot_calibration: {{"skip_launch": bool, "keep_bringup": bool}}
- start_recover_home: {{"dry_run": bool}}
- reset_simulator:    {{"reset_model": bool}}
- explain:            {{}}   ← 액션 실행 없이 reasoning 만 사용자에게
- noop:               {{}}

체인 예시 (프리셋 로드 후 시뮬 자동 시작):
{{
  "action": "load_preset", "args": {{"name": "staircase"}},
  "reasoning": "staircase 프리셋 로드 후 시뮬 시작",
  "next_action": {{
    "action": "start_simulation", "args": {{"auto_correct": true}},
    "reasoning": "yaw 자동 보정 켜고 시뮬", "next_action": null
  }}
}}
"""

_MODE_A_SUFFIX = """\

이 모드 (자연어 → 액션): 사용자 의도를 실행 가능한 액션으로. 위험한 동작
(start_run, start_alignment with dry_run=false) 은 사용자가 명시적으로
"실제로", "진짜로", "로봇으로" 같은 단어를 쓸 때만. 모호하면 dry_run=true.
"""

_MODE_B_SUFFIX = """\

이 모드 (분석/설명): 사용자에게 현재 상황을 한국어로 요약·분석·다음 동작
제안. action 은 항상 "explain", 답변은 **reasoning 필드에 모두 작성**
(3~8 줄, 마크다운 사용 가능). analysis / details 같은 별도 필드를 만들지
말고 reasoning 안에 풍부한 한국어로 작성. 가용 액션 (load_preset,
start_alignment 등) 을 자연어로 안내.
"""

_MODE_C_SUFFIX = """\

이 모드 (vision): 첨부된 이미지를 직접 분석. 큐브 위치·색·자세를 시맨틱으로
파악해 좌표를 추출하거나, "가장 왼쪽 빨간 큐브" 같은 자연어 참조 해석.
액션은 add_cube/move_cube/start_run 등 적절히. 시각 정보를 reasoning 에 명시.
"""


def build_system_prompt(mode: LlmMode | str,
                        model_snapshot: Optional[dict] = None,
                        dets: Optional[list[dict]] = None,
                        plan_summary: Optional[str] = None) -> str:
    """모드별 system prompt 조립. 컨텍스트(모델/dets/plan) 도 함께 주입."""
    mode_str = mode.value if isinstance(mode, LlmMode) else str(mode)
    suffix = {
        'A': _MODE_A_SUFFIX,
        'B': _MODE_B_SUFFIX,
        'C': _MODE_C_SUFFIX,
        'D': _MODE_A_SUFFIX,   # 음성도 결국 자연어 → 액션
    }.get(mode_str, '')

    ctx_lines: list[str] = ['\n현재 컨텍스트:']
    if model_snapshot is not None:
        cubes = model_snapshot.get('cubes', [])
        ctx_lines.append(
            f'- 모델 큐브 {len(cubes)} 개, pitch={model_snapshot.get("pitch_mm", 27)}mm, '
            f'cube_w={model_snapshot.get("cube_width_mm", 25)}mm'
        )
        if cubes:
            # 최대 8개만 표시 (prompt 폭주 방지)
            preview = ', '.join(
                f'L{c["layer"]}({c["gx"]:+.1f},{c["gy"]:+.1f})y{c["yaw_deg"]:+.0f}'
                for c in cubes[:8]
            )
            tail = '' if len(cubes) <= 8 else f' …+{len(cubes)-8}'
            ctx_lines.append(f'  배치: {preview}{tail}')
    if dets:
        ctx_lines.append(f'- 비전 검출 dets {len(dets)} 개 캐시됨 (정렬 후)')
    if plan_summary:
        ctx_lines.append(f'- {plan_summary}')

    return _BASE_INSTRUCTION + suffix + '\n'.join(ctx_lines)


def build_user_message(mode: LlmMode | str, user_text: str) -> str:
    """사용자 입력 텍스트 + (모드별 선택적 부가)."""
    mode_str = mode.value if isinstance(mode, LlmMode) else str(mode)
    if mode_str == 'B':
        return (
            f'사용자: {user_text}\n\n'
            '위 컨텍스트를 토대로 현재 상황을 한국어로 분석·설명·다음 동작 제안하세요. '
            'JSON action 은 "explain" 으로 두고 reasoning 에 풍부한 답변.'
        )
    if mode_str == 'C':
        return (
            f'사용자: {user_text}\n\n'
            '첨부 이미지를 분석해 위 좌표계 기준 좌표를 추출하거나, 자연어 참조를 '
            '해석해 적절한 액션 JSON 으로 출력하세요.'
        )
    return f'사용자: {user_text}'


def serialize_for_prompt(obj: Any, max_chars: int = 4000) -> str:
    """객체를 JSON 으로 직렬화 (prompt 안전 길이 제한)."""
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        s = str(obj)
    if len(s) > max_chars:
        return s[:max_chars] + f'\n... (truncated, +{len(s) - max_chars} chars)'
    return s
