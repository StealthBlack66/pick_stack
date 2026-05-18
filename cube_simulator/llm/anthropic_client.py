"""Anthropic Claude API wrapper — text + vision multimodal.

진입점: `AnthropicClient.send(system, user_text, image_bytes=None) -> str`.
모델은 vision 첨부 여부에 따라 자동 선택 — 기본 Haiku, vision 은 Sonnet.

키 우선순위:
  1) 명시 인자
  2) ANTHROPIC_API_KEY env var
  3) `.env` 파일의 ANTHROPIC_API_KEY
  4) `~/.claude/.credentials.json` 의 OAuth accessToken (Claude Code 인증)

키 prefix 로 인증 방식 자동 선택:
  - `sk-ant-api03-...` → x-api-key 헤더 (일반 API key)
  - `sk-ant-oat01-...` → Authorization: Bearer 헤더 (OAuth token)
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Optional


DEFAULT_TEXT_MODEL = 'claude-haiku-4-5-20251001'
# vision 도 Haiku 4.5 가 multimodal 지원 — Sonnet 보다 빠르고 rate limit 관대.
# Sonnet 강제하려면 CUBE_SIM_LLM_VISION_MODEL env 또는 명시 인자.
DEFAULT_VISION_MODEL = 'claude-haiku-4-5-20251001'

# .env 자동 탐색 위치 — 강의 디렉토리 (02_Doosan_Robot_제어/.env) 우선
_ENV_SEARCH_DIRS: tuple[Path, ...] = (
    Path(__file__).resolve().parent.parent.parent,   # 02_Doosan_Robot_제어/
    Path.cwd(),
    Path.home(),
)


class AnthropicError(RuntimeError):
    """LLM 호출 실패."""


_CLAUDE_CREDS_PATH = Path.home() / '.claude' / '.credentials.json'


def load_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """토큰 로드 — explicit > env > .env > Claude Code OAuth credentials.

    Claude Code 가 설치돼 있으면 `.credentials.json` 의 accessToken 자동 사용
    (subscriptionType=max/pro 구독 시 그대로 API 호출 가능 — auth_token Bearer).

    None 반환 시 키 없음. 호출자가 UI 에 가이드 메시지.
    """
    if explicit:
        return explicit.strip()
    key = os.environ.get('ANTHROPIC_API_KEY')
    if key:
        return key.strip()
    # .env 파일 탐색
    try:
        from dotenv import dotenv_values
        for d in _ENV_SEARCH_DIRS:
            env_path = d / '.env'
            if env_path.exists():
                vals = dotenv_values(env_path)
                v = vals.get('ANTHROPIC_API_KEY')
                if v:
                    return v.strip()
    except ImportError:
        pass
    # Claude Code OAuth fallback
    if _CLAUDE_CREDS_PATH.exists():
        try:
            with open(_CLAUDE_CREDS_PATH, 'r') as f:
                creds = json.load(f)
            oauth = creds.get('claudeAiOauth') or {}
            token = oauth.get('accessToken')
            expires_at = oauth.get('expiresAt')   # ms epoch
            if token and (not expires_at or time.time() * 1000 < expires_at):
                return str(token).strip()
        except Exception:
            pass
    return None


def _is_oauth_token(token: str) -> bool:
    """sk-ant-oat01- prefix → OAuth (Bearer), 그 외 (sk-ant-api03- 등) → API key."""
    return token.startswith('sk-ant-oat01-') or token.startswith('sk-ant-ort01-')


def _resolve_model(default_env_key: str, fallback: str) -> str:
    """환경변수 override 우선."""
    return os.environ.get(default_env_key, fallback)


class AnthropicClient:
    """Lazy-init 한 Claude 클라이언트. 메서드 1 개만 노출."""

    def __init__(self, api_key: Optional[str] = None,
                 text_model: Optional[str] = None,
                 vision_model: Optional[str] = None,
                 max_tokens: int = 1024):
        self._api_key = load_api_key(api_key)
        self._text_model = text_model or _resolve_model(
            'CUBE_SIM_LLM_MODEL', DEFAULT_TEXT_MODEL
        )
        self._vision_model = vision_model or _resolve_model(
            'CUBE_SIM_LLM_VISION_MODEL', DEFAULT_VISION_MODEL
        )
        self._max_tokens = int(max_tokens)
        self._client = None   # lazy

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise AnthropicError(
                '키 못 찾음. 다음 중 하나로 인증:\n'
                '  ① ANTHROPIC_API_KEY 환경변수 (sk-ant-api03-...)\n'
                '  ② .env 파일에 동일\n'
                '  ③ Claude Code 로그인 — ~/.claude/.credentials.json 자동 사용\n'
                '발급: https://console.anthropic.com/settings/keys'
            )
        try:
            import anthropic
        except ImportError as e:
            raise AnthropicError(
                f'anthropic SDK 미설치: {e}\n'
                '설치: pip install --user anthropic python-dotenv'
            )
        # prefix 보고 인증 방식 자동 선택
        if _is_oauth_token(self._api_key):
            self._client = anthropic.Anthropic(auth_token=self._api_key)
            self._auth_kind = 'oauth'
        else:
            self._client = anthropic.Anthropic(api_key=self._api_key)
            self._auth_kind = 'api_key'
        return self._client

    @property
    def auth_kind(self) -> str:
        """'oauth' | 'api_key' | 'unknown' — 디버그용."""
        return getattr(self, '_auth_kind', 'unknown')

    def send(self,
             system: str,
             user_text: str,
             image_bytes: Optional[bytes] = None,
             image_mime: str = 'image/png',
             timeout_sec: float = 30.0) -> str:
        """단발 호출 — system + user 메시지 보내고 text 응답 반환.

        image_bytes 가 있으면 vision 모델 자동 선택 + base64 첨부.
        """
        client = self._ensure_client()
        model = self._vision_model if image_bytes else self._text_model

        user_content: list = []
        if image_bytes:
            user_content.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': image_mime,
                    'data': base64.b64encode(image_bytes).decode('ascii'),
                },
            })
        user_content.append({'type': 'text', 'text': user_text})

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{'role': 'user', 'content': user_content}],
                timeout=timeout_sec,
            )
        except Exception as e:  # anthropic.* / httpx.* 등
            raise AnthropicError(f'Claude API 실패: {e!r}') from e

        # 응답에서 텍스트 블록만 수집 (tool_use 등은 무시)
        out_parts: list[str] = []
        for block in getattr(resp, 'content', []) or []:
            if getattr(block, 'type', None) == 'text':
                out_parts.append(block.text)
        return '\n'.join(out_parts).strip()
