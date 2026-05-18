"""음성 → 텍스트 (Mode D 백엔드).

`arecord` (ALSA) subprocess 로 WAV 녹음 → SpeechRecognition + Google Web Speech
API 로 STT. 가장 가벼운 조합 — PortAudio / PyAudio 없이 동작.

요구사항:
- 시스템에 `arecord` (ALSA — Ubuntu 표준)
- pip 패키지: `SpeechRecognition` (pure Python)
- 인터넷 (Google Web Speech 무료 API)

마이크 없거나 arecord 미설치면 RuntimeError + 명확한 메시지.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional


class STTError(RuntimeError):
    """STT 실패."""


def _check_arecord() -> Optional[str]:
    """arecord 가용성 확인. 사용 가능하면 binary path, 없으면 None."""
    try:
        r = subprocess.run(
            ['which', 'arecord'], capture_output=True, text=True, timeout=2,
        )
        path = r.stdout.strip()
        return path if path and os.path.exists(path) else None
    except Exception:
        return None


def record_wav(duration_sec: float = 5.0,
               device: Optional[str] = None) -> bytes:
    """ALSA arecord 로 N초 녹음. CD 품질 (16-bit stereo 44.1kHz).

    device=None 이면 ALSA default. 다른 디바이스는 'hw:1,0' 형태로.
    """
    binary = _check_arecord()
    if binary is None:
        raise STTError(
            'arecord 명령을 찾을 수 없음 (ALSA 미설치).\n'
            '설치: sudo apt install alsa-utils'
        )

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name
    try:
        cmd = [binary, '-q', '-f', 'cd', '-t', 'wav',
               '-d', str(int(round(duration_sec)))]
        if device:
            cmd += ['-D', device]
        cmd.append(wav_path)
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=duration_sec + 5,
        )
        if result.returncode != 0:
            raise STTError(
                f'arecord 실패 (rc={result.returncode}):\n{result.stderr.strip()}'
            )
        with open(wav_path, 'rb') as f:
            return f.read()
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def transcribe(wav_bytes: bytes, language: str = 'ko-KR') -> str:
    """WAV bytes → 텍스트 (Google Web Speech).

    무료 API — 인터넷 필요. 분당 ~60회 한도 (보통 강의 demo 에는 충분).
    """
    try:
        import speech_recognition as sr
    except ImportError as e:
        raise STTError(
            f'SpeechRecognition 미설치: {e}\n'
            '설치: pip install --user SpeechRecognition'
        ) from e

    # speech_recognition 은 file path 또는 BytesIO. NamedTemp 로 임시 저장.
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(wav_bytes)
        wav_path = f.name
    try:
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as src:
            audio = r.record(src)
        try:
            return r.recognize_google(audio, language=language).strip()
        except sr.UnknownValueError:
            raise STTError('음성을 인식하지 못함 — 다시 시도')
        except sr.RequestError as e:
            raise STTError(f'Google STT API 실패 (인터넷 확인): {e}')
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def record_and_transcribe(duration_sec: float = 5.0,
                          language: str = 'ko-KR',
                          device: Optional[str] = None) -> str:
    """녹음 + STT 한 번에. UI 호출 진입점."""
    wav = record_wav(duration_sec=duration_sec, device=device)
    return transcribe(wav, language=language)
