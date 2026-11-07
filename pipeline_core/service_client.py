"""Synchronous HTTP client used by Django to reach internal model services.

Django is the control/UI plane. Model inference belongs to the dedicated STT,
LLM/RAG, TTS, and voice-conversion containers so gunicorn workers never load
their own copies of GPU models.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import requests


class InternalServiceError(RuntimeError):
    def __init__(
        self,
        service: str,
        message: str,
        *,
        status_code: int = 503,
        code: str = "service_unavailable",
        retry_after: str | None = None,
    ):
        super().__init__(message)
        self.service = service
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after


@dataclass(frozen=True)
class SpeechServiceResult:
    audio: bytes
    metrics: dict[str, str | float | bool]


def _error_detail(response: requests.Response, fallback: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback
    detail = payload.get("detail") or payload.get("error") or payload.get("message")
    if isinstance(detail, dict):
        detail = detail.get("error") or detail.get("status")
    return str(detail or fallback)


def _error_code(response: requests.Response, fallback: str) -> str:
    try:
        value = response.json().get("code")
    except (AttributeError, ValueError):
        value = None
    return str(value or fallback)


def _speech_metrics(response: requests.Response) -> dict[str, str | float | bool]:
    metrics: dict[str, str | float | bool] = {}
    prefix = "x-metarang-"
    for name, raw_value in response.headers.items():
        normalized_name = name.lower()
        if not normalized_name.startswith(prefix):
            continue
        key = normalized_name[len(prefix) :].replace("-", "_")
        value = raw_value.strip()
        if value.lower() in {"true", "false"}:
            metrics[key] = value.lower() == "true"
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def _speech_response(response: requests.Response, *, service: str) -> SpeechServiceResult:
    if not response.ok:
        raise InternalServiceError(
            service,
            _error_detail(response, f"{service} request failed"),
            status_code=response.status_code,
            code=_error_code(response, f"{service}_failed"),
            retry_after=response.headers.get("Retry-After"),
        )
    if not response.content:
        raise InternalServiceError(
            service,
            f"{service} returned empty audio",
            status_code=502,
            code="empty_audio",
        )
    return SpeechServiceResult(audio=response.content, metrics=_speech_metrics(response))


def synthesize_speech(
    base_url: str,
    text: str,
    *,
    timeout_seconds: float,
    synthesis_settings: dict | None = None,
) -> SpeechServiceResult:
    payload: dict = {"text": text}
    if synthesis_settings:
        payload["settings"] = synthesis_settings
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/synthesize",
            json=payload,
            timeout=(3, timeout_seconds),
        )
    except requests.RequestException as exc:
        raise InternalServiceError(
            "tts", "TTS service is unavailable", code="tts_unavailable"
        ) from exc
    return _speech_response(response, service="tts")


def convert_voice_sample(
    base_url: str,
    source_audio: bytes,
    *,
    target_filename: str,
    timeout_seconds: float,
) -> SpeechServiceResult:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/convert",
            params={"target": target_filename},
            data=source_audio,
            headers={"Content-Type": "audio/wav"},
            timeout=(3, timeout_seconds),
        )
    except requests.RequestException as exc:
        raise InternalServiceError(
            "voice_conversion",
            "Voice conversion service is unavailable",
            code="voice_conversion_unavailable",
        ) from exc
    return _speech_response(response, service="voice_conversion")


def convert_voice_reference(
    base_url: str,
    source_audio: bytes,
    reference_audio: bytes,
    *,
    timeout_seconds: float,
    tau: float = 0.3,
) -> SpeechServiceResult:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/convert",
            files={
                "source_audio": ("source.wav", source_audio, "audio/wav"),
                "reference_audio": ("reference.wav", reference_audio, "audio/wav"),
            },
            data={"tau": str(tau)},
            timeout=(3, timeout_seconds),
        )
    except requests.RequestException as exc:
        raise InternalServiceError(
            "voice_conversion",
            "Voice conversion service is unavailable",
            code="voice_conversion_unavailable",
        ) from exc
    return _speech_response(response, service="voice_conversion")


def transcribe_pcm(
    base_url: str,
    pcm: bytes,
    *,
    session_id: str,
    turn_id: str,
    timeout_seconds: float,
) -> dict:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/transcriptions",
            data=pcm,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": "16000",
                "X-Channels": "1",
                "X-Session-Id": session_id,
                "X-Turn-Id": turn_id,
            },
            timeout=(3, timeout_seconds),
        )
    except requests.RequestException as exc:
        raise InternalServiceError("stt", "STT service is unavailable") from exc
    if not response.ok:
        raise InternalServiceError(
            "stt",
            _error_detail(response, "Transcription failed"),
            status_code=response.status_code,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise InternalServiceError("stt", "STT service returned invalid JSON", status_code=502) from exc


def stream_rag_chat(
    base_url: str,
    payload: dict,
    *,
    session_id: str,
    turn_id: str,
    timeout_seconds: float,
) -> Iterator[dict]:
    try:
        with requests.post(
            f"{base_url.rstrip('/')}/v1/chat/stream",
            json=payload,
            headers={"X-Session-Id": session_id, "X-Turn-Id": turn_id},
            stream=True,
            timeout=(3, timeout_seconds),
        ) as response:
            if not response.ok:
                raise InternalServiceError(
                    "llm",
                    _error_detail(response, "LLM/RAG request failed"),
                    status_code=response.status_code,
                )
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    yield json.loads(raw_line)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise InternalServiceError(
                        "llm", "LLM/RAG service returned an invalid stream", status_code=502
                    ) from exc
    except InternalServiceError:
        raise
    except requests.RequestException as exc:
        raise InternalServiceError("llm", "LLM/RAG service is unavailable") from exc


def run_llm_operation(
    base_url: str,
    operation: str,
    payload: dict,
    *,
    timeout_seconds: float,
) -> dict:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/vector-stores/{operation}",
            json=payload,
            timeout=(3, timeout_seconds),
        )
    except requests.RequestException as exc:
        raise InternalServiceError("llm", "LLM/RAG service is unavailable") from exc
    if not response.ok:
        raise InternalServiceError(
            "llm",
            _error_detail(response, "Vector-store operation failed"),
            status_code=response.status_code,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise InternalServiceError("llm", "LLM/RAG service returned invalid JSON", status_code=502) from exc
