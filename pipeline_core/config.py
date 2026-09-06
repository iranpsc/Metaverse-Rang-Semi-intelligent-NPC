from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return value


def env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return value


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Required environment variable {name} is not set")
    return value


def validate_url(name: str, value: str, schemes: set[str]) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in schemes or not parsed.netloc:
        expected = ", ".join(sorted(schemes))
        raise ConfigurationError(f"{name} must be an absolute URL using {expected}")
    return value.rstrip("/")


def safe_identifier(value: str, *, fallback: str, max_length: int = 64) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    cleaned = cleaned.strip("-_")[:max_length]
    return cleaned or fallback


@dataclass(frozen=True)
class AgentSettings:
    agent_name: str
    stt_url: str
    llm_url: str
    tts_url: str
    vc_url: str
    vc_failure_fallback: bool
    http_connect_timeout_seconds: float
    stt_timeout_seconds: float
    llm_timeout_seconds: float
    tts_timeout_seconds: float
    vc_timeout_seconds: float
    input_sample_rate: int
    output_sample_rate: int
    audio_stream_capacity_frames: int
    output_queue_ms: int
    participant_wait_timeout_seconds: float
    speech_threshold_dbfs: float
    speech_start_ms: int
    speech_end_silence_ms: int
    speech_preroll_ms: int
    speech_keep_silence_ms: int
    min_turn_ms: int
    max_turn_seconds: int
    sentence_queue_size: int
    event_topic: str
    metrics_port: int
    prometheus_multiproc_dir: str

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "metarang-agent"),
            stt_url=validate_url(
                "STT_SERVER_URL", os.getenv("STT_SERVER_URL", "http://stt:5001"), {"http", "https"}
            ),
            llm_url=validate_url(
                "LLM_SERVER_URL", os.getenv("LLM_SERVER_URL", "http://llm:5004"), {"http", "https"}
            ),
            tts_url=validate_url(
                "TTS_SERVER_URL", os.getenv("TTS_SERVER_URL", "http://tts:5002"), {"http", "https"}
            ),
            vc_url=validate_url(
                "VC_SERVER_URL", os.getenv("VC_SERVER_URL", "http://vc:5003"), {"http", "https"}
            ),
            vc_failure_fallback=env_bool("VC_FAILURE_FALLBACK", True),
            http_connect_timeout_seconds=env_float("HTTP_CONNECT_TIMEOUT_SECONDS", 3.0, minimum=0.1),
            stt_timeout_seconds=env_float("STT_TIMEOUT_SECONDS", 120.0, minimum=1.0),
            llm_timeout_seconds=env_float("LLM_TIMEOUT_SECONDS", 180.0, minimum=1.0),
            tts_timeout_seconds=env_float("TTS_TIMEOUT_SECONDS", 120.0, minimum=1.0),
            vc_timeout_seconds=env_float("VC_TIMEOUT_SECONDS", 120.0, minimum=1.0),
            input_sample_rate=env_int("AGENT_INPUT_SAMPLE_RATE", 16000, minimum=8000),
            output_sample_rate=env_int("AGENT_OUTPUT_SAMPLE_RATE", 48000, minimum=8000),
            audio_stream_capacity_frames=env_int("AUDIO_STREAM_CAPACITY_FRAMES", 500, minimum=10),
            output_queue_ms=env_int("AUDIO_OUTPUT_QUEUE_MS", 1000, minimum=100),
            participant_wait_timeout_seconds=env_float(
                "PARTICIPANT_WAIT_TIMEOUT_SECONDS", 45.0, minimum=1.0
            ),
            speech_threshold_dbfs=env_float("SPEECH_THRESHOLD_DBFS", 38.0, minimum=1.0) * -1.0,
            speech_start_ms=env_int("SPEECH_START_MS", 120, minimum=20),
            speech_end_silence_ms=env_int("SPEECH_END_SILENCE_MS", 900, minimum=100),
            speech_preroll_ms=env_int("SPEECH_PREROLL_MS", 240, minimum=0),
            speech_keep_silence_ms=env_int("SPEECH_KEEP_SILENCE_MS", 160, minimum=0),
            min_turn_ms=env_int("MIN_TURN_MS", 300, minimum=100),
            max_turn_seconds=env_int("MAX_TURN_SECONDS", 30, minimum=1),
            sentence_queue_size=env_int("SENTENCE_QUEUE_SIZE", 2, minimum=1),
            event_topic=os.getenv("LIVEKIT_EVENT_TOPIC", "metarang.events.v1"),
            metrics_port=env_int("AGENT_METRICS_PORT", 9090, minimum=1),
            prometheus_multiproc_dir=os.getenv(
                "PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus-agent"
            ),
        )


def data_root() -> Path:
    return Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parent.parent / "data")).resolve()
