from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Literal


EVENT_VERSION = 1
EVENT_TOPIC = "metarang.events.v1"


@dataclass(frozen=True)
class TranscriptionEvent:
    session_id: str
    turn_id: str
    speaker: Literal["user", "agent"]
    text: str
    final: bool
    timestamp: float
    type: Literal["transcription"] = "transcription"
    version: int = EVENT_VERSION

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        turn_id: str,
        speaker: Literal["user", "agent"],
        text: str,
        final: bool,
    ) -> "TranscriptionEvent":
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            final=final,
            timestamp=time.time(),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class PipelineStateEvent:
    session_id: str
    turn_id: str | None
    state: Literal["listening", "transcribing", "thinking", "speaking", "ready"]
    timestamp: float
    type: Literal["pipeline_state"] = "pipeline_state"
    version: int = EVENT_VERSION

    @classmethod
    def create(
        cls, *, session_id: str, state: str, turn_id: str | None = None
    ) -> "PipelineStateEvent":
        return cls(  # type: ignore[arg-type]
            session_id=session_id,
            turn_id=turn_id,
            state=state,
            timestamp=time.time(),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class PipelineErrorEvent:
    session_id: str
    turn_id: str | None
    stage: str
    code: str
    message: str
    recoverable: bool
    timestamp: float
    type: Literal["pipeline_error"] = "pipeline_error"
    version: int = EVENT_VERSION

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        stage: str,
        code: str,
        message: str,
        recoverable: bool,
        turn_id: str | None = None,
    ) -> "PipelineErrorEvent":
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            stage=stage,
            code=code,
            message=message,
            recoverable=recoverable,
            timestamp=time.time(),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

