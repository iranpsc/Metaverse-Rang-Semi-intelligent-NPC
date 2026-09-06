"""Timeout-bounded HTTP client for the four internal pipeline stages."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from pipeline_core.config import AgentSettings
from pipeline_core.observability import inject_trace_headers


class PipelineServiceError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class PipelineClient:
    def __init__(self, settings: AgentSettings):
        self.settings = settings
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.http_connect_timeout_seconds,
                read=max(
                    settings.stt_timeout_seconds,
                    settings.llm_timeout_seconds,
                    settings.tts_timeout_seconds,
                    settings.vc_timeout_seconds,
                ),
                write=30.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    @staticmethod
    def _headers(session_id: str, turn_id: str) -> dict[str, str]:
        return inject_trace_headers({"x-session-id": session_id, "x-turn-id": turn_id})

    async def close(self) -> None:
        await self.http.aclose()

    async def transcribe(self, pcm: bytes, *, session_id: str, turn_id: str) -> str:
        headers = self._headers(session_id, turn_id)
        headers.update(
            {
                "content-type": "application/octet-stream",
                "x-sample-rate": str(self.settings.input_sample_rate),
                "x-channels": "1",
            }
        )
        try:
            response = await self.http.post(
                f"{self.settings.stt_url}/v1/transcriptions",
                content=pcm,
                headers=headers,
                timeout=self.settings.stt_timeout_seconds,
            )
            response.raise_for_status()
            return str(response.json().get("text", "")).strip()
        except (httpx.HTTPError, ValueError) as exc:
            raise PipelineServiceError("stt", "Transcription failed") from exc

    async def stream_answer(
        self,
        question: str,
        *,
        session_id: str,
        turn_id: str,
        user_id: str,
        vector_store: str,
    ) -> AsyncIterator[str]:
        try:
            async with self.http.stream(
                "POST",
                f"{self.settings.llm_url}/v1/chat/stream",
                json={
                    "question": question,
                    "user_id": user_id,
                    "vector_store": vector_store,
                },
                headers=self._headers(session_id, turn_id),
                timeout=self.settings.llm_timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("type") == "token":
                        yield str(event.get("text", ""))
                    elif event.get("type") == "error":
                        raise PipelineServiceError("llm", str(event.get("code", "Generation failed")))
        except PipelineServiceError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise PipelineServiceError("llm", "LLM/RAG request failed") from exc

    async def synthesize(self, text: str, *, session_id: str, turn_id: str) -> bytes:
        try:
            response = await self.http.post(
                f"{self.settings.tts_url}/synthesize",
                json={"text": text},
                headers=self._headers(session_id, turn_id),
                timeout=self.settings.tts_timeout_seconds,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            raise PipelineServiceError("tts", "Speech synthesis failed") from exc

    async def convert_voice(
        self,
        wav: bytes,
        *,
        target: str,
        session_id: str,
        turn_id: str,
    ) -> bytes:
        try:
            response = await self.http.post(
                f"{self.settings.vc_url}/convert",
                params={"target": target},
                content=wav,
                headers={
                    **self._headers(session_id, turn_id),
                    "content-type": "audio/wav",
                },
                timeout=self.settings.vc_timeout_seconds,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            raise PipelineServiceError("voice_conversion", "Voice conversion failed") from exc
