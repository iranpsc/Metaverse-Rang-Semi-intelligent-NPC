"""Bounded, in-memory PCM adapter around the repository's Whisper STT model."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from pipeline_core.observability import configure_observability, extract_trace_context, get_tracer


SERVICE = "stt"
configure_observability(SERVICE)
logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

REQUESTS = Counter("metarang_stt_requests_total", "STT requests", ("result",))
DURATION = Histogram("metarang_stt_duration_seconds", "STT request duration")
QUEUE_DEPTH = Gauge("metarang_stt_queue_depth", "Queued STT turns")


@dataclass
class TranscriptionJob:
    pcm: bytes
    session_id: str
    turn_id: str
    result: asyncio.Future


class WhisperRuntime:
    def __init__(self) -> None:
        self.model = None
        self.error: str | None = None
        self.queue: asyncio.Queue[TranscriptionJob | None] = asyncio.Queue(
            maxsize=int(os.getenv("STT_QUEUE_SIZE", "8"))
        )
        self.workers: list[asyncio.Task] = []

    async def start(self) -> None:
        import torch
        import whisper

        model_name = os.getenv("WHISPER_MODEL", "turbo")
        requested_device = os.getenv("WHISPER_DEVICE", "auto").lower()
        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if requested_device == "auto" else requested_device
        try:
            self.model = await asyncio.to_thread(whisper.load_model, model_name, device=device)
        except Exception as exc:
            self.error = type(exc).__name__
            logger.exception("Whisper model failed to load", extra={"event": "model_load_failed"})
            return
        count = int(os.getenv("STT_INFERENCE_WORKERS", "1"))
        self.workers = [
            asyncio.create_task(self._worker(), name=f"stt-inference-{i}") for i in range(count)
        ]
        logger.info("Whisper model ready", extra={"event": "model_ready", "device": device})

    async def stop(self) -> None:
        for _ in self.workers:
            await self.queue.put(None)
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            QUEUE_DEPTH.set(self.queue.qsize())
            try:
                if job is None:
                    return
                samples = np.frombuffer(job.pcm, dtype="<i2").astype(np.float32) / 32768.0
                result = await asyncio.to_thread(
                    self.model.transcribe,
                    samples,
                    fp16=False,
                    temperature=0.0,
                    no_speech_threshold=float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6")),
                )
                if not job.result.cancelled():
                    job.result.set_result(result)
            except Exception as exc:
                if job is not None and not job.result.cancelled():
                    job.result.set_exception(exc)
            finally:
                self.queue.task_done()
                QUEUE_DEPTH.set(self.queue.qsize())


runtime = WhisperRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await runtime.start()
    yield
    await runtime.stop()


app = FastAPI(title="MetaRang internal STT service", version="1.0.0", lifespan=lifespan)


@app.get("/health/live")
async def live() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/health/ready")
async def ready() -> Response:
    if runtime.model is None:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": runtime.error})
    return Response(content='{"status":"ok","service":"stt"}', media_type="application/json")


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/transcriptions")
async def transcribe(
    request: Request,
    x_session_id: str = Header(default="unknown"),
    x_turn_id: str = Header(default="unknown"),
    x_sample_rate: int = Header(default=16000),
    x_channels: int = Header(default=1),
) -> dict:
    if runtime.model is None:
        raise HTTPException(status_code=503, detail="STT model is not ready")
    if x_sample_rate != 16000 or x_channels != 1:
        raise HTTPException(status_code=415, detail="Expected mono PCM16 at 16000 Hz")
    pcm = await request.body()
    max_bytes = int(os.getenv("STT_MAX_AUDIO_SECONDS", "35")) * x_sample_rate * 2
    if not pcm or len(pcm) % 2 or len(pcm) > max_bytes:
        raise HTTPException(status_code=400, detail="Invalid or oversized PCM16 body")
    if runtime.queue.full():
        REQUESTS.labels("backpressured").inc()
        raise HTTPException(status_code=429, detail="STT queue is full")

    started = time.monotonic()
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    context = extract_trace_context(request.headers)
    with tracer.start_as_current_span("stt.transcribe", context=context) as span:
        span.set_attribute("session.id", x_session_id)
        span.set_attribute("turn.id", x_turn_id)
        span.set_attribute("audio.duration_ms", len(pcm) * 500 / x_sample_rate)
        await runtime.queue.put(TranscriptionJob(pcm, x_session_id, x_turn_id, future))
        QUEUE_DEPTH.set(runtime.queue.qsize())
        try:
            result = await asyncio.wait_for(
                future, timeout=float(os.getenv("STT_INFERENCE_TIMEOUT_SECONDS", "120"))
            )
        except asyncio.TimeoutError as exc:
            REQUESTS.labels("timeout").inc()
            raise HTTPException(status_code=504, detail="STT inference timed out") from exc
        except Exception as exc:
            REQUESTS.labels("error").inc()
            logger.exception(
                "STT inference failed",
                extra={"event": "stt_error", "session_id": x_session_id, "turn_id": x_turn_id},
            )
            raise HTTPException(status_code=500, detail="STT inference failed") from exc
        elapsed = time.monotonic() - started
        DURATION.observe(elapsed)
        REQUESTS.labels("success").inc()
        span.set_attribute("stt.text_length", len(result.get("text", "")))
        logger.info(
            "STT final available",
            extra={
                "event": "stt_final",
                "session_id": x_session_id,
                "turn_id": x_turn_id,
                "duration_ms": round(elapsed * 1000, 1),
            },
        )
        return {
            "text": str(result.get("text", "")).strip(),
            "language": result.get("language"),
            "duration_ms": round(elapsed * 1000, 1),
            "final": True,
        }
