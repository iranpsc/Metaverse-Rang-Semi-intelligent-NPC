"""Streaming internal HTTP adapter around the existing RAGService."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from pipeline_core.config import data_root, safe_identifier
from pipeline_core.observability import configure_observability, extract_trace_context, get_tracer


SERVICE = "llm-rag"
configure_observability(SERVICE)
logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

REQUESTS = Counter("metarang_llm_requests_total", "LLM/RAG requests", ("result",))
DURATION = Histogram("metarang_llm_duration_seconds", "LLM/RAG generation duration")
FIRST_TOKEN = Histogram("metarang_llm_first_token_seconds", "LLM time to first token")
ACTIVE = Gauge("metarang_llm_active_requests", "Active LLM/RAG streams")

_STORE_COMPONENT = re.compile(r"[\w][\w-]{0,127}\Z", re.UNICODE)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    user_id: str = Field(min_length=1, max_length=128)
    vector_store: str = Field(default="main_store", min_length=1, max_length=128)
    model_name: str | None = Field(default=None, max_length=128)


class VectorFilesRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    file_paths: list[str] = Field(min_length=1, max_length=10)


class VectorRSSRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    rss_links: list[str] = Field(min_length=1, max_length=10)


class VectorURLRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1, max_length=4096)
    title: str | None = Field(default=None, max_length=1000)


class VectorRebuildRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    dataset_path: str = Field(min_length=1, max_length=4096)
    vector_store_path: str = Field(min_length=1, max_length=4096)


class Runtime:
    rag = None
    error: str | None = None
    semaphore: asyncio.Semaphore | None = None
    vector_lock: asyncio.Lock | None = None


runtime = Runtime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runtime.semaphore = asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENT_REQUESTS", "2")))
    runtime.vector_lock = asyncio.Lock()
    try:
        from LLM.rag_service import RAGService

        runtime.rag = await asyncio.to_thread(RAGService)
    except Exception as exc:
        runtime.error = type(exc).__name__
        logger.exception("RAG service failed to initialize", extra={"event": "rag_init_failed"})
    yield


app = FastAPI(title="MetaRang internal LLM/RAG service", version="1.0.0", lifespan=lifespan)


@app.get("/health/live")
async def live() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/health/ready")
async def ready() -> Response:
    if runtime.rag is None:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": runtime.error})
    ollama = urlparse(os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
    try:
        with socket.create_connection((ollama.hostname or "ollama", ollama.port or 11434), timeout=0.4):
            pass
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Ollama is unavailable") from exc
    return Response(content='{"status":"ok","service":"llm-rag"}', media_type="application/json")


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _resolve_store(name: str) -> Path:
    parts = name.split("/")
    if not parts or any(not _STORE_COMPONENT.fullmatch(part) for part in parts):
        raise HTTPException(status_code=400, detail="Invalid vector_store")
    root = (data_root() / "vectorstores").resolve()
    requested_parts = tuple(parts)

    # Select a path discovered beneath the trusted root instead of constructing
    # a filesystem path directly from request data. Resolving each discovered
    # path also excludes vector-store symlinks that escape the trusted root.
    for index_file in root.rglob("index.faiss"):
        try:
            store = index_file.parent.resolve(strict=True)
            relative = store.relative_to(root)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if relative.parts == requested_parts:
            return store

    raise HTTPException(status_code=404, detail="Vector store was not found")


def _user_memory_path(user_id: str) -> Path:
    # A fixed-length digest preserves stable per-user memory without allowing a
    # user-controlled identifier to become part of a filesystem path.
    user_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return data_root() / "memories" / f"user_{user_key}"


def _user_vector_paths(user_id: str) -> tuple[str, Path, Path, Path]:
    safe_user = safe_identifier(user_id, fallback="anonymous", max_length=128)
    root = data_root()
    vector_store = root / "vectorstores" / f"user_{safe_user}" / "rss_store"
    dataset = root / "rss_datasets" / f"user_{safe_user}" / "rss_dataset.csv"
    uploads = root / "uploads" / f"user_{safe_user}"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    vector_store.mkdir(parents=True, exist_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)
    return safe_user, vector_store, dataset, uploads


def _trusted_path(raw_path: str, *, root: Path, must_exist: bool) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root.resolve())
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Path is outside the shared data directory") from exc
    return resolved


def _rag_runtime():
    if runtime.rag is None or runtime.vector_lock is None:
        raise HTTPException(status_code=503, detail="LLM/RAG service is not ready")
    return runtime.rag


@app.post("/v1/vector-stores/files")
async def process_vector_files(payload: VectorFilesRequest) -> dict:
    rag = _rag_runtime()
    safe_user, vector_store, dataset, uploads = _user_vector_paths(payload.user_id)
    paths = [
        _trusted_path(path, root=uploads, must_exist=True)
        for path in payload.file_paths
    ]
    async with runtime.vector_lock:
        result = await asyncio.to_thread(
            rag.process_user_files,
            user_id=safe_user,
            file_paths=[str(path) for path in paths],
            dataset_path=str(dataset),
            vector_store_path=str(vector_store),
        )
    return {**result, "dataset_path": str(dataset), "vector_store_path": str(vector_store)}


@app.post("/v1/vector-stores/rss")
async def process_vector_rss(payload: VectorRSSRequest) -> dict:
    rag = _rag_runtime()
    safe_user, vector_store, dataset, _ = _user_vector_paths(payload.user_id)
    async with runtime.vector_lock:
        result = await asyncio.to_thread(
            rag.process_user_rss,
            user_id=safe_user,
            rss_links=payload.rss_links,
            dataset_path=str(dataset),
            vector_store_path=str(vector_store),
        )
    return {**result, "dataset_path": str(dataset), "vector_store_path": str(vector_store)}


@app.post("/v1/vector-stores/url")
async def process_vector_url(payload: VectorURLRequest) -> dict:
    rag = _rag_runtime()
    safe_user, vector_store, dataset, _ = _user_vector_paths(payload.user_id)
    async with runtime.vector_lock:
        result = await asyncio.to_thread(
            rag.process_single_url,
            user_id=safe_user,
            url=payload.url,
            dataset_path=str(dataset),
            vector_store_path=str(vector_store),
            title=payload.title,
        )
    return {**result, "dataset_path": str(dataset), "vector_store_path": str(vector_store)}


@app.post("/v1/vector-stores/rebuild")
async def rebuild_vector_store(payload: VectorRebuildRequest) -> dict:
    rag = _rag_runtime()
    root = data_root()
    dataset = _trusted_path(payload.dataset_path, root=root, must_exist=True)
    vector_store = _trusted_path(payload.vector_store_path, root=root, must_exist=False)
    vector_store.mkdir(parents=True, exist_ok=True)
    safe_user = safe_identifier(payload.user_id, fallback="system", max_length=128)
    async with runtime.vector_lock:
        result = await asyncio.to_thread(
            rag.build_vector_store,
            dataset_path=str(dataset),
            vector_store_path=str(vector_store),
            user_id=safe_user,
        )
    return {
        **result,
        "dataset_path": str(dataset),
        "vector_store_path": str(vector_store),
        "user_id": safe_user,
    }


@app.post("/v1/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    x_session_id: str = Header(default="unknown"),
    x_turn_id: str = Header(default="unknown"),
) -> StreamingResponse:
    if runtime.rag is None or runtime.semaphore is None:
        raise HTTPException(status_code=503, detail="LLM/RAG service is not ready")
    store_path = _resolve_store(payload.vector_store)
    safe_user = safe_identifier(payload.user_id, fallback="anonymous", max_length=128)
    memory_path = _user_memory_path(payload.user_id)
    memory_path.mkdir(parents=True, exist_ok=True)
    parent_context = extract_trace_context(request.headers)

    async def generate():
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(
            maxsize=int(os.getenv("LLM_STREAM_QUEUE_SIZE", "32"))
        )
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        started = 0.0

        def produce() -> None:
            def enqueue(item: tuple[str, object]) -> bool:
                while not stop.is_set():
                    future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                    try:
                        future.result(timeout=0.5)
                        return True
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                return False

            try:
                generator = runtime.rag.get_answer_for_user(
                    user_id=safe_user,
                    question=payload.question.strip(),
                    vector_store_path=str(store_path),
                    memory_path=str(memory_path),
                    model_name=payload.model_name,
                )
                for chunk in generator:
                    if stop.is_set():
                        break
                    if not enqueue(("token", str(chunk))):
                        return
            except Exception as exc:
                enqueue(("error", exc))
            finally:
                enqueue(("done", None))

        with tracer.start_as_current_span("rag_llm.stream", context=parent_context) as span:
            span.set_attribute("session.id", x_session_id)
            span.set_attribute("turn.id", x_turn_id)
            first = True
            task: asyncio.Task | None = None
            active = False
            try:
                async with runtime.semaphore:
                    started = time.monotonic()
                    ACTIVE.inc()
                    active = True
                    task = asyncio.create_task(
                        asyncio.to_thread(produce), name=f"llm-{x_turn_id}"
                    )
                    while True:
                        kind, value = await asyncio.wait_for(
                            queue.get(), timeout=float(os.getenv("LLM_GENERATION_TIMEOUT_SECONDS", "180"))
                        )
                        if kind == "done":
                            break
                        if kind == "error":
                            raise value  # type: ignore[misc]
                        if first:
                            first_elapsed = time.monotonic() - started
                            FIRST_TOKEN.observe(first_elapsed)
                            span.add_event("llm.first_token", {"duration_ms": first_elapsed * 1000})
                            first = False
                        yield json.dumps({"type": "token", "text": value}, ensure_ascii=False) + "\n"
                elapsed = time.monotonic() - started
                DURATION.observe(elapsed)
                REQUESTS.labels("success").inc()
                yield json.dumps({"type": "done", "duration_ms": round(elapsed * 1000, 1)}) + "\n"
            except asyncio.TimeoutError:
                REQUESTS.labels("timeout").inc()
                yield json.dumps({"type": "error", "code": "llm_timeout"}) + "\n"
            except Exception:
                REQUESTS.labels("error").inc()
                logger.exception(
                    "LLM/RAG stream failed",
                    extra={"event": "llm_error", "session_id": x_session_id, "turn_id": x_turn_id},
                )
                yield json.dumps({"type": "error", "code": "llm_failed"}) + "\n"
            finally:
                stop.set()
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if active:
                    ACTIVE.dec()

    return StreamingResponse(generate(), media_type="application/x-ndjson")
