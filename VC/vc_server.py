"""GPU-only OpenVoice V2 conversion service for the Tesla P40.

The process remains live when CUDA, compatibility, or VRAM checks prevent model
loading. Readiness reports the exact reason and a background loader retries at
a bounded interval. No CPU conversion fallback exists in this service.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
import uvicorn
import webrtcvad
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.concurrency import run_in_threadpool

from pipeline_core.observability import configure_observability, extract_trace_context, get_tracer
from VC.openvoice_runtime import OpenVoiceConverter, decode_audio


SERVICE = "voice-conversion"
configure_observability(SERVICE)
log = logging.getLogger(__name__)
tracer = get_tracer(__name__)

REQUESTS = Counter("metarang_vc_requests_total", "Voice conversion requests", ("result",))
CONVERSION_DURATION = Histogram("metarang_vc_duration_seconds", "Tone conversion duration")
TARGET_EMBEDDING_DURATION = Histogram(
    "metarang_vc_target_embedding_seconds", "Target embedding extraction duration"
)
SOURCE_EMBEDDING_DURATION = Gauge(
    "metarang_vc_source_embedding_seconds", "Canonical source embedding extraction duration"
)
QUEUE_WAIT = Histogram("metarang_vc_queue_wait_seconds", "Voice conversion queue wait")
MODEL_READY = Gauge("metarang_vc_model_ready", "Whether OpenVoice is ready")
MODEL_LOAD_DURATION = Gauge("metarang_vc_model_load_seconds", "OpenVoice model load duration")
GPU_PEAK_ALLOCATED = Gauge(
    "metarang_vc_gpu_peak_allocated_mib", "Peak allocated GPU memory measured during initialization"
)
GPU_PEAK_RESERVED = Gauge(
    "metarang_vc_gpu_peak_reserved_mib", "Peak reserved GPU memory measured during initialization"
)
CACHE_HITS = Counter("metarang_vc_target_cache_total", "Target embedding cache access", ("result",))

BASE_DIR = Path(__file__).resolve().parent
HOST = os.getenv("VC_HOST", "127.0.0.1")
PORT = int(os.getenv("VC_PORT", "5003"))
DEVICE = os.getenv("VC_DEVICE", "cuda:0").strip()
PRECISION = os.getenv("VC_PRECISION", "fp32").strip().lower()
EXPECTED_GPU_NAME = os.getenv("VC_EXPECTED_GPU_NAME", "Tesla P40").strip()
EXPECTED_CAPABILITY = os.getenv("VC_EXPECTED_COMPUTE_CAPABILITY", "6.1").strip()
CONFIG_PATH = Path(
    os.getenv("VC_CONFIG_PATH", BASE_DIR / "checkpoints/openvoice-v2/converter/config.json")
)
CHECKPOINT_PATH = Path(
    os.getenv("VC_CHECKPOINT_PATH", BASE_DIR / "checkpoints/openvoice-v2/converter/checkpoint.pth")
)
CHECKPOINT_REVISION = os.getenv(
    "VC_CHECKPOINT_REVISION", "ea66dc9353e660b1dbbe0bebebd4c2d8b105483c"
)
OPENVOICE_COMMIT = os.getenv(
    "VC_OPENVOICE_COMMIT", "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
)
SAMPLES_DIR = Path(os.getenv("VC_SAMPLES_DIR", BASE_DIR.parent / "media/voice_samples")).resolve()
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://tts:5002").rstrip("/")
MIN_FREE_VRAM_MIB = max(0, int(os.getenv("VC_MIN_FREE_VRAM_MB", "0")))
VRAM_SAFETY_MARGIN_MIB = max(0, int(os.getenv("VC_VRAM_SAFETY_MARGIN_MB", "512")))
RETRY_COOLDOWN_SECONDS = max(5, int(os.getenv("VC_MODEL_RETRY_COOLDOWN_SECONDS", "30")))
SOURCE_REFRESH_SECONDS = max(30, int(os.getenv("VC_SOURCE_REFRESH_SECONDS", "300")))
MAX_CONCURRENT = max(1, int(os.getenv("VC_MAX_CONCURRENT_REQUESTS", "1")))
QUEUE_MAX_SIZE = max(0, int(os.getenv("VC_QUEUE_MAX_SIZE", "4")))
QUEUE_TIMEOUT_SECONDS = max(0.1, float(os.getenv("VC_QUEUE_TIMEOUT_SECONDS", "30")))
RETRY_AFTER_SECONDS = max(1, int(os.getenv("VC_RETRY_AFTER_SECONDS", "5")))
CACHE_MAX_ENTRIES = max(1, int(os.getenv("VC_TARGET_CACHE_MAX_ENTRIES", "32")))
CACHE_TTL_SECONDS = max(1, int(os.getenv("VC_TARGET_CACHE_TTL_SECONDS", "1800")))
MIN_VOICED_SECONDS = max(0.1, float(os.getenv("VC_MIN_VOICED_SECONDS", "0.8")))
MAX_REFERENCE_SECONDS = max(1.0, float(os.getenv("VC_MAX_REFERENCE_SECONDS", "60")))
MAX_UPLOAD_BYTES = max(1024, int(os.getenv("VC_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))))
MAX_SOURCE_BYTES = max(1024, int(os.getenv("VC_MAX_SOURCE_BYTES", str(32 * 1024 * 1024))))

if DEVICE != "cuda:0":
    raise RuntimeError("VC_DEVICE must be cuda:0")
if PRECISION != "fp32":
    raise RuntimeError("Tesla P40 migration requires VC_PRECISION=fp32")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CHECKPOINT_SHA256 = _sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else "missing"
CONFIG_SHA256 = _sha256_file(CONFIG_PATH) if CONFIG_PATH.is_file() else "missing"


class ServiceProblem(RuntimeError):
    def __init__(self, status: int, code: str, message: str, *, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


@dataclass
class CacheEntry:
    embedding: torch.Tensor
    expires_at: float


class TargetEmbeddingCache:
    def __init__(self) -> None:
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> torch.Tensor | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                CACHE_HITS.labels("miss").inc()
                return None
            if entry.expires_at <= now:
                CACHE_HITS.labels("expired").inc()
                return None
            self._entries[key] = entry
            CACHE_HITS.labels("hit").inc()
            return entry.embedding

    def put(self, key: str, embedding: torch.Tensor) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = CacheEntry(
                embedding=embedding.detach().cpu().float().contiguous(),
                expires_at=time.monotonic() + CACHE_TTL_SECONDS,
            )
            while len(self._entries) > CACHE_MAX_ENTRIES:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class BoundedGPUQueue:
    def __init__(self) -> None:
        self._semaphore = threading.BoundedSemaphore(MAX_CONCURRENT)
        self._lock = threading.Lock()
        self._admitted = 0
        self._active = 0

    def acquire(self) -> float:
        with self._lock:
            if self._admitted >= MAX_CONCURRENT + QUEUE_MAX_SIZE:
                raise ServiceProblem(
                    429,
                    "vc_queue_full",
                    "Voice conversion queue is full.",
                    retry_after=RETRY_AFTER_SECONDS,
                )
            self._admitted += 1
        started = time.monotonic()
        if not self._semaphore.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
            with self._lock:
                self._admitted -= 1
            raise ServiceProblem(
                429,
                "vc_queue_timeout",
                "Timed out waiting for the voice conversion GPU.",
                retry_after=RETRY_AFTER_SECONDS,
            )
        waited = time.monotonic() - started
        with self._lock:
            self._active += 1
        QUEUE_WAIT.observe(waited)
        return waited

    def release(self) -> None:
        with self._lock:
            self._active -= 1
            self._admitted -= 1
        self._semaphore.release()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": self._active,
                "waiting": self._admitted - self._active,
                "capacity": QUEUE_MAX_SIZE,
            }


TARGET_CACHE = TargetEmbeddingCache()
GPU_QUEUE = BoundedGPUQueue()
GPU_INFERENCE_LOCK = threading.Lock()


def _normalized_reference(audio_bytes: bytes) -> tuple[bytes, float, float]:
    audio = decode_audio(audio_bytes, 16000)
    duration = audio.size / 16000
    if duration > MAX_REFERENCE_SECONDS:
        raise ValueError(f"reference audio must not exceed {MAX_REFERENCE_SECONDS:g} seconds")

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    vad = webrtcvad.Vad(2)
    frame_bytes = 16000 * 30 // 1000 * 2
    voiced_frames = 0
    total_frames = 0
    for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        total_frames += 1
        if vad.is_speech(pcm[offset : offset + frame_bytes], 16000):
            voiced_frames += 1
    voiced_seconds = voiced_frames * 0.03
    if total_frames == 0 or voiced_seconds < MIN_VOICED_SECONDS:
        raise ValueError(
            f"reference audio has only {voiced_seconds:.2f}s of usable speech; "
            f"at least {MIN_VOICED_SECONDS:.2f}s is required"
        )
    return pcm, duration, voiced_seconds


def _reference_cache_key(normalized_pcm: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(CHECKPOINT_SHA256.encode("ascii"))
    digest.update(OPENVOICE_COMMIT.encode("ascii"))
    digest.update(normalized_pcm)
    return digest.hexdigest()


def _resolve_sample(filename: str) -> Path:
    name = Path(filename or "").name
    if not name:
        raise FileNotFoundError("empty target filename")
    candidate = (SAMPLES_DIR / name).resolve()
    if candidate.parent != SAMPLES_DIR or not candidate.is_file():
        raise FileNotFoundError(f"voice sample not found: {name}")
    return candidate


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.converter: OpenVoiceConverter | None = None
        self.source_embedding: torch.Tensor | None = None
        self.source_key = ""
        self.error_code = "model_not_loaded"
        self.error_message = "OpenVoice model has not loaded yet."
        self.loading = False
        self.last_attempt_at = 0.0
        self.last_source_refresh = 0.0
        self.metrics: dict = {}
        self.recommended_min_free_mib = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="openvoice-loader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _gpu_diagnostics(self) -> dict:
        if not torch.cuda.is_available():
            raise ServiceProblem(
                503, "cuda_unavailable", "CUDA is unavailable; OpenVoice CPU fallback is disabled."
            )
        name = torch.cuda.get_device_name(0)
        capability_tuple = torch.cuda.get_device_capability(0)
        capability = f"{capability_tuple[0]}.{capability_tuple[1]}"
        free, total = torch.cuda.mem_get_info(0)
        if EXPECTED_GPU_NAME and EXPECTED_GPU_NAME not in name:
            raise ServiceProblem(
                503, "unsupported_gpu", f"Expected {EXPECTED_GPU_NAME}, detected {name}."
            )
        if EXPECTED_CAPABILITY and capability != EXPECTED_CAPABILITY:
            raise ServiceProblem(
                503,
                "unsupported_compute_capability",
                f"Expected compute capability {EXPECTED_CAPABILITY}, detected {capability}.",
            )
        return {
            "name": name,
            "capability": capability,
            "free_mib": free / 1024**2,
            "total_mib": total / 1024**2,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "arch_list": torch.cuda.get_arch_list(),
        }

    def _fetch_canonical(self) -> tuple[bytes, str]:
        response = requests.get(f"{TTS_SERVER_URL}/canonical-source", timeout=(3, 120))
        response.raise_for_status()
        piper_sha = response.headers.get("X-MetaRang-Piper-Model-SHA256", "unknown")
        calibration_version = response.headers.get("X-MetaRang-Calibration-Version", "unknown")
        canonical_sha = response.headers.get(
            "X-MetaRang-Canonical-SHA256", hashlib.sha256(response.content).hexdigest()
        )
        return response.content, f"{piper_sha}:{calibration_version}:{canonical_sha}:{CHECKPOINT_SHA256}"

    def _validate_source_embedding(
        self, converter: OpenVoiceConverter, canonical_embedding: torch.Tensor
    ) -> dict:
        validation_sentences = (
            "این یک نمونه کوتاه برای بررسی ثبات صدای پایه است.",
            "سامانه متارنگ پاسخ فارسی را روشن و طبیعی تولید می‌کند.",
            "کیفیت گفتار در جمله‌های متفاوت باید یکسان باقی بماند.",
        )
        similarities: list[float] = []
        for sentence in validation_sentences:
            response = requests.post(
                f"{TTS_SERVER_URL}/synthesize", json={"text": sentence}, timeout=(3, 120)
            )
            response.raise_for_status()
            embedding, _ = converter.extract_se(response.content)
            similarity = torch.nn.functional.cosine_similarity(
                canonical_embedding.flatten(), embedding.flatten(), dim=0
            ).item()
            similarities.append(float(similarity))
        return {
            "samples": len(similarities),
            "min_cosine_similarity": min(similarities),
            "mean_cosine_similarity": sum(similarities) / len(similarities),
        }

    def _load(self) -> None:
        diagnostics = self._gpu_diagnostics()
        required_mib = max(MIN_FREE_VRAM_MIB, self.recommended_min_free_mib)
        if diagnostics["free_mib"] < required_mib:
            raise ServiceProblem(
                503,
                "insufficient_vram",
                f"OpenVoice requires at least {required_mib:.0f} MiB free VRAM; "
                f"{diagnostics['free_mib']:.0f} MiB is available.",
                retry_after=RETRY_COOLDOWN_SECONDS,
            )
        # NVIDIA cubins are forward-compatible within the same compute major.
        # The official cu118 wheel reports sm_60 and has been execution-tested
        # on this Tesla P40 (compute capability 6.1).
        compatible_arches = {"sm_60", "sm_61"}
        if not compatible_arches.intersection(diagnostics["arch_list"]):
            raise ServiceProblem(
                503,
                "p40_architecture_missing",
                "The installed PyTorch build does not include a compute-6.x kernel "
                "compatible with Tesla P40.",
            )
        if not CONFIG_PATH.is_file() or not CHECKPOINT_PATH.is_file():
            raise ServiceProblem(503, "checkpoint_missing", "OpenVoice converter files are missing.")

        converter = OpenVoiceConverter(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)
        canonical_wav, source_key = self._fetch_canonical()
        with GPU_INFERENCE_LOCK:
            source_embedding, source_metrics = converter.extract_se(canonical_wav)
            validation = self._validate_source_embedding(converter, source_embedding)
        peak_allocated = max(
            converter.load_metrics["peak_allocated_mib"], source_metrics["peak_allocated_mib"]
        )
        recommended = int(math.ceil(peak_allocated + VRAM_SAFETY_MARGIN_MIB))

        with self._lock:
            self.converter = converter
            self.source_embedding = source_embedding
            self.source_key = source_key
            self.error_code = ""
            self.error_message = ""
            self.last_source_refresh = time.monotonic()
            self.recommended_min_free_mib = recommended
            self.metrics = {
                **diagnostics,
                **converter.load_metrics,
                "source_embedding_seconds": source_metrics["embedding_time"],
                "source_validation": validation,
                "recommended_min_free_vram_mib": recommended,
            }
        TARGET_CACHE.clear()
        MODEL_READY.set(1)
        MODEL_LOAD_DURATION.set(converter.load_seconds)
        SOURCE_EMBEDDING_DURATION.set(source_metrics["embedding_time"])
        GPU_PEAK_ALLOCATED.set(peak_allocated)
        GPU_PEAK_RESERVED.set(
            max(converter.load_metrics["peak_reserved_mib"], source_metrics["peak_reserved_mib"])
        )
        log.info(
            "OpenVoice V2 ready on Tesla P40",
            extra={
                "event": "vc_model_ready",
                "device": diagnostics["name"],
                "capability": diagnostics["capability"],
                "free_vram_mib": round(diagnostics["free_mib"], 1),
                "recommended_min_free_vram_mib": recommended,
                "model_load_seconds": round(converter.load_seconds, 3),
                "source_embedding_seconds": round(source_metrics["embedding_time"], 3),
                "source_similarity_min": round(validation["min_cosine_similarity"], 4),
            },
        )

    def _refresh_source_if_changed(self) -> None:
        with self._lock:
            converter = self.converter
            current_key = self.source_key
        if converter is None:
            return
        canonical_wav, source_key = self._fetch_canonical()
        if source_key == current_key:
            with self._lock:
                self.last_source_refresh = time.monotonic()
            return
        with GPU_INFERENCE_LOCK:
            embedding, metrics = converter.extract_se(canonical_wav)
        with self._lock:
            self.source_embedding = embedding
            self.source_key = source_key
            self.last_source_refresh = time.monotonic()
            self.metrics["source_embedding_seconds"] = metrics["embedding_time"]
        TARGET_CACHE.clear()
        log.info("Piper source embedding cache refreshed", extra={"event": "vc_source_refresh"})

    def _set_error(self, problem: ServiceProblem) -> None:
        with self._lock:
            self.converter = None
            self.source_embedding = None
            self.error_code = problem.code
            self.error_message = problem.message
        MODEL_READY.set(0)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                ready = self.converter is not None and self.source_embedding is not None
                refresh_due = time.monotonic() - self.last_source_refresh >= SOURCE_REFRESH_SECONDS
            try:
                if not ready:
                    with self._lock:
                        self.loading = True
                        self.last_attempt_at = time.time()
                    self._load()
                elif refresh_due:
                    self._refresh_source_if_changed()
            except ServiceProblem as problem:
                self._set_error(problem)
                log.warning(
                    "OpenVoice is not ready: %s",
                    problem.message,
                    extra={"event": "vc_not_ready", "code": problem.code},
                )
            except torch.cuda.OutOfMemoryError:
                free, _ = torch.cuda.mem_get_info(0)
                self.recommended_min_free_mib = max(
                    self.recommended_min_free_mib,
                    int(math.ceil(free / 1024**2 + VRAM_SAFETY_MARGIN_MIB)),
                )
                self._set_error(
                    ServiceProblem(
                        503,
                        "insufficient_vram",
                        "OpenVoice GPU allocation failed because available VRAM was insufficient.",
                        retry_after=RETRY_COOLDOWN_SECONDS,
                    )
                )
                torch.cuda.empty_cache()
                log.exception("OpenVoice model loading ran out of VRAM", extra={"event": "vc_oom"})
            except Exception as exc:
                self._set_error(
                    ServiceProblem(
                        503,
                        "model_initialization_failed",
                        f"OpenVoice initialization failed: {type(exc).__name__}: {exc}",
                    )
                )
                log.exception("OpenVoice initialization failed", extra={"event": "vc_init_error"})
            finally:
                with self._lock:
                    self.loading = False
            self._stop.wait(RETRY_COOLDOWN_SECONDS)

    def get_ready(self) -> tuple[OpenVoiceConverter, torch.Tensor]:
        with self._lock:
            if self.converter is None or self.source_embedding is None:
                raise ServiceProblem(503, self.error_code, self.error_message)
            return self.converter, self.source_embedding

    def snapshot(self) -> dict:
        with self._lock:
            ready = self.converter is not None and self.source_embedding is not None
            payload = {
                "status": "ok" if ready else "not_ready",
                "ready": ready,
                "loading": self.loading,
                "engine": "openvoice-v2",
                "device": DEVICE,
                "precision": PRECISION,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "openvoice_commit": OPENVOICE_COMMIT,
                "cached_targets": len(TARGET_CACHE),
                "queue": GPU_QUEUE.snapshot(),
                "error_code": self.error_code or None,
                "error": self.error_message or None,
                "configured_min_free_vram_mib": MIN_FREE_VRAM_MIB,
                "recommended_min_free_vram_mib": self.recommended_min_free_mib,
            }
            payload.update(self.metrics)
            return payload


MANAGER = ModelManager()
app = FastAPI(title="MetaRang OpenVoice V2", docs_url=None, redoc_url=None, openapi_url=None)


@app.on_event("startup")
def _startup() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    MANAGER.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    MANAGER.stop()


@app.exception_handler(ServiceProblem)
async def _service_problem_handler(_request: Request, problem: ServiceProblem) -> JSONResponse:
    headers = {"Retry-After": str(problem.retry_after)} if problem.retry_after else None
    return JSONResponse(
        status_code=problem.status,
        content={"error": problem.message, "code": problem.code},
        headers=headers,
    )


@app.get("/health/live")
@app.get("/health")
def health_live() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/health/ready")
def health_ready() -> JSONResponse:
    payload = MANAGER.snapshot()
    return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/samples/validate")
def validate_sample(target: str = Query(...)) -> dict:
    MANAGER.get_ready()
    try:
        sample_path = _resolve_sample(target)
        sample_bytes = sample_path.read_bytes()
        _, duration, voiced = _normalized_reference(sample_bytes)
    except FileNotFoundError as exc:
        raise ServiceProblem(404, "voice_sample_not_found", str(exc)) from exc
    except ValueError as exc:
        raise ServiceProblem(400, "invalid_voice_sample", str(exc)) from exc
    return {
        "status": "ok",
        "filename": sample_path.name,
        "size_bytes": len(sample_bytes),
        "duration_seconds": duration,
        "voiced_seconds": voiced,
    }


def _target_embedding(
    converter: OpenVoiceConverter, reference_bytes: bytes
) -> tuple[torch.Tensor, float, bool, float, float]:
    normalized_pcm, duration, voiced_seconds = _normalized_reference(reference_bytes)
    key = _reference_cache_key(normalized_pcm)
    cached = TARGET_CACHE.get(key)
    if cached is not None:
        return cached, 0.0, True, duration, voiced_seconds
    with GPU_INFERENCE_LOCK:
        embedding, metrics = converter.extract_se(reference_bytes)
    TARGET_CACHE.put(key, embedding)
    TARGET_EMBEDDING_DURATION.observe(metrics["embedding_time"])
    return embedding, metrics["embedding_time"], False, duration, voiced_seconds


def _convert(
    source_bytes: bytes,
    reference_bytes: bytes,
    request: Request,
    *,
    tau: float = 0.3,
) -> Response:
    queue_wait = GPU_QUEUE.acquire()
    try:
        converter, source_embedding = MANAGER.get_ready()
        context = extract_trace_context(request.headers)
        with tracer.start_as_current_span("voice_conversion.convert", context=context) as span:
            target_embedding, target_time, cache_hit, reference_duration, voiced_seconds = (
                _target_embedding(converter, reference_bytes)
            )
            with GPU_INFERENCE_LOCK:
                wav_bytes, conversion = converter.convert(
                    source_bytes, source_embedding, target_embedding, tau=tau
                )
            CONVERSION_DURATION.observe(conversion["conversion_time"])
            REQUESTS.labels("success").inc()
            span.set_attribute("vc.engine", "openvoice-v2")
            span.set_attribute("vc.device", DEVICE)
            span.set_attribute("vc.target_cache_hit", cache_hit)
            span.set_attribute("vc.conversion_rtf", conversion["conversion_rtf"])
            headers = {
                "X-MetaRang-VC-Engine": "openvoice-v2",
                "X-MetaRang-VC-Device": DEVICE,
                "X-MetaRang-Queue-Wait": f"{queue_wait:.6f}",
                "X-MetaRang-Source-Embedding-Time": "0.000000",
                "X-MetaRang-Target-Embedding-Time": f"{target_time:.6f}",
                "X-MetaRang-Target-Cache-Hit": str(cache_hit).lower(),
                "X-MetaRang-Conversion-Time": f"{conversion['conversion_time']:.6f}",
                "X-MetaRang-Conversion-RTF": f"{conversion['conversion_rtf']:.6f}",
                "X-MetaRang-Audio-Duration": f"{conversion['audio_duration']:.6f}",
                "X-MetaRang-Reference-Duration": f"{reference_duration:.6f}",
                "X-MetaRang-Reference-Voiced": f"{voiced_seconds:.6f}",
                "X-MetaRang-GPU-Peak-Allocated-MiB": f"{conversion['peak_allocated_mib']:.3f}",
                "X-MetaRang-GPU-Peak-Reserved-MiB": f"{conversion['peak_reserved_mib']:.3f}",
            }
            log.info(
                "OpenVoice conversion completed",
                extra={
                    "event": "vc_complete",
                    "queue_wait_ms": round(queue_wait * 1000, 1),
                    "target_embedding_ms": round(target_time * 1000, 1),
                    "conversion_ms": round(conversion["conversion_time"] * 1000, 1),
                    "conversion_rtf": round(conversion["conversion_rtf"], 4),
                    "target_cache_hit": cache_hit,
                },
            )
            return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
    except ServiceProblem:
        REQUESTS.labels("unavailable").inc()
        raise
    except ValueError as exc:
        REQUESTS.labels("invalid").inc()
        raise ServiceProblem(400, "invalid_audio", str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        REQUESTS.labels("oom").inc()
        raise ServiceProblem(
            503,
            "insufficient_vram",
            "OpenVoice conversion could not allocate enough GPU memory.",
            retry_after=RETRY_COOLDOWN_SECONDS,
        ) from exc
    except Exception as exc:
        REQUESTS.labels("error").inc()
        log.exception("OpenVoice conversion failed", extra={"event": "vc_error"})
        raise ServiceProblem(
            500, "voice_conversion_failed", f"Voice conversion failed: {type(exc).__name__}."
        ) from exc
    finally:
        GPU_QUEUE.release()


@app.post("/convert")
async def convert_persistent(request: Request, target: str = Query(...)) -> Response:
    try:
        reference_bytes = _resolve_sample(target).read_bytes()
    except FileNotFoundError as exc:
        raise ServiceProblem(404, "voice_sample_not_found", str(exc)) from exc
    content_length = int(request.headers.get("content-length", "0") or "0")
    if content_length <= 0 or content_length > MAX_SOURCE_BYTES:
        raise ServiceProblem(413, "invalid_source_size", "Source audio is empty or too large.")
    source_bytes = await request.body()
    return await run_in_threadpool(_convert, source_bytes, reference_bytes, request)


@app.post("/v1/convert")
async def convert_temporary(
    request: Request,
    source_audio: UploadFile = File(...),
    reference_audio: UploadFile = File(...),
    tau: float = Form(0.3),
) -> Response:
    if not 0.0 <= tau <= 1.0:
        raise ServiceProblem(400, "invalid_tau", "tau must be between 0 and 1.")
    source_bytes = await source_audio.read(MAX_SOURCE_BYTES + 1)
    reference_bytes = await reference_audio.read(MAX_UPLOAD_BYTES + 1)
    await source_audio.close()
    await reference_audio.close()
    if not source_bytes or len(source_bytes) > MAX_SOURCE_BYTES:
        raise ServiceProblem(413, "invalid_source_size", "Source audio is empty or too large.")
    if not reference_bytes or len(reference_bytes) > MAX_UPLOAD_BYTES:
        raise ServiceProblem(413, "invalid_reference_size", "Reference audio is empty or too large.")
    return await run_in_threadpool(
        _convert,
        source_bytes,
        reference_bytes,
        request,
        tau=tau,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, workers=1, access_log=False)
