"""Persistent CPU-only Mana Persian Piper synthesis service.

Public internal contract (kept backward compatible):
    POST /synthesize  JSON {"text": "..."} -> audio/wav

The service never selects an ONNX CUDA provider. It also exposes one private
canonical-source endpoint used by the OpenVoice service to derive the fixed
Mana speaker embedding without persisting that embedding or calibration WAV.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import resource
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from pipeline_core.observability import configure_observability, extract_trace_context, get_tracer
from pipeline_core.persian_text import PersianTextNormalizer


SERVICE = "tts"
configure_observability(SERVICE)
log = logging.getLogger(__name__)
tracer = get_tracer(__name__)

REQUESTS = Counter("metarang_tts_requests_total", "TTS requests", ("result", "temperature"))
DURATION = Histogram("metarang_tts_duration_seconds", "Piper synthesis duration")
AUDIO_DURATION = Histogram("metarang_tts_audio_duration_seconds", "Generated audio duration")
RTF = Histogram("metarang_tts_rtf", "Piper real-time factor")
MODEL_LOAD = Gauge("metarang_tts_model_load_seconds", "Piper model load duration")
MODEL_READY = Gauge("metarang_tts_model_ready", "Whether the Piper model is ready")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("TTS_MODEL_PATH", BASE_DIR / "models/mana/fa_IR-mana-medium.onnx"))
CONFIG_PATH = Path(os.getenv("TTS_CONFIG_PATH", f"{MODEL_PATH}.json"))
MODEL_REVISION = os.getenv(
    "TTS_MODEL_REVISION", "ad9dd8518bedf517bd7cbc9f63b8e5c844bf5bc0"
)
HOST = os.getenv("TTS_HOST", "127.0.0.1")
PORT = int(os.getenv("TTS_PORT", "5002"))
DEVICE = os.getenv("TTS_DEVICE", "cpu").strip().lower()
CPU_THREADS = max(1, int(os.getenv("TTS_CPU_THREADS", "4")))
MAX_TEXT_LENGTH = max(1, int(os.getenv("TTS_MAX_TEXT_LENGTH", "2000")))
MAX_NORMALIZED_TEXT_LENGTH = max(1, int(os.getenv("TTS_MAX_NORMALIZED_TEXT_LENGTH", "12000")))
SENTENCE_SILENCE = max(0.0, float(os.getenv("TTS_SENTENCE_SILENCE_SECONDS", "0.12")))
TRAILING_SILENCE = max(0.0, float(os.getenv("TTS_TRAILING_SILENCE_SECONDS", "0.15")))

CALIBRATION_VERSION = "mana-fa-v1"
CALIBRATION_SENTENCES = (
    "سلام، من صدای پایه سامانه متارنگ هستم.",
    "امروز هوا آرام است و این نمونه برای کالیبراسیون تبدیل صدا تولید شده است.",
    "اعداد یک، دو و سه را همراه با جمله‌های کوتاه و بلند، طبیعی و روشن می‌خوانم.",
)
CALIBRATION_TEXT = " ".join(CALIBRATION_SENTENCES)

if DEVICE != "cpu":
    raise RuntimeError("Mana Piper is CPU-only; TTS_DEVICE must be 'cpu'")

# Defense in depth: the image has no NVIDIA runtime, and these variables also
# prevent libraries imported later from discovering a GPU accidentally.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))

import onnxruntime  # noqa: E402
from piper import PiperVoice  # noqa: E402
from piper.config import PiperConfig  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_voice() -> tuple[PiperVoice, float]:
    if not MODEL_PATH.is_file() or not CONFIG_PATH.is_file():
        raise RuntimeError(f"Mana Piper model files are missing under {MODEL_PATH.parent}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = PiperConfig.from_dict(json.load(config_file))

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    started = time.perf_counter()
    session = onnxruntime.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    elapsed = time.perf_counter() - started
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected Piper execution providers: {providers}")
    return PiperVoice(session=session, config=config), elapsed


MODEL_SHA256 = _sha256(MODEL_PATH)
CONFIG_SHA256 = _sha256(CONFIG_PATH)
VOICE, MODEL_LOAD_SECONDS = _load_voice()
MODEL_LOAD.set(MODEL_LOAD_SECONDS)
MODEL_READY.set(1)
NORMALIZER = PersianTextNormalizer.from_env()
SYNTH_LOCK = threading.Lock()

log.info(
    "Mana Piper loaded on CPU",
    extra={
        "event": "tts_model_loaded",
        "device": "cpu",
        "providers": VOICE.session.get_providers(),
        "cpu_threads": CPU_THREADS,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "duration_ms": round(MODEL_LOAD_SECONDS * 1000, 1),
    },
)


def _validate_setting(data: dict, name: str, minimum: float, maximum: float) -> float | None:
    value = data.get(name)
    if value is None:
        return None
    value = float(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return value


def _audio_duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as recording:
        return recording.getnframes() / recording.getframerate()


def synthesize_wav_bytes(text: str, settings: dict | None = None) -> tuple[bytes, dict]:
    settings = settings or {}
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text exceeds {MAX_TEXT_LENGTH} characters")
    normalize_started = time.perf_counter()
    normalized = NORMALIZER.normalize(text).normalized
    normalization_time = time.perf_counter() - normalize_started
    if not normalized:
        raise ValueError("text is empty after normalization")
    if len(normalized) > MAX_NORMALIZED_TEXT_LENGTH:
        raise ValueError(f"normalized text exceeds {MAX_NORMALIZED_TEXT_LENGTH} characters")

    length_scale = _validate_setting(settings, "length_scale", 0.5, 2.0)
    noise_scale = _validate_setting(settings, "noise_scale", 0.0, 2.0)
    noise_w = _validate_setting(settings, "noise_w", 0.0, 2.0)

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    buffer = io.BytesIO()
    with SYNTH_LOCK, wave.open(buffer, "wb") as wav_file:
        VOICE.synthesize(
            normalized,
            wav_file,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
            sentence_silence=SENTENCE_SILENCE,
        )
        wav_file.writeframes(bytes(int(VOICE.config.sample_rate * TRAILING_SILENCE) * 2))

    wav_bytes = buffer.getvalue()
    elapsed = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    audio_seconds = _audio_duration(wav_bytes)
    metrics = {
        "normalization_time": normalization_time,
        "tts_time": elapsed,
        "audio_duration": audio_seconds,
        "tts_rtf": elapsed / audio_seconds if audio_seconds else 0.0,
        "cpu_time": cpu_seconds,
        "cpu_percent": (cpu_seconds / elapsed * 100.0) if elapsed else 0.0,
        "ram_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "sample_rate": VOICE.config.sample_rate,
    }
    return wav_bytes, metrics


# The first inference is deliberately captured at service initialization. It is
# both the cold-start benchmark and the canonical Mana source recording.
_cold_started = time.perf_counter()
CANONICAL_WAV, CANONICAL_METRICS = synthesize_wav_bytes(CALIBRATION_TEXT)
COLD_SYNTHESIS_SECONDS = time.perf_counter() - _cold_started
CANONICAL_SHA256 = hashlib.sha256(CANONICAL_WAV).hexdigest()
log.info(
    "Mana Piper cold synthesis and canonical source completed",
    extra={
        "event": "tts_cold_start_complete",
        "duration_ms": round(COLD_SYNTHESIS_SECONDS * 1000, 1),
        "audio_duration": CANONICAL_METRICS["audio_duration"],
        "tts_rtf": CANONICAL_METRICS["tts_rtf"],
    },
)


class Handler(BaseHTTPRequestHandler):
    server_version = "MetaRangManaPiper/1"

    def _send_json(self, status: int, payload: dict, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_wav(self, wav_bytes: bytes, metrics: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("X-MetaRang-TTS-Engine", "mana-piper")
        self.send_header("X-MetaRang-TTS-Device", "cpu")
        if metrics:
            self.send_header("X-MetaRang-Normalization-Time", f"{metrics['normalization_time']:.6f}")
            self.send_header("X-MetaRang-TTS-Time", f"{metrics['tts_time']:.6f}")
            self.send_header("X-MetaRang-Audio-Duration", f"{metrics['audio_duration']:.6f}")
            self.send_header("X-MetaRang-TTS-RTF", f"{metrics['tts_rtf']:.6f}")
            self.send_header("X-MetaRang-CPU-Time", f"{metrics['cpu_time']:.6f}")
            self.send_header("X-MetaRang-CPU-Percent", f"{metrics['cpu_percent']:.3f}")
            self.send_header("X-MetaRang-RAM-Peak-MiB", f"{metrics['ram_peak_mib']:.3f}")
            self.send_header("X-MetaRang-Sample-Rate", str(metrics["sample_rate"]))
        self.end_headers()
        self.wfile.write(wav_bytes)

    def do_GET(self) -> None:
        if self.path in {"/health", "/health/live"}:
            self._send_json(200, {"status": "ok", "service": SERVICE})
            return
        if self.path == "/health/ready":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "engine": "mana-piper",
                    "device": "cpu",
                    "providers": VOICE.session.get_providers(),
                    "cpu_threads": CPU_THREADS,
                    "model_revision": MODEL_REVISION,
                    "model_sha256": MODEL_SHA256,
                    "config_sha256": CONFIG_SHA256,
                    "model_load_seconds": MODEL_LOAD_SECONDS,
                    "cold_synthesis_seconds": COLD_SYNTHESIS_SECONDS,
                    "cold_metrics": CANONICAL_METRICS,
                },
            )
            return
        if self.path == "/canonical-source":
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(CANONICAL_WAV)))
            self.send_header("X-MetaRang-Calibration-Version", CALIBRATION_VERSION)
            self.send_header("X-MetaRang-Piper-Model-SHA256", MODEL_SHA256)
            self.send_header("X-MetaRang-Canonical-SHA256", CANONICAL_SHA256)
            self.end_headers()
            self.wfile.write(CANONICAL_WAV)
            return
        if self.path == "/metrics":
            body = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json(404, {"error": "not found", "code": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/synthesize":
            self._send_json(404, {"error": "not found", "code": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65_536:
                raise ValueError("invalid body length")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(data.get("text") or "").strip()
            if not text:
                raise ValueError("text is empty")
            settings = data.get("settings") or {}
            if not isinstance(settings, dict):
                raise ValueError("settings must be an object")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc), "code": "invalid_request"})
            return

        temperature = "warm"  # initialization always performs the one cold synthesis
        context = extract_trace_context(self.headers)
        with tracer.start_as_current_span("tts.synthesize", context=context) as span:
            try:
                wav_bytes, metrics = synthesize_wav_bytes(text, settings)
            except ValueError as exc:
                REQUESTS.labels("invalid", temperature).inc()
                self._send_json(400, {"error": str(exc), "code": "invalid_request"})
                return
            except Exception:
                REQUESTS.labels("error", temperature).inc()
                log.exception("Mana Piper synthesis failed", extra={"event": "tts_error"})
                self._send_json(500, {"error": "synthesis failed", "code": "tts_failed"})
                return

            REQUESTS.labels("success", temperature).inc()
            DURATION.observe(metrics["tts_time"])
            AUDIO_DURATION.observe(metrics["audio_duration"])
            RTF.observe(metrics["tts_rtf"])
            span.set_attribute("tts.engine", "mana-piper")
            span.set_attribute("tts.device", "cpu")
            span.set_attribute("tts.audio_duration", metrics["audio_duration"])
            span.set_attribute("tts.rtf", metrics["tts_rtf"])
            log.info(
                "Mana Piper synthesis completed",
                extra={
                    "event": "tts_complete",
                    "duration_ms": round(metrics["tts_time"] * 1000, 1),
                    "audio_duration": round(metrics["audio_duration"], 3),
                    "rtf": round(metrics["tts_rtf"], 4),
                    "cpu_percent": round(metrics["cpu_percent"], 1),
                    "ram_peak_mib": round(metrics["ram_peak_mib"], 1),
                },
            )
            self._send_wav(wav_bytes, metrics)

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s %s", self.address_string(), fmt % args)


class Server(ThreadingHTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    server = Server((HOST, PORT), Handler)
    log.info("Mana Piper server listening on http://%s:%s", HOST, PORT)
    server.serve_forever()
