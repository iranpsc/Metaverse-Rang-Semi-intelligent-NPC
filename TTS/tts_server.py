"""
Persistent TTS microservice.

Runs inside the dedicated TTS venv (TTS/tts/) because Coqui TTS needs
torch/transformers versions that conflict with the main Django environment.
The model is loaded once at startup; Django proxies synthesis requests here.

Start it with:  ./start_tts_server.sh   (or: ./tts/bin/python3 tts_server.py)

API:
    GET  /health      -> {"status": "ok", "cuda": bool}
    POST /synthesize  -> body {"text": "..."}  returns audio/wav
"""

import io
import json
import logging
import os
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from pipeline_core.observability import configure_observability, extract_trace_context, get_tracer

configure_observability("tts")
log = logging.getLogger(__name__)
tracer = get_tracer(__name__)
REQUESTS = Counter("metarang_tts_requests_total", "TTS requests", ("result",))
DURATION = Histogram("metarang_tts_duration_seconds", "TTS synthesis duration")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("TTS_MODEL_PATH", os.path.join(BASE_DIR, "best_model_91323.pth"))
CONFIG_PATH = os.environ.get("TTS_CONFIG_PATH", os.path.join(BASE_DIR, "config.json"))
HOST = os.environ.get("TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_PORT", "5002"))
MAX_TEXT_LENGTH = 2000

import numpy as np  # noqa: E402
import torch  # noqa: E402
from TTS.utils.synthesizer import Synthesizer  # noqa: E402

USE_CUDA = torch.cuda.is_available()
log.info("Loading model %s (cuda=%s)...", MODEL_PATH, USE_CUDA)
SYNTH = Synthesizer(
    tts_checkpoint=MODEL_PATH,
    tts_config_path=CONFIG_PATH,
    use_cuda=USE_CUDA,
)
log.info("Model loaded, sample rate %s", SYNTH.output_sample_rate)

# Warmup: the first GPU inference pays CUDA allocation overhead
try:
    SYNTH.tts("سلام")
    log.info("Warmup synthesis done")
except Exception:
    log.exception("Warmup synthesis failed")

# Coqui's synthesizer is not thread-safe; serialize synthesis calls.
SYNTH_LOCK = threading.Lock()


def clean_text(text):
    """Strip markdown syntax and normalize whitespace so it isn't read aloud."""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_#`>|~]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def synthesize_wav_bytes(text):
    with SYNTH_LOCK:
        wav = SYNTH.tts(text)
    # Trailing silence so back-to-back clips don't end abruptly
    pad = np.zeros(int(0.15 * SYNTH.output_sample_rate), dtype=np.float32)
    wav = np.concatenate([np.asarray(wav, dtype=np.float32), pad])
    buf = io.BytesIO()
    try:
        SYNTH.save_wav(wav, buf)
        return buf.getvalue()
    except Exception:
        # scipy versions that reject file-like objects: go through a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            SYNTH.save_wav(wav, tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in {"/health", "/health/live", "/health/ready"}:
            self._send_json(200, {"status": "ok", "cuda": USE_CUDA})
        elif self.path == "/metrics":
            body = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/synthesize":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 32_768:
                raise ValueError("invalid body length")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        text = clean_text(str(data.get("text") or ""))
        if not text:
            self._send_json(400, {"error": "text is empty"})
            return
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        started = time.monotonic()
        context = extract_trace_context(self.headers)
        with tracer.start_as_current_span("tts.synthesize", context=context) as span:
            try:
                wav_bytes = synthesize_wav_bytes(text)
            except Exception:
                REQUESTS.labels("error").inc()
                log.exception("Synthesis failed", extra={"event": "tts_error"})
                self._send_json(500, {"error": "synthesis failed"})
                return
            elapsed = time.monotonic() - started
            DURATION.observe(elapsed)
            REQUESTS.labels("success").inc()
            span.set_attribute("tts.text_length", len(text))
            span.add_event("tts.first_audio", {"duration_ms": elapsed * 1000})
            log.info(
                "Synthesis completed",
                extra={"event": "tts_complete", "duration_ms": round(elapsed * 1000, 1)},
            )

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.end_headers()
        self.wfile.write(wav_bytes)

    def log_message(self, format, *args):
        log.info("%s %s", self.address_string(), format % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("TTS server listening on http://%s:%s", HOST, PORT)
    server.serve_forever()
