"""
Persistent Voice-Conversion microservice (FreeVC).

Runs inside the dedicated FreeVC venv (VC/freevc-env/) because FreeVC needs
torch 1.13 / librosa 0.8 versions that conflict with the main Django
environment. The models are loaded once at startup; Django proxies conversion
requests here (mirrors the TTS microservice in TTS/tts_server.py).

Given a *source* clip (the words to speak — typically the TTS output) and a
*target* voice sample (a short recording of the voice to imitate), it returns
the source content re-voiced in the target's timbre.

Start it with:  ./start_vc_server.sh
    (or: ./freevc-env/bin/python vc_server.py)

API:
    GET  /health   -> {"status": "ok", "cuda": bool, "cached_targets": int}
    POST /convert  -> raw WAV body (audio/wav) of the source clip,
                      target sample chosen via ?target=<filename>.
                      Returns audio/wav (16 kHz) of the converted clip.
"""

import io
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [VC] %(message)s")
log = logging.getLogger(__name__)

# FreeVC's utils/models use paths relative to the repo root (e.g.
# 'wavlm/WavLM-Large.pt'), exactly like convert.py expects. Run from here.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

HOST = os.environ.get("VC_HOST", "127.0.0.1")
PORT = int(os.environ.get("VC_PORT", "5003"))

HPFILE = os.environ.get("VC_HPFILE", "configs/freevc.json")
PTFILE = os.environ.get("VC_PTFILE", "freevc.pth")
SPK_CKPT = os.environ.get("VC_SPK_CKPT", "speaker_encoder/ckpt/pretrained_bak_5805000.pt")

# Where the recorded target voice samples live. Django saves them to
# <project>/media/voice_samples/; this server reads them directly (same host).
SAMPLES_DIR = os.environ.get(
    "VC_SAMPLES_DIR",
    os.path.abspath(os.path.join(BASE_DIR, "..", "media", "voice_samples")),
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import librosa  # noqa: E402
from scipy.io.wavfile import write  # noqa: E402

import utils  # noqa: E402
from models import SynthesizerTrn  # noqa: E402
from speaker_encoder.voice_encoder import SpeakerEncoder  # noqa: E402

USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    # FreeVC's convert path hard-codes .cuda(); it will not run on CPU.
    log.warning("CUDA is not available. FreeVC requires a GPU and will fail to load.")

log.info("Loading FreeVC config %s ...", HPFILE)
hps = utils.get_hparams_from_file(HPFILE)
SAMPLING_RATE = hps.data.sampling_rate  # 16000

log.info("Building synthesizer and loading checkpoint %s ...", PTFILE)
net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
).cuda()
net_g.eval()
utils.load_checkpoint(PTFILE, net_g, None, True)

log.info("Loading WavLM content encoder ...")
cmodel = utils.get_cmodel(0)

USE_SPK = bool(hps.model.use_spk)
smodel = None
if USE_SPK:
    log.info("Loading speaker encoder %s ...", SPK_CKPT)
    smodel = SpeakerEncoder(SPK_CKPT)

log.info(
    "FreeVC ready (cuda=%s, use_spk=%s, samples_dir=%s)",
    USE_CUDA, USE_SPK, SAMPLES_DIR,
)

# FreeVC inference is not thread-safe and shares one GPU; serialize calls.
INFER_LOCK = threading.Lock()

# Cache the (expensive) target voice representation per sample file. Keyed by
# filename + mtime so re-recording under the same name invalidates the cache.
_target_cache = {}


def _resolve_sample(filename):
    """Map an untrusted filename to a real file under SAMPLES_DIR (no traversal)."""
    name = os.path.basename((filename or "").strip())
    if not name:
        raise FileNotFoundError("empty target filename")
    path = os.path.join(SAMPLES_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"voice sample not found: {name}")
    return path, name


def _target_repr(filename):
    """Return the cached target representation (speaker embedding or mel)."""
    path, name = _resolve_sample(filename)
    key = f"{name}:{os.path.getmtime(path)}"
    cached = _target_cache.get(key)
    if cached is not None:
        return cached

    wav_tgt, _ = librosa.load(path, sr=SAMPLING_RATE)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

    with INFER_LOCK:
        if USE_SPK:
            g = smodel.embed_utterance(wav_tgt)
            repr_ = ("spk", torch.from_numpy(g).unsqueeze(0).cuda())
        else:
            from mel_processing import mel_spectrogram_torch
            wav_t = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            mel = mel_spectrogram_torch(
                wav_t,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            repr_ = ("mel", mel)

    # Bound the cache so a long-running server can't grow without limit.
    if len(_target_cache) > 256:
        _target_cache.clear()
    _target_cache[key] = repr_
    return repr_


def convert(source_wav_bytes, target_filename):
    """Re-voice the source clip in the target sample's voice. Returns WAV bytes."""
    kind, repr_ = _target_repr(target_filename)

    # Source is the content to speak (e.g. the TTS output), a real WAV.
    wav_src, _ = librosa.load(io.BytesIO(source_wav_bytes), sr=SAMPLING_RATE)
    wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()

    with INFER_LOCK, torch.no_grad():
        c = utils.get_content(cmodel, wav_src)
        if kind == "spk":
            audio = net_g.infer(c, g=repr_)
        else:
            audio = net_g.infer(c, mel=repr_)
        audio = audio[0][0].data.cpu().float().numpy()

    # int16 PCM WAV for maximum browser compatibility.
    audio = np.clip(audio, -1.0, 1.0)
    buf = io.BytesIO()
    write(buf, SAMPLING_RATE, (audio * 32767.0).astype(np.int16))
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send_json(200, {
                "status": "ok",
                "cuda": USE_CUDA,
                "use_spk": USE_SPK,
                "cached_targets": len(_target_cache),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/convert":
            self._send_json(404, {"error": "not found"})
            return

        params = parse_qs(parsed.query)
        target = (params.get("target", [""])[0]).strip()
        if not target:
            self._send_json(400, {"error": "missing ?target=<filename>"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"error": "empty source audio body"})
            return
        source_bytes = self.rfile.read(length)

        try:
            wav_bytes = convert(source_bytes, target)
        except FileNotFoundError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        except Exception as exc:
            log.exception("Conversion failed for target %s", target)
            self._send_json(500, {"error": f"conversion failed: {exc}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.end_headers()
        self.wfile.write(wav_bytes)

    def log_message(self, format, *args):
        log.info("%s %s", self.address_string(), format % args)


if __name__ == "__main__":
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("VC server listening on http://%s:%s", HOST, PORT)
    server.serve_forever()
