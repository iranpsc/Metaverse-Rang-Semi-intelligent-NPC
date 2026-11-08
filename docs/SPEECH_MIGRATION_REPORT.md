# Speech architecture migration report

## Status

The production speech path has been migrated from Coqui TTS + FreeVC/WavLM to:

- Mana Persian Piper on CPU only.
- OpenVoice V2 ToneColorConverter on the NVIDIA Tesla P40 in FP32.
- One shared STT, LLM/RAG, TTS, and VC service path for Django pages, REST APIs,
  WebSockets, and the LiveKit/Unity worker.

Automated functional and performance acceptance has passed. Human listening
acceptance has **not** been performed, so this report does not claim Persian
naturalness or target-speaker similarity approval.

## Pinned model provenance

| Component | Revision | Files / SHA256 |
|---|---|---|
| Mana Persian Piper | `ad9dd8518bedf517bd7cbc9f63b8e5c844bf5bc0` | ONNX `e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126`; config `d102276a05436613a05e1fbad8f125abcf8f56759aaf4fd36adf216653949997` |
| OpenVoice V2 checkpoints | `ea66dc9353e660b1dbbe0bebebd4c2d8b105483c` | checkpoint `9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e`; config `9dfff60350b8c63f2c664efd92a61b2516efb22671466960f0e5dfebd881fa47` |
| OpenVoice source | `74a1d147b17a8c3092dd5430504bd83ef6c7eb23` | Minimal official converter runtime, MIT |

Only the required model files are downloaded. `scripts/download_speech_models.py
--verify-only` verifies the pinned checksums.

## Runtime behavior

- Mana uses ONNX Runtime `CPUExecutionProvider`, 4 intra-op threads, and 1
  inter-op thread. The TTS container has no NVIDIA device request.
- OpenVoice runs on `cuda:0`, FP32, PyTorch `2.5.1+cu118`. The Tesla P40 was
  verified as compute capability 6.1 with an `sm_60` compatible CUDA binary.
- The measured OpenVoice startup threshold is 765 MiB free VRAM, including the
  observed peak plus the configured safety margin.
- VC permits one active GPU request and four queued requests. Queue overflow or
  timeout returns HTTP 429 with `Retry-After`.
- Target embeddings use an in-memory CPU LRU cache (32 entries, 30-minute TTL).
  Source and target embeddings are moved to CUDA only for conversion.
- Persistent voice samples and direct, non-persistent multipart references are
  both supported. Uploads are normalized to mono 16 kHz PCM and checked for
  actual voiced duration; 6–15 seconds of clean speech is recommended.
- `VC_FAILURE_FALLBACK` defaults to `false`. When enabled, fallback is explicitly
  reported with `conversion_status=fallback` and `fallback_used=true`.

## Public API and Swagger

Existing endpoints remain compatible:

- `POST /api/tts/`
- `POST /api/vc/?sample_id=...`
- `POST /api/v1/voice-samples/`
- Existing LiveKit session creation and `voice_sample_id` behavior.

The combined endpoint is:

- `POST /api/v1/speech/jobs/`

It accepts Persian `text` plus exactly one of persistent `sample_id` or temporary
`reference_audio`. `voice_sample_id` remains accepted as a deprecated alias. It
returns JSON URLs for one-hour source/converted WAV artifacts and flat TTS/VC/E2E
metrics. Swagger is available at `/api/docs/` and the OpenAPI schema at
`/api/schema/`.

## Measured benchmark

The latest machine-readable report is
`artifacts/benchmarks/speech-pipeline-latest.json`; the readable report is
`artifacts/benchmarks/speech-pipeline-latest.md`. It was generated at
`2026-11-08T01:46:08.612601+00:00` from three real Persian API requests.

| Measurement | Result |
|---|---:|
| Mana model load | 1.542 s |
| Mana cold synthesis | 0.860 s |
| Mana cold RTF | 0.0620 |
| Mana cold RAM peak | 279.9 MiB |
| OpenVoice model load | 0.735 s |
| OpenVoice source embedding | 0.0087 s |
| OpenVoice peak allocated VRAM | 252.6 MiB |
| OpenVoice peak reserved VRAM | 274.0 MiB |
| First combined request | 0.644 s |
| Warm average combined request | 0.601 s |
| Warm average TTS RTF | 0.0546 |
| Warm average conversion RTF | 0.0231 |
| Warm end-to-end RTF | 0.1083 |
| Requests / failures | 3 / 0 |
| Peak sampled GPU utilization | 63% |

The sampled total GPU memory includes other processes on the shared P40; the
OpenVoice-specific CUDA allocation is reported separately above.

## Functional verification

- Django/API suite: 56 tests passed.
- Model checksum verification: passed.
- TTS candidate: valid mono 22,050 Hz Persian WAV; CPU provider only.
- OpenVoice: real CUDA conversion passed; cache miss/hit and bounded queue
  saturation behavior passed.
- Persistent `sample_id` combined API: completed with `fallback_used=false`.
- Temporary reference combined API: 3/3 completed with `fallback_used=false`.
- Project healthcheck: Django live/ready, LiveKit, Prometheus, and Grafana all
  passed (5/5).
- Full WebRTC pipeline: a real LiveKit client published Persian audio and
  received final user/agent transcript events, RAG retrieval, speaking state,
  two successful OpenVoice conversions, and non-silent returned agent audio.

The synthetic LiveKit input was not transcribed word-for-word by Whisper, so the
test proves transport and stage integration, not STT accuracy or speech quality.

## Rollback

The legacy checkpoints remain untouched and the immutable images are preserved:

- `metarang-npc-tts:rollback-coqui-20261102`
- `metarang-npc-vc:rollback-freevc-20261102`

Use `deploy/rollback/docker-compose.speech-legacy-20261102.yml` as documented in
`deploy/rollback/SPEECH_LEGACY_20261102.md`. The old and new engines must never be
loaded simultaneously.

## Deployment note

The deployed web control-plane source is currently a tested code-only overlay
(`docker/Dockerfile.web-offline`) on the preserved pre-migration web image. It
has no GPU device request, all model imports were removed/lazy-loaded, and the
56-test suite passed inside the baked image, so it does not instantiate speech
or training models. The normal root `Dockerfile` and `requirements/web.txt` are
ready to produce the smaller model-free image, but that packaging build could
not finish because the host repeatedly timed out/reset connections to official
PyPI. This is a disk/image-size cleanup item, not a runtime pipeline blocker.
Rebuild and replace only `web` after PyPI connectivity returns.

## Remaining acceptance work

Before declaring speech quality production-approved, perform listening tests
with authorized 6–15 second Persian references and score intelligibility, word
preservation, similarity, naturalness, prosody, artifacts, and cross-speaker
consistency. No MOS or similarity score has been inferred from automated tests.
