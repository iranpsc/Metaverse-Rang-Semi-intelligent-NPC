# MetaRang realtime avatar backend

MetaRang exposes the existing Whisper → RAG/LLM → Coqui TTS → optional FreeVC pipeline as one self-hosted LiveKit participant. Django is the control plane; microphone and response audio never travel through Django REST.

## Architecture

```text
Unity client
  │ POST /api/v1/agent/sessions/       (control only)
  ▼
Django ──creates room + explicit dispatch──► self-hosted LiveKit ◄──► Redis DB 1
                                                    │ WebRTC
                                  ┌─────────────────┴─────────────────┐
                                  │ agent-worker (LiveKit participant) │
                                  └───┬────────────┬────────────┬──────┘
                                      │            │            │
                                PCM16 16 kHz   streamed text  sentence WAV
                                      ▼            ▼            ▼
                                  Whisper      RAG + Ollama    Coqui TTS
                                                                    │
                                                          optional FreeVC
                                                                    │
                                      LiveKit audio, PCM16 mono 48 kHz
                                                                    ▼
                                                               Unity client

All services ── structured stdout + OTLP traces + /metrics ──►
OpenTelemetry Collector ──► Loki / Tempo    Prometheus ──► Grafana
```

The existing model implementations remain in `STT`, `LLM`, `TTS`, and `VC`. The new internal STT and LLM HTTP adapters do not expose public ports. `realtime_agent` is the only AI component coupled to LiveKit.

### Audio path

1. Unity publishes its LiveKit microphone track. WebRTC/Opus transport is decoded by the LiveKit SDK.
2. The agent requests mono, 16-bit, 16 kHz, 20 ms PCM frames and holds them in a bounded 500-frame SDK ring.
3. A non-semantic energy/silence endpoint creates one utterance in memory. No temporary audio file is written.
4. Whisper receives raw PCM16 and returns its final utterance transcript.
5. RAG retrieval and Ollama generation stream text tokens. A two-item bounded sentence queue lets TTS start before the whole answer is complete.
6. The current Coqui and FreeVC implementations return one WAV per sentence; they are not chunk-streaming providers. If enabled, FreeVC converts the sentence WAV. This is the remaining full-buffer audio boundary.
7. The agent explicitly decodes PCM16 WAV, downmixes if needed, resamples to mono 48 kHz, splits it into 20 ms frames, and publishes the `agent-audio` LiveKit track.

There is intentionally no semantic endpointing or barge-in. Microphone frames received while the agent is processing/speaking are discarded.

## Local startup

Requirements: Docker Engine with Compose v2, enough disk/RAM for the models, and NVIDIA Container Toolkit if voice conversion is enabled.

```bash
cp .env.example .env
```

Set a LiveKit secret of at least 32 random characters, for example:

```bash
openssl rand -hex 32
```

Then edit `.env` and point the model variables at real host paths. In particular:

- `EMBEDDING_MODEL_PATH_HOST` must contain the sentence-transformer model.
- `TTS_MODEL_PATH_HOST` must contain the Coqui checkpoint matching `TTS/config.json`.
- the three `VC_*_PATH_HOST` values must exist when using FreeVC.
- `LLM_MODEL` must already exist in the Ollama volume. For a registry model, run `docker compose exec ollama ollama pull <model>` after Ollama starts. For the existing custom `dorna2`, create/import it using the project’s original Ollama model workflow.

Start without the optional CUDA-only FreeVC service:

```bash
make up
make health
```

Start the complete voice-conversion path on an NVIDIA host:

```bash
make up-with-vc
```

For clients outside the Docker host, allow/forward `7881/tcp` and `7882/udp`
to the host running LiveKit. HTTPS/WSS reverse proxying covers signaling only;
WebRTC media connects directly to the public ICE address advertised by LiveKit.
Set `LIVEKIT_NODE_IP` to the host's real public IPv4 address when this profile
is reachable from remote clients; LiveKit uses that explicit address instead
of STUN discovery or the Docker bridge address. Do not leave the
documentation-only example address in `.env`.

Useful endpoints:

| Purpose | URL |
|---|---|
| Swagger UI | `http://localhost:8000/api/docs/` |
| OpenAPI schema | `http://localhost:8000/api/schema/` |
| Django liveness/readiness | `http://localhost:8000/api/v1/health/live/`, `/ready/` |
| LiveKit signaling | `ws://localhost:7880` |
| Grafana | `http://localhost:3000` |
| Prometheus | `http://localhost:9091` |

Internal model ports are exposed only to the Compose network.

## Session API

This first integration phase is intentionally unauthenticated. Do not expose it to an untrusted network until authentication and rate limiting are added.

```http
POST /api/v1/agent/sessions/
Content-Type: application/json

{
  "user_id": "player-42",
  "vector_store": "main_store",
  "enable_voice_conversion": false
}
```

For voice conversion, set `enable_voice_conversion` to `true` and provide an existing Django `voice_sample_id`.

```json
{
  "session_id": "2e24ec36-1000-4bd1-88e7-5ab66a216cc0",
  "room_name": "agent-session-2e24ec3610004bd188e7",
  "livekit_url": "ws://localhost:7880",
  "access_token": "<short-lived JWT>",
  "expires_at": "2026-09-01T12:10:00Z"
}
```

The token is server-signed, room-scoped, can subscribe, and can publish only a microphone source. It cannot publish data, administer rooms, or expose the API secret. The default TTL is ten minutes. Django creates a two-participant room and explicitly dispatches `metarang-agent` before returning.

Close a session early with:

```http
DELETE /api/v1/agent/sessions/{session_id}/
```

## User vector stores

Create or extend a persistent vector store for a user by uploading one or more
PDF, DOCX, TXT, or CSV files. The endpoint uses `multipart/form-data` and is
available in Swagger under the **Vector stores** tag.

```bash
curl -X POST http://localhost:8000/api/v1/vector-stores/ \
  -F 'user_id=player-42' \
  -F 'files=@./knowledge.pdf' \
  -F 'files=@./faq.txt'
```

The response includes the identifier to use when creating a session:

```json
{
  "user_id": "player-42",
  "vector_store": "user_player-42/rss_store",
  "ready": true,
  "created": true,
  "files_received": 2,
  "documents_added": 2,
  "message": "Added 2 files to knowledge base."
}
```

```json
{
  "user_id": "player-42",
  "vector_store": "user_player-42/rss_store",
  "enable_voice_conversion": false
}
```

Check whether a user's store is ready with
`GET /api/v1/vector-stores/{user_id}/`. Uploading additional files with the
same `user_id` extends that user's existing store. Upload limits are controlled
by `VECTOR_STORE_MAX_FILES` and `VECTOR_STORE_MAX_UPLOAD_BYTES`.

The same store can be created or refreshed from RSS or Atom feeds. Use
`rss_link` for one feed or `rss_links` for multiple feeds:

```bash
curl -X POST http://localhost:8000/api/v1/vector-stores/rss/ \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "player-42",
    "rss_links": [
      "https://example.com/feed.xml",
      "https://example.org/atom.xml"
    ]
  }'
```

```json
{
  "user_id": "player-42",
  "vector_store": "user_player-42/rss_store",
  "ready": true,
  "created": false,
  "rss_links_received": 2,
  "documents_added": 4,
  "message": "Successfully added 4 new articles."
}
```

The crawler records the newest processed publication date per feed. Calling the
endpoint again with the same user and feed URLs updates the existing index with
new articles instead of creating a separate store. The per-request feed limit
is controlled by `VECTOR_STORE_MAX_RSS_LINKS`.

The complete Unity-facing contract is in [docs/UNITY_LIVEKIT_CONTRACT.md](docs/UNITY_LIVEKIT_CONTRACT.md).

## Voice samples and voice conversion

Upload a target-speaker recording as multipart form data. WAV, MP3, M4A, OGG,
FLAC, and WebM inputs are accepted and normalized to mono 16 kHz PCM WAV for
FreeVC.

```bash
curl -X POST http://localhost:8000/api/v1/voice-samples/ \
  -F 'name=Narrator' \
  -F 'audio=@./narrator.webm'
```

The response includes the sample ID required by the session API:

```json
{
  "id": 7,
  "name": "Narrator",
  "filename": "5f08d93f1ddd43a5b89e5782a34ce943.wav",
  "audio_url": "/media/voice_samples/5f08d93f1ddd43a5b89e5782a34ce943.wav",
  "created_at": "2026-10-07T10:00:00Z"
}
```

Enable voice conversion when creating the LiveKit session:

```json
{
  "user_id": "player-42",
  "vector_store": "user_player-42/rss_store",
  "enable_voice_conversion": true,
  "voice_sample_id": 7
}
```

Before creating the room, Django verifies that the sample file exists in the
shared media volume and asks the FreeVC service to confirm it can see the same
normalized file. Start the stack with the `voice-conversion` profile when this
option is enabled. `GET /api/v1/voice-samples/` lists available sample IDs.
Delete an unused sample and its stored audio with
`DELETE /api/v1/voice-samples/{sample_id}/`.

## Realtime events

Events use LiveKit text streams on topic `metarang.events.v1`. Every payload has `version: 1`.

```json
{
  "type": "transcription",
  "version": 1,
  "session_id": "...",
  "turn_id": "...",
  "speaker": "user",
  "text": "hello world",
  "final": true,
  "timestamp": 1788264000.123
}
```

Agent transcript updates use the same schema with `speaker: "agent"`; they may be partial (`final: false`) and always end with a final event. The current Whisper integration is utterance-based, so user events are final-only. It was not replaced merely to manufacture partial results.

State and failure events are also sent on the same topic:

```json
{"type":"pipeline_state","version":1,"session_id":"...","turn_id":"...","state":"thinking","timestamp":1788264000.2}
```

```json
{"type":"pipeline_error","version":1,"session_id":"...","turn_id":"...","stage":"tts","code":"tts_failed","message":"The AI pipeline could not complete this turn.","recoverable":true,"timestamp":1788264000.3}
```

State values are `listening`, `transcribing`, `thinking`, `speaking`, and `ready` (reserved). Error messages are deliberately generic; operator logs and traces contain the diagnostic detail without exposing it to clients.

## Operations and debugging

```bash
make ps
make logs
make health
docker compose logs -f agent-worker stt llm tts
docker compose exec redis redis-cli -n 1 ping
```

Grafana provisions the **MetaRang Realtime Pipeline** dashboard. Prometheus includes LiveKit rooms/participants, queue depth, error counters, active jobs, and stage latency. Tempo traces use `session.id` and `turn.id`; JSON application logs include trace/span IDs. Loki ingests Docker stdout through the OpenTelemetry Collector. Useful LogQL starts with a container selector followed by `| json`, then filter on `session_id` or `event`.

Raw audio, prompts, retrieved document text, and complete transcripts are not logged. Application logs are sent directly to the collector over OTLP, avoiding host-specific access to Docker's private data root. Infrastructure container logs such as LiveKit, Redis, and Ollama remain available through `docker compose logs`.

## Tests and smoke test

```bash
make test
make config
make smoke WAV=path/to/spoken-test.wav
```

The smoke client creates a session, connects with the Python LiveKit SDK, publishes a WAV as a microphone track, waits for a final user transcript and agent audio, then deletes the room. It requires a working vector store, Ollama model, STT, and TTS; it does not require Unity.

## Scaling

```bash
docker compose up -d --scale agent-worker=4
docker compose up -d --scale stt=2 --scale llm=2 --scale tts=2
```

| Service | Scale behavior |
|---|---|
| `agent-worker` | Stateless per job. LiveKit distributes explicit dispatch jobs; this is the preferred first scaling axis. |
| `stt` | One bounded inference queue/model per replica. Use one worker per GPU unless measured otherwise. |
| `llm` | Safe for replicas on one Compose host; vector stores are read-only during chat and memory files use cross-process locks plus atomic replacement. Ollama may remain the bottleneck. |
| `tts` | Safe per replica; synthesis is serialized within each replica because the model is not thread-safe. |
| `vc` | Cache is local and safe, but each replica needs assigned GPU capacity. Do not blindly let replicas contend for one GPU. |
| `web` | Do not scale with the default SQLite database. Move session persistence to PostgreSQL first. |
| `redis`, `ollama` | Single instances in this Compose topology. Use an appropriately managed/HA topology if their availability requirements increase. |

Compose DNS provides coarse distribution for internal HTTP replicas; persistent keep-alive connections can make it uneven. Add a proper internal load balancer only when measurements justify it.

## Production

Production uses Linux host networking only for LiveKit’s latency-sensitive media server. Start it with:

```bash
make production-up
```

Do not run that command until DNS, certificates, public IP, firewall rules, secrets, allowed origins, model paths, and TURN topology are configured. See [docs/PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md) for the exact checklist and ports.

## Legacy interfaces and known limitations

The old browser panel, Django SSE endpoints, and Channels WebSocket routes remain available for existing testing workflows, but new clients should use LiveKit. The ASGI router now mounts those legacy WebSocket routes correctly.

Current limitations:

- no authentication or rate limiting on session creation;
- final-only user transcripts with Whisper;
- energy/silence endpointing only, with no semantic turn detection or barge-in;
- TTS and FreeVC buffer one sentence WAV because the existing providers are non-streaming;
- FreeVC is CUDA-only and opt-in through the `voice-conversion` profile;
- SQLite prevents safe Django horizontal scaling;
- Compose is a single-host topology; Kubernetes was intentionally not introduced.
