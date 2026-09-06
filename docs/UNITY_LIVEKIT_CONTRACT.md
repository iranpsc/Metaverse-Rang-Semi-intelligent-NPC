# Unity-facing LiveKit contract

Unity treats the AI as another participant. It does not call STT, RAG, LLM, TTS, or voice-conversion services.

## Connection sequence

1. Call `POST /api/v1/agent/sessions/` over HTTPS.
2. Read `livekit_url`, `access_token`, `room_name`, and `session_id` from the response.
3. Connect the LiveKit Unity SDK to `livekit_url` with `access_token` before the token expires.
4. Publish exactly one microphone audio track with LiveKit source `Microphone`.
5. Subscribe to the remote participant whose identity starts with `agent-` and play its audio track named `agent-audio`.
6. Register a LiveKit text-stream handler for topic `metarang.events.v1` before or immediately after connecting.
7. Parse every complete text stream as one UTF-8 JSON event.
8. Disconnect normally. Optionally call `DELETE /api/v1/agent/sessions/{session_id}/` to close the room immediately.

Do not send raw PCM to Django or open the legacy `/ws/audio-transcription/` route for new integrations. The LiveKit SDK handles capture encoding, WebRTC transport, jitter, and playback transport.

## Room invariants

- One room represents one conversation.
- One standard human participant and one agent participant are allowed (`max_participants=2`).
- The Unity token is scoped to that room and expires after ten minutes by default.
- The Unity participant may publish only a microphone source and may subscribe. It has no room-admin or data-publish permission.
- AI output is mono audio published at 48 kHz. Let the LiveKit SDK/playback device handle final device conversion.

## Text stream topic and schemas

Topic: `metarang.events.v1`

All timestamps are Unix seconds with fractional precision. `turn_id` correlates transcript, state, error, metrics, and trace data for one user utterance.

### Transcription

```json
{
  "type": "transcription",
  "version": 1,
  "session_id": "uuid",
  "turn_id": "opaque string",
  "speaker": "user or agent",
  "text": "display text",
  "final": true,
  "timestamp": 1788264000.123
}
```

User transcripts are currently final-only because the existing Whisper provider is utterance-based. Agent transcripts may arrive as cumulative partial updates followed by one final event. Consumers should replace the displayed text for the same `(turn_id, speaker)` when `final` is false, then commit it when `final` is true.

### Pipeline state

```json
{
  "type": "pipeline_state",
  "version": 1,
  "session_id": "uuid",
  "turn_id": "opaque string or null",
  "state": "listening",
  "timestamp": 1788264000.123
}
```

Valid states: `listening`, `transcribing`, `thinking`, `speaking`, and `ready`. Unknown future values should be ignored instead of treated as fatal.

### Pipeline error

```json
{
  "type": "pipeline_error",
  "version": 1,
  "session_id": "uuid",
  "turn_id": "opaque string or null",
  "stage": "tts",
  "code": "tts_failed",
  "message": "The AI pipeline could not complete this turn.",
  "recoverable": true,
  "timestamp": 1788264000.123
}
```

For a recoverable error, keep the room connected and wait for the next `listening` state. For a room/connection failure, use the LiveKit SDK reconnect behavior; request a new session if its token has expired or the room was deleted.

## Client behavior notes

- Echo cancellation, microphone permission, and device selection belong in the client.
- There is no barge-in in version 1. Audio spoken while the AI is processing or speaking is intentionally ignored.
- Never cache a session token beyond `expires_at`, log it, or ship the LiveKit API secret in the Unity build.
- Do not depend on exact agent identity suffixes; depend on participant kind/metadata or the documented `agent-` prefix and `agent-audio` track name.
