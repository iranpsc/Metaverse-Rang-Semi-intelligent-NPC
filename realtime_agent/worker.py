"""LiveKit worker: WebRTC audio ↔ existing STT/RAG/TTS/VC services."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, JobRequest, cli
from prometheus_client import Counter, Gauge, Histogram

from pipeline_core.config import AgentSettings
from pipeline_core.contracts import PipelineErrorEvent, PipelineStateEvent, TranscriptionEvent
from pipeline_core.observability import configure_observability, get_tracer

from .audio import EndpointConfig, EnergyEndpointDetector, pcm_frames, wav_to_pcm16_mono
from .client import PipelineClient, PipelineServiceError


SERVICE = "realtime-agent"
configure_observability(SERVICE)
logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
settings = AgentSettings.from_env()

ACTIVE_JOBS = Gauge("metarang_agent_active_jobs", "Active realtime agent jobs")
TURNS = Counter("metarang_agent_turns_total", "Conversation turns", ("result",))
FIRST_AUDIO = Histogram(
    "metarang_agent_speech_end_to_first_audio_seconds",
    "Time from user endpoint to first AI audio frame",
)
VC_FALLBACKS = Counter("metarang_agent_vc_fallbacks_total", "Voice conversion fallbacks")
RECONNECTS = Counter("metarang_agent_reconnects_total", "LiveKit room reconnect attempts")
PARTICIPANT_FAILURES = Counter(
    "metarang_agent_participant_failures_total", "Expected participant connection failures"
)
JOB_DURATION = Histogram(
    "metarang_agent_job_duration_seconds", "Duration of a realtime agent room assignment"
)

server = AgentServer(prometheus_port=settings.metrics_port)


class SentenceBuffer:
    def __init__(self, max_chars: int = 280):
        self.text = ""
        self.max_chars = max_chars

    def push(self, chunk: str) -> list[str]:
        self.text += chunk.replace("[END]", "")
        ready: list[str] = []
        while self.text:
            match = re.search(r"[.!?؟\n]", self.text)
            if match:
                end = match.end()
            elif len(self.text) >= self.max_chars:
                split_at = self.text.rfind(" ", 0, self.max_chars)
                end = split_at if split_at > 0 else self.max_chars
            else:
                break
            sentence = self.text[:end].strip()
            self.text = self.text[end:].lstrip()
            if sentence:
                ready.append(sentence)
        return ready

    def finish(self) -> str:
        value = self.text.replace("[END]", "").strip()
        self.text = ""
        return value


async def accept_request(request: JobRequest) -> None:
    try:
        metadata = json.loads(request.job.metadata or "{}")
    except json.JSONDecodeError:
        metadata = {}
    session_id = str(metadata.get("session_id", request.id))
    suffix = session_id.replace("-", "")[:20]
    await request.accept(
        name="MetaRang AI",
        identity=f"agent-{suffix}",
        metadata=json.dumps({"session_id": session_id}, separators=(",", ":")),
    )


async def send_event(ctx: JobContext, destination: str, event) -> None:
    await ctx.room.local_participant.send_text(
        event.to_json(),
        topic=settings.event_topic,
        destination_identities=[destination],
        attributes={"content-type": "application/json", "schema-version": "1"},
    )


async def play_wav(source: rtc.AudioSource, wav_bytes: bytes, on_first_frame=None) -> None:
    pcm = wav_to_pcm16_mono(wav_bytes, output_rate=settings.output_sample_rate)
    for frame_bytes, sample_count in pcm_frames(pcm, sample_rate=settings.output_sample_rate):
        if on_first_frame is not None:
            on_first_frame()
            on_first_frame = None
        await source.capture_frame(
            rtc.AudioFrame(
                frame_bytes,
                sample_rate=settings.output_sample_rate,
                num_channels=1,
                samples_per_channel=sample_count,
            )
        )


async def handle_turn(
    *,
    ctx: JobContext,
    pipeline: PipelineClient,
    source: rtc.AudioSource,
    metadata: dict,
    destination: str,
    pcm: bytes,
) -> None:
    session_id = str(metadata["session_id"])
    turn_id = uuid.uuid4().hex
    endpoint_time = time.monotonic()
    with tracer.start_as_current_span("conversation.turn") as span:
        span.set_attribute("session.id", session_id)
        span.set_attribute("turn.id", turn_id)
        span.set_attribute("room.name", ctx.room.name)
        try:
            await send_event(
                ctx,
                destination,
                PipelineStateEvent.create(session_id=session_id, turn_id=turn_id, state="transcribing"),
            )
            transcript = await pipeline.transcribe(pcm, session_id=session_id, turn_id=turn_id)
            if not transcript:
                TURNS.labels("empty").inc()
                return
            await send_event(
                ctx,
                destination,
                TranscriptionEvent.create(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker="user",
                    text=transcript,
                    final=True,
                ),
            )
            await send_event(
                ctx,
                destination,
                PipelineStateEvent.create(session_id=session_id, turn_id=turn_id, state="thinking"),
            )

            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(
                maxsize=settings.sentence_queue_size
            )
            answer_parts: list[str] = []
            buffer = SentenceBuffer()
            first_audio = True

            async def generate_text() -> None:
                published_length = 0
                async for token in pipeline.stream_answer(
                    transcript,
                    session_id=session_id,
                    turn_id=turn_id,
                    user_id=str(metadata["user_id"]),
                    vector_store=str(metadata["vector_store"]),
                ):
                    answer_parts.append(token.replace("[END]", ""))
                    current = "".join(answer_parts).strip()
                    if len(current) - published_length >= 48:
                        await send_event(
                            ctx,
                            destination,
                            TranscriptionEvent.create(
                                session_id=session_id,
                                turn_id=turn_id,
                                speaker="agent",
                                text=current,
                                final=False,
                            ),
                        )
                        published_length = len(current)
                    for sentence in buffer.push(token):
                        await sentence_queue.put(sentence)
                tail = buffer.finish()
                if tail:
                    await sentence_queue.put(tail)

            async def synthesize_and_play() -> None:
                nonlocal first_audio
                speaking_sent = False
                while True:
                    sentence = await sentence_queue.get()
                    try:
                        if sentence is None:
                            break
                        wav = await pipeline.synthesize(
                            sentence, session_id=session_id, turn_id=turn_id
                        )
                        target = str(metadata.get("voice_sample_filename") or "")
                        if metadata.get("voice_conversion_enabled") and target:
                            try:
                                wav = await pipeline.convert_voice(
                                    wav,
                                    target=target,
                                    session_id=session_id,
                                    turn_id=turn_id,
                                )
                            except PipelineServiceError:
                                if not settings.vc_failure_fallback:
                                    raise
                                VC_FALLBACKS.inc()
                                logger.exception(
                                    "Voice conversion failed; publishing the original TTS audio",
                                    extra={
                                        "event": "vc_fallback",
                                        "session_id": session_id,
                                        "turn_id": turn_id,
                                    },
                                )
                        if not speaking_sent:
                            await send_event(
                                ctx,
                                destination,
                                PipelineStateEvent.create(
                                    session_id=session_id, turn_id=turn_id, state="speaking"
                                ),
                            )
                            speaking_sent = True
                        def record_first_audio() -> None:
                            nonlocal first_audio
                            if not first_audio:
                                return
                            latency = time.monotonic() - endpoint_time
                            FIRST_AUDIO.observe(latency)
                            span.add_event("realtime.first_ai_audio", {"duration_ms": latency * 1000})
                            first_audio = False

                        await play_wav(source, wav, on_first_frame=record_first_audio)
                    finally:
                        sentence_queue.task_done()

            playback = asyncio.create_task(synthesize_and_play(), name=f"playback-{turn_id}")

            async def generate_and_close_queue() -> None:
                try:
                    await generate_text()
                finally:
                    while not playback.done():
                        try:
                            sentence_queue.put_nowait(None)
                            break
                        except asyncio.QueueFull:
                            await asyncio.sleep(0.02)

            generation = asyncio.create_task(generate_and_close_queue(), name=f"generation-{turn_id}")
            done, pending = await asyncio.wait(
                (generation, playback), return_when=asyncio.FIRST_EXCEPTION
            )
            failure = next(
                (task.exception() for task in done if not task.cancelled() and task.exception()),
                None,
            )
            if failure is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failure
            await asyncio.gather(generation, playback)
            await source.wait_for_playout()

            answer = "".join(answer_parts).replace("[END]", "").strip()
            if answer:
                await send_event(
                    ctx,
                    destination,
                    TranscriptionEvent.create(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker="agent",
                        text=answer,
                        final=True,
                    ),
                )
            TURNS.labels("success").inc()
        except PipelineServiceError as exc:
            TURNS.labels("error").inc()
            span.record_exception(exc)
            logger.exception(
                "Conversation turn failed",
                extra={
                    "event": "turn_error",
                    "stage": exc.stage,
                    "session_id": session_id,
                    "turn_id": turn_id,
                },
            )
            await send_event(
                ctx,
                destination,
                PipelineErrorEvent.create(
                    session_id=session_id,
                    turn_id=turn_id,
                    stage=exc.stage,
                    code=f"{exc.stage}_failed",
                    message="The AI pipeline could not complete this turn.",
                    recoverable=True,
                ),
            )
        except Exception as exc:
            TURNS.labels("error").inc()
            span.record_exception(exc)
            logger.exception(
                "Unexpected conversation turn failure",
                extra={"event": "turn_error", "session_id": session_id, "turn_id": turn_id},
            )
            await send_event(
                ctx,
                destination,
                PipelineErrorEvent.create(
                    session_id=session_id,
                    turn_id=turn_id,
                    stage="realtime",
                    code="turn_failed",
                    message="The AI pipeline could not complete this turn.",
                    recoverable=True,
                ),
            )


@server.rtc_session(agent_name=settings.agent_name, on_request=accept_request)
async def realtime_session(ctx: JobContext) -> None:
    try:
        metadata = json.loads(ctx.job.metadata or "{}")
        for key in ("session_id", "user_id", "participant_identity", "vector_store"):
            if not metadata.get(key):
                raise ValueError(f"Agent dispatch metadata is missing {key}")
    except (json.JSONDecodeError, ValueError):
        logger.exception("Invalid agent dispatch metadata", extra={"event": "dispatch_rejected"})
        return

    session_id = str(metadata["session_id"])
    destination = str(metadata["participant_identity"])
    ctx.log_context_fields = {
        "service": SERVICE,
        "session_id": session_id,
        "room_name": ctx.job.room.name,
    }
    ACTIVE_JOBS.inc()
    job_started = time.monotonic()
    pipeline = PipelineClient(settings)
    source: rtc.AudioSource | None = None
    stream: rtc.AudioStream | None = None
    tasks: set[asyncio.Task] = set()

    @ctx.room.on("reconnecting")
    def on_reconnecting() -> None:
        RECONNECTS.inc()
        logger.warning("LiveKit room reconnecting", extra={"event": "room_reconnecting"})

    @ctx.room.on("reconnected")
    def on_reconnected() -> None:
        logger.info("LiveKit room reconnected", extra={"event": "room_reconnected"})

    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        source = rtc.AudioSource(
            settings.output_sample_rate, 1, queue_size_ms=settings.output_queue_ms
        )
        track = rtc.LocalAudioTrack.create_audio_track("agent-audio", source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await ctx.room.local_participant.publish_track(track, options)

        try:
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(identity=destination),
                timeout=settings.participant_wait_timeout_seconds,
            )
        except asyncio.TimeoutError:
            PARTICIPANT_FAILURES.inc()
            raise RuntimeError("Expected Unity participant did not join before timeout")
        disconnected = asyncio.Event()

        @ctx.room.on("participant_disconnected")
        def on_participant_disconnected(leaving: rtc.RemoteParticipant) -> None:
            if leaving.identity == destination:
                disconnected.set()

        stream = rtc.AudioStream.from_participant(
            participant=participant,
            track_source=rtc.TrackSource.SOURCE_MICROPHONE,
            capacity=settings.audio_stream_capacity_frames,
            sample_rate=settings.input_sample_rate,
            num_channels=1,
            frame_size_ms=20,
        )
        detector = EnergyEndpointDetector(
            EndpointConfig(
                sample_rate=settings.input_sample_rate,
                threshold_dbfs=settings.speech_threshold_dbfs,
                start_ms=settings.speech_start_ms,
                end_silence_ms=settings.speech_end_silence_ms,
                preroll_ms=settings.speech_preroll_ms,
                keep_silence_ms=settings.speech_keep_silence_ms,
                min_turn_ms=settings.min_turn_ms,
                max_turn_seconds=settings.max_turn_seconds,
            )
        )
        turn_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)
        busy = asyncio.Event()

        async def read_audio() -> None:
            async for event in stream:
                if disconnected.is_set():
                    break
                if busy.is_set():
                    detector.reset()
                    continue
                complete_turn = detector.push(bytes(event.frame.data))
                if complete_turn:
                    busy.set()
                    turn_queue.put_nowait(complete_turn)
            await turn_queue.put(None)

        async def process_turns() -> None:
            while True:
                pcm = await turn_queue.get()
                try:
                    if pcm is None:
                        return
                    await handle_turn(
                        ctx=ctx,
                        pipeline=pipeline,
                        source=source,
                        metadata=metadata,
                        destination=destination,
                        pcm=pcm,
                    )
                finally:
                    turn_queue.task_done()
                    busy.clear()
                    if pcm is not None and not disconnected.is_set():
                        await send_event(
                            ctx,
                            destination,
                            PipelineStateEvent.create(
                                session_id=session_id, turn_id=None, state="listening"
                            ),
                        )

        await send_event(
            ctx,
            destination,
            PipelineStateEvent.create(session_id=session_id, turn_id=None, state="listening"),
        )
        reader = asyncio.create_task(read_audio(), name=f"audio-reader-{session_id}")
        processor = asyncio.create_task(process_turns(), name=f"turn-processor-{session_id}")
        tasks.update((reader, processor))
        disconnect_waiter = asyncio.create_task(disconnected.wait(), name=f"disconnect-{session_id}")
        tasks.add(disconnect_waiter)
        done, _ = await asyncio.wait(
            (reader, processor, disconnect_waiter), return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if not task.cancelled() and task.exception():
                raise task.exception()  # type: ignore[misc]
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Realtime agent job failed", extra={"event": "job_error"})
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if stream is not None:
            await stream.aclose()
        if source is not None:
            source.clear_queue()
            await source.aclose()
        await pipeline.close()
        ACTIVE_JOBS.dec()
        duration = time.monotonic() - job_started
        JOB_DURATION.observe(duration)
        logger.info(
            "Realtime agent job closed",
            extra={"event": "job_closed", "duration_ms": round(duration * 1000, 1)},
        )


if __name__ == "__main__":
    cli.run_app(server)
