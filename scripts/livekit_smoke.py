#!/usr/bin/env python3
"""Backend-only WebRTC smoke test; no Unity installation is required."""

from __future__ import annotations

import argparse
import asyncio
import array
import json
import pathlib

import httpx
from livekit import rtc

from realtime_agent.audio import pcm_frames, wav_to_pcm16_mono


TOPIC = "metarang.events.v1"


async def run(
    api_url: str,
    wav_file: pathlib.Path,
    timeout: float,
    livekit_url_override: str | None = None,
    voice_sample_id: int | None = None,
) -> None:
    async with httpx.AsyncClient(timeout=15) as http:
        session_request = {"user_id": "smoke-client", "vector_store": "main_store"}
        if voice_sample_id is not None:
            session_request.update(
                {"enable_voice_conversion": True, "voice_sample_id": voice_sample_id}
            )
        response = await http.post(
            f"{api_url.rstrip('/')}/api/v1/agent/sessions/",
            json=session_request,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"Session creation failed with HTTP {response.status_code}: {detail}"
            ) from exc
        session = response.json()

        room = rtc.Room()
        final_user_transcript = asyncio.Event()
        final_agent_transcript = asyncio.Event()
        agent_speaking = asyncio.Event()
        received_agent_audio = asyncio.Event()
        pipeline_error = asyncio.Event()
        pipeline_errors: list[dict] = []
        stream_tasks: set[asyncio.Task] = set()
        source: rtc.AudioSource | None = None

        def task_finished(task: asyncio.Task) -> None:
            stream_tasks.discard(task)
            if not task.cancelled():
                # Retrieve the exception so an expected disconnect during cleanup
                # does not become an unhandled-task warning.
                task.exception()

        async def consume_text(reader: rtc.TextStreamReader) -> None:
            payload = json.loads(await reader.read_all())
            print(json.dumps(payload, ensure_ascii=False))
            if payload.get("type") == "transcription" and payload.get("final"):
                if payload.get("speaker") == "user":
                    final_user_transcript.set()
                elif payload.get("speaker") == "agent":
                    final_agent_transcript.set()
            elif payload.get("type") == "pipeline_state" and payload.get("state") == "speaking":
                agent_speaking.set()
            elif payload.get("type") == "pipeline_error":
                pipeline_errors.append(payload)
                pipeline_error.set()

        def text_handler(reader: rtc.TextStreamReader, _identity: str) -> None:
            task = asyncio.create_task(consume_text(reader))
            stream_tasks.add(task)
            task.add_done_callback(task_finished)

        async def consume_agent_audio(track: rtc.RemoteAudioTrack) -> None:
            stream = rtc.AudioStream(track, capacity=100, sample_rate=48000, num_channels=1)
            try:
                async for event in stream:
                    samples = array.array("h")
                    samples.frombytes(bytes(event.frame.data))
                    if samples and max(abs(sample) for sample in samples) >= 64:
                        received_agent_audio.set()
                        break
            finally:
                await stream.aclose()

        def on_track_subscribed(track, _publication, participant) -> None:
            if isinstance(track, rtc.RemoteAudioTrack) and participant.identity.startswith("agent-"):
                task = asyncio.create_task(consume_agent_audio(track))
                stream_tasks.add(task)
                task.add_done_callback(task_finished)

        room.register_text_stream_handler(TOPIC, text_handler)
        room.on("track_subscribed", on_track_subscribed)

        try:
            livekit_url = livekit_url_override or session["livekit_url"]
            await room.connect(livekit_url, session["access_token"])
            source = rtc.AudioSource(48000, 1, queue_size_ms=500)
            track = rtc.LocalAudioTrack.create_audio_track("smoke-microphone", source)
            await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            pcm = wav_to_pcm16_mono(wav_file.read_bytes(), output_rate=48000)
            pcm += b"\x00\x00" * int(48000 * 1.2)
            for frame_data, samples in pcm_frames(pcm, sample_rate=48000):
                await source.capture_frame(rtc.AudioFrame(frame_data, 48000, 1, samples))
            await source.wait_for_playout()

            await asyncio.wait_for(final_user_transcript.wait(), timeout=timeout)
            completed = asyncio.create_task(final_agent_transcript.wait())
            failed = asyncio.create_task(pipeline_error.wait())
            done, pending = await asyncio.wait(
                (completed, failed), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if not done:
                raise TimeoutError("Timed out waiting for the agent pipeline result")
            if failed in done and pipeline_error.is_set():
                raise RuntimeError(f"Agent pipeline failed: {pipeline_errors[-1]}")
            await asyncio.wait_for(agent_speaking.wait(), timeout=timeout)
            await asyncio.wait_for(received_agent_audio.wait(), timeout=timeout)
            print("PASS: final user/agent transcripts and non-silent agent audio were received")
        finally:
            await room.disconnect()
            if source is not None:
                await source.aclose()
            for task in stream_tasks:
                task.cancel()
            if stream_tasks:
                await asyncio.gather(*stream_tasks, return_exceptions=True)
            await http.delete(
                f"{api_url.rstrip('/')}/api/v1/agent/sessions/{session['session_id']}/"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--wav-file", required=True, type=pathlib.Path)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument(
        "--livekit-url",
        help="Override the public session URL, for example ws://livekit:7880 inside Compose.",
    )
    parser.add_argument(
        "--voice-sample-id",
        type=int,
        help="Enable voice conversion using this existing voice-sample ID.",
    )
    args = parser.parse_args()
    if not args.wav_file.is_file():
        parser.error(f"WAV file does not exist: {args.wav_file}")
    asyncio.run(
        run(
            args.api_url,
            args.wav_file,
            args.timeout,
            args.livekit_url,
            args.voice_sample_id,
        )
    )


if __name__ == "__main__":
    main()
