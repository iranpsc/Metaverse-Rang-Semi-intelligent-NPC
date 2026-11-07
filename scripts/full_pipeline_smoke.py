#!/usr/bin/env python3
"""Provision test resources and exercise STT -> RAG/LLM -> TTS -> VC over LiveKit."""

from __future__ import annotations

import argparse
import asyncio
import array
import json
import time

import httpx
from livekit import rtc

from realtime_agent.audio import pcm_frames, wav_to_pcm16_mono


TOPIC = "metarang.events.v1"
KNOWLEDGE_TEXT = (
    "کد آزمایش یکپارچه متارنگ سیمرغ آبی است. "
    "اگر پرسیده شد کد آزمایش یکپارچه چیست، پاسخ دقیق سیمرغ آبی است."
)
QUESTION_TEXT = "کد آزمایش یکپارچه متارنگ چیست؟"
EXPECTED_WORDS = ("سیمرغ", "آبی")


async def synthesize(http: httpx.AsyncClient, tts_url: str, text: str) -> bytes:
    response = await http.post(f"{tts_url}/synthesize", json={"text": text}, timeout=120)
    response.raise_for_status()
    return response.content


async def run(api_url: str, tts_url: str, livekit_url: str, timeout: float) -> dict:
    run_id = str(int(time.time()))
    user_id = f"pipeline-smoke-{run_id}"
    voice_sample_id: int | None = None
    session_id: str | None = None
    room: rtc.Room | None = None
    source: rtc.AudioSource | None = None
    stream_tasks: set[asyncio.Task] = set()

    result = {
        "user_id": user_id,
        "vector_store": "",
        "voice_sample_id": None,
        "session_id": None,
        "user_transcript": "",
        "agent_transcript": "",
        "received_audio": False,
    }

    async with httpx.AsyncClient(timeout=30) as http:
        try:
            vector_response = await http.post(
                f"{api_url}/api/v1/vector-stores/",
                data={"user_id": user_id},
                files={
                    "files": (
                        "pipeline-smoke-knowledge.txt",
                        KNOWLEDGE_TEXT.encode("utf-8"),
                        "text/plain",
                    )
                },
                timeout=180,
            )
            vector_response.raise_for_status()
            vector_store = vector_response.json()["vector_store"]
            result["vector_store"] = vector_store
            print(f"PASS vector_store={vector_store}")

            target_voice = await synthesize(
                http,
                tts_url,
                "این یک نمونه صدای موقت برای آزمایش تبدیل صدای سامانه متارنگ است.",
            )
            voice_response = await http.post(
                f"{api_url}/api/v1/voice-samples/",
                data={"name": f"pipeline-smoke-{run_id}"},
                files={"audio": ("target.wav", target_voice, "audio/wav")},
                timeout=60,
            )
            voice_response.raise_for_status()
            voice_sample_id = int(voice_response.json()["id"])
            result["voice_sample_id"] = voice_sample_id
            print(f"PASS voice_sample_id={voice_sample_id}")

            session_response = await http.post(
                f"{api_url}/api/v1/agent/sessions/",
                json={
                    "user_id": user_id,
                    "vector_store": vector_store,
                    "enable_voice_conversion": True,
                    "voice_sample_id": voice_sample_id,
                },
            )
            session_response.raise_for_status()
            session = session_response.json()
            session_id = session["session_id"]
            result["session_id"] = session_id
            print(f"PASS livekit_session={session_id}")

            question_wav = await synthesize(http, tts_url, QUESTION_TEXT)
            room = rtc.Room()
            final_user_transcript = asyncio.Event()
            final_agent_transcript = asyncio.Event()
            agent_speaking = asyncio.Event()
            received_agent_audio = asyncio.Event()
            pipeline_error = asyncio.Event()
            pipeline_errors: list[dict] = []

            def task_finished(task: asyncio.Task) -> None:
                stream_tasks.discard(task)
                if not task.cancelled():
                    task.exception()

            async def consume_text(reader: rtc.TextStreamReader) -> None:
                payload = json.loads(await reader.read_all())
                event_type = payload.get("type")
                if event_type == "transcription" and payload.get("final"):
                    text = str(payload.get("text") or "")
                    if payload.get("speaker") == "user":
                        result["user_transcript"] = text
                        final_user_transcript.set()
                    elif payload.get("speaker") == "agent":
                        result["agent_transcript"] = text
                        final_agent_transcript.set()
                elif event_type == "pipeline_state" and payload.get("state") == "speaking":
                    agent_speaking.set()
                elif event_type == "pipeline_error":
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
                            result["received_audio"] = True
                            received_agent_audio.set()
                            break
                finally:
                    await stream.aclose()

            def on_track_subscribed(track, _publication, participant) -> None:
                if isinstance(track, rtc.RemoteAudioTrack) and participant.identity.startswith(
                    "agent-"
                ):
                    task = asyncio.create_task(consume_agent_audio(track))
                    stream_tasks.add(task)
                    task.add_done_callback(task_finished)

            room.register_text_stream_handler(TOPIC, text_handler)
            room.on("track_subscribed", on_track_subscribed)
            await room.connect(livekit_url, session["access_token"])

            source = rtc.AudioSource(48000, 1, queue_size_ms=500)
            track = rtc.LocalAudioTrack.create_audio_track("smoke-microphone", source)
            await room.local_participant.publish_track(
                track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            pcm = wav_to_pcm16_mono(question_wav, output_rate=48000)
            pcm += b"\x00\x00" * int(48000 * 1.2)
            for frame_data, samples in pcm_frames(pcm, sample_rate=48000):
                await source.capture_frame(rtc.AudioFrame(frame_data, 48000, 1, samples))
            await source.wait_for_playout()

            await asyncio.wait_for(final_user_transcript.wait(), timeout=timeout)
            print(f"PASS stt={result['user_transcript']}")

            completed = asyncio.create_task(final_agent_transcript.wait())
            failed = asyncio.create_task(pipeline_error.wait())
            done, pending = await asyncio.wait(
                (completed, failed),
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if not done:
                raise TimeoutError("Timed out waiting for the RAG/LLM response")
            if failed in done and pipeline_error.is_set():
                raise RuntimeError(f"Agent pipeline failed: {pipeline_errors[-1]}")

            agent_text = str(result["agent_transcript"])
            if not all(word in agent_text for word in EXPECTED_WORDS):
                raise AssertionError(
                    f"RAG answer did not contain {EXPECTED_WORDS}: {agent_text!r}"
                )
            print(f"PASS rag_llm={agent_text}")

            await asyncio.wait_for(agent_speaking.wait(), timeout=timeout)
            await asyncio.wait_for(received_agent_audio.wait(), timeout=timeout)
            print("PASS tts_vc_audio=non-silent")
            return result
        finally:
            if room is not None:
                await room.disconnect()
            if source is not None:
                await source.aclose()
            for task in stream_tasks:
                task.cancel()
            if stream_tasks:
                await asyncio.gather(*stream_tasks, return_exceptions=True)
            if session_id is not None:
                await http.delete(f"{api_url}/api/v1/agent/sessions/{session_id}/")
            if voice_sample_id is not None:
                await http.delete(f"{api_url}/api/v1/voice-samples/{voice_sample_id}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://web:8000")
    parser.add_argument("--tts-url", default="http://tts:5002")
    parser.add_argument("--livekit-url", default="ws://livekit:7880")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    result = asyncio.run(
        run(
            args.api_url.rstrip("/"),
            args.tts_url.rstrip("/"),
            args.livekit_url,
            args.timeout,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
