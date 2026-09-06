"""Explicit PCM endpointing and WAV conversion utilities."""

from __future__ import annotations

import io
import math
import wave
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EndpointConfig:
    sample_rate: int = 16000
    threshold_dbfs: float = -38.0
    start_ms: int = 120
    end_silence_ms: int = 900
    preroll_ms: int = 240
    keep_silence_ms: int = 160
    min_turn_ms: int = 300
    max_turn_seconds: int = 30


class EnergyEndpointDetector:
    """A deliberately simple energy/silence endpoint detector, not semantic VAD."""

    def __init__(self, config: EndpointConfig):
        self.config = config
        self._preroll: deque[tuple[bytes, float]] = deque()
        self.reset()

    def reset(self) -> None:
        self._recording = False
        self._speech_run_ms = 0.0
        self._silence_ms = 0.0
        self._duration_ms = 0.0
        self._frames: list[bytes] = []
        self._preroll.clear()

    @property
    def recording(self) -> bool:
        return self._recording

    @staticmethod
    def dbfs(pcm: bytes) -> float:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return -96.0
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return -96.0 if rms < 1.0 else 20.0 * math.log10(rms / 32768.0)

    def push(self, pcm: bytes) -> bytes | None:
        if not pcm:
            return None
        frame_ms = (len(pcm) / 2) * 1000.0 / self.config.sample_rate
        voiced = self.dbfs(pcm) >= self.config.threshold_dbfs

        if not self._recording:
            self._preroll.append((pcm, frame_ms))
            preroll_total = sum(duration for _, duration in self._preroll)
            while self._preroll and preroll_total > self.config.preroll_ms:
                _, removed = self._preroll.popleft()
                preroll_total -= removed
            self._speech_run_ms = self._speech_run_ms + frame_ms if voiced else 0.0
            if self._speech_run_ms < self.config.start_ms:
                return None
            self._recording = True
            self._frames = [frame for frame, _ in self._preroll]
            self._duration_ms = sum(duration for _, duration in self._preroll)
            self._silence_ms = 0.0
            self._preroll.clear()
            return None

        self._frames.append(pcm)
        self._duration_ms += frame_ms
        self._silence_ms = 0.0 if voiced else self._silence_ms + frame_ms
        reached_silence = self._silence_ms >= self.config.end_silence_ms
        reached_limit = self._duration_ms >= self.config.max_turn_seconds * 1000
        if not reached_silence and not reached_limit:
            return None

        audio = b"".join(self._frames)
        if reached_silence and self._silence_ms > self.config.keep_silence_ms:
            trim_ms = self._silence_ms - self.config.keep_silence_ms
            trim_bytes = int(trim_ms * self.config.sample_rate / 1000) * 2
            if trim_bytes < len(audio):
                audio = audio[:-trim_bytes]
        minimum_bytes = int(self.config.min_turn_ms * self.config.sample_rate / 1000) * 2
        self.reset()
        return audio if len(audio) >= minimum_bytes else None


def wav_to_pcm16_mono(wav_bytes: bytes, output_rate: int = 48000) -> bytes:
    """Decode PCM16 WAV, downmix, and linearly resample to LiveKit's output rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        source_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only PCM16 WAV is supported, received {sample_width * 8}-bit audio")
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1)
    else:
        samples = samples.astype(np.float32)
    if source_rate != output_rate and samples.size:
        output_length = max(1, round(samples.size * output_rate / source_rate))
        old_positions = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
        new_positions = np.linspace(0.0, 1.0, num=output_length, endpoint=True)
        samples = np.interp(new_positions, old_positions, samples)
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def pcm_frames(pcm: bytes, *, sample_rate: int = 48000, frame_ms: int = 20):
    samples_per_frame = sample_rate * frame_ms // 1000
    bytes_per_frame = samples_per_frame * 2
    for offset in range(0, len(pcm), bytes_per_frame):
        frame = pcm[offset : offset + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            frame += b"\x00" * (bytes_per_frame - len(frame))
        yield frame, samples_per_frame
