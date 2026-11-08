from __future__ import annotations

import subprocess
import audioop
import tempfile
import wave
from pathlib import Path

from django.core.files.base import ContentFile


SUPPORTED_VOICE_SAMPLE_EXTENSIONS = (".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm")


class VoiceSampleProcessingError(ValueError):
    pass


def estimate_voiced_seconds(pcm: bytes, *, frame_rate: int, sample_width: int) -> float:
    """Return usable-speech duration using a conservative 30 ms energy VAD.

    This dependency-free precheck rejects silence in the Django control plane.
    OpenVoice performs a second, stricter WebRTC VAD check before extracting an
    embedding, so no weak precheck result is trusted for conversion.
    """

    frame_bytes = frame_rate * 30 // 1000 * sample_width
    frames = [
        pcm[offset : offset + frame_bytes]
        for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes)
    ]
    if not frames:
        return 0.0
    energies = [audioop.rms(frame, sample_width) for frame in frames]
    sorted_energies = sorted(energies)
    quiet_count = max(1, len(sorted_energies) // 5)
    noise_floor = sum(sorted_energies[:quiet_count]) / quiet_count
    threshold = max(300.0, noise_floor * 3.0)
    return sum(energy >= threshold for energy in energies) * 0.03


def normalize_voice_sample(
    uploaded_file,
    *,
    min_duration_seconds: float,
    max_duration_seconds: float,
    min_voiced_seconds: float,
    timeout_seconds: float,
) -> tuple[ContentFile, float]:
    """Validate and convert an uploaded recording to OpenVoice's canonical WAV input."""

    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="metarang-voice-") as temporary_directory:
        input_path = Path(temporary_directory) / f"input{suffix}"
        output_path = Path(temporary_directory) / "sample.wav"

        with input_path.open("wb") as target:
            for chunk in uploaded_file.chunks():
                target.write(chunk)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise VoiceSampleProcessingError("Audio processing is unavailable.") from exc
        except subprocess.TimeoutExpired as exc:
            raise VoiceSampleProcessingError("Audio processing timed out.") from exc
        except subprocess.CalledProcessError as exc:
            raise VoiceSampleProcessingError(
                "The uploaded file is not a readable audio recording."
            ) from exc

        try:
            with wave.open(str(output_path), "rb") as recording:
                frame_rate = recording.getframerate()
                frame_count = recording.getnframes()
                channels = recording.getnchannels()
                sample_width = recording.getsampwidth()
                pcm = recording.readframes(frame_count)
        except (OSError, wave.Error) as exc:
            raise VoiceSampleProcessingError("The normalized recording is invalid.") from exc

        if frame_rate != 16000 or channels != 1 or sample_width != 2 or frame_count <= 0:
            raise VoiceSampleProcessingError("The normalized recording has an invalid format.")

        duration_seconds = frame_count / frame_rate
        if duration_seconds < min_duration_seconds:
            raise VoiceSampleProcessingError(
                f"The recording must be at least {min_duration_seconds:g} seconds long."
            )
        if duration_seconds > max_duration_seconds:
            raise VoiceSampleProcessingError(
                f"The recording must not exceed {max_duration_seconds:g} seconds."
            )

        voiced_seconds = estimate_voiced_seconds(
            pcm,
            frame_rate=frame_rate,
            sample_width=sample_width,
        )
        if voiced_seconds < min_voiced_seconds:
            raise VoiceSampleProcessingError(
                f"The recording contains only {voiced_seconds:.2f} seconds of usable speech; "
                f"at least {min_voiced_seconds:.2f} seconds is required."
            )

        return ContentFile(output_path.read_bytes(), name="sample.wav"), duration_seconds
