from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

from django.core.files.base import ContentFile


SUPPORTED_VOICE_SAMPLE_EXTENSIONS = (".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm")


class VoiceSampleProcessingError(ValueError):
    pass


def normalize_voice_sample(
    uploaded_file,
    *,
    min_duration_seconds: float,
    max_duration_seconds: float,
    timeout_seconds: float,
) -> tuple[ContentFile, float]:
    """Validate and convert an uploaded recording to FreeVC's canonical WAV input."""

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

        return ContentFile(output_path.read_bytes(), name="sample.wav"), duration_seconds
