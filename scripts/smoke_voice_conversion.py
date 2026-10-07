#!/usr/bin/env python3
"""Upload a temporary voice sample and exercise the live FreeVC conversion path."""

from __future__ import annotations

import io
import os
import wave

import requests


def main() -> None:
    api_url = os.getenv("SMOKE_API_URL", "http://web:8000").rstrip("/")
    tts_url = os.getenv("SMOKE_TTS_URL", "http://tts:5002").rstrip("/")
    vc_url = os.getenv("SMOKE_VC_URL", "http://vc:5003").rstrip("/")
    sample_id = None

    source_response = requests.post(
        f"{tts_url}/synthesize",
        json={"text": "This is a temporary voice conversion integration test recording."},
        timeout=120,
    )
    source_response.raise_for_status()
    source_wav = source_response.content

    try:
        upload_response = requests.post(
            f"{api_url}/api/v1/voice-samples/",
            data={"name": "automated-smoke-test"},
            files={"audio": ("smoke-source.wav", source_wav, "audio/wav")},
            timeout=60,
        )
        upload_response.raise_for_status()
        sample = upload_response.json()
        sample_id = sample["id"]
        filename = sample["filename"]

        visibility_response = requests.get(
            f"{vc_url}/samples/validate",
            params={"target": filename},
            timeout=10,
        )
        visibility_response.raise_for_status()

        conversion_response = requests.post(
            f"{vc_url}/convert",
            params={"target": filename},
            data=source_wav,
            headers={"Content-Type": "audio/wav"},
            timeout=180,
        )
        conversion_response.raise_for_status()
        with wave.open(io.BytesIO(conversion_response.content), "rb") as converted:
            if converted.getnframes() <= 0 or converted.getframerate() != 16000:
                raise RuntimeError("FreeVC returned an invalid WAV response")

        print(
            f"OK sample_id={sample_id} filename={filename} "
            f"converted_bytes={len(conversion_response.content)}"
        )
    finally:
        if sample_id is not None:
            delete_response = requests.delete(
                f"{api_url}/api/v1/voice-samples/{sample_id}/",
                timeout=10,
            )
            delete_response.raise_for_status()


if __name__ == "__main__":
    main()
