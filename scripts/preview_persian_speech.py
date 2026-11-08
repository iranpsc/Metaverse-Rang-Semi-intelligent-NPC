"""Export reproducible before/after Piper samples without any model download.

Run from the repository root. Text-only mode uses the Python standard library;
--model enables CPU synthesis using the installed piper-tts runtime.
"""

from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import sys
import time
import wave

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_core.persian_text import PersianTextNormalizer


SAMPLES = (
    ("money-percent", "قیمت ۱۲۵٬۰۰۰ تومان با ۲۰٪ تخفیف است."),
    ("decimal-units", "وزن ۲٫۵kg و دمای هوا −۵°C است."),
    ("date-time", "جلسه در تاریخ ۱۴۰۳/۰۷/۱۲ ساعت ۱۴:۳۰ است."),
    ("phone-code", "شماره تلفن ۰۹۱۲۱۲۳۴۵۶۷ و کد ورود ۰۰۴۲ است."),
    ("pronunciation", "کرم خاکی با کرم ضد آفتاب فرق دارد."),
)


def wav_stats(path: Path) -> dict:
    with wave.open(str(path), "rb") as recording:
        channels = recording.getnchannels()
        rate = recording.getframerate()
        frames = recording.getnframes()
        if channels != 1 or recording.getsampwidth() != 2 or not frames:
            raise RuntimeError(f"invalid Piper audio: {path}")
        samples = array("h", recording.readframes(frames))
    if not any(samples):
        raise RuntimeError(f"silent Piper audio: {path}")
    return {"sample_rate": rate, "audio_seconds": frames / rate, "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="existing Piper ONNX model; enables audio export")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", help="use one custom sentence instead of the bundled samples")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = PersianTextNormalizer.from_env()
    voice = None
    if arguments.model:
        from piper import PiperVoice

        voice = PiperVoice.load(str(arguments.model), use_cuda=False)
    report = {"engine": "mana-piper" if voice else None, "diacritization": "lexicon" if normalizer.lexicon else "off", "samples": []}
    cases = [("custom", arguments.text)] if arguments.text is not None else SAMPLES
    for name, raw in cases:
        started = time.perf_counter()
        prepared = normalizer.normalize(raw).normalized
        row = {"id": name, "original": raw, "normalized": prepared, "normalization_ms": (time.perf_counter() - started) * 1000}
        if voice:
            for variant, text in (("raw", raw), ("normalized", prepared)):
                path = arguments.output_dir / f"{name}-{variant}.wav"
                phonemes = voice.phonemize(text)
                started = time.perf_counter()
                with wave.open(str(path), "wb") as recording:
                    voice.synthesize(text, recording, sentence_silence=0.12)
                row[variant + "_audio"] = {
                    "file": path.name, "phonemes": phonemes,
                    "synthesis_seconds": time.perf_counter() - started, **wav_stats(path),
                }
        report["samples"].append(row)
        print(f"{name}: {prepared}", flush=True)
    target = arguments.output_dir / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {target}")


if __name__ == "__main__":
    main()
