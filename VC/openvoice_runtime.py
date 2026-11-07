"""Minimal OpenVoice V2 ToneColorConverter runtime.

The upstream OpenVoice model modules are vendored unmodified under
``third_party/openvoice``.  This adapter intentionally avoids the upstream
demo, MeloTTS, Whisper segmentation, and watermark dependencies.  Source and
target embeddings are extracted by the converter checkpoint's built-in
``ref_enc`` and returned to callers in CPU RAM.
"""

from __future__ import annotations

import io
import time
import wave
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from openvoice import utils
from openvoice.mel_processing import spectrogram_torch
from openvoice.models import SynthesizerTrn


class OpenVoiceRuntimeError(RuntimeError):
    pass


def decode_audio(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    try:
        audio, source_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("audio is not a readable WAV recording") from exc
    if audio.size == 0:
        raise ValueError("audio is empty")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=sample_rate)
    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(sample_rate)
        recording.writeframes((audio * 32767.0).astype("<i2").tobytes())
    return output.getvalue()


class OpenVoiceConverter:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: str = "cuda:0"):
        if device != "cuda:0":
            raise OpenVoiceRuntimeError("OpenVoice must use VC_DEVICE=cuda:0")
        if not torch.cuda.is_available():
            raise OpenVoiceRuntimeError("CUDA is unavailable; OpenVoice CPU fallback is disabled")

        self.device = torch.device(device)
        self.hps = utils.get_hparams_from_file(str(config_path))
        self.sample_rate = int(self.hps.data.sampling_rate)

        free_before, total = torch.cuda.mem_get_info(0)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        model = SynthesizerTrn(
            len(getattr(self.hps, "symbols", [])),
            self.hps.data.filter_length // 2 + 1,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(device=self.device, dtype=torch.float32)
        checkpoint = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            raise OpenVoiceRuntimeError(
                f"OpenVoice checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        del checkpoint
        model.eval()
        torch.cuda.synchronize(self.device)
        self.model = model
        self.load_seconds = time.perf_counter() - started
        free_after, _ = torch.cuda.mem_get_info(0)
        self.load_metrics = {
            "gpu_free_before_mib": free_before / 1024**2,
            "gpu_free_after_mib": free_after / 1024**2,
            "gpu_total_mib": total / 1024**2,
            "model_load_seconds": self.load_seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
        }

    def _spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        if audio.size < self.hps.data.filter_length:
            raise ValueError("audio is too short for speaker embedding extraction")
        tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device, dtype=torch.float32)
        return spectrogram_torch(
            tensor,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        ).to(self.device)

    def extract_se(self, audio_bytes: bytes) -> tuple[torch.Tensor, dict]:
        """Extract an in-memory speaker embedding using ToneColorConverter.ref_enc.

        This is the byte-oriented equivalent of upstream
        ``ToneColorConverter.extract_se``. It deliberately omits that method's
        optional on-disk ``se_save_path`` behavior.
        """
        audio = decode_audio(audio_bytes, self.sample_rate)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        with torch.inference_mode():
            spectrogram = self._spectrogram(audio)
            embedding = self.model.ref_enc(spectrogram.transpose(1, 2)).unsqueeze(-1)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        cpu_embedding = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
        del embedding, spectrogram
        return cpu_embedding, {
            "embedding_time": elapsed,
            "audio_duration": audio.size / self.sample_rate,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
        }

    # Kept as a private-adapter compatibility alias while callers migrate to
    # the upstream OpenVoice method name.
    extract_embedding = extract_se

    def convert(
        self,
        source_audio_bytes: bytes,
        source_embedding_cpu: torch.Tensor,
        target_embedding_cpu: torch.Tensor,
        *,
        tau: float = 0.3,
    ) -> tuple[bytes, dict]:
        source_audio = decode_audio(source_audio_bytes, self.sample_rate)
        audio_duration = source_audio.size / self.sample_rate
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        with torch.inference_mode():
            spectrogram = self._spectrogram(source_audio)
            lengths = torch.tensor([spectrogram.size(-1)], device=self.device, dtype=torch.long)
            source_embedding = source_embedding_cpu.to(self.device, dtype=torch.float32)
            target_embedding = target_embedding_cpu.to(self.device, dtype=torch.float32)
            converted = self.model.voice_conversion(
                spectrogram,
                lengths,
                sid_src=source_embedding,
                sid_tgt=target_embedding,
                tau=tau,
            )[0][0, 0]
            output = converted.detach().to(device="cpu", dtype=torch.float32).numpy()
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        del spectrogram, lengths, source_embedding, target_embedding, converted
        wav_bytes = encode_wav(output, self.sample_rate)
        return wav_bytes, {
            "conversion_time": elapsed,
            "audio_duration": audio_duration,
            "conversion_rtf": elapsed / audio_duration if audio_duration else 0.0,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
        }
