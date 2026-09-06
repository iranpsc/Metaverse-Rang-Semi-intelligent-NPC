import io
import unittest
import wave

import numpy as np

from realtime_agent.audio import EndpointConfig, EnergyEndpointDetector, pcm_frames, wav_to_pcm16_mono


def tone_frame(sample_rate: int = 16000, duration_ms: int = 20, amplitude: int = 8000) -> bytes:
    samples = np.full(sample_rate * duration_ms // 1000, amplitude, dtype="<i2")
    return samples.tobytes()


class EnergyEndpointDetectorTests(unittest.TestCase):
    def test_emits_one_bounded_turn_after_silence(self):
        detector = EnergyEndpointDetector(
            EndpointConfig(
                threshold_dbfs=-40,
                start_ms=40,
                end_silence_ms=60,
                preroll_ms=40,
                keep_silence_ms=0,
                min_turn_ms=40,
            )
        )
        silence = b"\x00\x00" * 320
        self.assertIsNone(detector.push(tone_frame()))
        self.assertIsNone(detector.push(tone_frame()))
        self.assertIsNone(detector.push(silence))
        self.assertIsNone(detector.push(silence))
        turn = detector.push(silence)
        self.assertIsNotNone(turn)
        self.assertGreaterEqual(len(turn), 16000 * 2 * 40 // 1000)
        self.assertFalse(detector.recording)

    def test_silence_does_not_start_a_turn(self):
        detector = EnergyEndpointDetector(EndpointConfig())
        silence = b"\x00\x00" * 320
        for _ in range(100):
            self.assertIsNone(detector.push(silence))


class AudioConversionTests(unittest.TestCase):
    def test_resamples_pcm16_wav_to_48khz_mono_frames(self):
        source = tone_frame(sample_rate=16000, duration_ms=100)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(source)
        output = wav_to_pcm16_mono(buffer.getvalue(), output_rate=48000)
        self.assertAlmostEqual(len(output), 48000 * 2 // 10, delta=4)
        self.assertEqual(len(list(pcm_frames(output))), 5)
