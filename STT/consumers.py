import json
import uuid
import httpx
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
from django.conf import settings


class AudioTranscriptionConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stt_ready = False
        self.http = None
        self.audio_buffer = b''
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration_ms = 1000  # 1 second chunks
        self.silence_threshold = -40  # dB
        self.min_silence_len = 500  # ms
        self.sentence_end_silence = 2000  # ms of silence to consider sentence end

    async def connect(self):
        """Handle WebSocket connection"""
        await self.accept()

        # Check the one shared STT microservice; never load Whisper in Django.
        await self.connect_stt_service()

        # Send connection confirmation
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'WebSocket connected successfully. Ready to receive audio.',
            'config': {
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'chunk_duration_ms': self.chunk_duration_ms
            }
        }))

    async def disconnect(self, close_code):
        if self.http is not None:
            await self.http.aclose()

    async def connect_stt_service(self):
        """Create a bounded HTTP client and verify the shared STT service."""
        try:
            self.http = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.STT_SERVICE_TIMEOUT_SECONDS, connect=3.0)
            )
            response = await self.http.get(f"{settings.STT_SERVER_URL}/health/ready")
            response.raise_for_status()
            self.stt_ready = True
            await self.send(text_data=json.dumps({
                'type': 'model_loaded',
                'message': f'Shared Whisper {settings.STT_MODEL_NAME} service is ready'
            }))
        except Exception as e:
            self.stt_ready = False
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'STT service is unavailable: {str(e)}'
            }))

    async def receive(self, text_data=None, bytes_data=None):
        """Handle incoming WebSocket messages"""
        if bytes_data:
            # Handle binary audio data
            await self.handle_audio_data(bytes_data)
        elif text_data:
            # Handle text commands
            try:
                data = json.loads(text_data)
                await self.handle_text_command(data)
            except json.JSONDecodeError:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))

    async def handle_text_command(self, data):
        """Handle text commands from client"""
        command = data.get('command')

        if command == 'start_recording':
            self.audio_buffer = b''
            await self.send(text_data=json.dumps({
                'type': 'recording_started',
                'message': 'Recording started. Send audio data.'
            }))

        elif command == 'stop_recording':
            # Process any remaining audio in buffer
            if len(self.audio_buffer) >= 4:  # At least 2 samples (4 bytes)
                # Pad buffer to even size
                if len(self.audio_buffer) % 2 != 0:
                    self.audio_buffer += b'\x00'
                await self.process_audio_chunk(self.audio_buffer)

            self.audio_buffer = b''
            await self.send(text_data=json.dumps({
                'type': 'recording_stopped',
                'message': 'Recording stopped.'
            }))

        elif command == 'process_chunk':
            if len(self.audio_buffer) >= self.get_chunk_size():
                chunk_data = self.audio_buffer[:self.get_chunk_size()]
                self.audio_buffer = self.audio_buffer[self.get_chunk_size():]
                await self.process_audio_chunk(chunk_data)

        elif command == 'get_status':
            await self.send(text_data=json.dumps({
                'type': 'status',
                'buffer_size': len(self.audio_buffer),
                'model_loaded': self.stt_ready
            }))

    async def handle_audio_data(self, audio_data):
        """Handle incoming audio data"""
        # Add audio data to buffer
        self.audio_buffer += audio_data

        # Process complete chunks only (multiples of 2 bytes for 16-bit audio)
        chunk_size = self.get_chunk_size()
        while len(self.audio_buffer) >= chunk_size:
            # Extract exactly one chunk
            chunk_data = self.audio_buffer[:chunk_size]
            self.audio_buffer = self.audio_buffer[chunk_size:]

            await self.process_audio_chunk(chunk_data)

    def get_chunk_size(self):
        """Calculate chunk size in bytes based on duration and sample rate"""
        # 16-bit audio = 2 bytes per sample
        bytes_per_sample = 2
        samples_per_chunk = (self.sample_rate * self.chunk_duration_ms) // 1000
        chunk_size = samples_per_chunk * self.channels * bytes_per_sample

        # Ensure chunk size is even (multiple of 2)
        if chunk_size % 2 != 0:
            chunk_size += 1

        return chunk_size

    async def process_audio_chunk(self, chunk_data):
        """Process a chunk of audio data"""
        if not self.stt_ready or not chunk_data:
            return
        # Ensure chunk size is valid
        if len(chunk_data) % 2 != 0:  # Must be multiple of 2 for 16-bit audio
            # Pad with zero if incomplete
            chunk_data += b'\x00'

        try:
            # Convert bytes to AudioSegment
            audio_segment = self.bytes_to_audio_segment(chunk_data)

            # Check if audio has enough energy (not just silence)
            if audio_segment.dBFS < self.silence_threshold:
                return

            # Split audio into sentences based on silence
            sentences = self.split_into_sentences(audio_segment)

            if sentences:
                # Process each sentence
                for i, sentence_audio in enumerate(sentences):
                    if len(sentence_audio) > 100:  # Only process if longer than 100ms
                        transcription = await self.transcribe_audio(sentence_audio)
                        if transcription.strip():
                            await self.send(text_data=json.dumps({
                                'type': 'transcription',
                                'text': transcription.strip(),
                                'sentence_index': i,
                                'total_sentences': len(sentences),
                                'timestamp': sentence_audio.duration_seconds
                            }))

            else:
                # If no sentences detected, process the whole chunk
                transcription = await self.transcribe_audio(audio_segment)
                if transcription.strip():
                    await self.send(text_data=json.dumps({
                        'type': 'transcription',
                        'text': transcription.strip(),
                        'sentence_index': 0,
                        'total_sentences': 1,
                        'timestamp': audio_segment.duration_seconds
                    }))

        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing audio chunk: {str(e)}'
            }))

    async def process_complete_audio(self):
        """Process the complete audio buffer"""
        if not self.stt_ready or not self.audio_buffer:
            return

        try:
            audio_segment = self.bytes_to_audio_segment(self.audio_buffer)
            sentences = self.split_into_sentences(audio_segment)

            full_transcription = ""
            for i, sentence_audio in enumerate(sentences):
                if len(sentence_audio) > 100:
                    transcription = await self.transcribe_audio(sentence_audio)
                    if transcription.strip():
                        full_transcription += transcription.strip() + " "

                        await self.send(text_data=json.dumps({
                            'type': 'transcription',
                            'text': transcription.strip(),
                            'sentence_index': i,
                            'total_sentences': len(sentences),
                            'timestamp': sentence_audio.duration_seconds
                        }))

            # Send complete transcription
            if full_transcription.strip():
                await self.send(text_data=json.dumps({
                    'type': 'complete_transcription',
                    'text': full_transcription.strip(),
                    'total_sentences': len(sentences)
                }))

                # Save to database
                await self.save_transcription(full_transcription.strip())

        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing complete audio: {str(e)}'
            }))

    def bytes_to_audio_segment(self, audio_bytes):
        """Convert raw audio bytes to AudioSegment"""
        try:
            # Ensure the buffer size is even
            if len(audio_bytes) % 2 != 0:
                # Trim to even size
                audio_bytes = audio_bytes[:len(audio_bytes) - 1]

            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Create AudioSegment from numpy array
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,  # 16-bit
                channels=self.channels
            )

            return audio_segment
        except Exception as e:
            # Fallback: create empty audio segment
            return AudioSegment.silent(duration=self.chunk_duration_ms)

    def split_into_sentences(self, audio_segment):
        """Split audio into sentences based on silence detection"""
        try:
            # Normalize audio first
            audio_segment = audio_segment.normalize()

            # Split on silence
            sentences = split_on_silence(
                audio_segment,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_threshold,
                keep_silence=200  # Keep 200ms of silence at the end
            )

            # Filter out very short segments
            sentences = [s for s in sentences if len(
                s) > 500]  # At least 500ms

            return sentences
        except Exception as e:
            # If splitting fails, return the original audio
            return [audio_segment]

    async def transcribe_audio(self, audio_segment):
        """Transcribe through the shared STT HTTP service."""
        if not self.stt_ready or self.http is None:
            return ""

        try:
            normalized = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            response = await self.http.post(
                f"{settings.STT_SERVER_URL}/v1/transcriptions",
                content=normalized.raw_data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Sample-Rate": "16000",
                    "X-Channels": "1",
                    "X-Session-Id": self.channel_name,
                    "X-Turn-Id": uuid.uuid4().hex,
                },
            )
            response.raise_for_status()
            return str(response.json().get("text", ""))

        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    @database_sync_to_async
    def save_transcription(self, transcription):
        """Save transcription to database"""
        try:
            from .models import AudioRecording
            recording = AudioRecording(
                transcript=transcription
            )
            recording.save()
        except Exception as e:
            print(f"Database save error: {e}")
