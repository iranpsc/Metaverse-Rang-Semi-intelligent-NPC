import json
import asyncio
import whisper
import os
import tempfile
import io
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
from django.conf import settings


class AudioTranscriptionConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.whisper_model = None
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

        # Load Whisper model asynchronously
        await self.load_whisper_model()

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
        """Handle WebSocket disconnection"""
        # Clean up any temporary files
        if hasattr(self, 'temp_files'):
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    async def load_whisper_model(self):
        """Load Whisper model asynchronously"""
        try:
            model_path = os.path.join(
                os.path.dirname(__file__), "model", "Tiny.pt")
            if os.path.exists(model_path):
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.whisper_model = await loop.run_in_executor(
                    None, whisper.load_model, model_path, 'cpu'
                )
            else:
                # Fallback to base model
                self.whisper_model = await loop.run_in_executor(
                    None, whisper.load_model, 'base', 'cpu'
                )

            await self.send(text_data=json.dumps({
                'type': 'model_loaded',
                'message': 'Whisper model loaded successfully'
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to load Whisper model: {str(e)}'
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
                'model_loaded': self.whisper_model is not None
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
        if not self.whisper_model or not self.audio_buffer:
            return
        # Ensure chunk size is valid
        if len(chunk_data) % 2 != 0:  # Must be multiple of 2 for 16-bit audio
            # Pad with zero if incomplete
            chunk_data += b'\x00'

        try:
            # Convert bytes to AudioSegment
            audio_segment = self.bytes_to_audio_segment(self.audio_buffer)

            # Check if audio has enough energy (not just silence)
            if audio_segment.dBFS < self.silence_threshold:
                # Clear buffer and return if audio is too quiet
                self.audio_buffer = b''
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

                # Clear buffer after processing
                self.audio_buffer = b''
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
                self.audio_buffer = b''

        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing audio chunk: {str(e)}'
            }))
            self.audio_buffer = b''

    async def process_complete_audio(self):
        """Process the complete audio buffer"""
        if not self.whisper_model or not self.audio_buffer:
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
        """Transcribe audio segment using Whisper"""
        if not self.whisper_model:
            return ""

        try:
            # Export audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                audio_segment.export(temp_file.name, format="wav")
                temp_file_path = temp_file.name

            try:
                # Run transcription in thread pool with correct parameters
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.whisper_model.transcribe(
                        temp_file_path,
                        fp16=False,
                        temperature=0.0,
                        best_of=1,
                        beam_size=3,
                        patience=1.0,
                        no_speech_threshold=0.6
                    )
                )

                return result["text"]
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

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
