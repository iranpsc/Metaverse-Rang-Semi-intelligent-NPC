#!/usr/bin/env python3
"""
Test WebSocket client for Unity integration
This script demonstrates how to connect to the WebSocket API and send audio data
"""

import asyncio
import websockets
import json
import pyaudio
import wave
import threading
import time
import os
from pydub import AudioSegment
from pydub.utils import which


class UnityWebSocketClient:
    def __init__(self, uri="ws://localhost:8000/ws/audio-transcription/"):
        self.uri = uri
        self.websocket = None
        self.is_recording = False
        self.audio_buffer = b''

        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None

    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            print("Connected to WebSocket server")

            # Start listening for messages
            await self.listen_for_messages()

        except Exception as e:
            print(f"Connection failed: {e}")

    async def listen_for_messages(self):
        """Listen for messages from the server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_server_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by server")
        except Exception as e:
            print(f"Error listening for messages: {e}")

    async def handle_server_message(self, data):
        """Handle messages from the server"""
        message_type = data.get('type')

        if message_type == 'connection_established':
            print(f"Server: {data.get('message')}")
            print(f"Config: {data.get('config')}")

        elif message_type == 'model_loaded':
            print(f"Server: {data.get('message')}")

        elif message_type == 'transcription':
            text = data.get('text', '')
            sentence_index = data.get('sentence_index', 0)
            total_sentences = data.get('total_sentences', 1)
            timestamp = data.get('timestamp', 0)

            print(
                f"[{timestamp:.2f}s] Sentence {sentence_index + 1}/{total_sentences}: {text}")

        elif message_type == 'complete_transcription':
            text = data.get('text', '')
            total_sentences = data.get('total_sentences', 0)
            print(
                f"\n=== Complete Transcription ({total_sentences} sentences) ===")
            print(text)
            print("=" * 50)

        elif message_type == 'error':
            print(f"Error: {data.get('message')}")

        elif message_type == 'recording_started':
            print(f"Server: {data.get('message')}")

        elif message_type == 'recording_stopped':
            print(f"Server: {data.get('message')}")

        elif message_type == 'status':
            buffer_size = data.get('buffer_size', 0)
            model_loaded = data.get('model_loaded', False)
            print(
                f"Status - Buffer: {buffer_size} bytes, Model loaded: {model_loaded}")

    async def send_command(self, command, **kwargs):
        """Send a command to the server"""
        if self.websocket:
            message = {'command': command, **kwargs}
            await self.websocket.send(json.dumps(message))

    async def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return

        self.is_recording = True
        await self.send_command('start_recording')

        # Start audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )

        print("Recording started... Press Ctrl+C to stop")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_recording and self.websocket:
            # Send audio data asynchronously
            asyncio.create_task(self.send_audio_data(in_data))
        return (in_data, pyaudio.paContinue)

    async def send_audio_data(self, audio_data):
        """Send audio data to the server"""
        if self.websocket:
            try:
                await self.websocket.send(audio_data)
            except Exception as e:
                print(f"Error sending audio data: {e}")

    async def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        await self.send_command('stop_recording')
        print("Recording stopped")

    async def send_audio_file(self, file_path):
        """Send an audio file to the server"""
        try:
            # Load and process audio file
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(
                self.sample_rate).set_channels(self.channels)

            # Convert to raw audio data
            raw_data = audio.raw_data

            # Send start command
            await self.send_command('start_recording')

            # Send audio data in chunks
            chunk_size = 1024
            for i in range(0, len(raw_data), chunk_size):
                chunk = raw_data[i:i + chunk_size]
                await self.websocket.send(chunk)
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

            # Send stop command
            await self.send_command('stop_recording')

        except Exception as e:
            print(f"Error sending audio file: {e}")

    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        if self.websocket:
            asyncio.create_task(self.websocket.close())


async def main():
    """Main function to run the test client"""
    client = UnityWebSocketClient()

    try:
        # Connect to server
        connect_task = asyncio.create_task(client.connect())

        # Wait a bit for connection to establish
        await asyncio.sleep(2)

        # Test with audio file if provided
        if len(os.sys.argv) > 1:
            audio_file = os.sys.argv[1]
            if os.path.exists(audio_file):
                print(f"Sending audio file: {audio_file}")
                await client.send_audio_file(audio_file)
            else:
                print(f"Audio file not found: {audio_file}")
        else:
            # Interactive mode
            print("\nWebSocket Audio Transcription Test Client")
            print("Commands:")
            print("  'start' - Start recording")
            print("  'stop' - Stop recording")
            print("  'status' - Get server status")
            print("  'quit' - Exit")
            print()

            while True:
                try:
                    command = input("Enter command: ").strip().lower()

                    if command == 'start':
                        await client.start_recording()
                    elif command == 'stop':
                        await client.stop_recording()
                    elif command == 'status':
                        await client.send_command('get_status')
                    elif command == 'quit':
                        break
                    else:
                        print("Unknown command")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")

        # Wait for any remaining messages
        await asyncio.sleep(2)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.cleanup()


if __name__ == "__main__":
    # Check if pyaudio is available
    try:
        import pyaudio
    except ImportError:
        print("PyAudio is required for microphone recording.")
        print("Install it with: pip install pyaudio")
        exit(1)

    asyncio.run(main())
