# WebSocket Audio Transcription API

This WebSocket API provides real-time audio transcription using Whisper models. It's designed to work with Unity clients and supports audio chunking for multiple sentences.

## Features

- **Real-time Audio Transcription**: Stream audio data and get live transcriptions
- **Audio Chunking**: Automatically splits audio into sentences based on silence detection
- **Multiple Sentence Support**: Handles audio containing multiple sentences
- **Unity Integration**: Includes C# script for Unity WebSocket integration
- **Web Test Interface**: HTML test page for easy testing
- **Python Test Client**: Command-line client for testing

## Installation

1. Install the required dependencies:
```bash
pip install -r requirments.txt
```

2. Make sure Redis is running:
```bash
redis-server
```

3. Run database migrations:
```bash
python manage.py migrate
```

4. Start the Django server with ASGI support:
```bash
python manage.py runserver
```

## WebSocket API

### Connection
Connect to: `ws://localhost:8000/ws/audio-transcription/`

### Message Types

#### Client to Server

**Text Commands:**
```json
{
    "command": "start_recording"
}
```

```json
{
    "command": "stop_recording"
}
```

```json
{
    "command": "get_status"
}
```

**Binary Audio Data:**
- Send raw 16-bit PCM audio data
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Format: 16-bit signed integers

#### Server to Client

**Connection Established:**
```json
{
    "type": "connection_established",
    "message": "WebSocket connected successfully. Ready to receive audio.",
    "config": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_duration_ms": 1000
    }
}
```

**Model Loaded:**
```json
{
    "type": "model_loaded",
    "message": "Whisper model loaded successfully"
}
```

**Transcription (per sentence):**
```json
{
    "type": "transcription",
    "text": "Hello, this is a test sentence.",
    "sentence_index": 0,
    "total_sentences": 2,
    "timestamp": 1.5
}
```

**Complete Transcription:**
```json
{
    "type": "complete_transcription",
    "text": "Hello, this is a test sentence. This is another sentence.",
    "total_sentences": 2
}
```

**Error Messages:**
```json
{
    "type": "error",
    "message": "Error description"
}
```

**Status Response:**
```json
{
    "type": "status",
    "buffer_size": 1024,
    "model_loaded": true
}
```

## Usage Examples

### Python Test Client

```bash
# Interactive mode
python test_websocket_client.py

# Send audio file
python test_websocket_client.py path/to/audio.wav
```

### Unity Integration

1. Add the `UnityWebSocketClient.cs` script to your Unity project
2. Configure the WebSocket URL in the inspector
3. Use the public methods:

```csharp
// Connect to server
unityClient.ConnectToServer();

// Start recording
unityClient.StartRecording();

// Stop recording
unityClient.StopRecording();

// Send audio file
unityClient.SendAudioFile("path/to/audio.wav");

// Subscribe to events
unityClient.OnTranscriptionReceived += (text) => {
    Debug.Log($"Transcription: {text}");
};
```

### Web Test Interface

Visit `http://localhost:8000/websocket-test/` in your browser to use the web interface.

## Audio Processing

### Chunking Strategy

The API automatically splits audio into sentences using silence detection:

- **Silence Threshold**: -40 dB
- **Minimum Silence Length**: 500ms
- **Sentence End Silence**: 2000ms
- **Minimum Sentence Length**: 500ms

### Audio Format Requirements

- **Sample Rate**: 16000 Hz
- **Channels**: 1 (mono)
- **Bit Depth**: 16-bit
- **Format**: PCM (uncompressed)

## Configuration

### WebSocket Consumer Settings

You can modify these settings in `STT/consumers.py`:

```python
self.sample_rate = 16000
self.channels = 1
self.chunk_duration_ms = 1000
self.silence_threshold = -40  # dB
self.min_silence_len = 500    # ms
self.sentence_end_silence = 2000  # ms
```

### Whisper Model Settings

The API uses the following Whisper parameters:

```python
result = model.transcribe(
    audio_path,
    fp16=False,
    temperature=0.0,
    best_of=1,
    beam_size=3,
    patience=1.0,
    no_speech_threshold=0.6
)
```

## Error Handling

The API handles various error conditions:

- **Connection Errors**: WebSocket connection failures
- **Audio Processing Errors**: Invalid audio format or processing failures
- **Model Loading Errors**: Whisper model loading failures
- **Transcription Errors**: Whisper transcription failures

All errors are sent to the client as JSON messages with type "error".

## Performance Considerations

### Memory Usage

- Audio buffer is cleared after each chunk processing
- Temporary files are automatically cleaned up
- Model is loaded once per connection

### Latency

- Real-time processing with 1-second chunks
- Asynchronous processing to prevent blocking
- Configurable chunk size for different latency requirements

### Scalability

- Each WebSocket connection runs independently
- Redis channel layer for horizontal scaling
- Stateless design for load balancing

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure Redis is running and Django server is started with ASGI
2. **Audio Not Processing**: Check audio format (16kHz, mono, 16-bit)
3. **Model Loading Errors**: Ensure Whisper model file exists in `STT/model/`
4. **High Latency**: Reduce chunk size or increase processing frequency

### Debug Mode

Enable debug logging in the consumer:

```python
# In STT/consumers.py
print(f"Debug: {debug_message}")
```

## API Endpoints

- **WebSocket**: `ws://localhost:8000/ws/audio-transcription/`
- **Test Page**: `http://localhost:8000/websocket-test/`
- **Health Check**: `http://localhost:8000/` (existing Django views)

## Dependencies

- Django 5.1.3
- Channels 4.0.0
- Channels-Redis 4.2.0
- Whisper 1.1.10
- PyDub 0.25.1
- NumPy 1.24.3
- WebSocketSharp (Unity)
- PyAudio (Python test client)

## License

This project uses the same license as the main MetaRangNPC project.
