# 🤖 MetaRang Semi-Intelligent NPC

> Backend infrastructure for **semi-intelligent NPCs** in the MetaRang Metaverse, combining **LLM-powered conversation, RAG, speech-to-text, text-to-speech, background task processing, and real-time WebSocket communication with Unity**.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.x-092E20?logo=django\&logoColor=white)](https://www.djangoproject.com/)
[![WebSocket](https://img.shields.io/badge/WebSocket-Realtime-010101?logo=socketdotio\&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis\&logoColor=white)](https://redis.io/)
[![Celery](https://img.shields.io/badge/Celery-5.3-37814A?logo=celery\&logoColor=white)](https://docs.celeryq.dev/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker\&logoColor=white)](https://www.docker.com/)

---

## 📌 Overview

**MetaRang Semi-Intelligent NPC** is the backend service responsible for powering AI-assisted NPCs inside the MetaRang virtual environment.

The project provides the infrastructure required to connect a virtual NPC to AI and speech services:

```text
                    ┌──────────────────────┐
                    │   MetaRang / Unity   │
                    │       Client         │
                    └──────────┬───────────┘
                               │
                         WebSocket / HTTP
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Django / ASGI      │
                    │      Backend         │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       ┌───────────┐     ┌───────────┐     ┌───────────┐
       │    STT    │     │    LLM    │     │    TTS    │
       │  Whisper  │     │ AI / RAG  │     │  Speech   │
       └───────────┘     └─────┬─────┘     └───────────┘
                               │
                         ┌─────▼─────┐
                         │  Vector   │
                         │ Retrieval │
                         └───────────┘

                    ┌──────────────────────┐
                    │ Redis + Celery       │
                    │ Background Tasks     │
                    └──────────────────────┘
```

The backend is designed around **real-time interaction** between the virtual world and AI-driven NPC services.

---

# ✨ Features

## 🧠 LLM Integration

The project contains an `LLM` application responsible for AI-related functionality.

The dependency stack includes components such as:

* LangChain
* LangChain Community
* LangChain HuggingFace
* LangChain Ollama
* Transformers
* Hugging Face Hub
* Sentence Transformers
* ChromaDB
* FAISS
* Ollama
* PyTorch

This provides the foundation for connecting NPC behavior and conversation to local or externally hosted language models.

---

## 🔎 RAG & Semantic Retrieval

The project includes the dependencies required for retrieval-augmented generation and semantic search.

The stack includes:

* ChromaDB
* FAISS
* Sentence Transformers
* Hugging Face models
* LangChain
* Embedding models

This allows NPC responses to be enhanced with contextual information instead of relying only on the language model's internal knowledge.

A typical conceptual flow is:

```text
User / Player
     │
     ▼
NPC Message
     │
     ▼
Query Processing
     │
     ▼
Embedding
     │
     ▼
Vector Search
     │
     ▼
Relevant Context
     │
     ▼
LLM
     │
     ▼
NPC Response
```

---

# 🎙️ Speech-to-Text

The `STT` application provides real-time speech transcription using **Whisper**.

The project exposes a WebSocket endpoint:

```text
/ws/audio-transcription/
```

The audio pipeline supports:

* Real-time audio streaming
* Audio chunking
* Silence detection
* Sentence segmentation
* Whisper transcription
* Incremental transcription messages
* Complete transcription messages
* Unity integration

The expected audio format is:

| Property    | Value     |
| ----------- | --------- |
| Sample Rate | 16,000 Hz |
| Channels    | Mono      |
| Bit Depth   | 16-bit    |
| Encoding    | PCM       |

The WebSocket implementation uses approximately 1-second chunks and supports configurable silence detection.

---

# 🔊 Text-to-Speech

The `TTS` component provides the speech-generation layer for NPC responses.

The Django configuration exposes the TTS service through:

```text
TTS_SERVER_URL
```

The default local configuration points to:

```text
http://127.0.0.1:5002
```

This allows the speech-generation service to run independently from the main Django application.

---

# 🔌 Real-Time WebSocket Communication

Real-time communication is implemented using:

* Django Channels
* ASGI
* Redis Channel Layer
* WebSocket

This is particularly important for the Unity integration, where the NPC needs to communicate without relying exclusively on traditional request/response HTTP APIs.

The repository also contains:

* `UnityWebSocketClient.cs`
* WebSocket test clients
* Browser-based WebSocket test interface
* Nginx WebSocket configuration documentation

The documented audio WebSocket endpoint is:

```text
ws://localhost:8000/ws/audio-transcription/
```

The API supports commands such as:

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

and returns structured events such as:

```json
{
  "type": "transcription",
  "text": "Hello, this is a test sentence.",
  "sentence_index": 0,
  "total_sentences": 2,
  "timestamp": 1.5
}
```

A final transcription can be returned using:

```json
{
  "type": "complete_transcription",
  "text": "Hello, this is a test sentence. This is another sentence.",
  "total_sentences": 2
}
```

---

# ⚙️ Background Processing

The project uses **Celery** with **Redis** as the broker/result backend.

The Docker Compose configuration provides separate services for:

```text
web
redis
celery-worker
celery-beat
flower
```

This allows background operations and scheduled jobs to be separated from the main web server.

### Celery Worker

Processes asynchronous tasks.

### Celery Beat

Runs scheduled/background tasks.

### Flower

Provides a monitoring interface for Celery workers and tasks.

The default Flower port is:

```text
5555
```

---

# 🏗️ Project Structure

The main repository is organized around the following components:

```text
Metaverse-Rang-Semi-intelligent-NPC/
│
├── .github/
│   └── workflows/
│
├── LLM/
│   └── LLM and RAG functionality
│
├── MetaRangNPC/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── STT/
│   └── Speech-to-Text / Whisper
│
├── TTS/
│   └── Text-to-Speech service
│
├── VC/
│   └── Voice Conversion components
│
├── staticfiles/
│   └── Django static assets
│
├── UnityWebSocketClient.cs
│
├── test_websocket_client.py
├── test_websocket_simple.py
│
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── manage.py
├── requirements.txt
│
├── WEBSOCKET_API_README.md
└── NGINX_WEBSOCKET_FIX.md
```

The repository currently contains 26 commits and the main application directories include `LLM`, `MetaRangNPC`, `STT`, `TTS`, and a voice-conversion component.

---

# 🧩 Technology Stack

## Backend

* Python 3.11
* Django
* Django Channels
* ASGI
* Uvicorn

## AI / LLM

* PyTorch
* Transformers
* Hugging Face
* LangChain
* Ollama
* Sentence Transformers

## RAG / Vector Search

* ChromaDB
* FAISS
* Embedding Models

## Speech

* OpenAI Whisper
* PyDub
* FFmpeg
* WebSocket audio streaming

## Async Processing

* Celery
* Redis
* Flower

## Client Integration

* Unity
* C#
* WebSocket

## Infrastructure

* Docker
* Docker Compose
* Nginx
* Redis

The project's dependency file currently includes Django, Channels, Redis, Celery, ChromaDB, FAISS, LangChain, Ollama, Transformers, Sentence Transformers, PyTorch, Whisper and related AI/audio packages.

---

# 🚀 Getting Started

## Requirements

Before running the project locally, install:

* Python 3.11+
* Git
* Redis
* FFmpeg
* Optional: Docker & Docker Compose
* Optional: Ollama
* Required AI models depending on the selected LLM/STT configuration

The Docker image is based on Python `3.11.9-slim` and installs FFmpeg and the Python dependencies from `requirements.txt`.

---

# 1. Clone the Repository

```bash
git clone https://github.com/iranpsc/Metaverse-Rang-Semi-intelligent-NPC.git

cd Metaverse-Rang-Semi-intelligent-NPC
```

---

# 2. Create a Virtual Environment

### Linux / macOS

```bash
python3.11 -m venv venv

source venv/bin/activate
```

### Windows

```powershell
py -3.11 -m venv venv

venv\Scripts\activate
```

---

# 3. Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

> **Note:** The dependency list contains several large AI/ML packages. Installation may require significant disk space and, depending on the selected models, a CUDA-compatible GPU may be beneficial.

---

# 4. Start Redis

If Redis is installed locally:

```bash
redis-server
```

Verify:

```bash
redis-cli ping
```

Expected:

```text
PONG
```

---

# 5. Run Django Migrations

```bash
python manage.py migrate
```

---

# 6. Start the Backend

For local development:

```bash
python manage.py runserver
```

The application will be available at:

```text
http://localhost:8000
```

For ASGI/WebSocket deployment, Uvicorn can be used:

```bash
uvicorn MetaRangNPC.asgi:application --host 0.0.0.0 --port 8000
```

The repository's Docker configuration uses Uvicorn with the Django ASGI application.

---

# 🐳 Docker

Docker Compose is the recommended way to run the complete service stack.

The provided configuration starts:

```text
Redis
   │
   ├── Django / ASGI
   │
   ├── Celery Worker
   │
   ├── Celery Beat
   │
   └── Flower
```

Start the stack:

```bash
docker compose up --build
```

Run in background:

```bash
docker compose up -d --build
```

Check running containers:

```bash
docker compose ps
```

Stop the services:

```bash
docker compose down
```

The current Compose configuration exposes:

| Service       |   Port |
| ------------- | -----: |
| Django / ASGI | `8000` |
| Flower        | `5555` |
| Redis         | `6379` |

---

# 🔐 Environment Configuration

Environment-specific values should be configured through environment variables rather than hard-coded values.

Examples used by the Docker configuration include:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434

EMBEDDING_MODEL_PATH=/app/models/sentence_embeddings

TTS_SERVER_URL=http://127.0.0.1:5002
```

For production, sensitive configuration such as Django's `SECRET_KEY`, database credentials, API keys and service URLs should be provided through environment variables or a secrets-management system.

> **Security note:** the current `settings.py` contains a hard-coded Django secret key and has `DEBUG = True`. These values must be changed before a production deployment.

---

# 🧠 LLM Configuration

The project supports an LLM-oriented architecture with integrations including Ollama, Hugging Face, Transformers and LangChain.

When using Ollama, configure:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

Inside Docker, the Compose configuration defaults to:

```text
http://host.docker.internal:11434
```

The actual model should be configured according to the implementation inside the `LLM` application.

---

# 🎤 WebSocket Speech-to-Text API

## Endpoint

```text
ws://localhost:8000/ws/audio-transcription/
```

## Connection

A successful connection returns:

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

## Audio

Send raw:

```text
16-bit PCM
16 kHz
Mono
```

audio data over the WebSocket.

## Status

```json
{
  "command": "get_status"
}
```

Possible response:

```json
{
  "type": "status",
  "buffer_size": 1024,
  "model_loaded": true
}
```

The complete WebSocket protocol is documented in:

```text
WEBSOCKET_API_README.md
```

---

# 🎮 Unity Integration

The repository includes:

```text
UnityWebSocketClient.cs
```

This client is intended to simplify communication between Unity and the NPC backend.

Typical operations include:

```csharp
unityClient.ConnectToServer();

unityClient.StartRecording();

unityClient.StopRecording();

unityClient.SendAudioFile("path/to/audio.wav");
```

Transcription events can be consumed through:

```csharp
unityClient.OnTranscriptionReceived += (text) =>
{
    Debug.Log($"Transcription: {text}");
};
```

---

# 🌐 Nginx + WebSocket

When the application is deployed behind Nginx, WebSocket upgrade headers must be forwarded correctly.

A basic configuration is:

```nginx
location /ws/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;

    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
}
```

This is required because WebSocket connections depend on the HTTP `Upgrade` mechanism. The repository contains a dedicated Nginx troubleshooting document for this configuration.

---

# 🧪 Testing

The repository includes WebSocket test clients:

```bash
python test_websocket_client.py
```

or:

```bash
python test_websocket_simple.py
```

The browser-based WebSocket test interface is available at:

```text
http://localhost:8000/websocket-test/
```

The API documentation also provides an example for testing an audio file:

```bash
python test_websocket_client.py path/to/audio.wav
```

---

# 📊 WebSocket Audio Processing

The current STT implementation uses configurable audio segmentation parameters.

Default values include:

| Parameter            |    Default |
| -------------------- | ---------: |
| Sample Rate          | `16000 Hz` |
| Channels             |        `1` |
| Chunk Duration       |  `1000 ms` |
| Silence Threshold    |   `-40 dB` |
| Minimum Silence      |   `500 ms` |
| Sentence-End Silence |  `2000 ms` |
| Minimum Sentence     |   `500 ms` |

This allows the backend to process a continuous audio stream while producing sentence-level transcription events.

---

# 🏭 Production Architecture

A recommended production topology is:

```text
                         ┌───────────────┐
                         │     Unity     │
                         │    Client     │
                         └───────┬───────┘
                                 │
                         HTTPS / WSS
                                 │
                                 ▼
                         ┌───────────────┐
                         │     Nginx     │
                         │ Reverse Proxy │
                         └───────┬───────┘
                                 │
                     ┌───────────▼───────────┐
                     │      Django ASGI      │
                     │       Uvicorn         │
                     └───────────┬───────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
        ┌─────────┐        ┌───────────┐       ┌─────────┐
        │   STT   │        │    LLM    │       │   TTS   │
        │ Whisper │        │   + RAG   │       │ Service │
        └─────────┘        └───────────┘       └─────────┘
                                 │
                                 ▼
                           Vector Storage
                                 │
                                 ▼
                              Redis
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
              Celery Worker              Celery Beat
                    │
                    ▼
                  Tasks
```

---

# 🔒 Production Security Checklist

Before deploying this project publicly, review the following:

* [ ] Set `DEBUG=False`
* [ ] Move `SECRET_KEY` to environment variables
* [ ] Restrict `ALLOWED_HOSTS`
* [ ] Restrict CORS origins
* [ ] Configure CSRF trusted origins
* [ ] Use HTTPS
* [ ] Use `wss://` instead of `ws://`
* [ ] Protect Django Admin
* [ ] Protect Flower
* [ ] Restrict Redis from public access
* [ ] Store model files outside the Git repository
* [ ] Store API keys outside source code
* [ ] Configure production logging
* [ ] Configure database backups
* [ ] Configure resource limits for AI workloads

In particular, the current settings use:

```python
DEBUG = True
```

and:

```python
CORS_ALLOW_ALL_ORIGINS = True
```

These settings are appropriate only for development/testing and should be reviewed before production deployment.

---

# 📁 Main Components

| Component                  | Responsibility                                         |
| -------------------------- | ------------------------------------------------------ |
| `MetaRangNPC`              | Django project configuration and ASGI/WSGI entrypoints |
| `LLM`                      | LLM and AI-related functionality                       |
| `STT`                      | Speech-to-Text / Whisper                               |
| `TTS`                      | Text-to-Speech integration                             |
| `VC`                       | Voice conversion functionality                         |
| `UnityWebSocketClient.cs`  | Unity ↔ backend real-time communication                |
| `test_websocket_client.py` | WebSocket client testing                               |
| `test_websocket_simple.py` | Simple WebSocket testing                               |
| `docker-compose.yml`       | Multi-service deployment                               |
| `Dockerfile`               | Backend container image                                |
| `NGINX_WEBSOCKET_FIX.md`   | WebSocket reverse-proxy configuration                  |
| `WEBSOCKET_API_README.md`  | STT WebSocket protocol documentation                   |

---

# 🔄 NPC Interaction Flow

A typical voice interaction can be represented as:

```text
Player speaks
     │
     ▼
Unity Microphone
     │
     ▼
WebSocket
     │
     ▼
STT / Whisper
     │
     ▼
Text
     │
     ▼
LLM / RAG
     │
     ├── Retrieve Context
     │
     └── Generate Response
     │
     ▼
NPC Response
     │
     ▼
TTS
     │
     ▼
Audio
     │
     ▼
Unity
     │
     ▼
NPC speaks
```

This architecture separates speech recognition, reasoning, retrieval and speech synthesis into independent layers, making the system easier to extend and maintain.

---

# 🛠️ Development

A typical development workflow is:

```bash
git checkout -b feature/my-feature

# Make changes

python manage.py migrate

python manage.py runserver

# Run relevant tests

git add .

git commit -m "feat: implement my feature"

git push origin feature/my-feature
```

Pull requests should include:

* Description of the change
* Reason for the change
* Testing performed
* Configuration changes
* Any required model/service changes

---

# 🐛 Troubleshooting

## WebSocket connection refused

Check that Redis and the ASGI server are running.

```bash
redis-cli ping
```

Then verify:

```text
http://localhost:8000
```

For production behind Nginx, verify that the `Upgrade` and `Connection` headers are forwarded correctly.

---

## Whisper model does not load

Verify that the required model is available in the expected model directory and that the runtime has sufficient memory.

The STT documentation specifically identifies missing Whisper model files as one possible cause of model-loading failures.

---

## Audio is not transcribed

Verify the incoming audio is:

```text
16 kHz
Mono
16-bit PCM
```

Incorrect audio format can prevent successful processing.

---

## High transcription latency

The STT implementation processes audio in chunks.

Potential adjustments include:

* Reduce chunk duration
* Optimize Whisper model size
* Use GPU acceleration
* Optimize audio preprocessing
* Separate STT workers
* Scale WebSocket consumers

The repository documentation specifically notes chunk size and processing frequency as latency-related configuration points.

---

# 📚 Documentation

Additional project documentation:

* `WEBSOCKET_API_README.md` — WebSocket audio transcription API
* `NGINX_WEBSOCKET_FIX.md` — Nginx/WebSocket deployment configuration

---

# 🗺️ Roadmap

Potential future improvements include:

* [ ] Production-ready environment configuration
* [ ] Authentication for WebSocket connections
* [ ] Per-NPC memory
* [ ] Persistent conversation history
* [ ] Improved RAG pipeline
* [ ] NPC personality and behavior profiles
* [ ] Multiple concurrent NPC sessions
* [ ] GPU worker management
* [ ] STT/TTS service isolation
* [ ] API authentication and rate limiting
* [ ] PostgreSQL production database
* [ ] Automated test coverage
* [ ] CI/CD quality gates
* [ ] Monitoring and observability
* [ ] Horizontal scaling of WebSocket workers
* [ ] Production model management

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test the affected components
5. Commit your changes
6. Open a Pull Request

Please keep changes focused and document any new environment variables, services or model requirements.

---

# 📄 License

The license for this project should be defined in the repository before publishing or distributing the software.

If this project is intended for internal MetaRang use, add the organization's approved license and copyright notice here.

---

# 👨‍💻 Project

**MetaRang Semi-Intelligent NPC**

Repository:

`iranpsc/Metaverse-Rang-Semi-intelligent-NPC`

Built as part of the **MetaRang Metaverse** ecosystem.

---
