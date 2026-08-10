# Contributing to MetaRang Semi-Intelligent NPC

Thank you for contributing to the **MetaRang Semi-Intelligent NPC** project.

This repository contains the backend infrastructure for AI-assisted NPCs in the MetaRang Metaverse, including LLM/RAG processing, Speech-to-Text, Text-to-Speech, real-time WebSocket communication, Unity integration, Celery background tasks, Redis, and Docker-based deployment.

This document defines the development workflow and contribution standards for the project.

---

## 📋 Table of Contents

* [Development Setup](#development-setup)
* [Project Structure](#project-structure)
* [Branching Strategy](#branching-strategy)
* [Branch Naming](#branch-naming)
* [Commit Convention](#commit-convention)
* [Pull Requests](#pull-requests)
* [Code Style](#code-style)
* [Testing](#testing)
* [WebSocket Development](#websocket-development)
* [AI / LLM / RAG Changes](#ai--llm--rag-changes)
* [STT / TTS Changes](#stt--tts-changes)
* [Unity Integration Changes](#unity-integration-changes)
* [Docker Changes](#docker-changes)
* [Environment Variables](#environment-variables)
* [Security Requirements](#security-requirements)
* [Documentation](#documentation)
* [Review Checklist](#review-checklist)

---

# 🚀 Development Setup

## Requirements

Before contributing, make sure you have the required development tools installed:

* Python 3.11+
* Git
* Redis
* FFmpeg
* Docker
* Docker Compose

Depending on the component you are working on, you may also need:

* CUDA
* NVIDIA GPU
* Ollama
* Whisper models
* Hugging Face models
* Unity

---

## Clone the Repository

```bash
git clone https://github.com/iranpsc/Metaverse-Rang-Semi-intelligent-NPC.git

cd Metaverse-Rang-Semi-intelligent-NPC
```

---

## Create a Virtual Environment

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

## Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

## Run Database Migrations

```bash
python manage.py migrate
```

---

## Run the Development Server

```bash
python manage.py runserver
```

For ASGI/WebSocket development:

```bash
uvicorn MetaRangNPC.asgi:application --host 0.0.0.0 --port 8000
```

---

# 🏗️ Project Structure

The main components of the repository are:

```text
Metaverse-Rang-Semi-intelligent-NPC/
│
├── .github/
│   └── workflows/
│
├── LLM/
│   └── LLM / RAG functionality
│
├── MetaRangNPC/
│   └── Django project configuration
│
├── STT/
│   └── Speech-to-Text
│
├── TTS/
│   └── Text-to-Speech
│
├── VC/
│   └── Voice conversion
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
│
├── WEBSOCKET_API_README.md
├── NGINX_WEBSOCKET_FIX.md
└── requirements.txt
```

Keep changes isolated to the component they belong to whenever possible.

For example:

```text
STT changes      → STT/
LLM changes      → LLM/
TTS changes      → TTS/
Django changes   → MetaRangNPC/
Unity changes    → UnityWebSocketClient.cs
Infrastructure   → Dockerfile / docker-compose.yml
WebSocket        → ASGI / Channels / STT
```

---

# 🌿 Branching Strategy

Do not commit directly to the `main` branch.

Create a dedicated branch for every feature, bug fix, refactor, documentation change, or infrastructure change.

Recommended workflow:

```text
main
 │
 ├── feature/...
 ├── fix/...
 ├── refactor/...
 ├── docs/...
 ├── test/...
 └── chore/...
```

Example:

```bash
git checkout main

git pull origin main

git checkout -b feature/npc-memory
```

---

# 🏷️ Branch Naming

Use descriptive branch names.

### Feature

```text
feature/npc-memory
feature/llm-rag-context
feature/websocket-authentication
```

### Bug Fix

```text
fix/whisper-model-loading
fix/websocket-disconnect
fix/tts-timeout
```

### Refactor

```text
refactor/stt-service
refactor/llm-pipeline
```

### Documentation

```text
docs/websocket-api
docs/deployment
```

### Tests

```text
test/stt-websocket
test/llm-rag
```

### Infrastructure

```text
chore/docker-update
chore/redis-config
chore/ci-pipeline
```

Avoid names such as:

```text
test
new
changes
update
fix
my-branch
final
final2
new-version
```

Branch names should communicate the purpose of the change.

---

# 📝 Commit Convention

Use clear and consistent commit messages.

The recommended format is:

```text
type(scope): description
```

Examples:

```text
feat(stt): add sentence-level transcription
```

```text
fix(websocket): handle unexpected client disconnect
```

```text
feat(llm): add contextual retrieval
```

```text
fix(tts): handle unavailable TTS service
```

```text
refactor(rag): simplify document retrieval pipeline
```

```text
test(stt): add websocket transcription tests
```

```text
docs(websocket): update audio protocol documentation
```

```text
chore(docker): optimize backend image
```

---

## Commit Types

| Type       | Purpose                                    |
| ---------- | ------------------------------------------ |
| `feat`     | New functionality                          |
| `fix`      | Bug fix                                    |
| `refactor` | Code restructuring without behavior change |
| `test`     | Adding or modifying tests                  |
| `docs`     | Documentation                              |
| `chore`    | Maintenance/configuration                  |
| `perf`     | Performance improvement                    |
| `build`    | Build/dependency changes                   |
| `ci`       | CI/CD changes                              |
| `revert`   | Revert a previous change                   |

---

# 🔀 Pull Requests

All changes should be submitted through a Pull Request.

A Pull Request should:

1. Have a clear title.
2. Explain what changed.
3. Explain why the change was necessary.
4. Include testing information.
5. Mention configuration changes.
6. Mention breaking changes.
7. Include relevant screenshots/logs when useful.
8. Keep unrelated changes out of the PR.

---

## Pull Request Title

Use the same convention as commits.

Good:

```text
feat(stt): add real-time sentence detection
```

```text
fix(websocket): prevent connection timeout
```

Bad:

```text
Update
```

```text
Changes
```

```text
Fix bug
```

```text
New version
```

---

# 📄 Pull Request Template

Use the following structure when creating a PR:

```markdown
## Description

Briefly describe the change.

## Why

Explain why this change is required.

## Changes

- Change 1
- Change 2
- Change 3

## Testing

- [ ] Unit tests
- [ ] Integration tests
- [ ] WebSocket tests
- [ ] Manual testing
- [ ] Docker testing

## Configuration Changes

- [ ] No configuration changes
- [ ] New environment variables
- [ ] Existing environment variables changed

## Breaking Changes

- [ ] No
- [ ] Yes

If yes, explain:

## Additional Notes

Add any additional information here.
```

---

# 🧪 Testing

Every functional change should be tested before opening a Pull Request.

At minimum:

```bash
python manage.py check
```

and:

```bash
python manage.py test
```

If the change affects WebSocket functionality, also run:

```bash
python test_websocket_simple.py
```

or:

```bash
python test_websocket_client.py
```

---

## Test Requirements

### Bug Fix

Every bug fix should ideally include a regression test.

Example:

```text
Bug:
WebSocket crashes when the client disconnects unexpectedly.

Expected:
Connection closes gracefully.

Test:
test_websocket_disconnect.py
```

### New Feature

New functionality should include tests covering:

* Expected behavior
* Invalid input
* Failure cases
* Edge cases

---

# 🔌 WebSocket Development

WebSocket functionality is a critical part of the NPC architecture.

The documented STT endpoint is:

```text
/ws/audio-transcription/
```

Changes to WebSocket behavior must be handled carefully because they can affect:

* Unity
* Browser clients
* STT
* Audio streaming
* Sentence detection
* Production Nginx configuration

Before modifying the WebSocket protocol:

1. Review `WEBSOCKET_API_README.md`.
2. Check existing clients.
3. Test connection establishment.
4. Test audio streaming.
5. Test disconnect/reconnect.
6. Test malformed messages.
7. Test timeout behavior.
8. Verify Unity compatibility.

---

## WebSocket Compatibility

Avoid changing existing message structures without a clear migration plan.

For example, changing:

```json
{
  "type": "transcription",
  "text": "Hello"
}
```

to:

```json
{
  "event": "text",
  "content": "Hello"
}
```

may break existing Unity clients.

If a breaking protocol change is required:

* Document it.
* Update the WebSocket documentation.
* Update the Unity client.
* Update test clients.
* Update integration tests.
* Clearly mark the change in the Pull Request.

---

# 🧠 AI / LLM / RAG Changes

Changes to the AI pipeline require additional care.

Before modifying LLM/RAG functionality, document:

* Model name
* Model version
* Embedding model
* Vector database
* Retrieval configuration
* Prompt changes
* Context window requirements
* Hardware requirements
* Expected performance impact

---

## Model Files

Do not commit large model files directly to Git.

Avoid committing:

```text
*.bin
*.safetensors
*.pt
*.pth
*.onnx
```

unless explicitly approved for the project.

Model artifacts should be stored using an appropriate model/artifact storage system.

---

## Prompt Changes

Prompt changes can significantly affect NPC behavior.

For significant prompt changes, include:

```text
Previous behavior:
...

New behavior:
...

Reason:
...

Example input:
...

Expected output:
...
```

This makes AI-related changes reviewable and reproducible.

---

# 🎤 STT / TTS Changes

Speech processing changes should be tested using real audio samples where possible.

The STT pipeline currently expects audio parameters such as:

```text
Sample Rate: 16000 Hz
Channels: 1
Bit Depth: 16-bit PCM
```

Changes to these requirements must be documented.

---

## STT Testing

When modifying STT functionality, test:

* Short speech
* Long speech
* Silence
* Background noise
* Multiple sentences
* Partial transcription
* Complete transcription
* Client disconnect
* Invalid audio
* Model loading failure

---

## TTS Testing

TTS changes should verify:

* Successful synthesis
* Empty input
* Long text
* Service unavailable
* Timeout
* Invalid response
* Audio format
* Unity playback compatibility

---

# 🎮 Unity Integration Changes

The repository includes:

```text
UnityWebSocketClient.cs
```

Changes affecting the Unity client should be tested against the backend WebSocket implementation.

If a backend message changes, update:

```text
UnityWebSocketClient.cs
```

and the WebSocket documentation together.

Do not merge backend protocol changes without verifying Unity compatibility.

---

# 🐳 Docker Changes

Docker-related changes should be tested locally.

Run:

```bash
docker compose build
```

Then:

```bash
docker compose up
```

Verify:

```bash
docker compose ps
```

Check logs:

```bash
docker compose logs
```

For a specific service:

```bash
docker compose logs web
```

or:

```bash
docker compose logs celery-worker
```

---

## Docker Best Practices

When modifying Docker configuration:

* Avoid unnecessary image layers.
* Do not include secrets.
* Do not copy local virtual environments.
* Keep model files outside the image unless explicitly required.
* Prefer reproducible dependency versions.
* Verify the image starts successfully.
* Verify WebSocket functionality.
* Verify Celery connectivity.

---

# 🔐 Environment Variables

Never commit secrets.

Do not commit:

```text
.env
.env.production
.env.local
```

or files containing:

* API keys
* Passwords
* Tokens
* Database credentials
* Django secret keys
* Cloud credentials

Instead, provide an example:

```text
.env.example
```

Example:

```env
DEBUG=False
SECRET_KEY=change-me
ALLOWED_HOSTS=localhost,127.0.0.1

REDIS_URL=redis://redis:6379/0

OLLAMA_BASE_URL=http://localhost:11434

TTS_SERVER_URL=http://localhost:5002
```

---

# 🛡️ Security Requirements

Security-sensitive changes require additional review.

Never:

* Commit credentials.
* Disable authentication to simplify testing.
* Expose Redis publicly.
* Expose Celery Flower publicly without protection.
* Enable unrestricted CORS in production.
* Use development secrets in production.
* Log API keys or access tokens.
* Store user audio unnecessarily.

Before production deployment, verify:

```text
DEBUG=False
```

and that CORS, CSRF, authentication and allowed hosts are properly restricted.

---

# 📚 Documentation

If your change modifies behavior, update the relevant documentation.

Examples:

| Change               | Documentation                 |
| -------------------- | ----------------------------- |
| WebSocket protocol   | `WEBSOCKET_API_README.md`     |
| Nginx/WebSocket      | `NGINX_WEBSOCKET_FIX.md`      |
| Environment variable | `.env.example` + README       |
| LLM model            | README / LLM documentation    |
| STT behavior         | WebSocket/STT documentation   |
| Unity protocol       | Unity client + WebSocket docs |
| Docker service       | README + Docker documentation |

Documentation should be updated in the same Pull Request whenever possible.

---

# 🧹 Code Quality

Keep code:

* Readable
* Modular
* Testable
* Explicit
* Consistent with existing project architecture

Avoid:

* Unnecessary global state
* Duplicate logic
* Hard-coded secrets
* Unexplained magic numbers
* Dead code
* Debug `print()` statements
* Unused imports
* Large unrelated refactors

Prefer meaningful names:

```python
audio_buffer
transcription_result
npc_response
retrieved_context
```

instead of:

```python
x
data2
tmp
result1
```

---

# ⚡ Performance

AI workloads can be computationally expensive.

Performance-sensitive changes should consider:

* CPU usage
* GPU usage
* RAM/VRAM consumption
* Model loading time
* WebSocket latency
* STT latency
* LLM response time
* TTS latency
* Redis load
* Celery worker concurrency

If a change significantly affects performance, include before/after measurements in the Pull Request.

Example:

```text
STT latency:

Before: 2.8s
After: 1.9s

VRAM:

Before: 5.2 GB
After: 5.6 GB
```

---

# 📊 Logging

Use structured and meaningful logs.

Good:

```python
logger.info(
    "STT transcription completed",
    extra={
        "session_id": session_id,
        "duration": duration,
    },
)
```

Avoid:

```python
print("AAAAAAAAAAAA")
```

Never log:

* Passwords
* API keys
* Access tokens
* Private user information
* Sensitive audio metadata

---

# 🔄 Recommended Development Workflow

The recommended workflow is:

```text
1. Pull latest main
        │
        ▼
2. Create feature/fix branch
        │
        ▼
3. Implement change
        │
        ▼
4. Run formatting / checks
        │
        ▼
5. Run tests
        │
        ▼
6. Test Docker if required
        │
        ▼
7. Commit changes
        │
        ▼
8. Push branch
        │
        ▼
9. Open Pull Request
        │
        ▼
10. Code Review
        │
        ▼
11. CI checks
        │
        ▼
12. Merge
```

---

# 🔍 Before Opening a Pull Request

Run the following checklist:

```bash
python manage.py check
```

```bash
python manage.py test
```

If WebSocket functionality was changed:

```bash
python test_websocket_simple.py
```

If Docker-related changes were made:

```bash
docker compose build
docker compose up
```

Then verify:

* [ ] Code works locally
* [ ] Tests pass
* [ ] No secrets were committed
* [ ] No unnecessary files were added
* [ ] Documentation is updated
* [ ] Environment variables are documented
* [ ] WebSocket compatibility was checked
* [ ] Unity compatibility was checked when applicable
* [ ] Docker was tested when applicable
* [ ] Commit messages follow the project convention

---

# 👀 Code Review Checklist

Reviewers should check:

### Functionality

* [ ] Does the implementation solve the intended problem?
* [ ] Are edge cases handled?
* [ ] Are errors handled correctly?

### Architecture

* [ ] Does the change belong in the selected component?
* [ ] Does it introduce unnecessary coupling?
* [ ] Is the implementation maintainable?

### Testing

* [ ] Are tests included?
* [ ] Are regression cases covered?
* [ ] Does CI pass?

### Security

* [ ] Are secrets protected?
* [ ] Are permissions appropriate?
* [ ] Is user data handled safely?

### Performance

* [ ] Does the change increase CPU/GPU usage?
* [ ] Does it increase memory usage?
* [ ] Does it introduce additional network latency?

### Documentation

* [ ] Are API changes documented?
* [ ] Are configuration changes documented?
* [ ] Are breaking changes clearly identified?

---

# 🚨 Breaking Changes

Breaking changes must be clearly identified in the Pull Request.

Examples:

* WebSocket message format changes
* Endpoint changes
* Environment variable removal
* Model replacement
* Required Python version changes
* Docker service changes
* Unity client protocol changes

Use:

```text
BREAKING CHANGE:
<description>
```

in the Pull Request description.

---

# 🧩 Adding a New Service

If a new service is introduced, the Pull Request should include:

* Service purpose
* Architecture impact
* Docker configuration
* Environment variables
* Health check
* Logging
* Error handling
* Tests
* Documentation
* Resource requirements

Example:

```text
New Service: NPC Memory Service

Purpose:
Persistent conversation memory for NPC sessions.

Dependencies:
Redis
PostgreSQL

Environment:
NPC_MEMORY_DATABASE_URL
```

---

# 📦 Dependency Changes

Before adding a new dependency, consider:

1. Is it already available in the project?
2. Is it actively maintained?
3. Does it introduce security concerns?
4. Does it significantly increase Docker image size?
5. Does it require GPU/CUDA?
6. Does it introduce license restrictions?
7. Is there a simpler alternative?

After modifying dependencies:

```bash
pip install -r requirements.txt
```

and run the relevant tests.

Document major dependency changes in the Pull Request.

---

# 🤖 AI-Specific Review

AI-related Pull Requests should additionally answer:

```text
Model:
...

Model Version:
...

Embedding Model:
...

Vector Store:
...

Prompt Changes:
...

Expected Behavior:
...

Hardware Requirements:
...

Performance Impact:
...
```

This is especially important because changing an AI model or prompt can change NPC behavior without producing conventional software errors.

---

# 📌 Issue Reporting

When opening a bug report, include:

```text
## Description

What happened?

## Expected Behavior

What should have happened?

## Actual Behavior

What actually happened?

## Steps to Reproduce

1.
2.
3.

## Environment

Python:
Django:
Docker:
OS:

## Logs

Paste relevant logs here.

## Additional Context

Anything else that may help reproduce the problem.
```

Do not include:

* API keys
* Passwords
* Private tokens
* Production credentials
* Sensitive user information

---

# 🤝 Contribution Principles

We value:

* Clear communication
* Small and focused Pull Requests
* Reproducible changes
* Automated testing
* Security-conscious development
* Performance awareness
* Good documentation
* Respectful code review

The goal is not only to make the code work, but to keep the NPC platform maintainable as the MetaRang ecosystem grows.

---

# 📜 License

Contributors must follow the project's applicable license and organizational policies.

If the project's license changes, this document should be updated accordingly.

---

# 📬 Questions

For questions about architecture, AI models, WebSocket behavior, Unity integration, deployment, or contribution workflow, open a GitHub Issue or discuss the change in the relevant Pull Request.

Thank you for contributing to **MetaRang Semi-Intelligent NPC**.
