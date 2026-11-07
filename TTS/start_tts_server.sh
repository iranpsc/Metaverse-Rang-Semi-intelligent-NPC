#!/bin/bash
# Starts Mana Persian Piper from the current Python environment.
# Production should use Compose. For local use:
#   nohup ./TTS/start_tts_server.sh > tts_server.log 2>&1 &
cd "$(dirname "$0")/.."
exec python TTS/tts_server.py "$@"
