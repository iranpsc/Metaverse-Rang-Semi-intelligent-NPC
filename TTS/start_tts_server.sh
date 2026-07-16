#!/bin/bash
# Starts the TTS microservice inside its dedicated venv (TTS/tts/).
# Run in the background with:  nohup ./start_tts_server.sh > tts_server.log 2>&1 &
cd "$(dirname "$0")"
exec ./tts/bin/python3 tts_server.py "$@"
