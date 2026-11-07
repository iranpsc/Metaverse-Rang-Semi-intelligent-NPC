#!/bin/bash
# Starts the OpenVoice V2 microservice from the current Python environment.
# Production should use the Compose voice-conversion profile. For local use:
#   nohup ./start_vc_server.sh > vc_server.log 2>&1 &
cd "$(dirname "$0")/.."
exec python VC/vc_server.py "$@"
