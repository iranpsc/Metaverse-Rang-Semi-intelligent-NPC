#!/bin/bash
# Starts the FreeVC voice-conversion microservice inside its dedicated venv
# (VC/freevc-env/). Run in the background with:
#   nohup ./start_vc_server.sh > vc_server.log 2>&1 &
cd "$(dirname "$0")"
exec ./freevc-env/bin/python vc_server.py "$@"
