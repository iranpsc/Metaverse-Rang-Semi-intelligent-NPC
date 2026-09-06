#!/bin/sh
set -eu

# Named volumes are initially owned by root. Prepare the persistent Whisper
# download cache, then permanently drop privileges before starting Uvicorn.
mkdir -p /cache/whisper
chown -R app:app /cache

exec setpriv --reuid=app --regid=app --init-groups "$@"
