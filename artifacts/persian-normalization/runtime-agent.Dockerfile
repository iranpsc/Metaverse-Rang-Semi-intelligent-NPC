FROM metarang-npc-agent-worker:before-normalization
WORKDIR /app
COPY --chown=app:app pipeline_core pipeline_core
COPY --chown=app:app realtime_agent/worker.py realtime_agent/worker.py
