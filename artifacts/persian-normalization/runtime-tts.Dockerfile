FROM metarang-npc-tts:before-normalization
WORKDIR /app
COPY --chown=app:app pipeline_core pipeline_core
COPY --chown=app:app TTS/tts_server.py TTS/tts_server.py
