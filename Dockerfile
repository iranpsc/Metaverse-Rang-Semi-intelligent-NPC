FROM python:3.11.16-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=20
WORKDIR /build
COPY requirements requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r requirements/web.txt

FROM python:3.11.16-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /app app
COPY --from=builder /wheels /wheels
COPY requirements requirements
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements/web.txt && rm -rf /wheels
WORKDIR /app
COPY --chown=app:app . .
RUN mkdir -p /app/runtime /app/staticfiles /app/data /app/media /cache/whisper \
    && chown -R app:app /app/runtime /app/staticfiles /app/data /app/media /cache \
    && chmod +x /app/entrypoint.sh
USER app
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "MetaRangNPC.asgi:application", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--access-logfile", "-"]
