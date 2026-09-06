from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Mapping


_configured_services: set[str] = set()


class JsonFormatter(logging.Formatter):
    _standard = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "service": getattr(record, "service", self.service),
            "logger": record.name,
            "message": record.getMessage(),
        }
        try:
            from opentelemetry import trace

            span_context = trace.get_current_span().get_span_context()
            if span_context.is_valid:
                payload["trace_id"] = format(span_context.trace_id, "032x")
                payload["span_id"] = format(span_context.span_id, "016x")
        except Exception:
            pass
        for key, value in record.__dict__.items():
            if key not in self._standard and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_observability(service: str) -> None:
    if service in _configured_services:
        return
    _configured_services.add(service)

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root.handlers.clear()
    root.addHandler(handler)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(tracer_provider)

        if os.getenv("OTEL_EXPORT_LOGS", "false").lower() in {"1", "true", "yes"}:
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
            from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=True))
            )
            root.addHandler(LoggingHandler(logger_provider=logger_provider))
    except Exception:
        logging.getLogger(__name__).exception(
            "OpenTelemetry exporter initialization failed; stdout logging remains active"
        )


def get_tracer(name: str):
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return _NoopTracer()


def inject_trace_headers(headers: dict[str, str]) -> dict[str, str]:
    try:
        from opentelemetry.propagate import inject

        inject(headers)
    except Exception:
        pass
    return headers


def extract_trace_context(headers: Mapping[str, str]):
    try:
        from opentelemetry.propagate import extract

        return extract(dict(headers))
    except Exception:
        return None


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def set_attribute(self, *_args, **_kwargs):
        return None

    def add_event(self, *_args, **_kwargs):
        return None

    def record_exception(self, *_args, **_kwargs):
        return None


class _NoopTracer:
    def start_as_current_span(self, *_args, **_kwargs):
        return _NoopSpan()
