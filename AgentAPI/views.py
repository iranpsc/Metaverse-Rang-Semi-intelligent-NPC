from __future__ import annotations

import logging
import socket
import uuid
from datetime import timedelta
from urllib.parse import urlparse

import aiohttp
from asgiref.sync import async_to_sync
from django.conf import settings
from django.db import connection
from django.utils import timezone
from drf_spectacular.utils import OpenApiResponse, extend_schema
from livekit import api
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from django.http import HttpResponse

from LLM.models import VoiceSample
from pipeline_core.config import ConfigurationError, safe_identifier

from .livekit import create_participant_token, delete_room, provision_room_and_agent
from .models import AgentSession
from .serializers import (
    AgentSessionCreateSerializer,
    AgentSessionResponseSerializer,
    ErrorSerializer,
    HealthSerializer,
)


logger = logging.getLogger(__name__)


class AgentSessionCreateView(APIView):
    # Intentionally unauthenticated for the first integration phase. Add an
    # authentication class here before exposing this endpoint to untrusted users.
    authentication_classes: list = []
    permission_classes = [AllowAny]

    @extend_schema(
        operation_id="create_agent_session",
        summary="Create an isolated LiveKit room and dispatch the AI agent",
        description=(
            "Development endpoint without authentication. The returned token is short-lived, "
            "room-scoped, microphone-only for publication, and never contains the API secret."
        ),
        request=AgentSessionCreateSerializer,
        responses={
            201: AgentSessionResponseSerializer,
            400: ErrorSerializer,
            503: OpenApiResponse(response=ErrorSerializer, description="LiveKit unavailable"),
        },
        tags=["Agent sessions"],
    )
    def post(self, request):
        serializer = AgentSessionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data

        session_id = uuid.uuid4()
        short_id = session_id.hex[:20]
        room_name = f"agent-session-{short_id}"
        participant_identity = f"user-{short_id}"
        user_id = safe_identifier(
            payload.get("user_id", participant_identity), fallback=participant_identity, max_length=128
        )

        voice_sample = None
        if payload.get("enable_voice_conversion"):
            try:
                voice_sample = VoiceSample.objects.get(pk=payload["voice_sample_id"])
            except VoiceSample.DoesNotExist:
                return Response(
                    {"error": "Voice sample was not found.", "code": "voice_sample_not_found"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        expires_at = timezone.now() + timedelta(seconds=settings.LIVEKIT_TOKEN_TTL_SECONDS)
        session = AgentSession.objects.create(
            id=session_id,
            room_name=room_name,
            participant_identity=participant_identity,
            user_id=user_id,
            vector_store=payload["vector_store"],
            voice_sample=voice_sample,
            expires_at=expires_at,
        )

        dispatch_metadata = {
            "session_id": str(session.id),
            "user_id": user_id,
            "participant_identity": participant_identity,
            "vector_store": session.vector_store,
            "voice_sample_filename": voice_sample.filename if voice_sample else "",
            "voice_conversion_enabled": bool(voice_sample),
        }

        try:
            token = create_participant_token(
                room_name=room_name,
                identity=participant_identity,
                session_id=str(session.id),
            )
            dispatch_id = async_to_sync(provision_room_and_agent)(
                room_name=room_name, metadata=dispatch_metadata
            )
        except (ConfigurationError, ValueError):
            session.status = AgentSession.Status.FAILED
            session.save(update_fields=("status", "updated_at"))
            logger.exception("Agent session configuration failed", extra={"session_id": str(session.id)})
            return Response(
                {"error": "Realtime service is not configured.", "code": "configuration_error"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except (api.TwirpError, aiohttp.ClientError, OSError):
            session.status = AgentSession.Status.FAILED
            session.save(update_fields=("status", "updated_at"))
            logger.exception(
                "LiveKit session provisioning failed",
                extra={"session_id": str(session.id), "room_name": room_name},
            )
            return Response(
                {"error": "Realtime service is unavailable.", "code": "livekit_unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        session.dispatch_id = dispatch_id
        session.status = AgentSession.Status.ACTIVE
        session.save(update_fields=("dispatch_id", "status", "updated_at"))
        logger.info(
            "Agent session created",
            extra={"session_id": str(session.id), "room_name": room_name, "event": "session_created"},
        )
        response = {
            "session_id": session.id,
            "room_name": room_name,
            "livekit_url": settings.LIVEKIT_URL,
            "access_token": token,
            "expires_at": expires_at,
        }
        return Response(AgentSessionResponseSerializer(response).data, status=status.HTTP_201_CREATED)


class AgentSessionDeleteView(APIView):
    authentication_classes: list = []
    permission_classes = [AllowAny]

    @extend_schema(
        operation_id="delete_agent_session",
        summary="Close an agent session and delete its LiveKit room",
        responses={204: None, 404: ErrorSerializer, 503: ErrorSerializer},
        tags=["Agent sessions"],
    )
    def delete(self, request, session_id):
        try:
            session = AgentSession.objects.get(pk=session_id)
        except AgentSession.DoesNotExist:
            return Response(
                {"error": "Session was not found.", "code": "session_not_found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            async_to_sync(delete_room)(session.room_name)
        except api.TwirpError as exc:
            if exc.code != api.TwirpErrorCode.NOT_FOUND:
                logger.exception("LiveKit room deletion failed", extra={"session_id": str(session.id)})
                return Response(
                    {"error": "Realtime service is unavailable.", "code": "livekit_unavailable"},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
        except (aiohttp.ClientError, OSError):
            logger.exception("LiveKit room deletion failed", extra={"session_id": str(session.id)})
            return Response(
                {"error": "Realtime service is unavailable.", "code": "livekit_unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        session.status = AgentSession.Status.CLOSED
        session.save(update_fields=("status", "updated_at"))
        return Response(status=status.HTTP_204_NO_CONTENT)


class LiveHealthView(APIView):
    authentication_classes: list = []
    permission_classes = [AllowAny]

    @extend_schema(responses={200: HealthSerializer}, tags=["Health"])
    def get(self, request):
        return Response({"status": "ok", "service": "django"})


class ReadyHealthView(APIView):
    authentication_classes: list = []
    permission_classes = [AllowAny]

    @extend_schema(responses={200: HealthSerializer, 503: HealthSerializer}, tags=["Health"])
    def get(self, request):
        dependencies = {"database": False, "livekit": False, "configuration": False}
        try:
            if not settings.LIVEKIT_API_KEY or not settings.LIVEKIT_API_SECRET:
                raise ConfigurationError("LiveKit credentials are missing")
            dependencies["configuration"] = True
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            dependencies["database"] = True

            parsed = urlparse(settings.LIVEKIT_API_URL)
            port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
            if not parsed.hostname:
                raise ConfigurationError("LIVEKIT_API_URL has no hostname")
            with socket.create_connection((parsed.hostname, port), timeout=0.5):
                dependencies["livekit"] = True
        except Exception:
            logger.warning("Django readiness check failed", exc_info=True)
            return Response(
                {"status": "not_ready", "service": "django", "dependencies": dependencies},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"status": "ok", "service": "django", "dependencies": dependencies})


class MetricsView(APIView):
    authentication_classes: list = []
    permission_classes = [AllowAny]

    @extend_schema(exclude=True)
    def get(self, request):
        return HttpResponse(generate_latest(), content_type=CONTENT_TYPE_LATEST)
