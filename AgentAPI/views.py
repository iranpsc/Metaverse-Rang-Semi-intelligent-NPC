from __future__ import annotations

import logging
import socket
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import requests
from asgiref.sync import async_to_sync
from django.conf import settings
from django.db import connection
from django.utils import timezone
from drf_spectacular.utils import OpenApiResponse, extend_schema
from filelock import FileLock, Timeout as FileLockTimeout
from livekit import api
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from django.http import HttpResponse
from django.utils.text import get_valid_filename

from LLM.models import VoiceSample
from LLM.rag_service import RAGService
from pipeline_core.config import ConfigurationError, safe_identifier

from .livekit import create_participant_token, delete_room, provision_room_and_agent
from .models import AgentSession
from .serializers import (
    AgentSessionCreateSerializer,
    AgentSessionResponseSerializer,
    ErrorSerializer,
    HealthSerializer,
    VoiceSampleCreateSerializer,
    VoiceSampleDeleteSerializer,
    VoiceSampleListSerializer,
    VoiceSampleResponseSerializer,
    VectorStoreCreateSerializer,
    VectorStoreRSSCreateSerializer,
    VectorStoreResponseSerializer,
    VectorStoreStatusSerializer,
)
from .voice_samples import VoiceSampleProcessingError, normalize_voice_sample


logger = logging.getLogger(__name__)


def _voice_sample_payload(sample: VoiceSample) -> dict:
    return {
        "id": sample.pk,
        "name": sample.name,
        "filename": sample.filename,
        "audio_url": sample.audio_file.url if sample.audio_file else "",
        "created_at": sample.created_at,
    }


class VoiceSampleListCreateView(APIView):
    authentication_classes: list = []
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        operation_id="list_voice_samples",
        summary="List target voice samples",
        responses={200: VoiceSampleListSerializer},
        tags=["Voice samples"],
    )
    def get(self, request):
        samples = [
            VoiceSampleResponseSerializer(_voice_sample_payload(sample)).data
            for sample in VoiceSample.objects.all()
        ]
        return Response({"samples": samples, "count": len(samples)})

    @extend_schema(
        operation_id="upload_voice_sample",
        summary="Upload a target voice sample",
        description=(
            "Uploads a target-speaker recording and normalizes it to mono 16 kHz PCM WAV, "
            "the format consumed by FreeVC. Use the returned `id` as `voice_sample_id` and "
            "set `enable_voice_conversion` to true when creating an agent session."
        ),
        request=VoiceSampleCreateSerializer,
        responses={
            201: VoiceSampleResponseSerializer,
            400: ErrorSerializer,
            500: ErrorSerializer,
        },
        tags=["Voice samples"],
    )
    def post(self, request):
        serializer = VoiceSampleCreateSerializer(
            data=request.data,
            context={"max_bytes": settings.VOICE_SAMPLE_MAX_UPLOAD_BYTES},
        )
        serializer.is_valid(raise_exception=True)

        try:
            normalized_audio, _ = normalize_voice_sample(
                serializer.validated_data["audio"],
                min_duration_seconds=settings.VOICE_SAMPLE_MIN_DURATION_SECONDS,
                max_duration_seconds=settings.VOICE_SAMPLE_MAX_DURATION_SECONDS,
                timeout_seconds=settings.VOICE_SAMPLE_PROCESSING_TIMEOUT_SECONDS,
            )
        except VoiceSampleProcessingError as exc:
            return Response(
                {"error": str(exc), "code": "invalid_voice_sample"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        sample = VoiceSample(name=serializer.validated_data["name"])
        try:
            sample.audio_file.save("sample.wav", normalized_audio, save=False)
            sample.save()
        except Exception:
            if sample.audio_file:
                sample.audio_file.delete(save=False)
            logger.exception("Failed to persist voice sample")
            return Response(
                {"error": "Voice sample could not be saved.", "code": "voice_sample_save_failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        payload = VoiceSampleResponseSerializer(_voice_sample_payload(sample)).data
        return Response(payload, status=status.HTTP_201_CREATED)


class VoiceSampleDeleteView(APIView):
    authentication_classes: list = []
    permission_classes = [AllowAny]

    @extend_schema(
        operation_id="delete_voice_sample",
        summary="Delete a target voice sample",
        responses={
            200: VoiceSampleDeleteSerializer,
            404: ErrorSerializer,
        },
        tags=["Voice samples"],
    )
    def delete(self, request, sample_id):
        try:
            sample = VoiceSample.objects.get(pk=sample_id)
        except VoiceSample.DoesNotExist:
            return Response(
                {"error": "Voice sample was not found.", "code": "voice_sample_not_found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        sample.audio_file.delete(save=False)
        sample.delete()
        return Response({"id": sample_id, "status": "deleted"})


def _user_vector_store_paths(user_id: str) -> tuple[Path, Path, Path]:
    data_root = Path(settings.DATA_ROOT)
    vector_store_path = data_root / "vectorstores" / f"user_{user_id}" / "rss_store"
    dataset_path = data_root / "rss_datasets" / f"user_{user_id}" / "rss_dataset.csv"
    upload_path = data_root / "uploads" / f"user_{user_id}"
    return vector_store_path, dataset_path, upload_path


class VectorStoreCreateView(APIView):
    """Build or extend a user-scoped FAISS vector store from uploaded files."""

    authentication_classes: list = []
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        operation_id="create_or_update_user_vector_store",
        summary="Create or update a vector store for a user",
        description=(
            "Uploads knowledge files and builds a persistent user-scoped FAISS index. "
            "Send the returned `vector_store` value when creating an agent session. "
            "Uploading more files for the same `user_id` extends the existing store."
        ),
        request=VectorStoreCreateSerializer,
        responses={
            200: VectorStoreResponseSerializer,
            201: VectorStoreResponseSerializer,
            400: ErrorSerializer,
            409: OpenApiResponse(response=ErrorSerializer, description="Store update in progress"),
            500: ErrorSerializer,
        },
        tags=["Vector stores"],
    )
    def post(self, request):
        serializer = VectorStoreCreateSerializer(
            data=request.data,
            context={
                "max_files": settings.VECTOR_STORE_MAX_FILES,
                "max_file_bytes": settings.VECTOR_STORE_MAX_UPLOAD_BYTES,
            },
        )
        serializer.is_valid(raise_exception=True)
        user_id = serializer.validated_data["user_id"]
        uploaded_files = serializer.validated_data["files"]
        vector_store_path, dataset_path, upload_path = _user_vector_store_paths(user_id)
        existed = (vector_store_path / "index.faiss").is_file()

        upload_path.mkdir(parents=True, exist_ok=True)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store_path.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        try:
            lock = FileLock(str(vector_store_path.parent / ".build.lock"), timeout=1)
            try:
                with lock:
                    for uploaded_file in uploaded_files:
                        safe_name = get_valid_filename(Path(uploaded_file.name).name) or "document"
                        destination_directory = upload_path / uuid.uuid4().hex
                        destination_directory.mkdir()
                        destination = destination_directory / safe_name
                        with destination.open("wb") as target:
                            for chunk in uploaded_file.chunks():
                                target.write(chunk)
                        saved_paths.append(str(destination))

                    result = RAGService().process_user_files(
                        user_id=user_id,
                        file_paths=saved_paths,
                        dataset_path=str(dataset_path),
                        vector_store_path=str(vector_store_path),
                    )
            except FileLockTimeout:
                return Response(
                    {
                        "error": "A vector-store update is already in progress for this user.",
                        "code": "vector_store_busy",
                    },
                    status=status.HTTP_409_CONFLICT,
                )

            ready = (vector_store_path / "index.faiss").is_file()
            if result.get("status") != "success" or not ready:
                logger.warning(
                    "Vector-store build did not produce an index",
                    extra={"user_id": user_id, "result_status": result.get("status")},
                )
                return Response(
                    {
                        "error": result.get("message", "No valid content could be extracted."),
                        "code": "vector_store_build_failed",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            response_payload = {
                "user_id": user_id,
                "vector_store": f"user_{user_id}/rss_store",
                "ready": True,
                "created": not existed,
                "files_received": len(uploaded_files),
                "documents_added": result.get("new_count", 0),
                "message": result.get("message", "Vector store updated."),
            }
            response_status = status.HTTP_201_CREATED if not existed else status.HTTP_200_OK
            return Response(
                VectorStoreResponseSerializer(response_payload).data,
                status=response_status,
            )
        except Exception:
            logger.exception("Vector-store build failed", extra={"user_id": user_id})
            return Response(
                {"error": "Vector store could not be built.", "code": "vector_store_build_failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class VectorStoreRSSCreateView(APIView):
    """Build or refresh a user-scoped FAISS vector store from RSS/Atom feeds."""

    authentication_classes: list = []
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    @extend_schema(
        operation_id="create_or_update_user_vector_store_from_rss",
        summary="Create or update a user's vector store from RSS feeds",
        description=(
            "Crawls RSS or Atom feeds, extracts new article content, and adds it to the "
            "same persistent user-scoped FAISS index used by file uploads. Feed state is "
            "remembered, so later calls add only newly published articles. Send the returned "
            "`vector_store` value when creating an agent session."
        ),
        request=VectorStoreRSSCreateSerializer,
        responses={
            200: VectorStoreResponseSerializer,
            201: VectorStoreResponseSerializer,
            400: ErrorSerializer,
            409: OpenApiResponse(response=ErrorSerializer, description="Store update in progress"),
            500: ErrorSerializer,
        },
        tags=["Vector stores"],
    )
    def post(self, request):
        serializer = VectorStoreRSSCreateSerializer(
            data=request.data,
            context={"max_links": settings.VECTOR_STORE_MAX_RSS_LINKS},
        )
        serializer.is_valid(raise_exception=True)
        user_id = serializer.validated_data["user_id"]
        rss_links = serializer.validated_data["rss_links"]
        vector_store_path, dataset_path, _ = _user_vector_store_paths(user_id)
        existed = (vector_store_path / "index.faiss").is_file()

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store_path.mkdir(parents=True, exist_ok=True)

        try:
            lock = FileLock(str(vector_store_path.parent / ".build.lock"), timeout=1)
            try:
                with lock:
                    result = RAGService().process_user_rss(
                        user_id=user_id,
                        rss_links=rss_links,
                        dataset_path=str(dataset_path),
                        vector_store_path=str(vector_store_path),
                    )
            except FileLockTimeout:
                return Response(
                    {
                        "error": "A vector-store update is already in progress for this user.",
                        "code": "vector_store_busy",
                    },
                    status=status.HTTP_409_CONFLICT,
                )

            ready = (vector_store_path / "index.faiss").is_file()
            if result.get("status") != "success" or not ready:
                logger.warning(
                    "RSS crawl did not produce a vector index",
                    extra={"user_id": user_id, "result_status": result.get("status")},
                )
                return Response(
                    {
                        "error": result.get(
                            "message", "No valid RSS article content could be extracted."
                        ),
                        "code": "vector_store_build_failed",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            response_payload = {
                "user_id": user_id,
                "vector_store": f"user_{user_id}/rss_store",
                "ready": True,
                "created": not existed,
                "rss_links_received": len(rss_links),
                "documents_added": result.get("new_count", 0),
                "message": result.get("message", "Vector store updated from RSS."),
            }
            response_status = status.HTTP_201_CREATED if not existed else status.HTTP_200_OK
            return Response(
                VectorStoreResponseSerializer(response_payload).data,
                status=response_status,
            )
        except Exception:
            logger.exception("RSS vector-store update failed", extra={"user_id": user_id})
            return Response(
                {"error": "Vector store could not be updated.", "code": "vector_store_build_failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class VectorStoreStatusView(APIView):
    authentication_classes: list = []
    permission_classes = [AllowAny]

    @extend_schema(
        operation_id="get_user_vector_store",
        summary="Get vector-store status for a user",
        responses={200: VectorStoreStatusSerializer},
        tags=["Vector stores"],
    )
    def get(self, request, user_id):
        vector_store_path, dataset_path, _ = _user_vector_store_paths(user_id)
        payload = {
            "user_id": user_id,
            "vector_store": f"user_{user_id}/rss_store",
            "ready": (vector_store_path / "index.faiss").is_file(),
            "dataset_exists": dataset_path.is_file(),
        }
        return Response(VectorStoreStatusSerializer(payload).data)


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
            503: OpenApiResponse(
                response=ErrorSerializer,
                description="LiveKit or voice-conversion service unavailable",
            ),
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
            if not voice_sample.audio_file or not voice_sample.audio_file.storage.exists(
                voice_sample.audio_file.name
            ):
                return Response(
                    {
                        "error": "Voice sample audio file was not found.",
                        "code": "voice_sample_file_missing",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            try:
                validation_response = requests.get(
                    f"{settings.VC_SERVER_URL}/samples/validate",
                    params={"target": voice_sample.filename},
                    timeout=settings.VC_SAMPLE_VALIDATION_TIMEOUT_SECONDS,
                )
            except requests.RequestException:
                logger.exception(
                    "Voice-conversion sample validation failed",
                    extra={"voice_sample_id": voice_sample.pk},
                )
                return Response(
                    {
                        "error": "Voice conversion service is unavailable.",
                        "code": "voice_conversion_unavailable",
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            if validation_response.status_code == 404:
                return Response(
                    {
                        "error": "Voice sample is not available to the conversion service.",
                        "code": "voice_sample_unavailable",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if validation_response.status_code != 200:
                return Response(
                    {
                        "error": "Voice conversion service is unavailable.",
                        "code": "voice_conversion_unavailable",
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
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
