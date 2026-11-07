from __future__ import annotations

import logging
import shutil
import socket
import time
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
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiResponse, extend_schema
from filelock import FileLock, Timeout as FileLockTimeout
from livekit import api
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from django.http import FileResponse, HttpResponse
from django.urls import reverse
from django.utils.text import get_valid_filename

from LLM.models import VoiceSample
from pipeline_core.config import ConfigurationError, safe_identifier
from pipeline_core.service_client import run_llm_operation
from pipeline_core.service_client import (
    InternalServiceError,
    convert_voice_reference,
    convert_voice_sample,
    synthesize_speech,
)

from .livekit import create_participant_token, delete_room, provision_room_and_agent
from .models import AgentSession
from .serializers import (
    AgentSessionCreateSerializer,
    AgentSessionResponseSerializer,
    ErrorSerializer,
    HealthSerializer,
    SpeechJobCreateSerializer,
    SpeechJobResponseSerializer,
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
            "the format consumed by OpenVoice V2. Recordings may be 1–60 seconds; 6–15 seconds "
            "of clean, single-speaker speech is recommended. A VAD check rejects empty or mostly "
            "silent input. Use the returned `id` as `voice_sample_id` and set "
            "`enable_voice_conversion` to true when creating an agent session."
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
                min_voiced_seconds=settings.VOICE_SAMPLE_MIN_VOICED_SECONDS,
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


def _speech_jobs_root() -> Path:
    return Path(settings.MEDIA_ROOT) / "speech_jobs"


def _cleanup_expired_speech_jobs() -> None:
    root = _speech_jobs_root()
    if not root.is_dir():
        return
    cutoff = timezone.now().timestamp() - settings.SPEECH_JOB_RETENTION_SECONDS
    for candidate in root.iterdir():
        try:
            uuid.UUID(candidate.name)
            if candidate.is_dir() and candidate.stat().st_mtime < cutoff:
                shutil.rmtree(candidate)
        except (OSError, ValueError):
            logger.warning(
                "Could not inspect or remove an expired speech job",
                extra={"speech_job": candidate.name},
            )


def _speech_audio_url(request, job_id: uuid.UUID, artifact: str) -> str:
    path = reverse(
        "speech-job-audio",
        kwargs={"job_id": job_id, "artifact": artifact},
    )
    return request.build_absolute_uri(path)


class SpeechJobCreateView(APIView):
    """Run Mana synthesis followed by optional-reference OpenVoice conversion."""

    authentication_classes: list = []
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        operation_id="create_speech_job",
        summary="Synthesize Persian speech and convert it to a target voice",
        description=(
            "Runs the combined Mana Piper CPU TTS → OpenVoice V2 GPU voice-conversion flow. "
            "Choose exactly one target: `sample_id` for a persistent uploaded sample, "
            "or `reference_audio` for a one-request temporary sample. Temporary reference audio "
            "is normalized but never stored. Generated audio is retained for the configured "
            "short-lived job retention period. The deprecated `voice_sample_id` alias remains "
            "accepted for compatibility. Voice-conversion fallback is disabled by default."
        ),
        request=SpeechJobCreateSerializer,
        responses={
            201: SpeechJobResponseSerializer,
            400: ErrorSerializer,
            404: ErrorSerializer,
            429: OpenApiResponse(response=ErrorSerializer, description="VC queue is full"),
            503: OpenApiResponse(
                response=ErrorSerializer,
                description="TTS or GPU voice-conversion service is unavailable",
            ),
        },
        tags=["Speech generation"],
    )
    def post(self, request):
        serializer = SpeechJobCreateSerializer(
            data=request.data,
            context={
                "max_bytes": settings.VOICE_SAMPLE_MAX_UPLOAD_BYTES,
                "max_text_length": settings.SPEECH_TEXT_MAX_LENGTH,
            },
        )
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        job_started = time.perf_counter()
        _cleanup_expired_speech_jobs()

        sample = None
        reference_bytes = None
        if "voice_sample_id" in payload:
            try:
                sample = VoiceSample.objects.get(pk=payload["voice_sample_id"])
            except VoiceSample.DoesNotExist:
                return Response(
                    {"error": "Voice sample was not found.", "code": "voice_sample_not_found"},
                    status=status.HTTP_404_NOT_FOUND,
                )
            if not sample.audio_file or not sample.audio_file.storage.exists(
                sample.audio_file.name
            ):
                return Response(
                    {
                        "error": "Voice sample audio file was not found.",
                        "code": "voice_sample_file_missing",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            try:
                normalized_reference, _ = normalize_voice_sample(
                    payload["reference_audio"],
                    min_duration_seconds=settings.VOICE_SAMPLE_MIN_DURATION_SECONDS,
                    max_duration_seconds=settings.VOICE_SAMPLE_MAX_DURATION_SECONDS,
                    min_voiced_seconds=settings.VOICE_SAMPLE_MIN_VOICED_SECONDS,
                    timeout_seconds=settings.VOICE_SAMPLE_PROCESSING_TIMEOUT_SECONDS,
                )
                reference_bytes = normalized_reference.read()
            except VoiceSampleProcessingError as exc:
                return Response(
                    {"error": str(exc), "code": "invalid_voice_sample"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        synthesis_settings = {
            key: payload[key]
            for key in ("length_scale", "noise_scale", "noise_w")
            if key in payload
        }
        try:
            synthesized = synthesize_speech(
                settings.TTS_SERVER_URL,
                payload["text"],
                synthesis_settings=synthesis_settings,
                timeout_seconds=settings.TTS_SERVICE_TIMEOUT_SECONDS,
            )
        except InternalServiceError as exc:
            return Response(
                {"error": str(exc), "code": exc.code, "stage": "tts"},
                status=exc.status_code if 400 <= exc.status_code < 600 else 503,
            )

        try:
            if sample is not None:
                converted = convert_voice_sample(
                    settings.VC_SERVER_URL,
                    synthesized.audio,
                    target_filename=sample.filename,
                    timeout_seconds=settings.VC_SERVICE_TIMEOUT_SECONDS,
                )
            else:
                converted = convert_voice_reference(
                    settings.VC_SERVER_URL,
                    synthesized.audio,
                    reference_bytes,
                    tau=payload["tau"],
                    timeout_seconds=settings.VC_SERVICE_TIMEOUT_SECONDS,
                )
        except InternalServiceError as exc:
            if not settings.VC_FAILURE_FALLBACK:
                response = Response(
                    {
                        "error": str(exc),
                        "code": exc.code,
                        "stage": "voice_conversion",
                        "fallback_used": False,
                    },
                    status=exc.status_code if 400 <= exc.status_code < 600 else 503,
                )
                if exc.retry_after:
                    response["Retry-After"] = exc.retry_after
                return response
            converted = None
            logger.warning(
                "Combined speech API used original TTS audio after VC failure",
                extra={"event": "vc_fallback", "error_code": exc.code},
            )

        job_id = uuid.uuid4()
        job_directory = _speech_jobs_root() / str(job_id)
        job_directory.mkdir(parents=True, exist_ok=False)
        (job_directory / "source.wav").write_bytes(synthesized.audio)
        if converted is not None:
            (job_directory / "converted.wav").write_bytes(converted.audio)

        fallback_used = converted is None
        total_time = time.perf_counter() - job_started
        vc_metrics = {} if fallback_used else converted.metrics
        audio_duration = float(
            vc_metrics.get("audio_duration", synthesized.metrics.get("audio_duration", 0.0))
            or 0.0
        )
        metrics = {
            "normalization_time": synthesized.metrics.get("normalization_time"),
            "tts_time": synthesized.metrics.get("tts_time"),
            "tts_rtf": synthesized.metrics.get("tts_rtf"),
            "cpu_time": synthesized.metrics.get("cpu_time"),
            "cpu_percent": synthesized.metrics.get("cpu_percent"),
            "ram_peak_mib": synthesized.metrics.get("ram_peak_mib"),
            "source_embedding_time": vc_metrics.get("source_embedding_time"),
            "target_embedding_time": vc_metrics.get("target_embedding_time"),
            "conversion_time": vc_metrics.get("conversion_time"),
            "conversion_rtf": vc_metrics.get("conversion_rtf"),
            "queue_wait_time": vc_metrics.get("queue_wait"),
            "target_cache_hit": vc_metrics.get("target_cache_hit"),
            "audio_duration": audio_duration,
            "total_time": total_time,
            "end_to_end_rtf": total_time / audio_duration if audio_duration else None,
        }
        source_url = _speech_audio_url(request, job_id, "source")
        response_payload = {
            "job_id": job_id,
            "status": "completed_with_fallback" if fallback_used else "completed",
            "conversion_status": "fallback" if fallback_used else "completed",
            "fallback_used": fallback_used,
            "source_audio_url": source_url,
            "converted_audio_url": (
                None if fallback_used else _speech_audio_url(request, job_id, "converted")
            ),
            "metrics": metrics,
            "expires_at": timezone.now()
            + timedelta(seconds=settings.SPEECH_JOB_RETENTION_SECONDS),
        }
        logger.info(
            "Combined speech job completed",
            extra={
                "event": "speech_job_complete",
                "speech_job_id": str(job_id),
                "fallback_used": fallback_used,
                "total_time_seconds": round(total_time, 6),
                "tts_time_seconds": synthesized.metrics.get("tts_time"),
                "conversion_time_seconds": vc_metrics.get("conversion_time"),
                "queue_wait_seconds": vc_metrics.get("queue_wait"),
            },
        )
        return Response(
            SpeechJobResponseSerializer(response_payload).data,
            status=status.HTTP_201_CREATED,
        )


class SpeechJobAudioView(APIView):
    authentication_classes: list = []
    permission_classes = [AllowAny]

    @extend_schema(
        operation_id="download_speech_job_audio",
        summary="Download source or converted WAV from a speech job",
        responses={
            (200, "audio/wav"): OpenApiTypes.BINARY,
            404: ErrorSerializer,
        },
        tags=["Speech generation"],
    )
    def get(self, request, job_id, artifact):
        if artifact not in {"source", "converted"}:
            return Response(
                {"error": "Audio artifact was not found.", "code": "audio_not_found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        audio_path = _speech_jobs_root() / str(job_id) / f"{artifact}.wav"
        if not audio_path.is_file():
            return Response(
                {"error": "Audio artifact was not found or has expired.", "code": "audio_not_found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        return FileResponse(audio_path.open("rb"), content_type="audio/wav")


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

                    result = run_llm_operation(
                        settings.LLM_SERVER_URL,
                        "files",
                        {"user_id": user_id, "file_paths": saved_paths},
                        timeout_seconds=settings.LLM_OPERATION_TIMEOUT_SECONDS,
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
                    result = run_llm_operation(
                        settings.LLM_SERVER_URL,
                        "rss",
                        {"user_id": user_id, "rss_links": rss_links},
                        timeout_seconds=settings.LLM_OPERATION_TIMEOUT_SECONDS,
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
