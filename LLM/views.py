# views.py
import json
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils.text import get_valid_filename
from django.views.decorators.csrf import csrf_exempt

from .models import VoiceSample
from .rss_ingestor import RSS_SCHEMA
from AgentAPI.voice_samples import VoiceSampleProcessingError, normalize_voice_sample
from pipeline_core.config import safe_identifier
from pipeline_core.service_client import (
    InternalServiceError,
    convert_voice_sample,
    run_llm_operation,
    synthesize_speech,
    stream_rag_chat,
)


# تنظیم مسیرهای وکتوراستور و مموری
DATA_BASE_PATH = str(settings.DATA_ROOT)
VECTOR_STORE_PATH = os.path.join(DATA_BASE_PATH, "vectorstores/main_store")
VECTOR_STORE_BASE_PATH = os.path.join(DATA_BASE_PATH, "vectorstores")

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(VECTOR_STORE_BASE_PATH, exist_ok=True)


def get_user_id(request):
    """
    Get user_id from authenticated user, session, or request data.
    Returns a string identifier for the user.
    """
    # Try to get from authenticated user first
    if request.user.is_authenticated:
        return str(request.user.id)
    
    # Try to get from session
    if 'user_id' in request.session:
        return str(request.session['user_id'])
    
    # Generate a session-based user_id if not exists
    if 'user_id' not in request.session:
        import uuid
        request.session['user_id'] = str(uuid.uuid4())
    
    return str(request.session['user_id'])


def get_user_vector_store_path(user_id: str) -> str:
    user_store_dir = os.path.join(VECTOR_STORE_BASE_PATH, f"user_{user_id}", "rss_store")
    os.makedirs(user_store_dir, exist_ok=True)
    return user_store_dir


def resolve_vector_store_path(user_id: str) -> str:
    user_store_dir = get_user_vector_store_path(user_id)
    index_file = os.path.join(user_store_dir, "index.faiss")
    if os.path.exists(index_file):
        return user_store_dir
    main_index = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if os.path.exists(main_index):
        return VECTOR_STORE_PATH
    for discovered_index in sorted(Path(VECTOR_STORE_BASE_PATH).rglob("index.faiss")):
        return str(discovered_index.parent)
    return VECTOR_STORE_PATH


def _vector_store_service_name(path: str) -> str:
    root = Path(VECTOR_STORE_BASE_PATH).resolve()
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Vector store must be inside the shared vector-store directory") from exc


def _service_error_response(exc: InternalServiceError) -> JsonResponse:
    status_code = exc.status_code if 400 <= exc.status_code < 600 else 503
    return JsonResponse(
        {"status": "error", "error": str(exc), "service": exc.service},
        status=status_code,
    )


def list_available_vector_stores():
    """
    Scan the vectorstores directory and return a list of available vector stores.
    Returns a list of dicts with 'path', 'name', and 'user_id' keys.
    """
    stores = []
    
    # Add main_store if it exists
    main_index = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if os.path.exists(main_index):
        stores.append({
            "path": VECTOR_STORE_PATH,
            "name": "Main Store (Default)",
            "user_id": "main",
            "display_name": "Main Store"
        })
    
    # Scan user stores
    if os.path.exists(VECTOR_STORE_BASE_PATH):
        for item in os.listdir(VECTOR_STORE_BASE_PATH):
            if item.startswith("user_"):
                user_id = item.replace("user_", "")
                # Check for rss_store subdirectory
                rss_store_path = os.path.join(VECTOR_STORE_BASE_PATH, item, "rss_store")
                index_file = os.path.join(rss_store_path, "index.faiss")
                if os.path.exists(index_file):
                    stores.append({
                        "path": rss_store_path,
                        "name": f"User {user_id}",
                        "user_id": user_id,
                        "display_name": f"User: {user_id[:8]}..."
                    })
                # Also check for other potential subdirectories
                for subdir in os.listdir(os.path.join(VECTOR_STORE_BASE_PATH, item)):
                    subdir_path = os.path.join(VECTOR_STORE_BASE_PATH, item, subdir)
                    if os.path.isdir(subdir_path):
                        subdir_index = os.path.join(subdir_path, "index.faiss")
                        if os.path.exists(subdir_index) and subdir != "rss_store":
                            stores.append({
                                "path": subdir_path,
                                "name": f"User {user_id} - {subdir}",
                                "user_id": user_id,
                                "display_name": f"User: {user_id[:8]}... ({subdir})"
                            })
    
    return stores


def chat_page(request):
    """
    صفحه‌ی اصلی چت که قالب HTML را رندر می‌کند.
    """
    user_id = get_user_id(request)
    context = {
        'user_id': user_id,
    }
    return render(request, "rag_chat.html", context)


@csrf_exempt
def rag_chat_api(request):
    """
    HTTP Streaming API endpoint for RAG chat using Server-Sent Events (SSE).
    Returns a StreamingHttpResponse that streams response tokens in real-time.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    question = (data.get("message") or "").strip()
    user_id = str(data.get("user_id") or get_user_id(request)).strip()
    
    # Allow explicit vector_store_path selection, otherwise use default resolution
    vector_store_path = data.get("vector_store_path", "").strip()
    if not vector_store_path:
        vector_store_path = resolve_vector_store_path(user_id)
    else:
        # Validate that the provided path exists and has an index
        if not os.path.isabs(vector_store_path):
            # If relative, assume it's relative to VECTOR_STORE_BASE_PATH
            vector_store_path = os.path.join(VECTOR_STORE_BASE_PATH, vector_store_path)
        
        # Normalize the path to handle any path issues
        vector_store_path = os.path.normpath(vector_store_path)
        
        index_file = os.path.join(vector_store_path, "index.faiss")
        if not os.path.exists(index_file):
            return JsonResponse({
                "error": f"Vector store not found at {vector_store_path}. Index file missing: {index_file}"
            }, status=404)

    if not question:
        return JsonResponse({"error": "Message is empty"}, status=400)

    try:
        vector_store_name = _vector_store_service_name(vector_store_path)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    session_id = str(request.session.session_key or user_id)
    turn_id = uuid.uuid4().hex

    def event_stream():
        """
        Generator function that streams Server-Sent Events (SSE) for HTTP streaming.
        Yields SSE-formatted messages containing response tokens in real-time.
        """
        import sys
        
        # Send initial connection confirmation
        initial_msg = f"data: {json.dumps({'type': 'start', 'message': 'Starting response generation...'}, ensure_ascii=False)}\n\n"
        yield initial_msg
        sys.stdout.flush()
        
        try:
            chunk_count = 0
            for event in stream_rag_chat(
                settings.LLM_SERVER_URL,
                {
                    "question": question,
                    "user_id": user_id,
                    "vector_store": vector_store_name,
                },
                session_id=session_id,
                turn_id=turn_id,
                timeout_seconds=settings.LLM_SERVICE_TIMEOUT_SECONDS,
            ):
                if event.get("type") == "error":
                    raise InternalServiceError(
                        "llm", str(event.get("code", "Generation failed")), status_code=502
                    )
                if event.get("type") != "token":
                    continue
                chunk = str(event.get("text", ""))
                if chunk:
                    chunk_count += 1
                    payload = json.dumps({"type": "token", "token": chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    sys.stdout.flush()
            
            # Send completion signal
            done_msg = f"data: {json.dumps({'type': 'done', 'user_id': user_id, 'chunk_count': chunk_count}, ensure_ascii=False)}\n\n"
            yield done_msg
            sys.stdout.flush()
            
        except GeneratorExit:
            # Client disconnected - this is normal for HTTP streaming
            raise
        except Exception as exc:
            # Send error message via SSE
            error_payload = json.dumps({
                "type": "error", 
                "message": str(exc)
            }, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"
            sys.stdout.flush()

    # Create HTTP streaming response with SSE content type
    response = StreamingHttpResponse(
        event_stream(), 
        content_type="text/event-stream"
    )
    
    # Set headers for proper SSE/HTTP streaming
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    response["X-Accel-Buffering"] = "no"  # Disable buffering in nginx
    
    # CORS headers
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    
    return response


@csrf_exempt
def tts_api(request):
    """
    Synthesize speech for a piece of text (typically one sentence of an LLM
    response). Proxies to the persistent CPU-only Mana Piper service.
    Returns audio/wav.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    text = (data.get("text") or "").strip()
    if not text:
        return JsonResponse({"error": "text is empty"}, status=400)

    try:
        result = synthesize_speech(
            settings.TTS_SERVER_URL,
            text,
            synthesis_settings=data.get("settings"),
            timeout_seconds=settings.TTS_SERVICE_TIMEOUT_SECONDS,
        )
    except InternalServiceError as exc:
        return JsonResponse(
            {"error": str(exc), "code": exc.code},
            status=exc.status_code if 400 <= exc.status_code < 600 else 503,
        )

    response = HttpResponse(result.audio, content_type="audio/wav")
    response["X-MetaRang-TTS-Engine"] = str(result.metrics.get("tts_engine", "mana-piper"))
    return response


@csrf_exempt
def upload_rss_feed(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    user_id = str(data.get("user_id") or get_user_id(request))
    
    # Support both single feed_url (backward compatibility) and rss_links (list)
    feed_url = data.get("feed_url")
    rss_links = data.get("rss_links")
    
    # If rss_links is provided, use it; otherwise fall back to feed_url
    if rss_links:
        if isinstance(rss_links, str):
            rss_links = [rss_links]
        elif isinstance(rss_links, list):
            rss_links = [link.strip() for link in rss_links if link and link.strip()]
        else:
            return JsonResponse({"error": "rss_links must be a string or list of strings"}, status=400)
    elif feed_url:
        rss_links = [feed_url.strip()]
    else:
        return JsonResponse({"error": "Either feed_url or rss_links is required"}, status=400)

    if not rss_links:
        return JsonResponse({"error": "No valid RSS links provided"}, status=400)

    try:
        result = run_llm_operation(
            settings.LLM_SERVER_URL,
            "rss",
            {"user_id": user_id, "rss_links": rss_links},
            timeout_seconds=settings.LLM_OPERATION_TIMEOUT_SECONDS,
        )
    except InternalServiceError as exc:
        return _service_error_response(exc)

    dataset_path = result.get("dataset_path", "")
    vector_store_path = result.get("vector_store_path", "")

    status_code = 200 if result.get("status") in ["success", "warning"] else 500
    
    # Check if vector store was created
    vector_store_exists = bool(vector_store_path) and os.path.exists(
        os.path.join(vector_store_path, "index.faiss")
    )
    
    response = {
        "message": result.get("message", "RSS processing completed"),
        "status": result.get("status"),
        "new_count": result.get("new_count", 0),
        "dataset": {
            "dataset_path": dataset_path,
            "dataset_name": f"rss_dataset_user_{user_id}",
            "num_rows": result.get("new_count", 0),
            "schema": RSS_SCHEMA,
        },
        "dataset_path": dataset_path,  # Keep for backward compatibility
        "vector_store_path": vector_store_path,
        "vector_store_exists": vector_store_exists,
        "rss_links": rss_links,
        "schema": RSS_SCHEMA,
    }
    return JsonResponse(response, status=status_code)


@csrf_exempt
def upload_single_url(request):
    """
    API endpoint to process a single article/webpage URL and add it to the user's knowledge base.
    Similar to RSS upload but for individual URLs.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    user_id = str(data.get("user_id") or get_user_id(request))
    url = data.get("url", "").strip()
    title = data.get("title", "").strip() or None  # Optional title
    
    if not url:
        return JsonResponse({"error": "URL is required"}, status=400)
    
    # Validate URL format (basic check)
    if not (url.startswith("http://") or url.startswith("https://")):
        return JsonResponse({"error": "Invalid URL format. URL must start with http:// or https://"}, status=400)

    try:
        result = run_llm_operation(
            settings.LLM_SERVER_URL,
            "url",
            {"user_id": user_id, "url": url, "title": title},
            timeout_seconds=settings.LLM_OPERATION_TIMEOUT_SECONDS,
        )
    except InternalServiceError as exc:
        return _service_error_response(exc)

    dataset_path = result.get("dataset_path", "")
    vector_store_path = result.get("vector_store_path", "")

    status_code = 200 if result.get("status") == "success" else 500
    
    # Check if vector store was created/updated
    vector_store_exists = os.path.exists(os.path.join(vector_store_path, "index.faiss"))
    
    response = {
        "message": result.get("message", "URL processing completed"),
        "status": result.get("status"),
        "new_count": result.get("new_count", 0),
        "title": result.get("title", ""),
        "url": result.get("url", url),
        "dataset_path": dataset_path,
        "vector_store_path": vector_store_path,
        "vector_store_exists": vector_store_exists,
    }
    return JsonResponse(response, status=status_code)


@csrf_exempt
def upload_files(request):
    """
    API endpoint to upload files (PDF, DOCX, TXT, CSV) and add them
    to the user's dataset and vector store.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    user_id = str(request.POST.get("user_id") or get_user_id(request))
    safe_user_id = safe_identifier(user_id, fallback="anonymous", max_length=128)

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No files uploaded"}, status=400)

    try:
        upload_root = Path(settings.DATA_ROOT) / "uploads" / f"user_{safe_user_id}"
        saved_paths = []
        for uploaded_file in files:
            safe_name = get_valid_filename(Path(uploaded_file.name).name) or "document"
            destination_dir = upload_root / uuid.uuid4().hex
            destination_dir.mkdir(parents=True, exist_ok=False)
            destination = destination_dir / safe_name
            with destination.open("wb") as target:
                for chunk in uploaded_file.chunks():
                    target.write(chunk)
            saved_paths.append(str(destination))

        result = run_llm_operation(
            settings.LLM_SERVER_URL,
            "files",
            {"user_id": safe_user_id, "file_paths": saved_paths},
            timeout_seconds=settings.LLM_OPERATION_TIMEOUT_SECONDS,
        )

        status_code = 200 if result.get("status") == "success" else 500
        dataset_path = result.get("dataset_path", "")
        vector_store_path = result.get("vector_store_path", "")
        vector_store_exists = bool(vector_store_path) and os.path.exists(
            os.path.join(vector_store_path, "index.faiss")
        )

        response = {
            "status": result.get("status"),
            "message": result.get("message", "File processing completed"),
            "new_count": result.get("new_count", 0),
            "processed_files": [Path(path).name for path in saved_paths],
            "dataset_path": dataset_path,
            "vector_store_path": vector_store_path,
            "vector_store_exists": vector_store_exists,
        }

        return JsonResponse(response, status=status_code)

    except InternalServiceError as exc:
        return _service_error_response(exc)
    except Exception as e:
        return JsonResponse(
            {"status": "error", "message": str(e)},
            status=500
        )



@csrf_exempt
def vector_store_rebuild_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    dataset_path = (data.get("dataset_path") or "").strip()
    vector_store_target = (data.get("vector_store_path") or "").strip()
    user_id = str(data.get("user_id") or "system")

    if not dataset_path:
        return JsonResponse({"error": "dataset_path is required"}, status=400)
    if not vector_store_target:
        return JsonResponse({"error": "vector_store_path is required"}, status=400)

    dataset_abs_path = (
        dataset_path
        if os.path.isabs(dataset_path)
        else os.path.join(DATA_BASE_PATH, dataset_path)
    )
    vector_store_abs_path = (
        vector_store_target
        if os.path.isabs(vector_store_target)
        else os.path.join(VECTOR_STORE_BASE_PATH, vector_store_target)
    )

    if not os.path.exists(dataset_abs_path):
        return JsonResponse({"error": f"Dataset not found at {dataset_abs_path}"}, status=404)

    try:
        result = run_llm_operation(
            settings.LLM_SERVER_URL,
            "rebuild",
            {
                "dataset_path": dataset_abs_path,
                "vector_store_path": vector_store_abs_path,
                "user_id": user_id,
            },
            timeout_seconds=settings.LLM_OPERATION_TIMEOUT_SECONDS,
        )
    except InternalServiceError as exc:
        return _service_error_response(exc)

    status_code = 200 if result.get("status") == "success" else 500
    return JsonResponse(
        {
            "dataset_path": dataset_abs_path,
            "vector_store_path": vector_store_abs_path,
            "user_id": user_id,
            "result": result,
        },
        status=status_code,
    )


@csrf_exempt
def list_vector_stores_api(request):
    """
    API endpoint to list all available vector stores.
    """
    if request.method != "GET":
        return JsonResponse({"error": "Only GET allowed"}, status=405)
    
    stores = list_available_vector_stores()
    return JsonResponse({
        "stores": stores,
        "count": len(stores)
    })


# ---------------------------------------------------------------------------
# Voice conversion (OpenVoice V2)
#
# Recorded target voices ("voice samples") are stored via the VoiceSample model
# under media/voice_samples/. The OpenVoice microservice (VC/vc_server.py) reads
# those files by basename and re-voices a source clip (typically the TTS output)
# in the chosen target voice. These views manage the sample library and proxy
# conversion requests, mirroring the TTS proxy above.
# ---------------------------------------------------------------------------

MAX_VOICE_SAMPLE_BYTES = settings.VOICE_SAMPLE_MAX_UPLOAD_BYTES


def _voice_sample_dict(sample):
    return {
        "id": sample.id,
        "name": sample.name,
        "filename": sample.filename,
        "url": sample.audio_file.url if sample.audio_file else "",
        "created_at": sample.created_at.isoformat(),
    }


def voice_samples_page(request):
    """Render the page for recording and managing target voice samples."""
    return render(request, "voice_samples.html", {"max_voice_sample_bytes": MAX_VOICE_SAMPLE_BYTES})


@csrf_exempt
def voice_samples_api(request):
    """List (GET) or create (POST, multipart) target voice samples."""
    if request.method == "GET":
        samples = [_voice_sample_dict(s) for s in VoiceSample.objects.all()]
        return JsonResponse({"samples": samples, "count": len(samples)})

    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        audio = request.FILES.get("audio")
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        if not audio:
            return JsonResponse({"error": "audio file is required"}, status=400)
        if audio.size and audio.size > MAX_VOICE_SAMPLE_BYTES:
            return JsonResponse({"error": "recording is too large"}, status=400)

        try:
            normalized_audio, _ = normalize_voice_sample(
                audio,
                min_duration_seconds=settings.VOICE_SAMPLE_MIN_DURATION_SECONDS,
                max_duration_seconds=settings.VOICE_SAMPLE_MAX_DURATION_SECONDS,
                min_voiced_seconds=settings.VOICE_SAMPLE_MIN_VOICED_SECONDS,
                timeout_seconds=settings.VOICE_SAMPLE_PROCESSING_TIMEOUT_SECONDS,
            )
        except VoiceSampleProcessingError as exc:
            return JsonResponse(
                {"error": str(exc), "code": "invalid_voice_sample"}, status=400
            )

        sample = VoiceSample(name=name[:64])
        try:
            sample.audio_file.save("sample.wav", normalized_audio, save=False)
            sample.save()
        except Exception:
            if sample.audio_file:
                sample.audio_file.delete(save=False)
            return JsonResponse(
                {"error": "Voice sample could not be saved."}, status=500
            )
        return JsonResponse({"sample": _voice_sample_dict(sample)}, status=201)

    return JsonResponse({"error": "Only GET and POST allowed"}, status=405)


@csrf_exempt
def voice_sample_delete_api(request, sample_id):
    """Delete a voice sample and its file."""
    if request.method not in ("POST", "DELETE"):
        return JsonResponse({"error": "Only POST or DELETE allowed"}, status=405)

    try:
        sample = VoiceSample.objects.get(pk=sample_id)
    except VoiceSample.DoesNotExist:
        return JsonResponse({"error": "voice sample not found"}, status=404)

    # Remove the stored file, then the row.
    sample.audio_file.delete(save=False)
    sample.delete()
    return JsonResponse({"status": "deleted", "id": sample_id})


@csrf_exempt
def vc_convert_api(request):
    """Convert a source WAV clip to a target voice via the OpenVoice V2 service.

    Body: raw WAV bytes of the source (e.g. the TTS output).
    Query: ?sample_id=<id> selecting the target voice sample.
    Returns audio/wav of the converted clip.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    sample_id = (request.GET.get("sample_id") or "").strip()
    if not sample_id:
        return JsonResponse({"error": "sample_id is required"}, status=400)

    try:
        sample = VoiceSample.objects.get(pk=sample_id)
    except (VoiceSample.DoesNotExist, ValueError):
        return JsonResponse({"error": "voice sample not found"}, status=404)

    source_bytes = request.body
    if not source_bytes:
        return JsonResponse({"error": "empty source audio"}, status=400)

    try:
        result = convert_voice_sample(
            settings.VC_SERVER_URL,
            source_bytes,
            target_filename=sample.filename,
            timeout_seconds=settings.VC_SERVICE_TIMEOUT_SECONDS,
        )
    except InternalServiceError as exc:
        if settings.VC_FAILURE_FALLBACK:
            response = HttpResponse(source_bytes, content_type="audio/wav")
            response["X-MetaRang-VC-Fallback"] = "true"
            response["X-MetaRang-VC-Error-Code"] = exc.code
            return response
        response = JsonResponse(
            {"error": str(exc), "code": exc.code, "fallback_used": False},
            status=exc.status_code if 400 <= exc.status_code < 600 else 503,
        )
        if exc.retry_after:
            response["Retry-After"] = exc.retry_after
        return response

    response = HttpResponse(result.audio, content_type="audio/wav")
    response["X-MetaRang-VC-Engine"] = str(
        result.metrics.get("vc_engine", "openvoice-v2")
    )
    response["X-MetaRang-VC-Fallback"] = "false"
    return response
