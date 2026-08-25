# views.py
import json
import os
from dataclasses import asdict
from pathlib import Path

import requests
from django.conf import settings
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import VoiceSample
from .rag_service import RAGService
from .rss_ingestor import RSSIngestor, RSS_SCHEMA


# تنظیم مسیرهای وکتوراستور و مموری
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_BASE_PATH = os.path.join(BASE_DIR, "data")
VECTOR_STORE_PATH = os.path.join(DATA_BASE_PATH, "vectorstores/main_store")
VECTOR_STORE_BASE_PATH = os.path.join(DATA_BASE_PATH, "vectorstores")
RSS_DATASET_BASE_PATH = os.path.join(DATA_BASE_PATH, "rss_datasets")
MEMORY_BASE_PATH = os.path.join(DATA_BASE_PATH, "memories")

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(VECTOR_STORE_BASE_PATH, exist_ok=True)
os.makedirs(RSS_DATASET_BASE_PATH, exist_ok=True)
os.makedirs(MEMORY_BASE_PATH, exist_ok=True)


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


def get_user_memory_path(user_id):
    """
    Get user-specific memory path.
    """
    user_memory_dir = os.path.join(MEMORY_BASE_PATH, f"user_{user_id}")
    os.makedirs(user_memory_dir, exist_ok=True)
    return user_memory_dir


def get_user_vector_store_path(user_id: str) -> str:
    user_store_dir = os.path.join(VECTOR_STORE_BASE_PATH, f"user_{user_id}", "rss_store")
    base_real = os.path.realpath(VECTOR_STORE_BASE_PATH)
    user_store_real = os.path.realpath(user_store_dir)

    if os.path.commonpath([base_real, user_store_real]) != base_real:
        raise ValueError("Invalid user_id path")

    os.makedirs(user_store_real, exist_ok=True)
    return user_store_real


def resolve_vector_store_path(user_id: str) -> str:
    user_store_dir = get_user_vector_store_path(user_id)
    index_file = os.path.join(user_store_dir, "index.faiss")
    if os.path.exists(index_file):
        return user_store_dir
    return VECTOR_STORE_PATH


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

    memory_path = get_user_memory_path(user_id)
    rag_service = RAGService()

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
            # Get answer generator from RAG service
            answer_generator = rag_service.get_answer_for_user(
                user_id=user_id,
                question=question,
                vector_store_path=vector_store_path,
                memory_path=memory_path
            )
            
            # Stream chunks directly from the generator in real-time
            chunk_count = 0
            for chunk in answer_generator:
                # Skip empty chunks
                if not chunk or not chunk.strip():
                    continue
                    
                chunk_count += 1
                
                # Format as SSE message
                payload = json.dumps({
                    "type": "token", 
                    "token": chunk
                }, ensure_ascii=False)
                sse_message = f"data: {payload}\n\n"
                yield sse_message
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
    response). Proxies to the TTS microservice (TTS/tts_server.py), which runs
    in its own venv because Coqui TTS conflicts with this environment's deps.
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
        tts_response = requests.post(
            f"{settings.TTS_SERVER_URL}/synthesize",
            json={"text": text},
            timeout=(3, 120),
        )
    except requests.exceptions.RequestException:
        return JsonResponse(
            {"error": "TTS service unavailable. Start it with TTS/start_tts_server.sh"},
            status=503,
        )

    if tts_response.status_code != 200:
        try:
            detail = tts_response.json().get("error", "synthesis failed")
        except ValueError:
            detail = "synthesis failed"
        return JsonResponse({"error": detail}, status=502)

    return HttpResponse(tts_response.content, content_type="audio/wav")


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

    # Build dataset path for user (single CSV file per user)
    user_dataset_dir = os.path.join(RSS_DATASET_BASE_PATH, f"user_{user_id}")
    os.makedirs(user_dataset_dir, exist_ok=True)
    dataset_path = os.path.join(user_dataset_dir, "rss_dataset.csv")
    
    # Get vector store path for user
    vector_store_path = get_user_vector_store_path(user_id)
    
    # Use RAGService.process_user_rss to fetch and process RSS feeds
    rag_service = RAGService()
    result = rag_service.process_user_rss(
        user_id=user_id,
        rss_links=rss_links,
        dataset_path=dataset_path,
        vector_store_path=vector_store_path
    )

    status_code = 200 if result.get("status") in ["success", "warning"] else 500
    
    # Check if vector store was created
    vector_store_exists = os.path.exists(os.path.join(vector_store_path, "index.faiss"))
    
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

    # Build dataset path for user (same CSV file as RSS)
    user_dataset_dir = os.path.join(RSS_DATASET_BASE_PATH, f"user_{user_id}")
    os.makedirs(user_dataset_dir, exist_ok=True)
    dataset_path = os.path.join(user_dataset_dir, "rss_dataset.csv")
    
    # Get vector store path for user
    vector_store_path = get_user_vector_store_path(user_id)
    
    # Use RAGService.process_single_url to fetch and process the URL
    rag_service = RAGService()
    result = rag_service.process_single_url(
        user_id=user_id,
        url=url,
        dataset_path=dataset_path,
        vector_store_path=vector_store_path,
        title=title
    )

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

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No files uploaded"}, status=400)

    # Build dataset path (same dataset as RSS & URL)
    user_dataset_dir = os.path.join(RSS_DATASET_BASE_PATH, f"user_{user_id}")
    os.makedirs(user_dataset_dir, exist_ok=True)
    dataset_path = os.path.join(user_dataset_dir, "rss_dataset.csv")

    # Vector store path
    vector_store_path = get_user_vector_store_path(user_id)

    rag_service = RAGService()

    try:
        result = rag_service.process_uploaded_files(
            user_id=user_id,
            files=files,
            dataset_path=dataset_path,
            vector_store_path=vector_store_path
        )

        status_code = 200 if result.get("status") == "success" else 500

        vector_store_exists = os.path.exists(
            os.path.join(vector_store_path, "index.faiss")
        )

        response = {
            "status": result.get("status"),
            "message": result.get("message", "File processing completed"),
            "new_count": result.get("new_count", 0),
            "processed_files": result.get("processed_files", []),
            "dataset_path": dataset_path,
            "vector_store_path": vector_store_path,
            "vector_store_exists": vector_store_exists,
        }

        return JsonResponse(response, status=status_code)

    except ValueError as e:
        return JsonResponse(
            {"status": "error", "message": str(e)},
            status=400
        )
    except Exception as e:
        traceback.print_exc()
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

    dataset_abs_path = dataset_path if os.path.isabs(dataset_path) else os.path.join(BASE_DIR, dataset_path)
    vector_store_abs_path = (
        vector_store_target
        if os.path.isabs(vector_store_target)
        else os.path.join(VECTOR_STORE_BASE_PATH, vector_store_target)
    )

    if not os.path.exists(dataset_abs_path):
        return JsonResponse({"error": f"Dataset not found at {dataset_abs_path}"}, status=404)

    os.makedirs(vector_store_abs_path, exist_ok=True)
    rag_service = RAGService()
    result = rag_service.build_vector_store(
        dataset_path=dataset_abs_path,
        vector_store_path=vector_store_abs_path,
        user_id=user_id,
    )

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
# Voice conversion (FreeVC)
#
# Recorded target voices ("voice samples") are stored via the VoiceSample model
# under media/voice_samples/. The FreeVC microservice (VC/vc_server.py) reads
# those files by basename and re-voices a source clip (typically the TTS output)
# in the chosen target voice. These views manage the sample library and proxy
# conversion requests, mirroring the TTS proxy above.
# ---------------------------------------------------------------------------

MAX_VOICE_SAMPLE_BYTES = 25 * 1024 * 1024  # 25 MB guard for uploaded recordings


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
    return render(request, "voice_samples.html", {})


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

        sample = VoiceSample.objects.create(name=name[:64], audio_file=audio)
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
    """Convert a source WAV clip to a target voice via the FreeVC microservice.

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
        vc_response = requests.post(
            f"{settings.VC_SERVER_URL}/convert",
            params={"target": sample.filename},
            data=source_bytes,
            headers={"Content-Type": "audio/wav"},
            timeout=(3, 120),
        )
    except requests.exceptions.RequestException:
        return JsonResponse(
            {"error": "Voice conversion service unavailable. Start it with VC/start_vc_server.sh"},
            status=503,
        )

    if vc_response.status_code != 200:
        try:
            detail = vc_response.json().get("error", "conversion failed")
        except ValueError:
            detail = "conversion failed"
        return JsonResponse({"error": detail}, status=502)

    return HttpResponse(vc_response.content, content_type="audio/wav")
