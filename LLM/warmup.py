import threading
import time
from pathlib import Path

from .config import CONFIG

_keepalive_started = False
_keepalive_lock = threading.Lock()


def _default_vector_store_path() -> str:
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    return str(data_dir / "vectorstores" / "main_store")


def _warmup():
    try:
        from .rag_service import RAGService
        from .rag_system import VectorStoreManager
    except Exception as exc:  # pragma: no cover
        print(f"[RAG Warmup] Failed to import services: {exc}")
        return

    rag_service = RAGService()

    # Warm vector store cache
    default_store = _default_vector_store_path()
    index_file = Path(default_store) / "index.faiss"
    if index_file.exists():
        try:
            vstore_manager = VectorStoreManager(rag_service.embeddings, user_id="system", verbose=False)
            vstore_manager.load(default_store)
            print(f"[RAG Warmup] Cached vector store at {default_store}")
        except Exception as exc:
            print(f"[RAG Warmup] Failed to warm vector store: {exc}")
    else:
        print(f"[RAG Warmup] Skipping vector store warmup; no index at {default_store}")

    # Warm LLM weights
    try:
        start_time = time.time()
        # Create LLM instance if not already created
        if rag_service.llm is None:
            rag_service.llm = rag_service._create_llm()
        rag_service.llm.invoke("سلام! این یک درخواست تست برای راه‌اندازی مدل است.")
        duration = time.time() - start_time
        print(f"[RAG Warmup] LLM responded to warmup prompt in {duration:.2f} seconds.")
    except Exception as exc:
        print(f"[RAG Warmup] LLM warmup failed: {exc}")

    _start_keepalive_thread()


def _keepalive_loop(interval: int):
    from .rag_service import RAGService

    rag_service = RAGService()
    while True:
        try:
            # Create LLM instance if not already created
            if rag_service.llm is None:
                rag_service.llm = rag_service._create_llm()
            rag_service.llm.invoke("ping")
            print("[RAG Warmup] Sent keep-alive ping to LLM.")
        except Exception as exc:
            print(f"[RAG Warmup] Keep-alive ping failed: {exc}")
        time.sleep(interval)


def _start_keepalive_thread():
    interval = CONFIG.get("KEEP_ALIVE_INTERVAL_SECONDS", 0)
    if interval <= 0:
        return

    global _keepalive_started
    with _keepalive_lock:
        if _keepalive_started:
            return
        thread = threading.Thread(
            target=_keepalive_loop, args=(interval,), name="rag-keepalive", daemon=True
        )
        thread.start()
        _keepalive_started = True


def schedule_rag_warmup():
    thread = threading.Thread(target=_warmup, name="rag-warmup", daemon=True)
    thread.start()

