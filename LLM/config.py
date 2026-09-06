"""Environment-driven settings for the existing RAG implementation."""

import os


CONFIG = {
    "LLM_MODEL": os.getenv("LLM_MODEL", "dorna2"),
    "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    "OLLAMA_REQUEST_TIMEOUT_SECONDS": float(
        os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "180")
    ),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL_PATH", "/models/sentence_embeddings"),
    "USE_GPU": os.getenv("LLM_USE_GPU", "true").lower() in {"1", "true", "yes"},
    "EMBEDDING_BATCH_SIZE": int(os.getenv("EMBEDDING_BATCH_SIZE", "256")),
    "NORMALIZE_EMBEDDINGS": True,
    "SEARCH_TYPE": "similarity",
    "TOP_K_RESULTS": int(os.getenv("RAG_TOP_K_RESULTS", "3")),
    "TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "NUM_PREDICT": int(os.getenv("LLM_NUM_PREDICT", "512")),
    "REPEAT_PENALTY": float(os.getenv("LLM_REPEAT_PENALTY", "1.1")),
    "TOP_K": int(os.getenv("LLM_TOP_K", "10")),
    "TOP_P": float(os.getenv("LLM_TOP_P", "0.5")),
    "STOP_TOKENS": [
        "[END]",
        "User:",
        "Question:",
        "**Question",
        "** Question",
        "سوال: ",
        "سوال:",
        "سوال:**",
        "سوال:** ",
        "---",
        "\nUser",
        "\nQuestion",
    ],
    "ENABLE_MEMORY": os.getenv("ENABLE_MEMORY", "true").lower() in {"1", "true", "yes"},
}
