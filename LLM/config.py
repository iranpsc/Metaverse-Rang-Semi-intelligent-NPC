# فایل: config.py

CONFIG = {
    # --- بخش مدل زبان (LLM) ---
    'LLM_MODEL': 'dorna2',
    'OLLAMA_BASE_URL': 'http://localhost:11434',
    
    # --- بخش مدل Embedding ---
    # مسیر دقیق مدل امبدینگ روی سرور شما
    'EMBEDDING_MODEL': '/home/npc/Documents/Fine-tune-LLM/models/sentence_embeddings',
    'USE_GPU': True,               # اگر گرافیک دارید True بگذارید تا سرعت ایندکس و پاسخ‌دهی بالا برود
    'EMBEDDING_BATCH_SIZE': 256,   # تعداد جملاتی که همزمان به وکتور تبدیل می‌شوند
    'NORMALIZE_EMBEDDINGS': True,  # برای جستجوی Cosine Similarity بهتر است True باشد
    
    # --- بخش پارامترهای جستجو (RAG Retrieval) ---
    'SEARCH_TYPE': 'similarity',
    'TOP_K_RESULTS': 3,            # چند تا تکه متن مرتبط از دیتابیس پیدا شود؟ (۳ تا معمولاً تعادل خوبی است)

    # --- بخش پارامترهای تولید متن (Generation) ---
    'TEMPERATURE': 0.1,            # دمای پایین (0.1) برای پاسخ‌های دقیق و بدون توهم
    'NUM_PREDICT': 512,            # حداکثر طول پاسخ مدل (کمی بیشتر کردم تا پاسخ نصفه نماند)
    'REPEAT_PENALTY': 1.1,         # جریمه برای جلوگیری از تکرار جملات
    'TOP_K': 10,
    'TOP_P': 0.5,

    # --- توکن‌های توقف (Stop Tokens) ---
    # لیست کلماتی که اگر مدل تولید کرد، بلافاصله باید ساکت شود
    'STOP_TOKENS': [
        "[END]",
        "User:",
        "Question:",
        "**Question",
        "** Question",
        "سوال: ",
        "سوال:",
        "سوال:**",
        "سوال:** ",
        "سوال:** ",
        "---",
        "\nUser",
        "\nQuestion"
    ],
    
    # --- بخش حافظه (Memory) ---
    'ENABLE_MEMORY': True,         # آیا تاریخچه مکالمه در پرامپت لحاظ شود؟
}