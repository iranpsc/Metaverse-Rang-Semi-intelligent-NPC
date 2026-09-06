"""Django settings for the MetaRang realtime avatar control plane."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


DEBUG = env_bool("DJANGO_DEBUG", True)
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "development-only-change-me")
if not DEBUG and SECRET_KEY == "development-only-change-me":
    raise RuntimeError("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is false")

ALLOWED_HOSTS = env_list(
    "DJANGO_ALLOWED_HOSTS",
    "npc.metarang.com,www.npc.metarang.com,npc.irpsc.com,www.npc.irpsc.com,localhost,127.0.0.1",
)
CSRF_TRUSTED_ORIGINS = env_list(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    "https://npc.metarang.com,https://www.npc.metarang.com,https://npc.irpsc.com,https://www.npc.irpsc.com",
)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "channels",
    "rest_framework",
    "drf_spectacular",
    "STT",
    "LLM",
    "AgentAPI",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

CORS_ALLOW_ALL_ORIGINS = env_bool("CORS_ALLOW_ALL_ORIGINS", DEBUG)
CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS")

ROOT_URLCONF = "MetaRangNPC.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]
WSGI_APPLICATION = "MetaRangNPC.wsgi.application"
ASGI_APPLICATION = "MetaRangNPC.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": Path(os.getenv("DATABASE_PATH", BASE_DIR / "db.sqlite3")),
        "OPTIONS": {"timeout": int(os.getenv("SQLITE_TIMEOUT_SECONDS", "20"))},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", BASE_DIR / "media"))
DATA_ROOT = Path(os.getenv("DATA_ROOT", BASE_DIR / "data"))
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://127.0.0.1:5002")
VC_SERVER_URL = os.getenv("VC_SERVER_URL", "http://127.0.0.1:5003")

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_SOCKET_TIMEOUT_SECONDS = float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "5"))
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [
                {
                    "address": REDIS_URL,
                    "socket_connect_timeout": REDIS_SOCKET_TIMEOUT_SECONDS,
                    "socket_timeout": REDIS_SOCKET_TIMEOUT_SECONDS,
                }
            ]
        },
    }
}

# The public URL is returned to Unity. The API URL is private service-to-service traffic.
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_URL = os.getenv("LIVEKIT_API_URL", "http://127.0.0.1:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
LIVEKIT_AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "metarang-agent")
LIVEKIT_TOKEN_TTL_SECONDS = int(os.getenv("LIVEKIT_TOKEN_TTL_SECONDS", "600"))
LIVEKIT_API_TIMEOUT_SECONDS = float(os.getenv("LIVEKIT_API_TIMEOUT_SECONDS", "5"))
LIVEKIT_ROOM_EMPTY_TIMEOUT_SECONDS = int(os.getenv("LIVEKIT_ROOM_EMPTY_TIMEOUT_SECONDS", "60"))
LIVEKIT_ROOM_DEPARTURE_TIMEOUT_SECONDS = int(
    os.getenv("LIVEKIT_ROOM_DEPARTURE_TIMEOUT_SECONDS", "20")
)

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
}
SPECTACULAR_SETTINGS = {
    "TITLE": "MetaRang Realtime Avatar API",
    "DESCRIPTION": (
        "Creates short-lived LiveKit sessions for the STT → RAG/LLM → TTS → optional voice "
        "conversion pipeline. Audio travels over WebRTC; this API is the control plane only."
    ),
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_HSTS_SECONDS = int(os.getenv("DJANGO_SECURE_HSTS_SECONDS", "0" if DEBUG else "31536000"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool("DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", False)
