"""ASGI entrypoint for HTTP and the two legacy browser WebSocket routes."""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MetaRangNPC.settings")

from pipeline_core.observability import configure_observability  # noqa: E402

configure_observability("django")

from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402
from django.core.asgi import get_asgi_application  # noqa: E402

from LLM.routing import websocket_urlpatterns as llm_websockets  # noqa: E402
from STT.routing import websocket_urlpatterns as stt_websockets  # noqa: E402


application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": URLRouter([*stt_websockets, *llm_websockets]),
    }
)
