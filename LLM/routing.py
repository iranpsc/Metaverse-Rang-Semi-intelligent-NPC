from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Try both with and without leading slash (Channels may handle it differently)
    re_path(r'^/?ws/rag-chat/?$', consumers.RAGChatConsumer.as_asgi()),
]

