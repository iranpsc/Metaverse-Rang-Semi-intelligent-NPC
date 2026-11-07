from django.urls import path

from .views import (
    AgentSessionCreateView,
    AgentSessionDeleteView,
    LiveHealthView,
    MetricsView,
    ReadyHealthView,
    VectorStoreCreateView,
    VectorStoreRSSCreateView,
    VectorStoreStatusView,
    VoiceSampleListCreateView,
    VoiceSampleDeleteView,
)


urlpatterns = [
    path("agent/sessions/", AgentSessionCreateView.as_view(), name="agent-session-create"),
    path(
        "agent/sessions/<uuid:session_id>/",
        AgentSessionDeleteView.as_view(),
        name="agent-session-delete",
    ),
    path("voice-samples/", VoiceSampleListCreateView.as_view(), name="voice-sample-list-create"),
    path(
        "voice-samples/<int:sample_id>/",
        VoiceSampleDeleteView.as_view(),
        name="voice-sample-delete",
    ),
    path("vector-stores/", VectorStoreCreateView.as_view(), name="vector-store-create"),
    path(
        "vector-stores/rss/",
        VectorStoreRSSCreateView.as_view(),
        name="vector-store-rss-create",
    ),
    path(
        "vector-stores/<slug:user_id>/",
        VectorStoreStatusView.as_view(),
        name="vector-store-status",
    ),
    path("health/live/", LiveHealthView.as_view(), name="health-live"),
    path("health/ready/", ReadyHealthView.as_view(), name="health-ready"),
    path("metrics/", MetricsView.as_view(), name="metrics"),
]
