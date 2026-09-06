from django.urls import path

from .views import (
    AgentSessionCreateView,
    AgentSessionDeleteView,
    LiveHealthView,
    MetricsView,
    ReadyHealthView,
)


urlpatterns = [
    path("agent/sessions/", AgentSessionCreateView.as_view(), name="agent-session-create"),
    path(
        "agent/sessions/<uuid:session_id>/",
        AgentSessionDeleteView.as_view(),
        name="agent-session-delete",
    ),
    path("health/live/", LiveHealthView.as_view(), name="health-live"),
    path("health/ready/", ReadyHealthView.as_view(), name="health-ready"),
    path("metrics/", MetricsView.as_view(), name="metrics"),
]
