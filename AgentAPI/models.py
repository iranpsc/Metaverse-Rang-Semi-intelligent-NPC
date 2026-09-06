from __future__ import annotations

import uuid

from django.db import models


class AgentSession(models.Model):
    class Status(models.TextChoices):
        CREATING = "creating", "Creating"
        ACTIVE = "active", "Active"
        FAILED = "failed", "Failed"
        CLOSED = "closed", "Closed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    room_name = models.CharField(max_length=128, unique=True)
    participant_identity = models.CharField(max_length=128, unique=True)
    user_id = models.CharField(max_length=128)
    vector_store = models.CharField(max_length=128, default="main_store")
    voice_sample = models.ForeignKey(
        "LLM.VoiceSample", null=True, blank=True, on_delete=models.SET_NULL
    )
    dispatch_id = models.CharField(max_length=128, blank=True)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.CREATING
    )
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return f"{self.room_name} ({self.status})"

