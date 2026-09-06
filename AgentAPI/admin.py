from django.contrib import admin

from .models import AgentSession


@admin.register(AgentSession)
class AgentSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "room_name", "participant_identity", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("room_name", "participant_identity", "user_id")
    readonly_fields = ("id", "created_at", "updated_at")

