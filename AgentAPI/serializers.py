from __future__ import annotations

from rest_framework import serializers


class AgentSessionCreateSerializer(serializers.Serializer):
    user_id = serializers.RegexField(
        r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$", required=False, allow_blank=False
    )
    vector_store = serializers.RegexField(
        r"^[A-Za-z0-9][A-Za-z0-9_/-]{0,127}$",
        required=False,
        default="main_store",
        help_text="Vector-store name relative to the configured data root.",
    )
    voice_sample_id = serializers.IntegerField(required=False, allow_null=True, min_value=1)
    enable_voice_conversion = serializers.BooleanField(required=False, default=False)

    def validate_vector_store(self, value: str) -> str:
        if ".." in value.split("/"):
            raise serializers.ValidationError("Path traversal is not allowed.")
        return value.strip("/")

    def validate(self, attrs):
        if attrs.get("enable_voice_conversion") and not attrs.get("voice_sample_id"):
            raise serializers.ValidationError(
                {"voice_sample_id": "Required when voice conversion is enabled."}
            )
        return attrs


class AgentSessionResponseSerializer(serializers.Serializer):
    session_id = serializers.UUIDField()
    room_name = serializers.CharField()
    livekit_url = serializers.CharField(help_text="WebSocket URL used by the LiveKit Unity SDK.")
    access_token = serializers.CharField()
    expires_at = serializers.DateTimeField()


class ErrorSerializer(serializers.Serializer):
    error = serializers.CharField()
    code = serializers.CharField()


class HealthSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=("ok", "not_ready"))
    service = serializers.CharField()
    dependencies = serializers.DictField(required=False)
