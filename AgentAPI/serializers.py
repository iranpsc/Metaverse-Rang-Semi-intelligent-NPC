from __future__ import annotations

from pathlib import Path

from rest_framework import serializers

from .voice_samples import SUPPORTED_VOICE_SAMPLE_EXTENSIONS


SUPPORTED_VECTOR_STORE_EXTENSIONS = (".csv", ".docx", ".pdf", ".txt")


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


class VoiceSampleCreateSerializer(serializers.Serializer):
    name = serializers.CharField(
        max_length=64,
        help_text="Human-readable label for the target voice.",
    )
    audio = serializers.FileField(
        help_text="A WAV, MP3, M4A, OGG, FLAC, or WebM voice recording.",
    )

    def validate_audio(self, audio):
        suffix = Path(audio.name).suffix.lower()
        if suffix not in SUPPORTED_VOICE_SAMPLE_EXTENSIONS:
            supported = ", ".join(SUPPORTED_VOICE_SAMPLE_EXTENSIONS)
            raise serializers.ValidationError(f"Unsupported audio type. Supported: {supported}.")
        if not audio.size:
            raise serializers.ValidationError("The audio recording is empty.")
        max_bytes = self.context["max_bytes"]
        if audio.size > max_bytes:
            max_megabytes = max_bytes // (1024 * 1024)
            raise serializers.ValidationError(
                f"The audio recording exceeds the {max_megabytes} MB limit."
            )
        return audio


class VoiceSampleResponseSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    filename = serializers.CharField(
        help_text="Internal normalized WAV filename used by the voice-conversion service."
    )
    audio_url = serializers.CharField()
    created_at = serializers.DateTimeField()


class VoiceSampleListSerializer(serializers.Serializer):
    samples = VoiceSampleResponseSerializer(many=True)
    count = serializers.IntegerField()


class VoiceSampleDeleteSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    status = serializers.CharField()


class VectorStoreCreateSerializer(serializers.Serializer):
    user_id = serializers.RegexField(
        r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
        help_text=(
            "Stable application user ID. The resulting store identifier can be used "
            "when creating an agent session for this user."
        ),
    )
    files = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False,
        help_text="One or more PDF, DOCX, TXT, or CSV knowledge files.",
    )

    def validate_files(self, files):
        max_files = self.context["max_files"]
        max_file_bytes = self.context["max_file_bytes"]
        if len(files) > max_files:
            raise serializers.ValidationError(f"At most {max_files} files may be uploaded at once.")

        for uploaded_file in files:
            filename = uploaded_file.name.lower()
            if not filename.endswith(SUPPORTED_VECTOR_STORE_EXTENSIONS):
                supported = ", ".join(SUPPORTED_VECTOR_STORE_EXTENSIONS)
                raise serializers.ValidationError(
                    f"Unsupported file type for '{uploaded_file.name}'. Supported: {supported}."
                )
            if uploaded_file.size > max_file_bytes:
                max_megabytes = max_file_bytes // (1024 * 1024)
                raise serializers.ValidationError(
                    f"'{uploaded_file.name}' exceeds the {max_megabytes} MB limit."
                )
        return files


class VectorStoreResponseSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    vector_store = serializers.CharField(
        help_text="Pass this value as vector_store when creating an agent session."
    )
    ready = serializers.BooleanField()
    created = serializers.BooleanField()
    files_received = serializers.IntegerField(required=False)
    rss_links_received = serializers.IntegerField(required=False)
    documents_added = serializers.IntegerField()
    message = serializers.CharField()


class VectorStoreRSSCreateSerializer(serializers.Serializer):
    user_id = serializers.RegexField(
        r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
        help_text="Stable application user ID that owns the resulting vector store.",
    )
    rss_link = serializers.URLField(
        required=False,
        help_text="A single RSS or Atom feed URL.",
    )
    rss_links = serializers.ListField(
        child=serializers.URLField(),
        required=False,
        allow_empty=False,
        help_text="One or more RSS or Atom feed URLs.",
    )

    def validate(self, attrs):
        links = list(attrs.get("rss_links", []))
        single_link = attrs.pop("rss_link", None)
        if single_link:
            links.append(single_link)

        # Preserve request order while avoiding duplicate crawling work.
        links = list(dict.fromkeys(links))
        if not links:
            raise serializers.ValidationError(
                {"rss_links": "Provide rss_link or at least one item in rss_links."}
            )
        max_links = self.context["max_links"]
        if len(links) > max_links:
            raise serializers.ValidationError(
                {"rss_links": f"At most {max_links} RSS links may be processed at once."}
            )
        attrs["rss_links"] = links
        return attrs


class VectorStoreStatusSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    vector_store = serializers.CharField()
    ready = serializers.BooleanField()
    dataset_exists = serializers.BooleanField()


class HealthSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=("ok", "not_ready"))
    service = serializers.CharField()
    dependencies = serializers.DictField(required=False)
