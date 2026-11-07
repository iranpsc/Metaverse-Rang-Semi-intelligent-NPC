import io
import tempfile
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from livekit import api

from AgentAPI.livekit import create_participant_token
from AgentAPI.models import AgentSession
from LLM.models import VoiceSample
from pipeline_core.config import ConfigurationError
from pipeline_core.service_client import InternalServiceError, SpeechServiceResult


LIVEKIT_TEST_SETTINGS = {
    "LIVEKIT_URL": "ws://livekit.test:7880",
    "LIVEKIT_API_URL": "http://livekit.test:7880",
    "LIVEKIT_API_KEY": "test-key",
    "LIVEKIT_API_SECRET": "test-secret-that-is-long-enough-for-signing",
    "LIVEKIT_AGENT_NAME": "metarang-agent",
    "LIVEKIT_TOKEN_TTL_SECONDS": 600,
}


@override_settings(**LIVEKIT_TEST_SETTINGS)
class AgentSessionApiTests(TestCase):
    @patch("AgentAPI.views.provision_room_and_agent", new_callable=AsyncMock)
    def test_create_session_provisions_room_dispatch_and_scoped_token(self, provision):
        provision.return_value = "dispatch-123"

        response = self.client.post(
            "/api/v1/agent/sessions/",
            {"user_id": "unity-user", "vector_store": "main_store"},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["livekit_url"], "ws://livekit.test:7880")
        self.assertTrue(payload["room_name"].startswith("agent-session-"))
        session = AgentSession.objects.get(pk=payload["session_id"])
        self.assertEqual(session.status, AgentSession.Status.ACTIVE)
        self.assertEqual(session.dispatch_id, "dispatch-123")
        provision.assert_awaited_once()

        claims = api.TokenVerifier(
            LIVEKIT_TEST_SETTINGS["LIVEKIT_API_KEY"],
            LIVEKIT_TEST_SETTINGS["LIVEKIT_API_SECRET"],
        ).verify(payload["access_token"])
        self.assertEqual(claims.video.room, payload["room_name"])
        self.assertTrue(claims.video.room_join)
        self.assertTrue(claims.video.can_subscribe)
        self.assertFalse(claims.video.can_publish_data)
        self.assertEqual(claims.video.can_publish_sources, ["microphone"])
        self.assertFalse(bool(claims.video.room_admin))

    def test_voice_conversion_requires_a_voice_sample(self):
        response = self.client.post(
            "/api/v1/agent/sessions/",
            {"enable_voice_conversion": True},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    @patch("AgentAPI.views.requests.get")
    @patch("AgentAPI.views.provision_room_and_agent", new_callable=AsyncMock)
    def test_voice_sample_is_validated_and_dispatched_to_agent(self, provision, vc_get):
        provision.return_value = "dispatch-vc"
        vc_get.return_value.status_code = 200

        with tempfile.TemporaryDirectory() as media_root, override_settings(
            MEDIA_ROOT=Path(media_root),
            VC_SERVER_URL="http://vc.test:5003",
            VC_SAMPLE_VALIDATION_TIMEOUT_SECONDS=3,
        ):
            sample = VoiceSample.objects.create(
                name="Target voice",
                audio_file=SimpleUploadedFile("target.wav", _wav_bytes(2.0)),
            )

            response = self.client.post(
                "/api/v1/agent/sessions/",
                {
                    "user_id": "unity-user",
                    "vector_store": "main_store",
                    "enable_voice_conversion": True,
                    "voice_sample_id": sample.pk,
                },
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 201)
        vc_get.assert_called_once_with(
            "http://vc.test:5003/samples/validate",
            params={"target": sample.filename},
            timeout=3,
        )
        metadata = provision.await_args.kwargs["metadata"]
        self.assertTrue(metadata["voice_conversion_enabled"])
        self.assertEqual(metadata["voice_sample_filename"], sample.filename)

    @patch("AgentAPI.views.create_participant_token")
    def test_configuration_error_does_not_expose_exception_details(self, create_token):
        create_token.side_effect = ConfigurationError("secret backend details")

        response = self.client.post(
            "/api/v1/agent/sessions/",
            {"user_id": "unity-user", "vector_store": "main_store"},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"error": "Realtime service is not configured.", "code": "configuration_error"},
        )
        self.assertNotIn("secret backend details", response.content.decode())

    def test_swagger_schema_is_available(self):
        response = self.client.get("/api/schema/")
        self.assertEqual(response.status_code, 200)


@override_settings(**LIVEKIT_TEST_SETTINGS)
class TokenTests(TestCase):
    def test_token_is_room_scoped_and_contains_no_secret(self):
        token = create_participant_token(
            room_name="agent-session-test", identity="user-test", session_id="session-test"
        )
        self.assertNotIn(LIVEKIT_TEST_SETTINGS["LIVEKIT_API_SECRET"], token)
        claims = api.TokenVerifier(
            LIVEKIT_TEST_SETTINGS["LIVEKIT_API_KEY"],
            LIVEKIT_TEST_SETTINGS["LIVEKIT_API_SECRET"],
        ).verify(token)
        self.assertEqual(claims.identity, "user-test")
        self.assertEqual(claims.video.room, "agent-session-test")


def _wav_bytes(duration_seconds: float, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(sample_rate)
        recording.writeframes(b"\x00\x00" * int(duration_seconds * sample_rate))
    return output.getvalue()


class VoiceSampleApiTests(TestCase):
    def setUp(self):
        self.media_directory = tempfile.TemporaryDirectory()
        self.settings_override = override_settings(
            MEDIA_ROOT=Path(self.media_directory.name),
            VOICE_SAMPLE_MAX_UPLOAD_BYTES=1024 * 1024,
            VOICE_SAMPLE_MIN_DURATION_SECONDS=1,
            VOICE_SAMPLE_MAX_DURATION_SECONDS=60,
            VOICE_SAMPLE_MIN_VOICED_SECONDS=0.8,
            VOICE_SAMPLE_PROCESSING_TIMEOUT_SECONDS=30,
        )
        self.settings_override.enable()

    def tearDown(self):
        self.settings_override.disable()
        self.media_directory.cleanup()

    @patch("AgentAPI.voice_samples.estimate_voiced_seconds", return_value=1.2)
    def test_upload_normalizes_and_persists_voice_sample_for_openvoice(self, _vad):
        upload = SimpleUploadedFile(
            "browser-recording.wav",
            _wav_bytes(1.5, sample_rate=48000),
            content_type="audio/wav",
        )

        response = self.client.post(
            "/api/v1/voice-samples/",
            {"name": "Narrator", "audio": upload},
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["name"], "Narrator")
        self.assertTrue(payload["filename"].endswith(".wav"))
        sample = VoiceSample.objects.get(pk=payload["id"])
        self.assertTrue(sample.audio_file.storage.exists(sample.audio_file.name))
        with wave.open(sample.audio_file.path, "rb") as normalized:
            self.assertEqual(normalized.getframerate(), 16000)
            self.assertEqual(normalized.getnchannels(), 1)
            self.assertEqual(normalized.getsampwidth(), 2)

        list_response = self.client.get("/api/v1/voice-samples/")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["count"], 1)
        self.assertEqual(list_response.json()["samples"][0]["id"], sample.pk)

        stored_path = Path(sample.audio_file.path)
        delete_response = self.client.delete(f"/api/v1/voice-samples/{sample.pk}/")
        self.assertEqual(delete_response.status_code, 200)
        self.assertFalse(VoiceSample.objects.filter(pk=sample.pk).exists())
        self.assertFalse(stored_path.exists())

    def test_upload_rejects_recording_without_usable_speech(self):
        upload = SimpleUploadedFile(
            "silence.wav",
            _wav_bytes(1.5),
            content_type="audio/wav",
        )
        response = self.client.post(
            "/api/v1/voice-samples/",
            {"name": "Silence", "audio": upload},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_voice_sample")

    @patch("AgentAPI.voice_samples.estimate_voiced_seconds", return_value=1.2)
    def test_rag_chat_upload_uses_the_same_normalization_pipeline(self, _vad):
        upload = SimpleUploadedFile(
            "browser-recording.wav",
            _wav_bytes(1.5, sample_rate=48000),
            content_type="audio/wav",
        )

        response = self.client.post(
            "/api/voice-samples/",
            {"name": "RAG narrator", "audio": upload},
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()["sample"]
        self.assertTrue(payload["filename"].endswith(".wav"))
        sample = VoiceSample.objects.get(pk=payload["id"])
        with wave.open(sample.audio_file.path, "rb") as normalized:
            self.assertEqual(normalized.getframerate(), 16000)
            self.assertEqual(normalized.getnchannels(), 1)
            self.assertEqual(normalized.getsampwidth(), 2)

    def test_rejects_voice_sample_that_is_too_short(self):
        upload = SimpleUploadedFile(
            "short.wav",
            _wav_bytes(0.25),
            content_type="audio/wav",
        )

        response = self.client.post(
            "/api/v1/voice-samples/",
            {"name": "Too short", "audio": upload},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_voice_sample")


class SpeechJobApiTests(TestCase):
    def setUp(self):
        self.media_directory = tempfile.TemporaryDirectory()
        self.settings_override = override_settings(
            MEDIA_ROOT=Path(self.media_directory.name),
            VOICE_SAMPLE_MAX_UPLOAD_BYTES=1024 * 1024,
            VOICE_SAMPLE_MIN_DURATION_SECONDS=1,
            VOICE_SAMPLE_MAX_DURATION_SECONDS=60,
            VOICE_SAMPLE_PROCESSING_TIMEOUT_SECONDS=30,
            SPEECH_JOB_RETENTION_SECONDS=3600,
            SPEECH_TEXT_MAX_LENGTH=2000,
            TTS_SERVICE_TIMEOUT_SECONDS=120,
            VC_SERVICE_TIMEOUT_SECONDS=120,
            VC_FAILURE_FALLBACK=False,
        )
        self.settings_override.enable()

    def tearDown(self):
        self.settings_override.disable()
        self.media_directory.cleanup()

    @patch("AgentAPI.views.convert_voice_sample")
    @patch("AgentAPI.views.synthesize_speech")
    def test_combined_speech_job_uses_persistent_sample_and_exposes_audio(
        self, synthesize, convert
    ):
        source_wav = _wav_bytes(1.2)
        converted_wav = _wav_bytes(1.1)
        synthesize.return_value = SpeechServiceResult(
            source_wav, {"tts_engine": "mana-piper", "tts_rtf": 0.08}
        )
        convert.return_value = SpeechServiceResult(
            converted_wav, {"vc_engine": "openvoice-v2", "target_cache_hit": False}
        )
        sample = VoiceSample.objects.create(
            name="Target",
            audio_file=SimpleUploadedFile("target.wav", _wav_bytes(2.0)),
        )

        response = self.client.post(
            "/api/v1/speech/jobs/",
            {"text": "سلام دنیا", "sample_id": sample.pk},
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertFalse(payload["fallback_used"])
        self.assertEqual(payload["conversion_status"], "completed")
        self.assertEqual(payload["metrics"]["tts_rtf"], 0.08)
        self.assertIn("total_time", payload["metrics"])
        synthesize.assert_called_once()
        convert.assert_called_once()

        source_response = self.client.get(payload["source_audio_url"])
        converted_response = self.client.get(payload["converted_audio_url"])
        self.assertEqual(b"".join(source_response.streaming_content), source_wav)
        self.assertEqual(b"".join(converted_response.streaming_content), converted_wav)

    @patch("AgentAPI.views.normalize_voice_sample")
    @patch("AgentAPI.views.convert_voice_reference")
    @patch("AgentAPI.views.synthesize_speech")
    def test_temporary_reference_is_normalized_without_creating_a_sample(
        self, synthesize, convert, normalize
    ):
        synthesize.return_value = SpeechServiceResult(_wav_bytes(1.0), {})
        convert.return_value = SpeechServiceResult(_wav_bytes(1.0), {})
        normalized = _wav_bytes(2.0)
        normalize.return_value = (ContentFile(normalized, name="sample.wav"), 2.0)
        upload = SimpleUploadedFile("reference.wav", _wav_bytes(2.0), content_type="audio/wav")

        response = self.client.post(
            "/api/v1/speech/jobs/",
            {"text": "آزمایش صدا", "reference_audio": upload, "tau": "0.4"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(VoiceSample.objects.count(), 0)
        self.assertEqual(convert.call_args.args[2], normalized)
        self.assertEqual(convert.call_args.kwargs["tau"], 0.4)

    @patch("AgentAPI.views.convert_voice_sample")
    @patch("AgentAPI.views.synthesize_speech")
    def test_vc_failure_is_explicit_when_fallback_is_disabled(self, synthesize, convert):
        synthesize.return_value = SpeechServiceResult(_wav_bytes(1.0), {})
        convert.side_effect = InternalServiceError(
            "voice_conversion",
            "VC queue is full",
            status_code=429,
            code="queue_full",
            retry_after="5",
        )
        sample = VoiceSample.objects.create(
            name="Target",
            audio_file=SimpleUploadedFile("target.wav", _wav_bytes(2.0)),
        )

        response = self.client.post(
            "/api/v1/speech/jobs/",
            {"text": "سلام", "voice_sample_id": sample.pk},
        )

        self.assertEqual(response.status_code, 429)
        self.assertFalse(response.json()["fallback_used"])
        self.assertEqual(response.json()["code"], "queue_full")
        self.assertEqual(response["Retry-After"], "5")

    @patch("AgentAPI.views.convert_voice_sample")
    @patch("AgentAPI.views.synthesize_speech")
    @override_settings(VC_FAILURE_FALLBACK=True)
    def test_enabled_fallback_is_clearly_reported(self, synthesize, convert):
        synthesize.return_value = SpeechServiceResult(_wav_bytes(1.0), {})
        convert.side_effect = InternalServiceError(
            "voice_conversion", "VC unavailable", code="voice_conversion_unavailable"
        )
        sample = VoiceSample.objects.create(
            name="Target",
            audio_file=SimpleUploadedFile("target.wav", _wav_bytes(2.0)),
        )

        response = self.client.post(
            "/api/v1/speech/jobs/",
            {"text": "سلام", "voice_sample_id": sample.pk},
        )

        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["fallback_used"])
        self.assertEqual(response.json()["conversion_status"], "fallback")
        self.assertIsNone(response.json()["converted_audio_url"])

    def test_exactly_one_voice_target_is_required(self):
        response = self.client.post("/api/v1/speech/jobs/", {"text": "سلام"})
        self.assertEqual(response.status_code, 400)

    def test_sample_id_and_legacy_alias_cannot_be_sent_together(self):
        response = self.client.post(
            "/api/v1/speech/jobs/",
            {"text": "سلام", "sample_id": 1, "voice_sample_id": 1},
        )
        self.assertEqual(response.status_code, 400)

    def test_swagger_contains_combined_speech_endpoints(self):
        response = self.client.get("/api/schema/", HTTP_ACCEPT="application/json")
        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]
        self.assertIn("/api/v1/speech/jobs/", paths)
        self.assertIn("/api/v1/speech/jobs/{job_id}/{artifact}/", paths)


class VectorStoreApiTests(TestCase):
    def setUp(self):
        self.data_directory = tempfile.TemporaryDirectory()
        self.settings_override = override_settings(
            DATA_ROOT=Path(self.data_directory.name),
            VECTOR_STORE_MAX_FILES=10,
            VECTOR_STORE_MAX_RSS_LINKS=10,
            VECTOR_STORE_MAX_UPLOAD_BYTES=1024 * 1024,
        )
        self.settings_override.enable()

    def tearDown(self):
        self.settings_override.disable()
        self.data_directory.cleanup()

    @patch("AgentAPI.views.run_llm_operation")
    def test_create_user_vector_store_returns_session_identifier(self, run_operation):
        def build_index(*args, **kwargs):
            store_path = Path(self.data_directory.name) / "vectorstores" / "user_player-42" / "rss_store"
            store_path.mkdir(parents=True, exist_ok=True)
            (store_path / "index.faiss").write_bytes(b"test-index")
            return {
                "status": "success",
                "message": "Added 1 files to knowledge base.",
                "new_count": 1,
            }

        run_operation.side_effect = build_index
        upload = SimpleUploadedFile("knowledge.txt", b"MetaRang knowledge")

        response = self.client.post(
            "/api/v1/vector-stores/",
            {"user_id": "player-42", "files": [upload]},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["vector_store"], "user_player-42/rss_store")
        self.assertTrue(response.json()["ready"])
        run_operation.assert_called_once()
        self.assertEqual(run_operation.call_args.args[1], "files")

    @patch("AgentAPI.views.run_llm_operation")
    def test_create_user_vector_store_from_rss(self, run_operation):
        def build_index(*args, **kwargs):
            store_path = Path(self.data_directory.name) / "vectorstores" / "user_player-42" / "rss_store"
            store_path.mkdir(parents=True, exist_ok=True)
            (store_path / "index.faiss").write_bytes(b"test-index")
            return {
                "status": "success",
                "message": "Successfully added 3 new articles.",
                "new_count": 3,
            }

        run_operation.side_effect = build_index

        response = self.client.post(
            "/api/v1/vector-stores/rss/",
            {
                "user_id": "player-42",
                "rss_link": "https://example.com/feed.xml",
            },
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["vector_store"], "user_player-42/rss_store")
        self.assertEqual(payload["rss_links_received"], 1)
        self.assertEqual(payload["documents_added"], 3)
        run_operation.assert_called_once_with(
            settings.LLM_SERVER_URL,
            "rss",
            {"user_id": "player-42", "rss_links": ["https://example.com/feed.xml"]},
            timeout_seconds=settings.LLM_OPERATION_TIMEOUT_SECONDS,
        )

    @patch("AgentAPI.views.run_llm_operation")
    def test_refresh_existing_rss_vector_store_returns_200(self, run_operation):
        store_path = (
            Path(self.data_directory.name)
            / "vectorstores"
            / "user_player-42"
            / "rss_store"
        )
        store_path.mkdir(parents=True)
        (store_path / "index.faiss").write_bytes(b"existing-index")
        run_operation.return_value = {
            "status": "success",
            "message": "No new articles found in provided RSS feeds.",
            "new_count": 0,
        }

        response = self.client.post(
            "/api/v1/vector-stores/rss/",
            {
                "user_id": "player-42",
                "rss_links": [
                    "https://example.com/feed.xml",
                    "https://example.org/atom.xml",
                ],
            },
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["created"])
        self.assertEqual(response.json()["rss_links_received"], 2)
        self.assertEqual(response.json()["documents_added"], 0)

    def test_rss_vector_store_requires_a_feed_link(self):
        response = self.client.post(
            "/api/v1/vector-stores/rss/",
            {"user_id": "player-42"},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Provide rss_link", str(response.json()))

    def test_rejects_unsupported_file_type(self):
        upload = SimpleUploadedFile("malware.exe", b"not executable")

        response = self.client.post(
            "/api/v1/vector-stores/",
            {"user_id": "player-42", "files": [upload]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", str(response.json()))

    def test_get_user_vector_store_status(self):
        root = Path(self.data_directory.name)
        store_path = root / "vectorstores" / "user_player-42" / "rss_store"
        dataset_path = root / "rss_datasets" / "user_player-42" / "rss_dataset.csv"
        store_path.mkdir(parents=True)
        dataset_path.parent.mkdir(parents=True)
        (store_path / "index.faiss").write_bytes(b"test-index")
        dataset_path.write_text("title,content\nTest,Knowledge\n", encoding="utf-8")

        response = self.client.get("/api/v1/vector-stores/player-42/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["vector_store"], "user_player-42/rss_store")
        self.assertTrue(response.json()["ready"])
        self.assertTrue(response.json()["dataset_exists"])

    def test_swagger_schema_contains_vector_store_endpoints(self):
        response = self.client.get("/api/schema/", HTTP_ACCEPT="application/json")

        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]
        self.assertIn("/api/v1/vector-stores/", paths)
        self.assertIn("/api/v1/vector-stores/rss/", paths)
        self.assertIn("/api/v1/vector-stores/{user_id}/", paths)
        self.assertIn("/api/v1/voice-samples/", paths)
