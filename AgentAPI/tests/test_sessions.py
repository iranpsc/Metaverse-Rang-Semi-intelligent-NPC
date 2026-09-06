from unittest.mock import AsyncMock, patch

from django.test import TestCase, override_settings
from livekit import api

from AgentAPI.livekit import create_participant_token
from AgentAPI.models import AgentSession


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
