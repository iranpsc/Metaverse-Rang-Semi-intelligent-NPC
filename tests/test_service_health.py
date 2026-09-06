import asyncio
import unittest

from services import llm_server, stt_server


class InternalServiceHealthTests(unittest.TestCase):
    def test_stt_liveness_does_not_require_model_readiness(self):
        self.assertEqual(asyncio.run(stt_server.live()), {"status": "ok", "service": "stt"})

    def test_llm_liveness_does_not_make_provider_call(self):
        self.assertEqual(
            asyncio.run(llm_server.live()), {"status": "ok", "service": "llm-rag"}
        )
