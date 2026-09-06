import json
import unittest

from pipeline_core.contracts import EVENT_VERSION, TranscriptionEvent


class TranscriptionContractTests(unittest.TestCase):
    def test_transcription_schema_is_stable_and_versioned(self):
        event = TranscriptionEvent.create(
            session_id="s1", turn_id="t1", speaker="user", text="hello", final=True
        )
        payload = json.loads(event.to_json())
        self.assertEqual(
            set(payload),
            {"type", "version", "session_id", "turn_id", "speaker", "text", "final", "timestamp"},
        )
        self.assertEqual(payload["type"], "transcription")
        self.assertEqual(payload["version"], EVENT_VERSION)
        self.assertTrue(payload["final"])
