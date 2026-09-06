import os
import unittest
from unittest.mock import patch

from pipeline_core.config import AgentSettings, ConfigurationError, safe_identifier


class ConfigurationTests(unittest.TestCase):
    def test_rejects_non_absolute_internal_url(self):
        with patch.dict(os.environ, {"STT_SERVER_URL": "stt:5001"}, clear=False):
            with self.assertRaises(ConfigurationError):
                AgentSettings.from_env()

    def test_rejects_invalid_boolean(self):
        with patch.dict(os.environ, {"VC_FAILURE_FALLBACK": "sometimes"}, clear=False):
            with self.assertRaises(ConfigurationError):
                AgentSettings.from_env()

    def test_identifier_is_sanitized_and_bounded(self):
        self.assertEqual(safe_identifier("../../user name", fallback="anon", max_length=12), "user-name")
