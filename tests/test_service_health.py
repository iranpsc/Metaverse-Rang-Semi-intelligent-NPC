import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, Request

from services import llm_server, stt_server


class InternalServiceHealthTests(unittest.TestCase):
    def test_stt_liveness_does_not_require_model_readiness(self):
        self.assertEqual(asyncio.run(stt_server.live()), {"status": "ok", "service": "stt"})

    def test_llm_liveness_does_not_make_provider_call(self):
        self.assertEqual(
            asyncio.run(llm_server.live()), {"status": "ok", "service": "llm-rag"}
        )


class LlmPathSecurityTests(unittest.TestCase):
    def test_store_is_selected_from_trusted_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = root / "vectorstores" / "tenant" / "main_store"
            store.mkdir(parents=True)
            (store / "index.faiss").touch()

            with patch.object(llm_server, "data_root", return_value=root):
                self.assertEqual(llm_server._resolve_store("tenant/main_store"), store)

    def test_store_rejects_traversal_and_unknown_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "vectorstores").mkdir()

            with patch.object(llm_server, "data_root", return_value=root):
                for value in ("../secret", "/absolute", "tenant//store", "tenant\\store"):
                    with self.subTest(value=value):
                        with self.assertRaises(HTTPException) as raised:
                            llm_server._resolve_store(value)
                        self.assertEqual(raised.exception.status_code, 400)

                with self.assertRaises(HTTPException) as raised:
                    llm_server._resolve_store("missing")
                self.assertEqual(raised.exception.status_code, 404)

    def test_memory_path_does_not_contain_user_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(llm_server, "data_root", return_value=root):
                memory_path = llm_server._user_memory_path("../../private/user")
                repeated_path = llm_server._user_memory_path("../../private/user")

            self.assertEqual(memory_path.parent, root / "memories")
            self.assertNotIn("private", memory_path.name)
            self.assertEqual(memory_path, repeated_path)


class LlmConcurrencyTests(unittest.TestCase):
    def test_generation_starts_only_after_semaphore_is_acquired(self):
        async def exercise():
            class FakeRag:
                def __init__(self):
                    self.started = threading.Event()

                def get_answer_for_user(self, **_kwargs):
                    self.started.set()
                    yield "hello"

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                store = root / "vectorstores" / "main_store"
                store.mkdir(parents=True)
                (store / "index.faiss").touch()
                fake_rag = FakeRag()
                semaphore = asyncio.Semaphore(1)
                await semaphore.acquire()

                previous_rag = llm_server.runtime.rag
                previous_semaphore = llm_server.runtime.semaphore
                llm_server.runtime.rag = fake_rag
                llm_server.runtime.semaphore = semaphore
                try:
                    request = Request({"type": "http", "headers": []})
                    payload = llm_server.ChatRequest(
                        question="Hi", user_id="user", vector_store="main_store"
                    )
                    with patch.object(llm_server, "data_root", return_value=root):
                        response = await llm_server.chat_stream(payload, request)

                        async def consume():
                            return [chunk async for chunk in response.body_iterator]

                        consumer = asyncio.create_task(consume())
                        await asyncio.sleep(0.05)
                        self.assertFalse(fake_rag.started.is_set())
                        semaphore.release()
                        chunks = await asyncio.wait_for(consumer, timeout=2)
                finally:
                    llm_server.runtime.rag = previous_rag
                    llm_server.runtime.semaphore = previous_semaphore

            self.assertTrue(fake_rag.started.is_set())
            body = "".join(
                chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in chunks
            )
            self.assertIn('"type": "done"', body)

        asyncio.run(exercise())
