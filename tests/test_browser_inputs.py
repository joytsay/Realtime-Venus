import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from demos.model.adapter import AdapterError, RealtimeVenusOmniAdapter, _Session
from demos.server.session import WebAgentSession
from harness.bridge.serving import TextAppend


class TypedInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = SimpleNamespace(streaming_prefill=MagicMock())
        self.adapter = RealtimeVenusOmniAdapter(
            self.model, special_token_ids={}, model_revision="test"
        )
        self.session = _Session("typed", 1, "Realtime-Venus-Omni")
        self.adapter._session = self.session

    async def test_text_drives_prefill_without_microphone_audio(self):
        accepted = await self.adapter.append_text({
            "session_id": "typed", "incarnation": 1,
            "event_seq": 1, "text": "Hello Venus",
        })
        self.assertEqual(accepted["input_seq"], 1)
        bucket = self.adapter._collect_bucket(self.session)
        np.testing.assert_array_equal(bucket.audio, np.zeros(16000))
        await self.adapter._prefill_bucket(self.session, bucket, is_backend=False)
        self.assertEqual(self.model.streaming_prefill.call_args.args[2], ["Hello Venus"])
        self.assertEqual(self.session.consumed_input_seq, 1)
        self.assertIsNone(self.adapter._collect_bucket(self.session))

    async def test_invalid_text_does_not_advance_input_fence(self):
        for text in [None, " ", "x" * 4001]:
            with self.assertRaises(AdapterError):
                await self.adapter.append_text({
                    "session_id": "typed", "incarnation": 1,
                    "event_seq": 1, "text": text,
                })
        self.assertEqual(self.session.event_seq, 0)
        with self.assertRaises(ValueError):
            TextAppend("typed", 1, 1, " ")

    async def test_browser_text_is_forwarded_and_acknowledged(self):
        web = WebAgentSession(MagicMock(), MagicMock(), mode="omni", input_source="text")
        web.host = SimpleNamespace(append_text=AsyncMock())
        web.send = AsyncMock()
        await web.handle("text", {"text": " Hello "})
        web.host.append_text.assert_awaited_once_with("Hello")
        web.send.assert_awaited_once_with({"type": "user_text", "text": "Hello"})

    async def test_audio_upload_is_accepted_by_both_model_variants(self):
        for mode in ["audio", "omni"]:
            web = WebAgentSession(MagicMock(), MagicMock(), mode=mode, input_source="audio_file")
            web._upload_media = AsyncMock()
            stream = object()
            await web.upload_audio(stream)
            web._upload_media.assert_awaited_once_with(stream)


if __name__ == "__main__":
    unittest.main()
