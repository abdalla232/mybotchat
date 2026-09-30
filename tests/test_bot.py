import base64
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import httpx2
from openai import APIConnectionError, APIStatusError, RateLimitError
from telegram.error import TelegramError

from bot import ChatBot, Settings, build_application


class ImageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = Settings("123:ABC", "test", "gpt-4.1-mini", frozenset({1}), False,
                            600, 4, 3000, 3, "Reply briefly.")
        self.generate = AsyncMock(return_value=NS(data=[NS(b64_json=base64.b64encode(b"jpeg-bytes").decode())]))
        self.client = NS(images=NS(generate=self.generate), responses=NS(create=AsyncMock()))
        self.bot = ChatBot(self.cfg, self.client)
        self.message = NS(text="/image city at sunset", reply_text=AsyncMock(), reply_photo=AsyncMock())
        self.update = NS(message=self.message, effective_user=NS(id=1), effective_chat=NS(id=1))
        self.context = NS(bot=NS(send_chat_action=AsyncMock()))

    async def test_image_delivery_preserves_text_history(self):
        history = [{"role": "user", "content": "hello"}]
        self.bot.session(1)["history"] = history.copy()
        async def receive(**kwargs):
            self.assertEqual(kwargs["photo"].read(), b"jpeg-bytes")
        self.message.reply_photo.side_effect = receive
        await self.bot.image(self.update, self.context)
        self.message.reply_photo.assert_awaited_once()
        self.assertEqual(self.bot.session(1)["history"], history)
        self.assertEqual(self.generate.call_args.kwargs["prompt"], "city at sunset")

    async def test_invalid_input_does_not_call_api(self):
        for value in ("/image", "/image   ", "/image " + "x" * 3001):
            self.message.text = value
            await self.bot.image(self.update, self.context)
        self.generate.assert_not_awaited()

    async def test_unauthorized_user(self):
        self.update.effective_user.id = 2
        await self.bot.image(self.update, self.context)
        self.generate.assert_not_awaited()

    async def test_allow_all(self):
        self.bot.settings = replace(self.cfg, allow_all=True)
        self.update.effective_user.id = 2
        await self.bot.image(self.update, self.context)
        self.generate.assert_awaited_once()

    async def test_cooldown_survives_reset(self):
        with patch("bot.time.monotonic", return_value=100):
            await self.bot.image(self.update, self.context)
            await self.bot.reset(self.update, self.context)
        with patch("bot.time.monotonic", return_value=104):
            await self.bot.image(self.update, self.context)
        self.generate.assert_awaited_once()
        with patch("bot.time.monotonic", return_value=160):
            await self.bot.image(self.update, self.context)
        self.assertEqual(self.generate.await_count, 2)

    async def test_chat_cooldown_applies_to_images(self):
        self.bot.session(1)["last_request"] = 99
        with patch("bot.time.monotonic", return_value=100):
            await self.bot.image(self.update, self.context)
        self.generate.assert_not_awaited()

    async def test_invalid_payloads(self):
        for data in ([], [NS(b64_json=None)], [NS(b64_json="!invalid!")]):
            self.bot.sessions.clear()
            self.generate.return_value = NS(data=data)
            await self.bot.image(self.update, self.context)
        self.message.reply_photo.assert_not_awaited()

    async def test_api_errors_are_reported_without_retry(self):
        request = httpx2.Request("POST", "https://api.openai.com/v1/images/generations")
        errors = [APIConnectionError(request=request),
                  RateLimitError("quota", response=httpx2.Response(429, request=request), body=None),
                  APIStatusError("denied", response=httpx2.Response(403, request=request), body=None),
                  APIStatusError("blocked", response=httpx2.Response(400, request=request),
                                 body={"code": "moderation_blocked"})]
        for error in errors:
            self.bot.sessions.clear()
            self.generate.reset_mock()
            self.generate.side_effect = error
            await self.bot.image(self.update, self.context)
            self.generate.assert_awaited_once()
        self.message.reply_photo.assert_not_awaited()

    async def test_delivery_failure_does_not_regenerate(self):
        self.message.reply_photo.side_effect = TelegramError("upload failed")
        await self.bot.image(self.update, self.context)
        self.generate.assert_awaited_once()
        self.assertIn("تيليغرام", self.message.reply_text.call_args.args[0])

    async def test_indicator_failure_does_not_block_image(self):
        self.context.bot.send_chat_action.side_effect = TelegramError("failed")
        await self.bot.image(self.update, self.context)
        self.message.reply_photo.assert_awaited_once()

    async def test_text_chat_still_stores_history(self):
        self.message.text = "hello"
        self.client.responses.create.return_value = NS(output_text="hi", status="completed", usage=None)
        await self.bot.text(self.update, self.context)
        self.assertEqual(self.bot.session(1)["history"][-1], {"role": "assistant", "content": "hi"})
        self.generate.assert_not_awaited()

    async def test_image_command_registered(self):
        app = build_application(self.cfg)
        self.assertTrue(any("image" in getattr(handler, "commands", ()) for handler in app.handlers[0]))
        await app.post_shutdown(app)


class SettingsTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:ABC", "OPENAI_API_KEY": "test"}, clear=True):
            self.assertEqual(Settings.from_env().image_quality, "low")
            for name, value in (("IMAGE_QUALITY", "bad"), ("IMAGE_COOLDOWN_SECONDS", "-1")):
                with patch.dict(os.environ, {name: value}):
                    with self.assertRaises(ValueError):
                        Settings.from_env()


if __name__ == "__main__":
    unittest.main()
