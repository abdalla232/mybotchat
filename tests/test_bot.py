import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.error import TimedOut

from bot import (
    ChatBot,
    MENU_BUTTONS,
    MENU_IMAGE,
    MENU_YOUTUBE,
    Settings,
    build_application,
    split_text,
)
from media_tools import is_youtube_url


def settings(**overrides):
    values = dict(
        telegram_token="123456:TEST_ONLY",
        openai_key="test-only",
        model="gpt-6-luna",
        image_model="gpt-image-2.5-sunburst",
        image_quality="high",
        transcription_model="gpt-transcribe",
        allowed_ids=frozenset({7}),
        allow_all=False,
        max_output=600,
        history_turns=2,
        max_input=100,
        cooldown=0,
        max_audio_mb=20,
        max_audio_seconds=1200,
        max_youtube_seconds=1800,
        system_prompt="Be helpful",
    )
    values.update(overrides)
    return Settings(**values)


def message_update(text="Hello", user=7):
    message = SimpleNamespace(
        text=text,
        photo=[],
        voice=None,
        audio=None,
        reply_text=AsyncMock(),
        reply_photo=AsyncMock(),
    )
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user),
        effective_chat=SimpleNamespace(id=user),
        message=message,
    )


def context(args=None):
    return SimpleNamespace(
        args=args or [],
        bot=SimpleNamespace(send_chat_action=AsyncMock(), get_file=AsyncMock()),
    )


class UnitTests(unittest.TestCase):
    def test_unicode_chunks(self):
        source = "مرحبا😀" * 2000
        parts = split_text(source)
        self.assertEqual("".join(parts), source)
        self.assertTrue(all(len(p.encode("utf-16-le")) // 2 <= 4000 for p in parts))

    def test_youtube_domain_allowlist(self):
        self.assertTrue(is_youtube_url("https://youtu.be/abc"))
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=abc"))
        self.assertFalse(is_youtube_url("https://youtube.com.example.org/watch?v=abc"))
        self.assertFalse(is_youtube_url("https://example.org/"))

    def test_missing_config(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN"):
                Settings.from_env()


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        response = SimpleNamespace(output_text="أهلًا", status="completed", usage=None)
        self.client = SimpleNamespace(
            responses=SimpleNamespace(create=AsyncMock(return_value=response)),
            images=SimpleNamespace(generate=AsyncMock()),
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=AsyncMock())
            ),
        )
        self.bot = ChatBot(settings(), self.client)
        self.context = context()

    async def test_authorized_start_shows_one_time_menu(self):
        update = message_update()
        await self.bot.start(update, self.context)
        markup = update.message.reply_text.call_args.kwargs["reply_markup"]
        self.assertIsInstance(markup, ReplyKeyboardMarkup)
        self.assertFalse(markup.is_persistent)
        self.assertTrue(markup.one_time_keyboard)
        labels = tuple(button.text for row in markup.keyboard for button in row)
        self.assertEqual(labels, MENU_BUTTONS)

    async def test_normal_text_reply_does_not_reattach_menu(self):
        update = message_update()
        await self.bot.send_menu_text(update.message, "جواب")
        markup = update.message.reply_text.call_args.kwargs["reply_markup"]
        self.assertIsNone(markup)

    async def test_unauthorized_user_is_blocked(self):
        update = message_update(user=9)
        await self.bot.text(update, self.context)
        self.client.responses.create.assert_not_awaited()
        call = update.message.reply_text.call_args
        self.assertIn("خاص", call.args[0])
        self.assertIsInstance(call.kwargs["reply_markup"], ReplyKeyboardRemove)

    async def test_image_button_sets_mode(self):
        update = message_update(MENU_IMAGE)
        await self.bot.menu_button(update, self.context)
        self.assertEqual(self.bot.session(7)["mode"], "image")
        markup = update.message.reply_text.call_args.kwargs["reply_markup"]
        self.assertIsInstance(markup, ReplyKeyboardRemove)

    async def test_youtube_button_sets_mode(self):
        update = message_update(MENU_YOUTUBE)
        await self.bot.menu_button(update, self.context)
        self.assertEqual(self.bot.session(7)["mode"], "youtube")

    async def test_oversize_audio_is_rejected_before_download(self):
        update = message_update()
        update.message.voice = SimpleNamespace(
            file_id="voice", file_size=21 * 1024 * 1024, duration=10
        )
        await self.bot.audio(update, self.context)
        self.context.bot.get_file.assert_not_awaited()
        self.assertIn("أكبر", update.message.reply_text.call_args.args[0])

    async def test_invalid_youtube_url_is_rejected(self):
        update = message_update()
        await self.bot.summarize_youtube(update, "https://example.org/video")
        self.assertIn("YouTube", update.message.reply_text.call_args.args[0])

    async def test_telegram_photo_download_retries_after_timeout(self):
        telegram_file = SimpleNamespace(download_to_memory=AsyncMock())
        self.context.bot.get_file.side_effect = [TimedOut(), telegram_file]
        expected = b"test-image"

        async def write_image(out, **kwargs):
            out.write(expected)

        telegram_file.download_to_memory.side_effect = write_image
        with patch("bot.asyncio.sleep", new=AsyncMock()):
            result = await self.bot.download_telegram_bytes(
                self.context.bot,
                "photo-id",
            )
        self.assertEqual(result, expected)
        self.assertEqual(self.context.bot.get_file.await_count, 2)

    async def test_chat_preserves_history(self):
        update = message_update()
        await self.bot.chat(update, self.context, "سؤال")
        self.assertEqual(len(self.bot.session(7)["history"]), 2)
        self.assertFalse(self.client.responses.create.call_args.kwargs["store"])

    async def test_application_builds_without_network(self):
        app = build_application(settings())
        self.assertEqual(len(app.handlers[0]), 13)
        await app.post_shutdown(app)


if __name__ == "__main__":
    unittest.main()
