import os
import unittest
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APIStatusError
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
        image_fallback_model="gpt-image-2.5-flare",
        image_quality="high",
        transcription_model="gpt-transcribe",
        allowed_ids=frozenset({7}),
        image_allowed_ids=frozenset({7}),
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
        caption=None,
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
            images=SimpleNamespace(generate=AsyncMock(), edit=AsyncMock()),
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

    async def test_allowed_friend_cannot_use_or_see_image_feature(self):
        restricted = ChatBot(
            settings(
                allowed_ids=frozenset({7, 8}),
                image_allowed_ids=frozenset({7}),
            ),
            self.client,
        )
        update = message_update(user=8)

        await restricted.start(update, self.context)
        markup = update.message.reply_text.call_args.kwargs["reply_markup"]
        labels = tuple(button.text for row in markup.keyboard for button in row)
        self.assertNotIn(MENU_IMAGE, labels)

        update.message.reply_text.reset_mock()
        await restricted.image_command(update, context(["a cat"]))
        self.client.images.generate.assert_not_awaited()
        self.assertIn("صاحب البوت", update.message.reply_text.call_args.args[0])

    async def test_image_model_falls_back_when_primary_is_unavailable(self):
        unavailable = APIStatusError(
            "model unavailable",
            response=httpx.Response(
                404,
                request=httpx.Request("POST", "https://api.openai.com/v1/images/generations"),
            ),
            body={"error": {"code": "model_not_found"}},
        )
        generated = SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(b"png").decode("ascii"))]
        )
        self.client.images.generate.side_effect = [unavailable, generated]
        update = message_update()

        await self.bot.generate_image(update, "a blue bird")

        models = [call.kwargs["model"] for call in self.client.images.generate.await_args_list]
        self.assertEqual(models, ["gpt-image-2.5-sunburst", "gpt-image-2.5-flare"])
        update.message.reply_photo.assert_awaited_once()
        upload = update.message.reply_photo.call_args.kwargs
        self.assertEqual(upload["read_timeout"], 120)
        self.assertEqual(upload["write_timeout"], 120)

    async def test_image_mode_stores_photo_until_edit_prompt_arrives(self):
        update = message_update()
        update.message.photo = [SimpleNamespace(file_id="source-photo")]
        self.bot.session(7)["mode"] = "image"

        await self.bot.photo(update, self.context)

        state = self.bot.session(7)
        self.assertEqual(state["mode"], "image_edit")
        self.assertEqual(state["pending_image_file_id"], "source-photo")
        self.client.images.edit.assert_not_awaited()

    async def test_photo_can_be_edited_from_followup_text(self):
        telegram_file = SimpleNamespace(download_to_memory=AsyncMock())

        async def write_image(out, **kwargs):
            out.write(b"source-image")

        telegram_file.download_to_memory.side_effect = write_image
        self.context.bot.get_file.return_value = telegram_file
        self.client.images.edit.return_value = SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(b"edited").decode("ascii"))]
        )
        update = message_update("add snow")
        state = self.bot.session(7)
        state["mode"] = "image_edit"
        state["pending_image_file_id"] = "source-photo"

        await self.bot.text(update, self.context)

        self.client.images.edit.assert_awaited_once()
        edit_request = self.client.images.edit.call_args.kwargs
        self.assertEqual(edit_request["model"], "gpt-image-2.5-sunburst")
        self.assertEqual(edit_request["prompt"], "add snow")
        update.message.reply_photo.assert_awaited_once()
        self.assertEqual(state["mode"], "chat")
        self.assertIsNone(state["pending_image_file_id"])

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
