"""Private Telegram AI bot: chat, images, OCR, audio and YouTube summaries."""
import asyncio
import base64
import logging
import os
import subprocess
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from telegram import BotCommand, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.error import NetworkError, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from media_tools import convert_to_mp3, download_youtube_audio, is_youtube_url

log = logging.getLogger("ai_bot")

MENU_CHAT = "💬 محادثة"
MENU_IMAGE = "🎨 إنشاء صورة"
MENU_OCR = "🖼 استخراج نص من صورة"
MENU_AUDIO = "🎙 تفريغ صوت"
MENU_YOUTUBE = "▶️ تلخيص YouTube"
MENU_RESET = "🧹 مسح المحادثة"
MENU_ID = "🆔 رقمي"
MENU_HELP = "ℹ️ المساعدة"
MENU_BUTTONS = (
    MENU_CHAT, MENU_IMAGE, MENU_OCR, MENU_AUDIO,
    MENU_YOUTUBE, MENU_RESET, MENU_ID, MENU_HELP,
)


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_key: str
    model: str
    image_model: str
    image_fallback_model: str
    image_quality: str
    transcription_model: str
    allowed_ids: frozenset[int]
    allow_all: bool
    max_output: int
    history_turns: int
    max_input: int
    cooldown: int
    max_audio_mb: int
    max_audio_seconds: int
    max_youtube_seconds: int
    system_prompt: str

    @classmethod
    def from_env(cls):
        def required(name):
            value = os.getenv(name, "").strip()
            if not value or value.startswith("YOUR_"):
                raise ValueError(f"Set {name} in Railway Variables or .env")
            return value

        def integer(name, default, low, high):
            try:
                value = int(os.getenv(name, str(default)))
            except ValueError:
                raise ValueError(f"{name} must be an integer") from None
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
            return value

        try:
            ids = frozenset(
                int(value.strip())
                for value in os.getenv("ALLOWED_USER_IDS", "").split(",")
                if value.strip()
            )
            if any(value <= 0 for value in ids):
                raise ValueError
        except ValueError:
            raise ValueError(
                "ALLOWED_USER_IDS must contain positive numeric IDs separated by commas"
            ) from None

        allow_all = os.getenv("ALLOW_ALL_USERS", "false").strip().lower()
        if allow_all not in ("true", "false"):
            raise ValueError("ALLOW_ALL_USERS must be true or false")

        image_quality = os.getenv("OPENAI_IMAGE_QUALITY", "high").strip().lower() or "high"
        if image_quality not in ("auto", "low", "medium", "high", "xhigh", "max"):
            raise ValueError(
                "OPENAI_IMAGE_QUALITY must be auto, low, medium, high, xhigh, or max"
            )

        return cls(
            telegram_token=required("TELEGRAM_BOT_TOKEN"),
            openai_key=required("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-6-luna").strip() or "gpt-6-luna",
            image_model=os.getenv(
                "OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst"
            ).strip() or "gpt-image-2.5-sunburst",
            image_fallback_model=os.getenv(
                "OPENAI_IMAGE_FALLBACK_MODEL", "gpt-image-2.5-flare"
            ).strip() or "gpt-image-2.5-flare",
            image_quality=image_quality,
            transcription_model=os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-transcribe").strip(),
            allowed_ids=ids,
            allow_all=allow_all == "true",
            max_output=integer("MAX_OUTPUT_TOKENS", 1800, 16, 4096),
            history_turns=integer("HISTORY_TURNS", 4, 0, 20),
            max_input=integer("MAX_INPUT_CHARS", 5000, 1, 20000),
            cooldown=integer("COOLDOWN_SECONDS", 3, 0, 3600),
            max_audio_mb=integer("MAX_AUDIO_MB", 20, 1, 24),
            max_audio_seconds=integer("MAX_AUDIO_SECONDS", 1200, 10, 7200),
            max_youtube_seconds=integer("MAX_YOUTUBE_SECONDS", 1800, 30, 7200),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "أنت مساعد ذكي ومفيد. أجب بالعربية ما لم يطلب المستخدم لغة أخرى. "
                "قدّم إجابات واضحة ومفصلة عند الحاجة.",
            ),
        )


def split_text(text, limit=4000):
    parts, current, size = [], [], 0
    for char in text:
        width = 2 if ord(char) > 0xFFFF else 1
        if size + width > limit:
            parts.append("".join(current))
            current, size = [], 0
        current.append(char)
        size += width
    if current:
        parts.append("".join(current))
    return parts


class ChatBot:
    def __init__(self, settings, client):
        self.settings = settings
        self.client = client
        self.sessions = OrderedDict()

    def allowed(self, user_id):
        return self.settings.allow_all or user_id in self.settings.allowed_ids

    def session(self, user_id):
        if user_id not in self.sessions:
            if len(self.sessions) >= 1000:
                self.sessions.popitem(last=False)
            self.sessions[user_id] = {
                "history": [],
                "mode": "chat",
                "last_request": float("-inf"),
                "pending_image_file_id": None,
            }
        self.sessions.move_to_end(user_id)
        return self.sessions[user_id]

    @staticmethod
    def menu_keyboard():
        return ReplyKeyboardMarkup(
            [
                [MENU_CHAT, MENU_IMAGE],
                [MENU_OCR, MENU_AUDIO],
                [MENU_YOUTUBE, MENU_RESET],
                [MENU_ID, MENU_HELP],
            ],
            resize_keyboard=True,
            is_persistent=False,
            one_time_keyboard=True,
        )

    async def send_menu_text(self, message, text, reply_markup=None):
        parts = split_text(text)
        for index, part in enumerate(parts):
            await message.reply_text(
                part,
                parse_mode=None,
                reply_markup=reply_markup if index == len(parts) - 1 else None,
            )

    async def download_telegram_bytes(self, bot, file_id):
        """Download a Telegram file with longer timeouts and transient retries."""
        last_error = None
        for attempt in range(3):
            try:
                telegram_file = await bot.get_file(
                    file_id,
                    read_timeout=30,
                    write_timeout=30,
                    connect_timeout=30,
                    pool_timeout=30,
                )
                output = BytesIO()
                await telegram_file.download_to_memory(
                    out=output,
                    read_timeout=60,
                    write_timeout=60,
                    connect_timeout=30,
                    pool_timeout=30,
                )
                data = output.getvalue()
                if not data:
                    raise TelegramError("Downloaded Telegram file is empty")
                return data
            except TelegramError as exc:
                last_error = exc
                log.warning(
                    "Telegram media download attempt %s failed: %s",
                    attempt + 1,
                    type(exc).__name__,
                )
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_error

    async def deny(self, update):
        user_id = update.effective_user.id
        await update.message.reply_text(
            "⛔ هذا البوت خاص وغير متاح لحسابك.\n\n"
            f"رقم حسابك: {user_id}\n"
            "أرسل الرقم لصاحب البوت ليضيفه إلى المستخدمين المسموحين.",
            reply_markup=ReplyKeyboardRemove(),
        )

    async def require_access(self, update):
        if self.allowed(update.effective_user.id):
            return True
        await self.deny(update)
        return False

    async def begin_expensive_request(self, update):
        if not await self.require_access(update):
            return False
        state = self.session(update.effective_user.id)
        now = time.monotonic()
        if now - state["last_request"] < self.settings.cooldown:
            await self.send_menu_text(update.message, "انتظر قليلًا قبل إرسال طلب آخر.")
            return False
        state["last_request"] = now
        return True

    async def start(self, update, context):
        if not await self.require_access(update):
            return
        await self.send_menu_text(
            update.message,
            "👋 أهلًا بك في بوت الذكاء الاصطناعي\n\n"
            "اختر ميزة من القائمة أو أرسل سؤالك مباشرة.\n"
            "لإظهار القائمة لاحقًا استخدم /menu.",
            reply_markup=self.menu_keyboard(),
        )

    async def menu_command(self, update, context):
        if not await self.require_access(update):
            return
        await self.send_menu_text(
            update.message,
            "اختر الميزة التي تريدها:",
            reply_markup=self.menu_keyboard(),
        )

    async def help_command(self, update, context):
        if not await self.require_access(update):
            return
        await self.send_menu_text(
            update.message,
            "الميزات المتاحة:\n"
            "• محادثة نصية مع ذاكرة قصيرة.\n"
            "• إنشاء صورة من وصفك.\n"
            "• استخراج النص ووصف محتوى الصور.\n"
            "• تحويل الرسائل والملفات الصوتية إلى نص.\n"
            "• تلخيص فيديو YouTube عام ضمن حد المدة.\n\n"
            "استخدم /menu لإظهار لوحة الأزرار من جديد.\n"
            "يمكنك أيضًا استخدام /image و/youtube و/chat و/reset و/id.",
            reply_markup=ReplyKeyboardRemove(),
        )

    async def identify(self, update, context):
        if self.allowed(update.effective_user.id):
            await self.send_menu_text(
                update.message,
                f"🆔 رقم حسابك: {update.effective_user.id}",
                reply_markup=ReplyKeyboardRemove(),
            )
        else:
            await self.deny(update)

    async def reset(self, update, context):
        if not await self.require_access(update):
            return
        state = self.session(update.effective_user.id)
        state["history"] = []
        state["mode"] = "chat"
        state["pending_image_file_id"] = None
        await self.send_menu_text(
            update.message,
            "✅ تم مسح المحادثة وبدء محادثة جديدة.",
            reply_markup=ReplyKeyboardRemove(),
        )

    async def set_mode(self, update, mode, prompt):
        if not await self.require_access(update):
            return
        state = self.session(update.effective_user.id)
        state["mode"] = mode
        state["pending_image_file_id"] = None
        await self.send_menu_text(
            update.message,
            prompt,
            reply_markup=ReplyKeyboardRemove(),
        )

    async def chat_command(self, update, context):
        await self.set_mode(update, "chat", "💬 وضع المحادثة مفعّل. أرسل سؤالك.")

    async def image_command(self, update, context):
        if context.args:
            if not await self.require_access(update):
                return
            await self.generate_image(update, " ".join(context.args))
        else:
            await self.set_mode(
                update,
                "image",
                "🎨 أرسل وصفًا لإنشاء صورة جديدة، أو أرسل صورة مع التعديل المطلوب في التعليق.",
            )

    async def youtube_command(self, update, context):
        if context.args:
            if not await self.require_access(update):
                return
            await self.summarize_youtube(update, " ".join(context.args))
        else:
            await self.set_mode(update, "youtube", "▶️ أرسل رابط فيديو YouTube العام.")

    async def menu_button(self, update, context):
        choice = update.message.text
        if choice == MENU_CHAT:
            await self.chat_command(update, context)
        elif choice == MENU_IMAGE:
            await self.set_mode(
                update,
                "image",
                "🎨 أرسل وصفًا لإنشاء صورة جديدة، أو أرسل صورة مع التعديل المطلوب في التعليق.",
            )
        elif choice == MENU_OCR:
            await self.set_mode(update, "ocr", "🖼 أرسل الصورة الآن، وسأستخرج النص وأصف محتواها.")
        elif choice == MENU_AUDIO:
            await self.set_mode(update, "audio", "🎙 أرسل رسالة صوتية أو ملفًا صوتيًا الآن.")
        elif choice == MENU_YOUTUBE:
            await self.set_mode(update, "youtube", "▶️ أرسل رابط فيديو YouTube العام.")
        elif choice == MENU_RESET:
            await self.reset(update, context)
        elif choice == MENU_ID:
            await self.identify(update, context)
        elif choice == MENU_HELP:
            await self.help_command(update, context)

    async def text(self, update, context):
        if not await self.require_access(update):
            return
        text = update.message.text.strip()
        if not text or len(text) > self.settings.max_input:
            await self.send_menu_text(
                update.message,
                f"أرسل نصًا بين 1 و{self.settings.max_input} حرف.",
            )
            return

        mode = self.session(update.effective_user.id)["mode"]
        if mode == "image":
            await self.generate_image(update, text)
        elif mode == "image_edit":
            file_id = self.session(update.effective_user.id).get("pending_image_file_id")
            if file_id:
                await self.edit_image(update, context, file_id, text)
            else:
                self.session(update.effective_user.id)["mode"] = "image"
                await self.send_menu_text(update.message, "أرسل الصورة التي تريد تعديلها أولًا.")
        elif mode == "youtube":
            await self.summarize_youtube(update, text)
        elif mode == "ocr":
            await self.send_menu_text(update.message, "أرسل صورة، وليس وصفًا نصيًا.")
        elif mode == "audio":
            await self.send_menu_text(update.message, "أرسل رسالة صوتية أو ملفًا صوتيًا.")
        else:
            await self.chat(update, context, text)

    async def chat(self, update, context, text):
        if not await self.begin_expensive_request(update):
            return
        state = self.session(update.effective_user.id)
        messages = [*state["history"], {"role": "user", "content": text}]
        try:
            await context.bot.send_chat_action(update.effective_chat.id, "typing")
        except TelegramError:
            pass
        try:
            response = await self.client.responses.create(
                model=self.settings.model,
                instructions=self.settings.system_prompt,
                input=messages,
                max_output_tokens=self.settings.max_output,
                store=False,
            )
            answer = (response.output_text or "").strip()
            if not answer:
                await self.send_menu_text(update.message, "لم يصل جواب نصي. جرّب مرة أخرى.")
                return
            if response.status == "incomplete":
                answer += "\n\n[توقف الرد عند الحد المحدد؛ يمكنك طلب المتابعة.]"
            await self.send_menu_text(update.message, answer)
            state["history"] = (
                messages + [{"role": "assistant", "content": answer}]
            )[-2 * self.settings.history_turns :] if self.settings.history_turns else []
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            await self.api_error(update.message, exc)

    async def generate_image(self, update, prompt):
        if not await self.begin_expensive_request(update):
            return
        await self.send_menu_text(update.message, "🎨 عم جهّز الصورة، قد يستغرق ذلك قليلًا…")
        try:
            try:
                result = await self.generate_image_request(
                    self.settings.image_model,
                    prompt,
                )
            except APIStatusError as exc:
                fallback = self.settings.image_fallback_model
                if (
                    getattr(exc, "status_code", None) in (403, 404)
                    and fallback
                    and fallback != self.settings.image_model
                ):
                    log.warning(
                        "Image model %s unavailable (status=%s); retrying with %s",
                        self.settings.image_model,
                        exc.status_code,
                        fallback,
                    )
                    await self.send_menu_text(
                        update.message,
                        "⚠️ المحرك الأقوى غير متاح لحسابك، سأجرّب المحرك الاحتياطي.",
                    )
                    result = await self.generate_image_request(fallback, prompt)
                else:
                    raise
            image_bytes = base64.b64decode(result.data[0].b64_json)
            image = BytesIO(image_bytes)
            image.name = "generated.png"
            await update.message.reply_photo(
                photo=image,
                caption="✅ تم إنشاء الصورة.",
                reply_markup=ReplyKeyboardRemove(),
                read_timeout=120,
                write_timeout=120,
                connect_timeout=30,
                pool_timeout=30,
            )
            state = self.session(update.effective_user.id)
            state["mode"] = "chat"
            state["pending_image_file_id"] = None
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            await self.api_error(update.message, exc)

    async def generate_image_request(self, model, prompt):
        return await self.client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
            quality=self.settings.image_quality,
            output_format="png",
            n=1,
        )

    async def edit_image(self, update, context, file_id, prompt):
        if not await self.begin_expensive_request(update):
            return
        await self.send_menu_text(update.message, "🎨 عم عدّل الصورة، قد يستغرق ذلك قليلًا…")
        try:
            source_bytes = await self.download_telegram_bytes(context.bot, file_id)
            try:
                result = await self.edit_image_request(
                    self.settings.image_model,
                    source_bytes,
                    prompt,
                )
            except APIStatusError as exc:
                fallback = self.settings.image_fallback_model
                if (
                    getattr(exc, "status_code", None) in (403, 404)
                    and fallback
                    and fallback != self.settings.image_model
                ):
                    log.warning(
                        "Image edit model %s unavailable (status=%s); retrying with %s",
                        self.settings.image_model,
                        exc.status_code,
                        fallback,
                    )
                    result = await self.edit_image_request(fallback, source_bytes, prompt)
                else:
                    raise

            edited_bytes = base64.b64decode(result.data[0].b64_json)
            edited = BytesIO(edited_bytes)
            edited.name = "edited.png"
            await update.message.reply_photo(
                photo=edited,
                caption="✅ تم تعديل الصورة.",
                reply_markup=ReplyKeyboardRemove(),
                read_timeout=120,
                write_timeout=120,
                connect_timeout=30,
                pool_timeout=30,
            )
            state = self.session(update.effective_user.id)
            state["mode"] = "chat"
            state["pending_image_file_id"] = None
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            await self.api_error(update.message, exc)
        except TelegramError:
            await self.send_menu_text(update.message, "تعذّر تنزيل الصورة من Telegram.")

    async def edit_image_request(self, model, image_bytes, prompt):
        source = BytesIO(image_bytes)
        source.name = "source.jpg"
        return await self.client.images.edit(
            model=model,
            image=source,
            prompt=prompt,
            size="1024x1024",
            quality=self.settings.image_quality,
            output_format="png",
            n=1,
        )

    async def photo(self, update, context):
        if not await self.require_access(update):
            return
        state = self.session(update.effective_user.id)
        if state["mode"] in ("image", "image_edit"):
            file_id = update.message.photo[-1].file_id
            prompt = (update.message.caption or "").strip()
            if prompt:
                await self.edit_image(update, context, file_id, prompt)
            else:
                state["mode"] = "image_edit"
                state["pending_image_file_id"] = file_id
                await self.send_menu_text(
                    update.message,
                    "✅ وصلت الصورة. اكتب الآن التعديل المطلوب، مثل: غيّر الخلفية إلى شاطئ.",
                )
            return
        if not await self.begin_expensive_request(update):
            return
        await self.send_menu_text(update.message, "🖼 عم حلّل الصورة وأستخرج النص…")
        try:
            data = await self.download_telegram_bytes(
                context.bot,
                update.message.photo[-1].file_id,
            )
            encoded = base64.b64encode(data).decode("ascii")
            instruction = (
                "استخرج كل النص الظاهر في الصورة بدقة مع الحفاظ على ترتيب الأسطر قدر الإمكان. "
                "بعده أضف وصفًا عربيًا موجزًا لمحتوى الصورة. إذا لم يوجد نص، قل ذلك ثم صف الصورة."
            )
            response = await self.client.responses.create(
                model=self.settings.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": instruction},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{encoded}",
                                "detail": "high",
                            },
                        ],
                    }
                ],
                max_output_tokens=self.settings.max_output,
                store=False,
            )
            await self.send_menu_text(update.message, response.output_text.strip())
            self.session(update.effective_user.id)["mode"] = "chat"
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            await self.api_error(update.message, exc)
        except TelegramError:
            await self.send_menu_text(update.message, "تعذّر تنزيل الصورة من Telegram.")

    async def audio(self, update, context):
        if not await self.begin_expensive_request(update):
            return
        media = update.message.voice or update.message.audio
        if media.file_size and media.file_size > self.settings.max_audio_mb * 1024 * 1024:
            await self.send_menu_text(
                update.message, f"الملف أكبر من الحد المسموح ({self.settings.max_audio_mb} MB)."
            )
            return
        if media.duration and media.duration > self.settings.max_audio_seconds:
            await self.send_menu_text(update.message, "المقطع الصوتي أطول من الحد المسموح.")
            return

        await self.send_menu_text(update.message, "🎙 عم حوّل الصوت إلى نص…")
        try:
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source"
                target = Path(directory) / "audio.mp3"
                telegram_file = await context.bot.get_file(media.file_id)
                await telegram_file.download_to_drive(custom_path=source)
                await asyncio.to_thread(convert_to_mp3, source, target)
                transcript = await self.transcribe_file(target)
            await self.send_menu_text(update.message, "📝 التفريغ:\n\n" + transcript)
            self.session(update.effective_user.id)["mode"] = "chat"
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            await self.api_error(update.message, exc)
        except (TelegramError, subprocess.SubprocessError, OSError):
            await self.send_menu_text(update.message, "تعذّر معالجة الملف الصوتي. جرّب MP3 أو رسالة صوتية أقصر.")

    async def transcribe_file(self, path):
        with path.open("rb") as audio_file:
            result = await self.client.audio.transcriptions.create(
                model=self.settings.transcription_model,
                file=audio_file,
            )
        return result.text.strip()

    async def summarize_youtube(self, update, url):
        if not is_youtube_url(url):
            await self.send_menu_text(update.message, "أرسل رابط YouTube صحيحًا من youtube.com أو youtu.be.")
            return
        if not await self.begin_expensive_request(update):
            return
        await self.send_menu_text(update.message, "▶️ عم نزّل الصوت وأفرّغه ثم ألخّصه… قد يستغرق عدة دقائق.")
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = await asyncio.to_thread(
                    download_youtube_audio,
                    url,
                    Path(directory),
                    self.settings.max_youtube_seconds,
                )
                if path.stat().st_size > 24 * 1024 * 1024:
                    raise ValueError("audio_too_large")
                transcript = await self.transcribe_file(path)
            response = await self.client.responses.create(
                model=self.settings.model,
                instructions=(
                    "لخّص تفريغ الفيديو بالعربية. اذكر الفكرة الرئيسية ثم أهم النقاط "
                    "والنتائج أو النصائح. لا تضف معلومات غير موجودة في التفريغ."
                ),
                input=transcript,
                max_output_tokens=self.settings.max_output,
                store=False,
            )
            await self.send_menu_text(update.message, "📌 ملخص الفيديو:\n\n" + response.output_text.strip())
            self.session(update.effective_user.id)["mode"] = "chat"
        except ValueError as exc:
            messages = {
                "video_too_long": "الفيديو أطول من الحد المسموح.",
                "live_video": "لا يمكن تلخيص البث المباشر.",
                "audio_too_large": "صوت الفيديو أكبر من حد خدمة التفريغ.",
            }
            await self.send_menu_text(update.message, messages.get(str(exc), "الرابط غير صالح أو الفيديو غير متاح."))
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            await self.api_error(update.message, exc)
        except Exception as exc:
            log.warning("YouTube processing failed: %s", type(exc).__name__)
            await self.send_menu_text(
                update.message,
                "تعذّر جلب الفيديو. تأكد أنه عام وغير مقيّد، ثم جرّب مجددًا.",
            )

    async def unsupported(self, update, context):
        if not await self.require_access(update):
            return
        await self.send_menu_text(
            update.message,
            "هذا النوع غير مدعوم حاليًا. أرسل نصًا أو صورة أو رسالة/ملفًا صوتيًا.",
        )

    async def api_error(self, message, exc):
        if isinstance(exc, RateLimitError):
            text = "تعذّر الطلب: راجع رصيد OpenAI وحدود الاستخدام."
        elif isinstance(exc, APIConnectionError):
            text = "تعذّر الاتصال بـOpenAI. حاول لاحقًا."
        else:
            status = getattr(exc, "status_code", None)
            request_id = getattr(exc, "request_id", None)
            log.warning("OpenAI API status=%s request_id=%s", status or "unknown", request_id)
            if status == 401:
                text = "تعذّر الطلب: مفتاح OpenAI غير صالح. راجع OPENAI_API_KEY."
            elif status == 403:
                text = (
                    "تعذّر الطلب: حساب OpenAI لا يملك صلاحية هذه الميزة. "
                    "أكمل Organization Verification من منصة OpenAI."
                )
            elif status == 404:
                text = (
                    "تعذّر الطلب: النموذج غير متاح لمشروع OpenAI الحالي. "
                    "راجع اسم النموذج أو استخدم المحرك الاحتياطي."
                )
            elif status == 400:
                text = "تعذّر الطلب: أحد إعدادات النموذج غير مدعوم. راجع متغيرات Railway."
            else:
                text = "تعذّر طلب OpenAI مؤقتًا. راجع Deploy Logs في Railway."
        await self.send_menu_text(message, text)

    async def post_init(self, application):
        await application.bot.set_my_commands(
            [
                BotCommand("start", "فتح القائمة"),
                BotCommand("menu", "إظهار لوحة الأزرار"),
                BotCommand("chat", "وضع المحادثة"),
                BotCommand("image", "إنشاء صورة"),
                BotCommand("youtube", "تلخيص فيديو YouTube"),
                BotCommand("reset", "مسح المحادثة"),
                BotCommand("id", "رقم حسابك"),
                BotCommand("help", "عرض المساعدة"),
            ]
        )

    async def shutdown(self, application):
        await self.client.close()


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Handler error: %s", type(context.error).__name__)
    if isinstance(context.error, NetworkError):
        # Telegram may accept a media upload even when the client times out
        # before receiving the confirmation. Avoid reporting a false failure.
        return
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("حدث خطأ مؤقت. حاول لاحقًا.")
        except TelegramError:
            pass


def build_application(settings):
    client = AsyncOpenAI(api_key=settings.openai_key, timeout=120.0, max_retries=0)
    bot = ChatBot(settings, client)
    app = (
        Application.builder().token(settings.telegram_token)
        .concurrent_updates(False).post_init(bot.post_init)
        .post_shutdown(bot.shutdown).build()
    )
    private = filters.ChatType.PRIVATE
    commands = (
        ("start", bot.start), ("menu", bot.menu_command),
        ("chat", bot.chat_command),
        ("image", bot.image_command), ("youtube", bot.youtube_command),
        ("reset", bot.reset), ("id", bot.identify), ("help", bot.help_command),
    )
    for name, handler in commands:
        app.add_handler(CommandHandler(name, handler, filters=private))
    app.add_handler(MessageHandler(private & filters.Text(MENU_BUTTONS), bot.menu_button))
    app.add_handler(MessageHandler(private & filters.PHOTO, bot.photo))
    app.add_handler(MessageHandler(private & (filters.VOICE | filters.AUDIO), bot.audio))
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, bot.text))
    app.add_handler(MessageHandler(private, bot.unsupported))
    app.add_error_handler(on_error)
    return app


def main():
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "telegram", "yt_dlp"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        log.error("Configuration: %s", exc)
        raise SystemExit(1) from None
    try:
        build_application(settings).run_polling(
            allowed_updates=["message"], drop_pending_updates=True, bootstrap_retries=3
        )
    except Exception as exc:
        log.error("Startup/runtime failure: %s", type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
