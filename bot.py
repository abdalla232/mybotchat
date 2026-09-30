"""Small, single-worker Telegram chatbot. Run: python bot.py"""
import base64
import binascii
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from telegram import BotCommand, Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

log = logging.getLogger("ai_bot")


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_key: str
    model: str
    allowed_ids: frozenset[int]
    allow_all: bool
    max_output: int
    history_turns: int
    max_input: int
    cooldown: int
    system_prompt: str
    image_model: str = "gpt-image-1.5"
    image_quality: str = "low"
    image_cooldown: int = 60

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
            ids = frozenset(int(v.strip()) for v in os.getenv("ALLOWED_USER_IDS", "").split(",") if v.strip())
            if any(v <= 0 for v in ids):
                raise ValueError
        except ValueError:
            raise ValueError("ALLOWED_USER_IDS must contain positive numeric IDs separated by commas") from None
        allow_all = os.getenv("ALLOW_ALL_USERS", "false").strip().lower()
        if allow_all not in ("true", "false"):
            raise ValueError("ALLOW_ALL_USERS must be true or false")
        image_quality = os.getenv("IMAGE_QUALITY", "low").strip().lower()
        if image_quality not in ("low", "medium", "high"):
            raise ValueError("IMAGE_QUALITY must be low, medium, or high")
        return cls(
            required("TELEGRAM_BOT_TOKEN"), required("OPENAI_API_KEY"),
            os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
            ids, allow_all == "true",
            integer("MAX_OUTPUT_TOKENS", 600, 16, 4096),
            integer("HISTORY_TURNS", 4, 0, 20),
            integer("MAX_INPUT_CHARS", 3000, 1, 10000),
            integer("COOLDOWN_SECONDS", 3, 0, 3600),
            os.getenv("SYSTEM_PROMPT", "أنت مساعد ودود. أجب بالعربية ما لم يطلب المستخدم لغة أخرى. اجعل إجاباتك واضحة ومختصرة."),
            os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5").strip() or "gpt-image-1.5",
            image_quality,
            integer("IMAGE_COOLDOWN_SECONDS", 60, 0, 3600),
        )


def split_text(text, limit=4000):
    """Conservative UTF-16 bound also handles long emoji-only answers."""
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
            self.sessions[user_id] = {"history": [], "last_request": float("-inf"), "last_image": float("-inf")}
        self.sessions.move_to_end(user_id)
        return self.sessions[user_id]

    async def start(self, update, context):
        await update.message.reply_text(
            "أهلًا! أرسل رسالة نصية لنتحدث.\n"
            "/image وصف الصورة — إنشاء صورة بالذكاء الاصطناعي\n"
            "/reset — بدء محادثة جديدة\n/id — معرفة رقم حسابك\n"
            "نص رسائلك والسياق الحديث يُرسلان إلى OpenAI للإجابة. "
            "الذاكرة مؤقتة وتُمسح عند إعادة تشغيل البوت."
        )

    async def identify(self, update, context):
        await update.message.reply_text(f"رقم حسابك: {update.effective_user.id}")

    async def reset(self, update, context):
        if self.allowed(update.effective_user.id):
            # Keep the rate limit when clearing conversation history.
            self.session(update.effective_user.id)["history"] = []
            await update.message.reply_text("تم مسح سياق المحادثة. لنبدأ من جديد!")

    async def text(self, update, context):
        user_id = update.effective_user.id
        message = update.message
        cfg = self.settings
        if not self.allowed(user_id):
            await message.reply_text("حسابك غير مفعّل لهذا البوت. استخدم /id وأضف الرقم إلى ALLOWED_USER_IDS في إعدادات البوت.")
            return
        text = message.text.strip()
        if not text or len(text) > cfg.max_input:
            await message.reply_text(f"أرسل نصًا بين 1 و{cfg.max_input} حرف.")
            return
        state = self.session(user_id)
        now = time.monotonic()
        if now - state["last_request"] < cfg.cooldown:
            await message.reply_text("انتظر قليلًا قبل إرسال رسالة أخرى.")
            return
        state["last_request"] = now
        messages = [*state["history"], {"role": "user", "content": text}]
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        except TelegramError:
            pass  # A failed typing indicator must not prevent the answer.
        try:
            response = await self.client.responses.create(
                model=cfg.model, instructions=cfg.system_prompt,
                input=messages, max_output_tokens=cfg.max_output, store=False,
            )
            answer = (response.output_text or "").strip()
            if not answer:
                await message.reply_text("لم يصل جواب نصي. جرّب سؤالًا أقصر أو راجع إعداد النموذج وحد الرد.")
                return
            displayed = answer
            if response.status == "incomplete":
                displayed += "\n\n[توقف الرد عند الحد المحدد لطوله. يمكنك طلب المتابعة.]"
            for part in split_text(displayed):
                await message.reply_text(part, parse_mode=None)
            # Commit history only after successful generation and delivery.
            state["history"] = (messages + [{"role": "assistant", "content": answer}])[-2 * cfg.history_turns:] if cfg.history_turns else []
            if response.usage:
                log.info("OpenAI usage: input=%s output=%s", response.usage.input_tokens, response.usage.output_tokens)
        except RateLimitError:
            log.warning("OpenAI RateLimitError")
            await message.reply_text("تعذّر إكمال الطلب: قد يكون رصيد API غير كافٍ أو حد الطلبات وصل. راجع Usage وBilling في OpenAI.")
        except APIConnectionError:
            log.warning("OpenAI connection/timeout error")
            await message.reply_text("تعذّر الاتصال بخدمة الذكاء الاصطناعي. حاول لاحقًا.")
        except APIStatusError as exc:
            log.warning("OpenAI API status=%s", exc.status_code)
            await message.reply_text("تعذّر إكمال الطلب. على صاحب البوت مراجعة مفتاح OpenAI واسم النموذج وصلاحياته في الإعدادات.")

    async def image(self, update, context):
        user_id = update.effective_user.id
        message = update.message
        cfg = self.settings
        if not self.allowed(user_id):
            await message.reply_text("حسابك غير مفعّل لهذا البوت. استخدم /id وأضف الرقم إلى ALLOWED_USER_IDS في إعدادات البوت.")
            return
        parts = message.text.split(maxsplit=1)
        prompt = parts[1].strip() if len(parts) == 2 else ""
        if not prompt:
            await message.reply_text("اكتب وصف الصورة بعد الأمر، مثل:\n/image مدينة مستقبلية وقت الغروب")
            return
        if len(prompt) > cfg.max_input:
            await message.reply_text(f"أرسل وصفًا لا يتجاوز {cfg.max_input} حرف.")
            return
        state = self.session(user_id)
        now = time.monotonic()
        if now - state["last_request"] < cfg.cooldown or now - state["last_image"] < cfg.image_cooldown:
            await message.reply_text("انتظر قليلًا قبل طلب صورة أخرى.")
            return
        await message.reply_text("جارٍ إنشاء الصورة، قد يستغرق ذلك بضع دقائق...")
        state["last_request"] = state["last_image"] = now
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
        except TelegramError:
            pass
        try:
            # Images need a longer timeout; disable retries to avoid duplicate paid requests.
            response = await self.client.images.generate(
                model=cfg.image_model, prompt=prompt, n=1,
                size="1024x1024", quality=cfg.image_quality,
                output_format="jpeg", timeout=300.0,
            )
            encoded = response.data[0].b64_json if response.data else None
            if not encoded:
                await message.reply_text("لم تصل صورة من الخدمة. جرّب وصفًا آخر.")
                return
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
                if not image_bytes:
                    raise ValueError("Empty image")
            except (binascii.Error, ValueError):
                log.warning("Invalid image payload")
                await message.reply_text("وصلت صورة غير صالحة من الخدمة. حاول لاحقًا.")
                return
            with BytesIO(image_bytes) as photo:
                await message.reply_photo(photo=photo, filename="generated-image.jpg", write_timeout=60, read_timeout=60)
        except RateLimitError:
            log.warning("OpenAI image RateLimitError")
            await message.reply_text("تعذّر إنشاء الصورة: راجع رصيد API وحدود الطلبات في OpenAI.")
        except APIConnectionError:
            log.warning("OpenAI image connection/timeout error")
            await message.reply_text("تعذّر الاتصال بخدمة الصور أو انتهت مهلة الانتظار. حاول لاحقًا.")
        except APIStatusError as exc:
            log.warning("OpenAI image API status=%s", exc.status_code)
            if exc.code in ("moderation_blocked", "content_policy_violation"):
                await message.reply_text("تعذّر إنشاء الصورة بهذا الوصف. جرّب وصفًا مختلفًا.")
            else:
                await message.reply_text("تعذّر إنشاء الصورة. راجع صلاحية توليد الصور وموديل OPENAI_IMAGE_MODEL في إعدادات OpenAI.")
        except TelegramError:
            log.warning("Telegram image delivery failed")
            await message.reply_text("تم إنشاء الصورة لكن تعذّر إرسالها إلى تيليغرام. قد يكون الطلب احتُسب؛ لم نكرر التوليد تلقائيًا.")

    async def unsupported(self, update, context):
        await update.message.reply_text("أرسل سؤالًا نصيًا أو استخدم /image مع وصف لإنشاء صورة. تعديل الصور المرفقة والصوت غير مدعومين.")

    async def post_init(self, application):
        await application.bot.set_my_commands([
            BotCommand("start", "بدء المحادثة"), BotCommand("help", "طريقة الاستخدام"),
            BotCommand("reset", "مسح سياق المحادثة"), BotCommand("id", "رقم حسابك"),
            BotCommand("image", "إنشاء صورة من وصف"),
        ])
        log.info("Bot initialized; starting polling")

    async def shutdown(self, application):
        await self.client.close()


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    # Avoid exception bodies/URLs: Telegram URLs contain the bot token.
    log.error("Handler error: %s", type(context.error).__name__)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("حدث خطأ مؤقت. حاول لاحقًا.")
        except TelegramError:
            pass


def build_application(settings):
    client = AsyncOpenAI(api_key=settings.openai_key, timeout=45.0, max_retries=0)
    bot = ChatBot(settings, client)
    app = (Application.builder().token(settings.telegram_token)
           .concurrent_updates(False).post_init(bot.post_init)
           .post_shutdown(bot.shutdown).build())
    private = filters.ChatType.PRIVATE
    for name, handler in (("start", bot.start), ("help", bot.start), ("id", bot.identify), ("reset", bot.reset), ("image", bot.image)):
        app.add_handler(CommandHandler(name, handler, filters=private))
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, bot.text))
    app.add_handler(MessageHandler(private, bot.unsupported))
    app.add_error_handler(on_error)
    return app


def main():
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "telegram"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        log.error("Configuration: %s", exc)
        raise SystemExit(1) from None
    try:
        app = build_application(settings)
        app.run_polling(allowed_updates=["message"], drop_pending_updates=True, bootstrap_retries=3)
    except Exception as exc:
        log.error("Startup/runtime failure: %s; check credentials and network", type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
