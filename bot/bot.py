import os
import json
import asyncio
import logging
import tempfile
import html
import re
import traceback
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.error import BadRequest

# ======================================================================
# CONFIG
# ======================================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
ADMIN_TELEGRAM_ID = os.getenv("ADMIN_TELEGRAM_ID")
ADMIN_TELEGRAM_ID = (
    int(ADMIN_TELEGRAM_ID)
    if ADMIN_TELEGRAM_ID and ADMIN_TELEGRAM_ID.isdigit()
    else None
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты — ассистент транскрибации и техподдержки. "
    "Отвечай по делу, на человеческом русском. "
    "Если чего-то не знаешь — говори честно, не выдумывай.",
)

DATA_DIR = os.getenv("BOT_DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
USERS_FILE = os.path.join(DATA_DIR, "users.json")

if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN not found in env")

STATUS_MESSAGES = {
    "CONVERTING": "Шаг 1/5: Конвертирую файл…",
    "DIARIZING": "Шаг 2/5: Разделяю речь по дикторам…",
    "NO_SPEECH_FOUND": "✅ В файле не найдено речи.",
    "SLICING": "Шаг 3/5: Нарезаю аудио…",
    "TRANSCRIBING": "Шаг 4/5: Распознаю речь…",
    "DOCUMENTING": "Шаг 5/5: Собираю итоговый отчёт…",
}

HELP_MESSAGE = (
    "<b>Как пользоваться</b>\n"
    "1) Отправьте файл (аудио/видео/голосовое).\n"
    "2) Выберите режим, язык, модель и формат.\n"
    "3) Нажмите «🚀 Отправить». Во время обработки можно отменить.\n\n"
    "<b>Модели</b>: S / M / L — Small / Medium / Large.\n\n"
    "<b>Команды:</b>\n"
    "/settings — текущие настройки\n"
    "/lang ru|en|auto — язык\n"
    "/diarize on|off — диаризация\n"
    "/summarize on|off — суммаризация LLM\n"
    "/search &lt;запрос&gt; — поиск по архиву\n"
    "/history — последние транскрипты\n"
    "/translate en|de|fr — перевод\n"
    "/addspeaker &lt;имя&gt; — голосовой профиль\n"
    "/speakers — список спикеров\n"
    "/removespeaker &lt;имя&gt; — удалить профиль\n"
    "/admin — панель управления"
)

# ======================================================================
# LOGGER
# ======================================================================

logger = logging.getLogger("bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)

# ======================================================================
# LLM HELPERS (Ollama)
# ======================================================================


@asynccontextmanager
async def ollama_stream(messages, model=None):
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "stream": True,
    }
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            yield resp.aiter_lines()


async def ask_llm(chat_history: list) -> str:
    full_answer: list[str] = []
    async with ollama_stream(chat_history) as lines:
        async for line in lines:
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("done"):
                break
            delta = (
                data.get("message", {}).get("content")
                or data.get("response")
                or ""
            )
            if delta:
                full_answer.append(delta)
    return "".join(full_answer).strip()


# ======================================================================
# USERS (JSON file)
# ======================================================================
# status: 0 = pending, 1 = approved, 2 = admin


def setup_users_file():
    if not os.path.exists(USERS_FILE):
        users: Dict[str, Any] = {}
        if ADMIN_TELEGRAM_ID:
            users[str(ADMIN_TELEGRAM_ID)] = {"status": 2, "username": "admin"}
            logger.info("Created users.json, added admin %s", ADMIN_TELEGRAM_ID)
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)


def load_users() -> Dict[str, Any]:
    if not os.path.exists(USERS_FILE):
        setup_users_file()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(data: Dict[str, Any]):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_auth(user_id: int) -> Optional[int]:
    try:
        users = load_users()
        return users.get(str(user_id), {}).get("status")
    except Exception:
        logger.exception("users.json read error, recreating")
        setup_users_file()
        return None


def is_admin(user_id: int) -> bool:
    return check_auth(user_id) == 2


def get_user_prefs(user_id: int) -> dict:
    users = load_users()
    return dict(users.get(str(user_id), {}).get("prefs", {}))


def save_user_prefs(user_id: int, prefs: dict):
    users = load_users()
    uid = str(user_id)
    if uid in users:
        users[uid]["prefs"] = prefs
        save_users(users)


# ======================================================================
# KEYBOARDS
# ======================================================================


def main_menu_kb(is_admin_flag: bool = False) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("📝 Транскрибация", callback_data="menu_transcribe")],
        [
            InlineKeyboardButton("📚 История", callback_data="menu_history"),
            InlineKeyboardButton("🔍 Поиск", callback_data="menu_search"),
        ],
        [
            InlineKeyboardButton("🎤 Спикеры", callback_data="menu_speakers"),
            InlineKeyboardButton("⚙️ Настройки", callback_data="menu_settings"),
        ],
        [
            InlineKeyboardButton("🤖 Чат с LLM", callback_data="menu_llm"),
            InlineKeyboardButton("ℹ️ Помощь", callback_data="menu_help"),
        ],
    ]
    if is_admin_flag:
        rows.append(
            [InlineKeyboardButton("👑 Админ", callback_data="menu_admin")]
        )
    return InlineKeyboardMarkup(rows)


def main_reply_kb(user_id: Optional[int] = None) -> ReplyKeyboardMarkup:
    rows = [["📝 Транскрибация"], ["🏠 Меню", "ℹ️ Помощь"]]
    if user_id and is_admin(user_id):
        rows.append(["👑 Админ"])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def options_kb(ud: Dict[str, Any]) -> InlineKeyboardMarkup:
    def m(key, val):
        return "✅ " if ud.get(key) == val else ""

    rows = [
        [
            InlineKeyboardButton(
                f"{m('mode','full')}🎤 Диаризация", callback_data="opt:mode:full"
            ),
            InlineKeyboardButton(
                f"{m('mode','text_only')}📄 Без дикторов",
                callback_data="opt:mode:text_only",
            ),
        ],
        [
            InlineKeyboardButton(
                f"{m('language','ru')}🇷🇺", callback_data="opt:language:ru"
            ),
            InlineKeyboardButton(
                f"{m('language','en')}🇬🇧", callback_data="opt:language:en"
            ),
            InlineKeyboardButton(
                f"{m('language','auto')}🌐 Авто", callback_data="opt:language:auto"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{m('summarize','on')}📋 Суммаризация", callback_data="opt:summarize:on"
            ),
            InlineKeyboardButton(
                f"{m('summarize','off')}Без резюме", callback_data="opt:summarize:off"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{m('output_format','docx')}📄 DOCX",
                callback_data="opt:output_format:docx",
            ),
            InlineKeyboardButton(
                f"{m('output_format','txt')}📝 TXT",
                callback_data="opt:output_format:txt",
            ),
            InlineKeyboardButton(
                f"{m('output_format','srt')}🎬 SRT",
                callback_data="opt:output_format:srt",
            ),
        ],
        [
            InlineKeyboardButton(
                f"{m('deliver','file')}📤 Файл", callback_data="opt:deliver:file"
            ),
            InlineKeyboardButton(
                f"{m('deliver','chat')}💬 В чат", callback_data="opt:deliver:chat"
            ),
        ],
        [
            InlineKeyboardButton("🚀 Отправить", callback_data="send_to_backend"),
            InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


def settings_kb(ud: Dict[str, Any]) -> InlineKeyboardMarkup:
    lang = ud.get("language", "ru")
    mode = ud.get("mode", "full")
    summ = ud.get("summarize", "off")
    rows = [
        [
            InlineKeyboardButton(
                f"{'✅ ' if lang == 'ru' else ''}🇷🇺 RU", callback_data="stg:language:ru"
            ),
            InlineKeyboardButton(
                f"{'✅ ' if lang == 'en' else ''}🇬🇧 EN", callback_data="stg:language:en"
            ),
            InlineKeyboardButton(
                f"{'✅ ' if lang == 'auto' else ''}🌐 Авто", callback_data="stg:language:auto"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{'✅ ' if mode == 'full' else ''}🎤 Диаризация", callback_data="stg:mode:full"
            ),
            InlineKeyboardButton(
                f"{'✅ ' if mode == 'text_only' else ''}📄 Без дикторов",
                callback_data="stg:mode:text_only",
            ),
        ],
        [
            InlineKeyboardButton(
                f"{'✅ ' if summ == 'on' else ''}📋 Суммаризация вкл",
                callback_data="stg:summarize:on",
            ),
            InlineKeyboardButton(
                f"{'❌ ' if summ != 'on' else ''}Суммаризация выкл",
                callback_data="stg:summarize:off",
            ),
        ],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")],
    ]
    return InlineKeyboardMarkup(rows)


def result_kb(task_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📄 DOCX", callback_data=f"dl:{task_id}:docx"),
            InlineKeyboardButton("📝 TXT", callback_data=f"dl:{task_id}:txt"),
            InlineKeyboardButton("🎬 SRT", callback_data=f"dl:{task_id}:srt"),
            InlineKeyboardButton("📊 JSON", callback_data=f"dl:{task_id}:json"),
        ],
        [
            InlineKeyboardButton("🇬🇧 Перевод EN", callback_data=f"tr:{task_id}:en"),
            InlineKeyboardButton("🇩🇪 Перевод DE", callback_data=f"tr:{task_id}:de"),
            InlineKeyboardButton("🇫🇷 Перевод FR", callback_data=f"tr:{task_id}:fr"),
        ],
    ])


# ======================================================================
# TEXT UTILS
# ======================================================================


async def send_safe(bot_or_update, chat_id: int, text: str, **kwargs):
    sender = (
        bot_or_update
        if hasattr(bot_or_update, "send_message")
        else bot_or_update.bot
    )
    try:
        return await sender.send_message(
            chat_id, text, parse_mode="HTML", **kwargs
        )
    except BadRequest:
        return await sender.send_message(chat_id, text, **kwargs)


def chunk_text(s: str, limit: int = 4096):
    for i in range(0, len(s), limit):
        yield s[i : i + limit]


# ======================================================================
# MARKDOWN → TELEGRAM HTML CONVERTER
# ======================================================================

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$", re.MULTILINE)
_ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a"}
_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9-]*)(?:\s+[^<>]*?)?>")


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def md_to_telegram_html(md: str) -> str:
    if not md:
        return ""
    t = md.replace("\r\n", "\n").replace("\r", "\n")

    # fenced code → placeholders
    blocks: list[str] = []

    def _fence(m):
        idx = len(blocks)
        blocks.append(f"<pre>{_esc(m.group(1))}</pre>")
        return f"__CODE_BLOCK_{idx}__"

    t = _FENCE_RE.sub(_fence, t)

    # headings
    t = _HEADING_RE.sub(lambda m: f"<b>{_esc(m.group(1))}</b>", t)

    # escape everything (will double-escape inside tags, fix below)
    t = _esc(t)

    # inline code
    t = _INLINE_CODE_RE.sub(lambda m: f"<code>{_esc(m.group(1))}</code>", t)

    # bold / italic / strike
    t = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", t)
    t = _ITALIC_RE.sub(lambda m: f"<i>{m.group(1)}</i>", t)
    t = _STRIKE_RE.sub(lambda m: f"<s>{m.group(1)}</s>", t)

    # links
    t = _LINK_RE.sub(
        lambda m: f'<a href="{_esc(m.group(2))}">{_esc(m.group(1))}</a>', t
    )

    # bullet lists
    t = re.sub(r"(?m)^\s*[-*]\s+", "• ", t)

    # restore code blocks
    for i, block in enumerate(blocks):
        t = t.replace(f"__CODE_BLOCK_{i}__", block)

    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _close_open_tags(fragment: str):
    stack: list[str] = []
    for m in _TAG_RE.finditer(fragment):
        tag = m.group(1).lower()
        if tag not in _ALLOWED_TAGS:
            continue
        is_close = fragment[m.start() + 1] == "/"
        if not is_close:
            stack.append(tag)
        else:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k] == tag:
                    stack.pop(k)
                    break
    closing = "".join(f"</{t}>" for t in reversed(stack))
    return fragment + closing, stack


def split_html_for_telegram(html_text: str, limit: int = 3500) -> list[str]:
    chunks: list[str] = []
    i, n = 0, len(html_text)

    while i < n:
        end = min(i + limit, n)
        if end == n:
            part = html_text[i:end]
            part, _ = _close_open_tags(part)
            chunks.append(part.strip())
            break

        window = html_text[i:end]

        # try to cut at paragraph or tag boundary
        safe_pos = None
        for m in re.finditer(r"(</(?:b|i|u|s|code|pre|a)>|\n\n)", window):
            safe_pos = m.end()
        if safe_pos is None:
            last_space = window.rfind(" ")
            safe_pos = (
                last_space
                if (last_space > 0 and (len(window) - last_space) < 120)
                else len(window)
            )

        part = window[:safe_pos]
        part_balanced, reopen_stack = _close_open_tags(part)
        chunks.append(part_balanced.strip())

        prefix = "".join(f"<{t}>" for t in reopen_stack)
        i += safe_pos
        if i < n and prefix:
            html_text = html_text[:i] + prefix + html_text[i:]
            n = len(html_text)

        while i < n and html_text[i] == "\n":
            i += 1

    return [c for c in chunks if c]


# ======================================================================
# MENU
# ======================================================================


async def show_main_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    edit_message=None,
):
    uid = update.effective_user.id
    kb_inline = main_menu_kb(is_admin(uid))
    kb_reply = main_reply_kb(uid)
    if edit_message:
        try:
            await edit_message.edit_message_text(
                "Выберите действие:", reply_markup=kb_inline
            )
        except (BadRequest, AttributeError):
            try:
                await edit_message.edit_text(
                    "Выберите действие:", reply_markup=kb_inline
                )
            except BadRequest:
                await context.bot.send_message(
                    uid, "Выберите действие:", reply_markup=kb_inline
                )
    else:
        await context.bot.send_message(
            uid, "Клавиатура включена ↓", reply_markup=kb_reply
        )
        await context.bot.send_message(
            uid, "Выберите действие:", reply_markup=kb_inline
        )


# ======================================================================
# LLM MODE
# ======================================================================


async def enter_llm_mode(query, context: ContextTypes.DEFAULT_TYPE):
    uid = query.from_user.id
    ud = context.user_data
    ud["llm_mode"] = True
    ud["llm_history"] = [{"role": "system", "content": SYSTEM_PROMPT}]

    kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]]
    )
    try:
        await query.edit_message_text(
            "🤖 Режим чата с моделью активирован.\n"
            "Спроси меня о чём-нибудь.\n\n"
            "Команда /exit — вернуться в меню.",
            reply_markup=kb,
        )
    except BadRequest:
        await context.bot.send_message(
            uid,
            "🤖 Режим чата активирован. /exit — выход.",
            reply_markup=kb,
        )


# ======================================================================
# COMMANDS
# ======================================================================


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    setup_users_file()
    uid = update.effective_user.id

    auth = check_auth(uid)
    if auth is None:
        users = load_users()
        users[str(uid)] = {
            "status": 0,
            "username": update.effective_user.username
            or update.effective_user.first_name,
        }
        save_users(users)
        await update.message.reply_text(
            "Заявка на доступ принята. Напишите администратору для авторизации."
        )
        if ADMIN_TELEGRAM_ID:
            await context.bot.send_message(
                ADMIN_TELEGRAM_ID,
                f"Запрос доступа: @{update.effective_user.username or update.effective_user.first_name} (ID: {uid})",
            )
        return

    await show_main_menu(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        HELP_MESSAGE,
        parse_mode="HTML",
        reply_markup=main_reply_kb(update.effective_user.id),
    )


# ======================================================================
# TEXT HANDLER (MENU + LLM)
# ======================================================================


async def handle_text_menu(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    uid = update.effective_user.id
    text = (update.message.text or "").strip()
    text_lower = text.lower()
    ud = context.user_data

    # --- Search mode ---
    if ud.get("search_mode"):
        ud.pop("search_mode", None)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{BACKEND_URL}/api/search", params={"q": text, "limit": 10}
                )
            results = r.json().get("results", [])
            if not results:
                await update.message.reply_text(f'🔍 По запросу «{text}» ничего не найдено')
            else:
                lines = [f'🔍 Результаты по «{text}»:\n']
                for hit in results[:10]:
                    ts = int(hit.get("start", 0))
                    m, s = divmod(ts, 60)
                    title = (hit.get("title") or hit.get("task_id", ""))[:30]
                    speaker = hit.get("speaker", "")
                    txt = hit.get("text", "")[:80]
                    lines.append(f"[{m:02d}:{s:02d}] {title}\n   {speaker}: {txt}...")
                await update.message.reply_text("\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"❌ {e}")
        return

    # --- Speaker name input ---
    if ud.get("awaiting_speaker_name"):
        ud.pop("awaiting_speaker_name", None)
        ud["awaiting_voice_sample"] = text
        await update.message.reply_text(
            f'🎤 Жду голосовой образец для «{text}».\n'
            f"Отправь голосовое сообщение или аудиофайл с речью этого человека."
        )
        return

    # --- LLM mode ---
    if ud.get("llm_mode"):
        if text_lower in {"/exit", "выход", "выйти", "🏠 меню", "меню"}:
            ud["llm_mode"] = False
            await show_main_menu(update, context)
            return

        await update.message.chat.send_action(action="typing")
        ud.setdefault(
            "llm_history", [{"role": "system", "content": SYSTEM_PROMPT}]
        )
        ud["llm_history"].append({"role": "user", "content": text})

        try:
            answer = await ask_llm(ud["llm_history"])
        except Exception as e:
            logger.exception("LLM error")
            safe_err = html.escape(str(e))[:500]
            await update.message.reply_text(
                f"⚠️ Ошибка обращения к модели:\n{safe_err}",
                parse_mode="HTML",
            )
            return

        if not answer:
            answer = "…модель не ответила."

        ud["llm_history"].append({"role": "assistant", "content": answer})

        html_text = md_to_telegram_html(answer)
        parts = split_html_for_telegram(html_text, limit=3500)
        for part in parts:
            try:
                await update.message.reply_text(part, parse_mode="HTML")
            except BadRequest:
                await update.message.reply_text(part)
        return

    # --- Menu shortcuts ---
    if text_lower in {"🏠 меню", "меню"}:
        await show_main_menu(update, context)
        return

    if text_lower in {"ℹ️ помощь", "помощь", "/help"}:
        await help_command(update, context)
        return

    if text_lower in {"👑 админ", "админ", "/admin"}:
        if not is_admin(uid):
            await update.message.reply_text("Нет прав.")
            return
        await admin_panel_cmd(update, context)
        return

    if text_lower in {"📝 транскрибация", "транскрибация"}:
        await update.message.reply_text(
            "Отправьте аудио/видео/голосовое. После получения покажу настройки.",
            reply_markup=main_reply_kb(uid),
        )
        return

    if any(k in text_lower for k in ("llm", "чат с llm", "🤖")):
        ud["llm_mode"] = True
        ud["llm_history"] = [{"role": "system", "content": SYSTEM_PROMPT}]
        await update.message.reply_text(
            "🤖 Режим чата активирован. /exit — выход.",
            reply_markup=main_reply_kb(uid),
        )
        return

    await update.message.reply_text(
        "Выбери действие в меню или напиши /help.",
        reply_markup=main_reply_kb(uid),
    )


# ======================================================================
# ADMIN
# ======================================================================


async def admin_panel_cmd(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    users = load_users()
    pending = {uid: u for uid, u in users.items() if u.get("status") == 0}
    if not pending:
        await update.message.reply_text(
            "Нет пользователей, ожидающих авторизации.",
            reply_markup=main_reply_kb(update.effective_user.id),
        )
        return
    keyboard = []
    for uid, info in pending.items():
        uname = info.get("username")
        label = (
            f"✅ Авторизовать @{uname}" if uname else f"✅ Авторизовать ID {uid}"
        )
        keyboard.append(
            [InlineKeyboardButton(label, callback_data=f"approve_{uid}")]
        )
    keyboard.append(
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
    )
    await update.message.reply_text(
        "Пользователи, ожидающие авторизации:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def admin_panel_inline(query, context):
    users = load_users()
    pending = {uid: u for uid, u in users.items() if u.get("status") == 0}
    back = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]]
    )
    if not pending:
        await query.edit_message_text(
            "Нет пользователей, ожидающих авторизации.",
            reply_markup=back,
        )
        return
    keyboard = []
    for uid, info in pending.items():
        uname = info.get("username")
        label = (
            f"✅ Авторизовать @{uname}" if uname else f"✅ Авторизовать ID {uid}"
        )
        keyboard.append(
            [InlineKeyboardButton(label, callback_data=f"approve_{uid}")]
        )
    keyboard.append(
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
    )
    await query.edit_message_text(
        "Пользователи, ожидающие авторизации:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Нет прав.")
        return
    if not context.args:
        await update.message.reply_text("Использование: /approve <telegram_id>")
        return
    uid = context.args[0]
    users = load_users()
    if uid in users:
        users[uid]["status"] = 1
        save_users(users)
        await update.message.reply_text(f"Пользователь {uid} авторизован.")
        try:
            await context.bot.send_message(
                int(uid),
                "✅ Ваш аккаунт активирован! Теперь вы можете отправлять файлы.",
            )
        except Exception:
            pass
    else:
        await update.message.reply_text("Нет такого пользователя.")


async def ban_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Нет прав.")
        return
    if not context.args:
        await update.message.reply_text("Использование: /ban <telegram_id>")
        return
    uid = context.args[0]
    users = load_users()
    if uid in users:
        users[uid]["status"] = 0
        save_users(users)
        await update.message.reply_text(
            f"Пользователь {uid} переведён в статус ожидания."
        )
    else:
        await update.message.reply_text("Нет такого пользователя.")


async def list_pending_cmd(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Нет прав.")
        return
    users = load_users()
    pending = [
        f"{uid} (@{u.get('username')})" if u.get("username") else uid
        for uid, u in users.items()
        if u.get("status") == 0
    ]
    txt = "Ожидают авторизации:\n" + ("\n".join(pending) if pending else "—")
    await update.message.reply_text(txt)


# ======================================================================
# FILE HANDLING
# ======================================================================


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id

    auth = check_auth(uid)
    if auth is None:
        users = load_users()
        users[str(uid)] = {
            "status": 0,
            "username": update.effective_user.username
            or update.effective_user.first_name,
        }
        save_users(users)
        await update.message.reply_text(
            "Заявка принята. Ожидайте авторизации."
        )
        if ADMIN_TELEGRAM_ID:
            await context.bot.send_message(
                ADMIN_TELEGRAM_ID,
                f"Запрос доступа: @{update.effective_user.username or ''} (ID: {uid})",
            )
        return
    if auth == 0:
        await update.message.reply_text(
            "Ваш аккаунт ещё не авторизован. Ожидайте подтверждения."
        )
        return

    file = (
        update.message.document
        or update.message.video
        or update.message.video_note
        or update.message.audio
        or update.message.voice
    )
    if not file:
        await update.message.reply_text("Не удалось распознать файл.")
        return

    ud = context.user_data
    ud["file_id"] = file.file_id

    file_name = getattr(file, "file_name", None)
    if not file_name:
        mime = getattr(file, "mime_type", "application/octet-stream")
        ext = mime.split("/")[-1] if "/" in mime else "dat"
        file_name = f"media_{uid}.{ext}"
    ud["file_name"] = file_name

    # load saved prefs as defaults
    prefs = get_user_prefs(uid)
    ud.setdefault("mode", prefs.get("mode", "full"))
    ud.setdefault("language", prefs.get("language", "ru"))
    ud.setdefault("model", prefs.get("model", "medium"))
    ud.setdefault("output_format", prefs.get("output_format", "docx"))
    ud.setdefault("deliver", prefs.get("deliver", "file"))
    ud.setdefault("summarize", prefs.get("summarize", "on"))

    await update.message.reply_text(
        "⚙️ Настройте параметры и нажмите «Отправить».\n"
        "<b>S/M/L</b> — Small/Medium/Large (скорость ↔ качество).",
        parse_mode="HTML",
        reply_markup=options_kb(ud),
    )


# ======================================================================
# CALLBACK HANDLER
# ======================================================================


async def callback_query_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    query = update.callback_query
    uid = query.from_user.id
    await query.answer()
    data = query.data

    back_kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]]
    )

    if data == "menu_transcribe":
        await query.edit_message_text(
            "📝 Отправьте аудио/видео/голосовое сообщение.\n"
            "Я покажу настройки и начну обработку.",
            reply_markup=back_kb,
        )
        return

    if data == "menu_history":
        await _show_history_inline(query, context)
        return

    if data == "menu_search":
        context.user_data["search_mode"] = True
        await query.edit_message_text(
            "🔍 Напишите поисковый запрос текстом.\n"
            "Например: <i>бюджет на квартал</i>\n\n"
            "Для выхода нажмите Назад.",
            parse_mode="HTML",
            reply_markup=back_kb,
        )
        return

    if data == "menu_speakers":
        await _show_speakers_inline(query, context)
        return

    if data == "menu_settings":
        await query.edit_message_text(
            "⚙️ Настройки транскрибации:",
            reply_markup=settings_kb(context.user_data),
        )
        return

    if data == "menu_llm":
        await enter_llm_mode(query, context)
        return

    if data == "menu_help":
        await query.edit_message_text(
            HELP_MESSAGE, parse_mode="HTML", reply_markup=back_kb,
        )
        return

    if data == "menu_admin":
        if not is_admin(uid):
            await query.edit_message_text("Нет прав.")
            return
        await admin_panel_inline(query, context)
        return

    if data == "back_to_menu":
        context.user_data.pop("llm_mode", None)
        await show_main_menu(update, context, edit_message=query)
        return

    if data.startswith("approve_"):
        if not is_admin(uid):
            await context.bot.send_message(uid, "Нет прав.")
            return
        target_id = data.split("_", 1)[1]
        users = load_users()
        if target_id in users:
            users[target_id]["status"] = 1
            save_users(users)
            try:
                await context.bot.send_message(
                    int(target_id),
                    "✅ Ваш аккаунт активирован! Отправляйте файлы.",
                )
            except Exception:
                pass
            await admin_panel_inline(query, context)
        else:
            await query.edit_message_text("Пользователь не найден.")
        return

    # --- file upload settings (opt:) ---
    if data.startswith("opt:"):
        parts = data.split(":", 2)
        if len(parts) == 3:
            _, param, value = parts
            context.user_data[param] = value
            if param == "deliver" and value == "chat":
                context.user_data["output_format"] = "md"
            _save_prefs(uid, context.user_data)
        try:
            await query.edit_message_reply_markup(
                reply_markup=options_kb(context.user_data)
            )
        except BadRequest:
            pass
        return

    # --- global settings (stg:) ---
    if data.startswith("stg:"):
        parts = data.split(":", 2)
        if len(parts) == 3:
            _, param, value = parts
            context.user_data[param] = value
            _save_prefs(uid, context.user_data)
        try:
            await query.edit_message_reply_markup(
                reply_markup=settings_kb(context.user_data)
            )
        except BadRequest:
            pass
        return

    # --- history item ---
    if data.startswith("hist:"):
        task_id = data.split(":", 1)[1]
        await context.bot.send_message(
            uid,
            f"📋 Транскрипт <code>{task_id[:8]}</code>\n\nВыберите формат:",
            parse_mode="HTML",
            reply_markup=result_kb(task_id),
        )
        return

    # --- add speaker flow ---
    if data == "speaker_add":
        context.user_data["awaiting_speaker_name"] = True
        await query.edit_message_text(
            "🎤 Напишите имя спикера текстом.\nНапример: <i>Иван Петров</i>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("🔙 Назад", callback_data="menu_speakers")]]
            ),
        )
        return

    if data.startswith("speaker_del:"):
        name = data.split(":", 1)[1]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.delete(f"{BACKEND_URL}/api/speakers/{name}")
            await _show_speakers_inline(query, context)
        except Exception as e:
            await context.bot.send_message(uid, f"❌ Ошибка: {e}")
        return

    if data.startswith("cancel_job:"):
        job_id = data.split(":", 1)[1]
        await cancel_job(uid, job_id, context)
        return

    # Download buttons: dl:{task_id}:{ext}
    if data.startswith("dl:"):
        parts = data.split(":")
        if len(parts) == 3:
            _, dl_task_id, ext = parts
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.get(
                        f"{BACKEND_URL}/result/{dl_task_id}",
                        params={"ext": ext},
                    )
                if r.status_code == 200:
                    await context.bot.send_document(
                        uid,
                        document=r.content,
                        filename=f"{dl_task_id}.{ext}",
                    )
                else:
                    await context.bot.send_message(
                        uid, f"Файл .{ext} не найден"
                    )
            except Exception as e:
                await context.bot.send_message(uid, f"❌ Ошибка: {e}")
        return

    # Translate buttons: tr:{task_id}:{lang}
    if data.startswith("tr:"):
        parts = data.split(":")
        if len(parts) == 3:
            _, tr_task_id, lang = parts
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.post(
                        f"{BACKEND_URL}/api/translate/{tr_task_id}",
                        data={"target_language": lang},
                    )
                if r.status_code == 200:
                    await context.bot.send_message(
                        uid, f"🔄 Перевод на {lang} поставлен в очередь"
                    )
                else:
                    await context.bot.send_message(uid, f"❌ {r.text[:200]}")
            except Exception as e:
                await context.bot.send_message(uid, f"❌ Ошибка: {e}")
        return

    if data == "send_to_backend":
        await start_transcription_flow(update, context, from_query=query)
        return


# ======================================================================
# INLINE HELPERS (history, speakers, prefs)
# ======================================================================


def _save_prefs(uid: int, ud: dict):
    save_user_prefs(
        uid,
        {
            k: ud[k]
            for k in ("mode", "language", "model", "output_format", "deliver", "summarize")
            if k in ud
        },
    )


async def _show_history_inline(query, context):
    back_kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]]
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{BACKEND_URL}/api/transcripts", params={"limit": 10})
        transcripts = r.json().get("transcripts", [])
        if not transcripts:
            await query.edit_message_text("📚 Архив пуст", reply_markup=back_kb)
            return
        rows = []
        for t in transcripts:
            title = t.get("title") or t.get("file_name") or t["task_id"][:8]
            dur = f"{t.get('duration', 0) / 60:.0f}м" if t.get("duration") else ""
            spk = len(t.get("speakers", []))
            label = f"📄 {title[:25]} ({dur}, {spk}сп.)"
            rows.append(
                [InlineKeyboardButton(label, callback_data=f"hist:{t['task_id']}")]
            )
        rows.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        await query.edit_message_text(
            "📚 Последние транскрипты:", reply_markup=InlineKeyboardMarkup(rows)
        )
    except Exception as e:
        await query.edit_message_text(f"❌ {e}", reply_markup=back_kb)


async def _show_speakers_inline(query, context):
    back_kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]]
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{BACKEND_URL}/api/speakers")
        speakers = r.json().get("speakers", [])
        rows = []
        for name in speakers:
            rows.append([
                InlineKeyboardButton(f"🎤 {name}", callback_data=f"_noop"),
                InlineKeyboardButton("🗑", callback_data=f"speaker_del:{name}"),
            ])
        rows.append([InlineKeyboardButton("➕ Добавить спикера", callback_data="speaker_add")])
        rows.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        header = "🎤 Спикеры:" if speakers else "📭 Нет спикеров"
        await query.edit_message_text(header, reply_markup=InlineKeyboardMarkup(rows))
    except Exception as e:
        await query.edit_message_text(f"❌ {e}", reply_markup=back_kb)


# ======================================================================
# TRANSCRIPTION CORE
# ======================================================================


async def cancel_job(
    chat_id: int, job_id: str, context: ContextTypes.DEFAULT_TYPE
):
    try:
        timeout = httpx.Timeout(connect=5, read=10, write=10, pool=10)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{BACKEND_URL}/cancel/{job_id}")
        if r.status_code == 200:
            await context.bot.send_message(chat_id, "⛔️ Задача отменена.")
        else:
            await context.bot.send_message(
                chat_id, f"Не удалось отменить (код {r.status_code})."
            )
    except Exception as e:
        logger.exception("Cancel error")
        await context.bot.send_message(chat_id, f"Ошибка отмены: {e}")


async def start_transcription_flow(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    from_query=None,
):
    uid = update.effective_user.id
    ud = context.user_data

    if not ud.get("file_id"):
        msg = "Ошибка: информация о файле утеряна. Отправьте файл заново."
        if from_query:
            await from_query.edit_message_text(msg)
        else:
            await context.bot.send_message(uid, msg)
        return

    status_msg = await context.bot.send_message(uid, "📥 Скачиваю файл…")

    # download file
    try:
        tg_file = await context.bot.get_file(ud["file_id"])
        safe_name = "".join(
            c
            for c in ud.get("file_name", "input.bin")
            if c.isalnum() or c in (" ", ".", "_", "-")
        )
        fd, tmp_path = tempfile.mkstemp(
            prefix="tg_", suffix="_" + os.path.basename(safe_name)
        )
        os.close(fd)
        await tg_file.download_to_drive(custom_path=tmp_path)
    except Exception as e:
        logger.exception("File download failed")
        await status_msg.edit_text(f"❌ Ошибка скачивания файла: {e}")
        return

    # send to backend
    await status_msg.edit_text("📤 Файл принят. Отправляю в сервис…")
    try:
        timeout = httpx.Timeout(connect=15, read=60, write=60, pool=60)
        with open(tmp_path, "rb") as f:
            files = {
                "file": (
                    os.path.basename(safe_name),
                    f,
                    "application/octet-stream",
                )
            }
            form_data = {
                "model": ud.get("model", "medium"),
                "language": ud.get("language", "ru"),
                "mode": ud.get("mode", "full"),
                "summarize": "true" if ud.get("summarize") == "on" else "false",
            }
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    f"{BACKEND_URL}/transcribe",
                    files=files,
                    data=form_data,
                )

        if r.status_code == 503:
            await status_msg.edit_text(f"❌ Сервер перегружен: {r.text}")
            return

        r.raise_for_status()
        resp = r.json()
        job_id = resp.get("job_id")
        task_id = resp.get("task_id")
        if not job_id or not task_id:
            await status_msg.edit_text("❌ Backend не вернул необходимые ID.")
            return
    except httpx.HTTPStatusError as e:
        await status_msg.edit_text(
            f"❌ Ошибка: {e}\n{getattr(e.response, 'text', '')[:500]}"
        )
        return
    except Exception as e:
        logger.exception("Transcription request failed")
        await status_msg.edit_text(f"❌ Непредвиденная ошибка: {e}")
        return
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # show cancel button
    cancel_kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "⛔️ Отменить",
                    callback_data=f"cancel_job:{job_id}",
                )
            ]
        ]
    )
    try:
        await status_msg.edit_text(
            "⏳ Задача отправлена. Жду статус…", reply_markup=cancel_kb
        )
    except BadRequest:
        pass

    # listen for status via SSE, fallback to polling
    try:
        await listen_status_sse(uid, status_msg, job_id, task_id, context)
    except Exception as e:
        logger.warning("SSE failed (%s), switching to polling", e)
        await listen_status_polling(uid, status_msg, job_id, task_id, context)


async def listen_status_sse(
    chat_id: int,
    status_msg,
    job_id: str,
    task_id: str,
    context: ContextTypes.DEFAULT_TYPE,
):
    url = f"{BACKEND_URL}/events/{job_id}"
    timeout = httpx.Timeout(connect=10, read=None, write=10, pool=10)
    cancel_kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "⛔️ Отменить",
                    callback_data=f"cancel_job:{job_id}",
                )
            ]
        ]
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            last_txt = ""
            async for line in resp.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue

                payload = line[len("data:") :].strip()
                try:
                    info = json.loads(payload)
                except Exception:
                    continue

                status = info.get("status", "")
                meta = info.get("info")
                if isinstance(meta, dict):
                    phase = meta.get("status") or meta.get("phase")
                else:
                    phase = None

                new_txt = f"⏳ Задача в очереди… ({status})"

                if status == "PROGRESS" and phase:
                    new_txt = STATUS_MESSAGES.get(
                        phase, f"Обработка… ({phase})"
                    )
                elif status in ("FAILURE", "FAILED"):
                    err = info.get("result") or info.get("error") or "Неизвестно"
                    safe_err = html.escape(str(err)[:1000])
                    try:
                        await status_msg.edit_text(
                            f"❌ Задача провалена.\n\n"
                            f"Причина: <code>{safe_err}</code>",
                            parse_mode="HTML",
                        )
                    except BadRequest:
                        pass
                    return
                elif status == "SUCCESS":
                    await deliver_result(
                        chat_id, task_id, context, status_msg
                    )
                    return

                if new_txt != last_txt:
                    try:
                        await status_msg.edit_text(
                            new_txt,
                            parse_mode="HTML",
                            reply_markup=cancel_kb,
                        )
                    except BadRequest:
                        pass
                    last_txt = new_txt


async def listen_status_polling(
    chat_id: int,
    status_msg,
    job_id: str,
    task_id: str,
    context: ContextTypes.DEFAULT_TYPE,
):
    timeout = httpx.Timeout(connect=10, read=15, write=10, pool=10)
    cancel_kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "⛔️ Отменить",
                    callback_data=f"cancel_job:{job_id}",
                )
            ]
        ]
    )
    last_txt = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(360):
            await asyncio.sleep(3 if i < 10 else 5)
            try:
                r = await client.get(f"{BACKEND_URL}/status/{job_id}")
            except Exception:
                continue
            if r.status_code != 200:
                continue

            data = r.json()
            status = data.get("status")
            meta = data.get("info")

            new_txt = f"⏳ Задача в очереди… ({status})"

            if status == "PROGRESS" and isinstance(meta, dict):
                phase = meta.get("status") or meta.get("phase")
                if phase:
                    new_txt = STATUS_MESSAGES.get(
                        phase, f"Обработка… ({phase})"
                    )
            elif status in ("FAILURE", "FAILED"):
                err = (
                    data.get("result")
                    or (meta if isinstance(meta, str) else "Неизвестно")
                )
                safe_err = html.escape(str(err)[:1000])
                try:
                    await status_msg.edit_text(
                        f"❌ Задача провалена.\n\n"
                        f"Причина: <code>{safe_err}</code>",
                        parse_mode="HTML",
                    )
                except BadRequest:
                    pass
                return
            elif status == "SUCCESS":
                await deliver_result(chat_id, task_id, context, status_msg)
                return

            if new_txt != last_txt:
                try:
                    await status_msg.edit_text(
                        new_txt, reply_markup=cancel_kb
                    )
                except BadRequest:
                    pass
                last_txt = new_txt

        try:
            await status_msg.edit_text("❌ Таймаут ожидания статуса.")
        except BadRequest:
            pass


async def deliver_result(
    chat_id: int,
    task_id: str,
    context: ContextTypes.DEFAULT_TYPE,
    status_msg,
):
    ud = context.user_data
    deliver = ud.get("deliver", "file")
    output_format = ud.get("output_format", "md")

    timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(
            f"{BACKEND_URL}/result/{task_id}", params={"ext": output_format}
        )
        if r.status_code != 200:
            # fallback to md
            r = await client.get(
                f"{BACKEND_URL}/result/{task_id}", params={"ext": "md"}
            )
        if r.status_code != 200:
            try:
                await status_msg.edit_text(
                    f"Не удалось скачать результат (код {r.status_code})"
                )
            except BadRequest:
                pass
            return
        content = r.content

    # Send the primary file
    if deliver == "chat":
        try:
            text = content.decode("utf-8", errors="replace")
            if not text.strip():
                raise ValueError("empty")
            html_text = md_to_telegram_html(text)
            parts = split_html_for_telegram(html_text, limit=3500)
            for part in parts:
                try:
                    await context.bot.send_message(
                        chat_id, part, parse_mode="HTML"
                    )
                except BadRequest:
                    await context.bot.send_message(chat_id, part)
        except Exception:
            await context.bot.send_document(
                chat_id, document=content, filename=f"{task_id}.txt"
            )
    else:
        await context.bot.send_document(
            chat_id, document=content, filename=f"{task_id}.{output_format}"
        )

    # Always send download/translate buttons
    try:
        await status_msg.edit_text(
            "✅ Готово! Скачайте в нужном формате:",
            reply_markup=result_kb(task_id),
        )
    except BadRequest:
        await context.bot.send_message(
            chat_id,
            "✅ Готово! Скачайте в нужном формате:",
            reply_markup=result_kb(task_id),
        )

    context.user_data["last_task_id"] = task_id
    ud.pop("file_id", None)
    ud.pop("file_name", None)


# ======================================================================
# NEW COMMANDS: search, history, translate, speakers, settings
# ======================================================================


async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Поиск по архиву транскриптов."""
    if not context.args:
        await update.message.reply_text(
            "Использование: /search <запрос>\nПример: /search бюджет на следующий квартал"
        )
        return
    query = " ".join(context.args)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{BACKEND_URL}/api/search", params={"q": query, "limit": 10}
            )
        if r.status_code != 200:
            await update.message.reply_text("Ошибка поиска.")
            return
        results = r.json().get("results", [])
        if not results:
            await update.message.reply_text(f'🔍 По запросу «{query}» ничего не найдено')
            return
        lines = [f'🔍 Результаты по «{query}»:\n']
        for hit in results[:10]:
            ts = int(hit.get("start", 0))
            m, s = divmod(ts, 60)
            title = hit.get("title", hit.get("task_id", ""))[:30]
            speaker = hit.get("speaker", "")
            text = hit.get("text", "")[:100]
            lines.append(f"[{m:02d}:{s:02d}] {title}]\n   {speaker}: {text}...")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка поиска: {e}")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Список последних транскриптов."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{BACKEND_URL}/api/transcripts", params={"limit": 15}
            )
        if r.status_code != 200:
            await update.message.reply_text("Ошибка загрузки истории.")
            return
        transcripts = r.json().get("transcripts", [])
        if not transcripts:
            await update.message.reply_text("📚 Архив пуст")
            return
        lines = ["📚 Последние транскрипты:\n"]
        for t in transcripts:
            dur = f"{t.get('duration', 0):.0f}мин"
            spk = t.get("speakers", [])
            title = t.get("title") or t.get("file_name") or t["task_id"][:8]
            tid = t["task_id"][:8]
            created = t.get("created_at", "")[:10]
            lines.append(
                f"• {title} ({dur}, {len(spk)} спик.)\n  ID: {tid} | {created}"
            )
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def cmd_translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Перевести последний транскрипт."""
    if not context.args:
        await update.message.reply_text(
            "Использование: /translate en\nКоды: en, ru, uk, de, fr, es, zh, ja"
        )
        return
    lang = context.args[0].lower().strip()
    task_id = context.user_data.get("last_task_id")
    if not task_id:
        await update.message.reply_text(
            "Нет транскриптов для перевода. Сначала отправь аудио."
        )
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{BACKEND_URL}/api/translate/{task_id}",
                data={"target_language": lang},
            )
        if r.status_code == 200:
            await update.message.reply_text(
                f"🔄 Перевод задачи {task_id[:8]} на {lang} поставлен в очередь"
            )
        else:
            await update.message.reply_text(f"❌ Ошибка: {r.text[:200]}")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def cmd_addspeaker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начать добавление голосового профиля спикера."""
    if not context.args:
        await update.message.reply_text(
            "Использование: /addspeaker Иван\n"
            "Затем отправь голосовое сообщение или аудио этого человека."
        )
        return
    name = " ".join(context.args)
    context.user_data["awaiting_voice_sample"] = name
    await update.message.reply_text(
        f'🎤 Жду голосовой образец для «{name}».\n'
        f"Отправь голосовое сообщение или аудиофайл с речью этого человека."
    )


async def cmd_speakers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Список зарегистрированных спикеров."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{BACKEND_URL}/api/speakers")
        speakers = r.json().get("speakers", [])
        if not speakers:
            await update.message.reply_text(
                "📭 Нет зарегистрированных спикеров.\nДобавь: /addspeaker <имя>"
            )
            return
        lines = ["🎤 Зарегистрированные спикеры:\n"]
        for name in speakers:
            lines.append(f"  • {name}")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def cmd_removespeaker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Удалить профиль спикера."""
    if not context.args:
        await update.message.reply_text("Использование: /removespeaker Иван")
        return
    name = " ".join(context.args)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.delete(f"{BACKEND_URL}/api/speakers/{name}")
        if r.status_code == 200:
            await update.message.reply_text(f'✅ Профиль «{name}» удалён')
        else:
            await update.message.reply_text(f'❌ Профиль «{name}» не найден')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def cmd_settings_show(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать текущие настройки пользователя."""
    ud = context.user_data
    lang = ud.get("language", "auto")
    diar = ud.get("mode", "full")
    summ = ud.get("summarize", "off")
    await update.message.reply_text(
        f"⚙️ Текущие настройки:\n\n"
        f"Язык: {lang}\n"
        f"Диаризация: {'✅ вкл' if diar == 'full' else '❌ выкл'}\n"
        f"Суммаризация: {'✅ вкл' if summ == 'on' else '❌ выкл'}\n\n"
        f"Изменить: /lang, /diarize, /summarize"
    )


async def cmd_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Установить язык транскрибации."""
    if not context.args:
        lang = context.user_data.get("language", "auto")
        await update.message.reply_text(
            f"Текущий язык: {lang}\nИспользование: /lang ru, /lang en, /lang auto"
        )
        return
    lang = context.args[0].lower().strip()
    context.user_data["language"] = lang
    display = "auto-определение" if lang == "auto" else lang
    await update.message.reply_text(f"✅ Язык: {display}")


async def cmd_diarize_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Включить/выключить диаризацию."""
    if not context.args:
        mode = context.user_data.get("mode", "full")
        await update.message.reply_text(
            f"Диаризация: {'✅ вкл' if mode == 'full' else '❌ выкл'}\n"
            f"Использование: /diarize on или /diarize off"
        )
        return
    val = context.args[0].lower()
    context.user_data["mode"] = "full" if val in ("on", "1", "yes", "вкл") else "text_only"
    await update.message.reply_text(
        f"✅ Диаризация: {'вкл' if context.user_data['mode'] == 'full' else 'выкл'}"
    )


async def cmd_summarize_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Включить/выключить суммаризацию."""
    if not context.args:
        val = context.user_data.get("summarize", "off")
        await update.message.reply_text(
            f"Суммаризация: {'✅ вкл' if val == 'on' else '❌ выкл'}\n"
            f"Использование: /summarize on или /summarize off"
        )
        return
    val = context.args[0].lower()
    context.user_data["summarize"] = "on" if val in ("on", "1", "yes", "вкл") else "off"
    await update.message.reply_text(
        f"✅ Суммаризация: {'вкл' if context.user_data['summarize'] == 'on' else 'выкл'}"
    )


# ======================================================================
# ERROR HANDLER
# ======================================================================


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception(
        "Unhandled error:\n%s", traceback.format_exc()
    )


# ======================================================================
# MAIN
# ======================================================================


def main():
    setup_users_file()
    logger.info("Bot starting. BACKEND_URL=%s", BACKEND_URL)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Core commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("admin", admin_panel_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CommandHandler("ban", ban_cmd))
    app.add_handler(CommandHandler("list_pending", list_pending_cmd))

    # New feature commands
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("translate", cmd_translate))
    app.add_handler(CommandHandler("addspeaker", cmd_addspeaker))
    app.add_handler(CommandHandler("speakers", cmd_speakers))
    app.add_handler(CommandHandler("removespeaker", cmd_removespeaker))
    app.add_handler(CommandHandler("settings", cmd_settings_show))
    app.add_handler(CommandHandler("lang", cmd_lang))
    app.add_handler(CommandHandler("diarize", cmd_diarize_toggle))
    app.add_handler(CommandHandler("summarize", cmd_summarize_toggle))

    # Text (menu + LLM)
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_menu)
    )

    # Files (audio/video/documents/voice)
    app.add_handler(
        MessageHandler(
            filters.AUDIO
            | filters.VIDEO
            | filters.VIDEO_NOTE
            | filters.Document.ALL
            | filters.VOICE,
            handle_file,
        )
    )

    # Inline buttons
    app.add_handler(CallbackQueryHandler(callback_query_handler))

    app.add_error_handler(_on_error)

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
