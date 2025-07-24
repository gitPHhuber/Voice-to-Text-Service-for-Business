import os
import sys
import asyncio
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
BACKEND_URL = "http://10.20.205.15:8001"

HELP_MESSAGE = """
Привет! Я бот для транскрибации аудио и видео.

Просто отправь мне аудио, видео или голосовое сообщение.

**Выбор модели:**
Вы можете выбрать "скорость" и "качество" распознавания, указав модель в подписи к файлу.
Доступные модели:
- `tiny` (самая быстрая, низкое качество)
- `base`
- `small`
- `medium` (по умолчанию, хороший баланс)
- `large` (самая медленная, лучшее качество)

**Пример:** Отправьте аудиофайл и в подписи к нему напишите `large`.
"""

ALLOWED_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

STATUS_MESSAGES = {
    "CONVERTING": "Шаг 1/5: Конвертирую файл...",
    "DIARIZING": "Шаг 2/5: Разделяю речь по дикторам...",
    "NO_SPEECH_FOUND": "✅ В файле не найдено речи.",
    "SLICING": "Шаг 3/5: Нарезаю аудио...",
    "TRANSCRIBING": "Шаг 4/5: Распознаю речь...",
    "DOCUMENTING": "Шаг 5/5: Собираю итоговый отчет...",
}

if not TELEGRAM_BOT_TOKEN:
    print("Ошибка: Не найден TELEGRAM_BOT_TOKEN в .env файле.", file=sys.stderr)
    sys.exit(1)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь /help, чтобы увидеть все возможности.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_MESSAGE, parse_mode='Markdown')

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_to_download = update.message.document or update.message.video or update.message.audio or update.message.voice
    if not file_to_download: return
    
    model_choice = "medium"
    if update.message.caption:
        caption_text = update.message.caption.strip().lower()
        if caption_text in ALLOWED_MODELS:
            model_choice = caption_text
    
    status_message = await update.message.reply_text(f'📥 Файл принят. Модель: `{model_choice}`. Отправляю в сервис...')
    
    downloaded_file = await file_to_download.get_file()
    file_bytes = await downloaded_file.download_as_bytearray()
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            files = {'file': (getattr(file_to_download, 'file_name', 'voice_message.ogg'), bytes(file_bytes), "application/octet-stream")}
            data = {'model': model_choice}
            
            response = await client.post(f"{BACKEND_URL}/transcribe", files=files, data=data)
            response.raise_for_status()
            
            data = response.json()
            job_id = data.get("job_id")
            task_id = data.get("task_id")

            if not job_id or not task_id:
                await status_message.edit_text("❌ Backend не вернул необходимые ID.")
                return

            last_status_text = ""
            for _ in range(360):
                await asyncio.sleep(5)
                status_response = await client.get(f"{BACKEND_URL}/status/{job_id}")
                status_data = status_response.json()
                
                current_status = status_data.get("status")
                current_info = status_data.get("info")
                
                new_status_text = f"⏳ Задача в очереди... ({current_status})"
                if current_status == "PROGRESS" and current_info and 'status' in current_info:
                    new_status_text = STATUS_MESSAGES.get(current_info['status'], "Обработка...")
                
                if new_status_text != last_status_text:
                    await status_message.edit_text(new_status_text)
                    last_status_text = new_status_text

                if current_status == "SUCCESS":
                    await status_message.edit_text("✅ Готово! Скачиваю и отправляю результат...")
                    result_resp = await client.get(f"{BACKEND_URL}/result/{task_id}")
                    if result_resp.status_code == 200:
                        await update.message.reply_document(document=result_resp.content, filename=f"{task_id}.md")
                    else:
                        await update.message.reply_text(f"Не удалось скачать файл результата. Статус: {result_resp.status_code}")
                    return
                
                elif current_status == "FAILED":
                    await status_message.edit_text(f"❌ Задача провалена на стороне backend. ID: `{job_id}`")
                    return

            await status_message.edit_text("❌ Таймаут ожидания. Задача обрабатывалась слишком долго.")

    except httpx.RequestError as e:
        await status_message.edit_text(f"❌ Ошибка подключения к backend: {e}")
    except Exception as e:
        await status_message.edit_text(f"❌ Произошла непредвиденная ошибка: {e}")

def main():
    print("Бот-клиент с улучшенной логикой запущен...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VIDEO | filters.Document.ALL | filters.VOICE, handle_file))
    application.run_polling()

if __name__ == '__main__':
    main()
