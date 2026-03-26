"""
Celery задачи для транскрибации и перевода.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from celery import Celery

from app.config import settings
from app.engine import process, translate_segments, Segment, TranscriptionResult
from app.export import to_docx, to_txt, to_json, to_srt
from app import db, storage

logger = logging.getLogger(__name__)

app = Celery(
    "transcriber",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
app.conf.broker_connection_retry_on_startup = True
app.conf.result_expires = 3600 * 12
app.conf.worker_prefetch_multiplier = 1
app.conf.task_acks_late = True
app.conf.worker_max_tasks_per_child = 10


def _update_status(
    path: Path,
    status: str,
    message: str = "",
    data: dict | None = None,
    progress: int = 0,
    step: str = "",
):
    payload = {
        "status": status,
        "message": message,
        "updated_at": datetime.now().isoformat(),
    }
    if progress:
        payload["progress"] = progress
    if step:
        payload["step"] = step
    if data:
        payload["data"] = data
    path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")


@app.task(bind=True, name="app.tasks.transcribe_task")
def transcribe_task(
    self,
    task_id: str,
    file_paths: list[str],
    language: str = "",
    diarize: bool = True,
    summarize: bool = False,
    num_speakers: int = 0,
    title: str = "Транскрипт встречи",
    output_name: str = "",
    summary_template: str = "meeting",
    telegram_chat_id: int | None = None,
    webhook_url: str = "",
):
    """
    Основная задача транскрибации.
    Вызывается через Celery, выполняется в worker-контейнере.
    """
    task_dir = settings.output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    status_path = task_dir / "status.json"

    _update_status(status_path, "processing", "Запуск...", step="init")

    def _on_step(step, percent, message):
        _update_status(status_path, "processing", message, step=step, progress=percent)
        self.update_state(state="PROGRESS", meta={"status": step.upper(), "step": step, "progress": percent, "message": message})

    try:
        result = process(
            audio_paths=file_paths,
            language=language,
            diarize_flag=diarize,
            summarize_flag=summarize,
            summary_template=summary_template,
            num_speakers=num_speakers,
            on_step=_on_step,
        )

        # Export all formats
        name = output_name or task_id
        base = str(task_dir / name)
        to_docx(result, base + ".docx", title)
        to_txt(result, base + ".txt")
        to_json(result, base + ".json")
        to_srt(result, base + ".srt")

        # Save to database
        file_name = Path(file_paths[0]).name if file_paths else ""
        db.save_transcript(task_id, result, title=title, file_name=file_name)

        # Upload to S3
        storage.upload_results(task_id, str(task_dir))

        # Status done
        done_data = {
            "duration": result.duration,
            "processing_time": result.processing_time,
            "speakers": len(result.speakers),
            "segments": len(result.segments),
            "topics": len(result.topics),
            "summary": result.summary[:500] if result.summary else "",
        }
        _update_status(status_path, "done", "Готово", data=done_data)

        # Notifications
        if telegram_chat_id:
            try:
                docx_path = base + ".docx"
                txt_path = base + ".txt"
                _notify_telegram(telegram_chat_id, task_id, result, docx_path, txt_path)
            except Exception as e:
                logger.error("Telegram notify failed: %s", e)

        if webhook_url:
            _notify_webhook(webhook_url, task_id, "done", result)

        return {"status": "done", "task_id": task_id}

    except Exception as e:
        logger.exception("Task %s failed", task_id)
        _update_status(status_path, "error", str(e)[:500])
        if telegram_chat_id:
            _notify_telegram_error(telegram_chat_id, str(e)[:300])
        if webhook_url:
            _notify_webhook(webhook_url, task_id, "error", error=str(e))
        raise


@app.task(bind=True, name="app.tasks.translate_task")
def translate_task(self, task_id: str, target_language: str):
    """Задача перевода существующего транскрипта."""
    task_dir = settings.output_dir / task_id
    status_path = task_dir / "status.json"

    transcript = db.get_transcript(task_id)
    if not transcript:
        raise ValueError(f"Transcript {task_id} not found")

    _update_status(status_path, "translating", f"Перевод на {target_language}...", step="translate")

    segments = [
        Segment(start=s["start"], end=s.get("end", 0), text=s["text"], speaker=s.get("speaker", ""))
        for s in transcript.get("segments", [])
    ]

    translated = translate_segments(segments, target_language)

    # Re-export with translated text
    result = TranscriptionResult(
        segments=translated,
        speakers=transcript.get("speakers", []),
        duration=transcript.get("duration", 0),
    )

    name = f"{task_id}_{target_language}"
    base = str(task_dir / name)
    to_txt(result, base + ".txt")
    to_docx(result, base + ".docx", f"Перевод ({target_language})")

    _update_status(status_path, "done", f"{target_language} готов", step="translated")
    logger.info("Translation %s → %s done", task_id, target_language)
    return {"status": "done", "task_id": task_id, "language": target_language}


def _notify_telegram(chat_id: int, task_id: str, result, docx_path: str, txt_path: str):
    """Отправляет результат в Telegram."""
    import httpx
    bot_token = settings.telegram_bot_token
    if not bot_token:
        return

    base_url = f"https://api.telegram.org/bot{bot_token}"

    text = (
        f"✅ Транскрипт готов!\n\n"
        f"⏱ Длительность: {result.duration:.0f} мин\n"
        f"🌐 Язык: {result.language}\n"
        f"👥 Спикеров: {len(result.speakers)}\n"
        f"📝 Сегментов: {len(result.segments)}\n"
        f"⚡ Обработка: {result.processing_time:.0f}с"
    )
    if result.summary:
        text += f"\n📋 Резюме:\n{result.summary[:500]}"

    keyboard = {
        "inline_keyboard": [
            [
                {"text": "📄 DOCX", "callback_data": f"dl:{task_id}:docx"},
                {"text": "📝 TXT", "callback_data": f"dl:{task_id}:txt"},
                {"text": "🎬 SRT", "callback_data": f"dl:{task_id}:srt"},
            ],
            [
                {"text": "🇬🇧 Перевод EN", "callback_data": f"tr:{task_id}:en"},
                {"text": "🇩🇪 Перевод DE", "callback_data": f"tr:{task_id}:de"},
            ],
        ]
    }

    # Send message
    httpx.post(
        f"{base_url}/sendMessage",
        json={"chat_id": chat_id, "text": text, "reply_markup": keyboard},
        timeout=30,
    )

    # Send DOCX
    if Path(docx_path).exists():
        with open(docx_path, "rb") as f:
            httpx.post(
                f"{base_url}/sendDocument",
                data={"chat_id": str(chat_id)},
                files={"document": (Path(docx_path).name, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                timeout=60,
            )

    # Send TXT
    if Path(txt_path).exists():
        with open(txt_path, "rb") as f:
            httpx.post(
                f"{base_url}/sendDocument",
                data={"chat_id": str(chat_id)},
                files={"document": (Path(txt_path).name, f, "text/plain")},
                timeout=60,
            )

    # Send SRT if exists
    srt_path = str(Path(txt_path).with_suffix(".srt"))
    if Path(srt_path).exists():
        with open(srt_path, "rb") as f:
            httpx.post(
                f"{base_url}/sendDocument",
                data={"chat_id": str(chat_id)},
                files={"document": (Path(srt_path).name, f, "text/plain")},
                timeout=60,
            )


def _notify_webhook(url: str, task_id: str, status: str, result=None, error: str = ""):
    """POST на webhook URL при завершении задачи."""
    import httpx
    payload = {"task_id": task_id, "status": status}
    if error:
        payload["error"] = error
    if result:
        payload.update({
            "duration": result.duration,
            "processing_time": result.processing_time,
            "speakers": len(result.speakers),
            "segments": len(result.segments),
            "language": result.language,
            "summary": result.summary[:500] if result.summary else "",
        })
    try:
        r = httpx.post(url, json=payload, timeout=30)
        logger.info("Webhook %s: %s", url, r.status_code)
    except Exception as e:
        logger.error("Webhook failed: %s", e)


def _notify_telegram_error(chat_id: int, error_msg: str):
    import httpx
    bot_token = settings.telegram_bot_token
    if not bot_token:
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": f"❌ Ошибка транскрибации:\n{error_msg}"},
            timeout=30,
        )
    except Exception:
        pass
