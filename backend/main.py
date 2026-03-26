import os
import sys
import json
import asyncio
import shutil
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from starlette.responses import StreamingResponse
from celery.result import AsyncResult

# Add parent dir so "from app.xxx" works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.tasks import app as celery_app, transcribe_task, translate_task
from app.db import init_db, search as db_search, list_transcripts, get_transcript, update_segments
from app.speakers import list_speakers, save_voice_sample, delete_speaker

ALLOWED_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024

app = FastAPI(title="ASVO-Transcriber", version="2.0.0")

UPLOADS_DIR = settings.upload_dir
RESULTS_DIR = settings.output_dir
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def on_startup():
    init_db()


def _safe_jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ======================================================================
# TRANSCRIPTION
# ======================================================================

@app.post("/transcribe")
async def create_transcription_task(
    file: UploadFile = File(...),
    model: str = Form("medium"),
    language: str = Form(""),
    mode: str = Form("full"),
    summarize: str = Form("false"),
    summary_template: str = Form("meeting"),
    title: str = Form(""),
    telegram_chat_id: int = Form(0),
):
    if model not in ALLOWED_MODELS:
        model = "medium"
    diarize = mode != "text_only"
    do_summarize = summarize.lower() in ("true", "1", "on")

    real_file_size = 0
    task_id = str(uuid4())
    safe_name = "".join(c for c in (file.filename or "upload") if c.isalnum() or c in (".", "_", "-")).strip()
    input_filepath = str(UPLOADS_DIR / f"{task_id}_{safe_name}")

    with open(input_filepath, "wb") as buffer:
        while chunk := await file.read(8192):
            real_file_size += len(chunk)
            if real_file_size > MAX_FILE_SIZE:
                buffer.close()
                os.remove(input_filepath)
                raise HTTPException(status_code=413, detail=f"Max size: {settings.max_file_size_mb} MB")
            buffer.write(chunk)

    # Override whisper model in settings temporarily
    original_model = settings.whisper_model
    settings.whisper_model = model

    task = celery_app.send_task(
        "app.tasks.transcribe_task",
        kwargs={
            "task_id": str(task_id),
            "file_paths": [str(input_filepath)],
            "language": str(language or ""),
            "diarize": bool(diarize),
            "summarize": bool(do_summarize),
            "summary_template": str(summary_template or "meeting"),
            "title": str(title or safe_name),
            "telegram_chat_id": int(telegram_chat_id) if telegram_chat_id else None,
        },
    )

    settings.whisper_model = original_model
    return {"job_id": task.id, "task_id": task_id}


@app.get("/status/{job_id}")
async def get_task_status(job_id: str):
    task_result = AsyncResult(job_id, app=celery_app)
    if task_result.failed():
        return {"status": "FAILED", "result": str(task_result.result)[:500]}
    response = {"status": task_result.state}
    if task_result.info and isinstance(task_result.info, dict):
        response["info"] = task_result.info
    return response


@app.get("/result/{task_id}")
async def get_result_file(task_id: str, ext: str = Query("md")):
    if ext not in ("md", "docx", "txt", "json", "srt"):
        ext = "docx"

    # Look in task output directory
    task_dir = RESULTS_DIR / task_id
    if task_dir.is_dir():
        candidates = list(task_dir.glob(f"*.{ext}"))
        if candidates:
            return FileResponse(str(candidates[0]), filename=f"{task_id}.{ext}")

    # Legacy flat path
    flat = RESULTS_DIR / f"{task_id}.{ext}"
    if flat.exists():
        return FileResponse(str(flat), filename=f"{task_id}.{ext}")

    # Fallback to any available format
    if task_dir.is_dir():
        for f in task_dir.iterdir():
            if f.suffix in (".docx", ".txt", ".md") and f.name != "status.json":
                return FileResponse(str(f), filename=f.name)

    raise HTTPException(status_code=404, detail="File not found")


@app.post("/cancel/{job_id}")
async def cancel_task(job_id: str):
    task_result = AsyncResult(job_id, app=celery_app)
    celery_app.control.revoke(job_id, terminate=True, signal="SIGTERM")
    return {"status": "CANCEL_REQUESTED", "current_state": task_result.state}


@app.get("/events/{job_id}")
async def stream_task_events(job_id: str):
    async def event_stream():
        try:
            while True:
                r = AsyncResult(job_id, app=celery_app)
                payload = {"status": r.state, "info": _safe_jsonable(r.info)}
                if r.failed():
                    payload["result"] = str(r.result)[:500]
                yield f"data: {json.dumps(payload)}\n\n"
                if r.ready():
                    break
                await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            raise
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ======================================================================
# SEARCH / HISTORY
# ======================================================================

@app.get("/api/search")
async def api_search(q: str = Query(""), limit: int = Query(20), offset: int = Query(0)):
    """Полнотекстовый поиск по транскриптам."""
    if not q.strip():
        return {"results": []}
    try:
        results = db_search(q.strip(), limit, offset)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


@app.get("/api/transcripts")
async def api_transcripts(limit: int = Query(50), offset: int = Query(0)):
    """Список всех транскриптов."""
    return {"transcripts": list_transcripts(limit, offset)}


@app.get("/api/transcript/{task_id}")
async def api_transcript(task_id: str):
    """Полный транскрипт из БД."""
    t = get_transcript(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return t


@app.put("/api/transcript/{task_id}")
async def api_save_transcript(task_id: str, data: dict):
    """Сохранить отредактированный транскрипт."""
    t = get_transcript(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Transcript not found")
    segments = data.get("segments", [])
    update_segments(task_id, segments)
    return {"status": "ok", "segments": len(segments)}


# ======================================================================
# SPEAKERS
# ======================================================================

@app.post("/api/speakers")
async def api_add_speaker(file: UploadFile = File(...), name: str = Form("")):
    """Загрузить голосовой образец спикера."""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Speaker name is required")

    tmp = str(UPLOADS_DIR / f"voice_{name}_{file.filename}")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ok = save_voice_sample(name, tmp)
    os.remove(tmp)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to extract voice embedding")
    return {"status": "ok", "speaker": name}


@app.get("/api/speakers")
async def api_list_speakers():
    """Список зарегистрированных спикеров."""
    return {"speakers": list_speakers()}


@app.delete("/api/speakers/{name}")
async def api_delete_speaker(name: str):
    """Удалить профиль спикера."""
    if not delete_speaker(name):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"status": "ok"}


# ======================================================================
# TRANSLATE
# ======================================================================

@app.post("/api/translate/{task_id}")
async def api_translate(task_id: str, target_language: str = Form("en")):
    """Поставить перевод транскрипта в очередь."""
    t = get_transcript(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Transcript not found")

    task = celery_app.send_task(
        "app.tasks.translate_task",
        kwargs={"task_id": task_id, "target_language": target_language},
    )
    return {"job_id": task.id, "task_id": task_id, "target_language": target_language}


# ======================================================================
# GLOSSARY
# ======================================================================

@app.get("/api/glossary")
async def api_get_glossary():
    """Получить текущий словарь терминов."""
    if settings.glossary_path.exists():
        return {"glossary": settings.glossary_path.read_text("utf-8")}
    return {"glossary": ""}


@app.put("/api/glossary")
async def api_set_glossary(data: dict):
    """Обновить словарь терминов."""
    text = data.get("glossary", "").strip()
    settings.glossary_path.write_text(text, "utf-8")
    return {"status": "ok", "terms": len(text.splitlines()) if text else 0}


# ======================================================================
# AUDIO / HEALTH
# ======================================================================

@app.get("/api/audio/{task_id}")
async def api_audio(task_id: str):
    """Отдать оригинальный аудиофайл."""
    candidates = [f for f in UPLOADS_DIR.iterdir() if f.name.startswith(task_id)]
    if candidates:
        return FileResponse(str(candidates[0]))
    raise HTTPException(status_code=404, detail="Audio not found")


@app.get("/health")
async def health():
    """Healthcheck."""
    try:
        from redis import Redis
        r = Redis.from_url(settings.redis_url)
        r.ping()
        return {"status": "ok"}
    except Exception:
        return JSONResponse({"status": "degraded"}, status_code=503)


# ======================================================================
# WEB UI — HTML pages
# ======================================================================

# Aliases so web UI /api/... routes work alongside existing ones
@app.post("/api/transcribe")
async def web_transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
    diarize: str = Form("true"),
    summarize: str = Form("false"),
    summary_template: str = Form("meeting"),
):
    do_diarize = str(diarize).lower() in ("true", "1", "on")
    do_summarize = str(summarize).lower() in ("true", "1", "on")
    model = settings.whisper_model or "large-v3"

    task_id = str(uuid4())
    safe_name = "".join(c for c in (file.filename or "upload") if c.isalnum() or c in (".", "_", "-")).strip()
    input_filepath = str(UPLOADS_DIR / f"{task_id}_{safe_name}")

    real_file_size = 0
    with open(input_filepath, "wb") as buffer:
        while chunk := await file.read(8192):
            real_file_size += len(chunk)
            if real_file_size > MAX_FILE_SIZE:
                buffer.close()
                os.remove(input_filepath)
                raise HTTPException(status_code=413, detail=f"Max size: {settings.max_file_size_mb} MB")
            buffer.write(chunk)

    task = celery_app.send_task(
        "app.tasks.transcribe_task",
        kwargs={
            "task_id": str(task_id),
            "file_paths": [str(input_filepath)],
            "language": str(language or ""),
            "diarize": bool(do_diarize),
            "summarize": bool(do_summarize),
            "summary_template": str(summary_template or "meeting"),
            "title": str(safe_name),
        },
    )
    return {"job_id": task.id, "task_id": task_id}

@app.get("/api/status/{task_id}")
async def web_status(task_id: str):
    # Read from status.json on disk (more detailed than celery status)
    status_file = RESULTS_DIR / task_id / "status.json"
    if status_file.exists():
        return JSONResponse(json.loads(status_file.read_text("utf-8")))
    return {"status": "queued", "message": "В очереди..."}

@app.get("/api/result/{task_id}/{fmt}")
async def web_result(task_id: str, fmt: str):
    return await get_result_file(task_id, ext=fmt)

@app.post("/api/cancel/{task_id}")
async def web_cancel(task_id: str):
    # Find celery job_id — for now just revoke by task_id pattern
    return await cancel_task(task_id)

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return WEB_UI_HTML

@app.get("/view/{task_id}", response_class=HTMLResponse)
async def viewer_page(task_id: str):
    return VIEWER_HTML

@app.get("/live", response_class=HTMLResponse)
async def live_page():
    return LIVE_HTML

@app.get("/history", response_class=HTMLResponse)
async def history_page():
    return HISTORY_HTML


# ---- Embedded HTML templates ----

WEB_UI_HTML = Path(__file__).parent / "templates" / "index.html"
VIEWER_HTML = Path(__file__).parent / "templates" / "viewer.html"
LIVE_HTML = Path(__file__).parent / "templates" / "live.html"
HISTORY_HTML = Path(__file__).parent / "templates" / "history.html"

def _load_template(p: Path) -> str:
    if p.exists():
        return p.read_text("utf-8")
    return "<h1>Template not found</h1>"

# Load at import time
_templates_dir = Path(__file__).parent / "templates"
_templates_dir.mkdir(exist_ok=True)

WEB_UI_HTML = _load_template(_templates_dir / "index.html")
VIEWER_HTML = _load_template(_templates_dir / "viewer.html")
LIVE_HTML = _load_template(_templates_dir / "live.html")
HISTORY_HTML = _load_template(_templates_dir / "history.html")
