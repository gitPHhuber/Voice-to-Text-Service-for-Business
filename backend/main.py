import os
import shutil
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from celery.result import AsyncResult
from tasks import app as celery_app, transcribe_task

celery_app.autodiscover_tasks(['tasks'])

MAX_FILE_SIZE = 500 * 1024 * 1024

app = FastAPI()

UPLOADS_DIR = "/app/data/uploads"
RESULTS_DIR = "/app/data/results"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/transcribe")
async def create_transcription_task(file: UploadFile = File(...)):
    real_file_size = 0
    task_id = str(uuid4())
    input_filepath = os.path.join(UPLOADS_DIR, f"{task_id}_{file.filename}")

    with open(input_filepath, "wb") as buffer:
        while chunk := await file.read(8192):
            real_file_size += len(chunk)
            if real_file_size > MAX_FILE_SIZE:
                buffer.close()
                os.remove(input_filepath)
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл слишком большой. Максимальный размер: {MAX_FILE_SIZE // 1024 // 1024} MB."
                )
            buffer.write(chunk)
    
    
    task = celery_app.send_task('tasks.transcribe_task', args=[input_filepath, task_id])
    return {"job_id": task.id, "task_id": task_id}

@app.get("/status/{job_id}")
async def get_task_status(job_id: str):
    task_result = AsyncResult(job_id, app=celery_app)
    if task_result.failed():
        return {"status": "FAILED", "result": str(task_result.result)[:500]}
    return {"status": task_result.state}

@app.get("/result/{task_id}")
async def get_result_file(task_id: str):
    file_path = os.path.join(RESULTS_DIR, f"{task_id}.md")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or not ready yet")
    return FileResponse(file_path, filename=f"{task_id}.md")
