import os
import subprocess
import shutil
import logging
import glob
from celery import Celery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Celery('tasks', broker=os.environ.get('REDIS_URL', 'redis://redis:6379/0'), backend=os.environ.get('REDIS_URL', 'redis://redis:6379/0'))

@app.task
def transcribe_task(input_filepath, task_id):
    work_dir = f"/app/data/tmp/{task_id}"
    results_dir = "/app/data/results"
    processing_scripts_dir = "/app/processing_scripts"

    converted_wav = os.path.join(work_dir, "out.wav")
    timings_list = os.path.join(work_dir, "timings.list")
    final_md = os.path.join(results_dir, f"{task_id}.md")
    
    error_occurred = False
    try:
        os.makedirs(work_dir, exist_ok=True)
        
        subprocess.run(['ffmpeg', '-i', input_filepath, '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', converted_wav], check=True)
        
        with open(timings_list, "w") as f:
            subprocess.run(['python', f'{processing_scripts_dir}/diarizer.py', converted_wav], stdout=f, check=True)
        
        subprocess.run(['python', f'{processing_scripts_dir}/slicer.py', converted_wav, timings_list], cwd=work_dir, check=True)
        
        clip_files = sorted(glob.glob(os.path.join(work_dir, 'clip-*.wav')))
        if not clip_files:
            raise RuntimeError("Slicer не создал ни одного аудио-клипа.")
        
        whisper_command = [
            'whisper', '--device=cuda', '--language=ru', '--model=medium',
            '--output_dir=.', '--output_format=txt'
        ] + clip_files
        subprocess.run(whisper_command, cwd=work_dir, check=True)
        
        subprocess.run([f'{processing_scripts_dir}/documenting.sh', final_md], cwd=work_dir, check=True)
            
        return "SUCCESS"
    except Exception as e:
        error_occurred = True
        logger.exception("!!! ОШИБКА В ЗАДАЧЕ")
        if isinstance(e, subprocess.CalledProcessError):
            logger.error(f"!!! STDERR: {e.stderr}")
        raise
    finally:
        if error_occurred:
            logger.error(f"Задача провалена. Временные файлы для отладки сохранены в: {work_dir}")
        else:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
        
        if os.path.exists(input_filepath):
            os.remove(input_filepath)
