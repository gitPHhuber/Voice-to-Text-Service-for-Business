import os
import subprocess
import shutil
import logging
import glob
from celery import Celery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Celery('tasks', broker=os.environ.get('REDIS_URL', 'redis://redis:6379/0'), backend=os.environ.get('REDIS_URL', 'redis://redis:6379/0'))


@app.task(bind=True)
def transcribe_task(self, input_filepath, task_id, model="medium", language="ru", mode="full"):
    work_dir = f"/app/data/tmp/{task_id}"
    results_dir = "/app/data/results"
    processing_scripts_dir = "/app/processing_scripts"

    converted_wav = os.path.join(work_dir, "out.wav")
    timings_list = os.path.join(work_dir, "timings.list")
    final_md = os.path.join(results_dir, f"{task_id}.md")
    final_docx = os.path.join(results_dir, f"{task_id}.docx")

    def update_progress(status):
        self.update_state(state='PROGRESS', meta={'status': status})
        logger.info(f"[{task_id}] {status}")

    error_occurred = False
    try:
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Step 1: Convert to WAV
        update_progress("CONVERTING")
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_filepath,
             '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
             converted_wav],
            check=True, capture_output=True,
        )

        # Step 2: Diarization (only in full mode)
        if mode == "full":
            update_progress("DIARIZING")
            with open(timings_list, "w") as f:
                subprocess.run(
                    ['python', f'{processing_scripts_dir}/diarizer.py', converted_wav],
                    stdout=f, check=True,
                )
        else:
            # text_only: get duration and create single segment
            update_progress("DIARIZING")
            result = subprocess.run(
                ['ffprobe', '-v', 'error',
                 '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1',
                 converted_wav],
                capture_output=True, text=True,
            )
            try:
                duration = float(result.stdout.strip())
            except ValueError:
                duration = 0.0
            if duration <= 0:
                duration = 1.0
            with open(timings_list, "w") as f:
                f.write(f"SPEAKER_00 0.0 {duration:.1f}\n")

        # Step 3: Slice audio by speakers
        update_progress("SLICING")
        subprocess.run(
            ['python', f'{processing_scripts_dir}/slicer.py', converted_wav, timings_list],
            cwd=work_dir, check=True,
        )

        clip_files = sorted(glob.glob(os.path.join(work_dir, 'clip-*.wav')))
        if not clip_files:
            update_progress("NO_SPEECH_FOUND")
            return "NO_SPEECH"

        # Step 4: Transcribe with Whisper
        update_progress("TRANSCRIBING")
        whisper_command = [
            'whisper', '--device=cuda',
            f'--model={model}',
            '--output_dir=.', '--output_format=txt',
        ]
        if language and language != "auto":
            whisper_command.append(f'--language={language}')
        whisper_command += clip_files
        subprocess.run(whisper_command, cwd=work_dir, check=True)

        # Step 5: Assemble final document
        update_progress("DOCUMENTING")
        subprocess.run(
            [f'{processing_scripts_dir}/documenting.sh', final_md],
            cwd=work_dir, check=True,
        )

        # Step 5b: Generate .docx via pandoc
        try:
            subprocess.run(
                ['pandoc', final_md, '-o', final_docx],
                check=True, capture_output=True,
            )
        except Exception as e:
            logger.warning("Pandoc docx conversion failed: %s", e)

        return "SUCCESS"
    except Exception as e:
        error_occurred = True
        logger.exception("!!! TASK ERROR")
        if isinstance(e, subprocess.CalledProcessError):
            logger.error(f"!!! STDERR: {e.stderr}")
        raise
    finally:
        if error_occurred:
            logger.error(f"Task failed. Temp files kept for debug: {work_dir}")
        else:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)

        if os.path.exists(input_filepath):
            os.remove(input_filepath)
