"""
Ядро транскрибации и диаризации.
faster-whisper + pyannote.audio
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Callable

import math

logger = logging.getLogger(__name__)

_whisper_model = None
_diarization_pipeline = None


# ======================================================================
# Device / compute helpers
# ======================================================================

def _resolve_device_and_compute() -> tuple[str, str]:
    """Определяет device и compute_type с автоматическим fallback на CPU."""
    import torch
    from app.config import settings

    device = settings.whisper_device
    compute = settings.whisper_compute_type

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable — falling back to CPU")
        device = "cpu"

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    elif compute == "float16" and device == "cpu":
        logger.warning("float16 not supported on CPU — switching to int8")
        compute = "int8"

    return device, compute


def get_whisper_model():
    """Lazy-load faster-whisper model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        from app.config import settings
        device, compute = _resolve_device_and_compute()
        logger.info("Loading Whisper model: %s on %s/%s", settings.whisper_model, device, compute)
        t0 = time.time()
        _whisper_model = WhisperModel(settings.whisper_model, device=device, compute_type=compute)
        logger.info("Whisper loaded in %.1fs", time.time() - t0)
    return _whisper_model


def get_diarization_pipeline():
    """Lazy-load pyannote diarization pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        from app.config import settings
        if not settings.hf_token:
            logger.warning("HF_TOKEN not set — diarization unavailable")
            return None
        try:
            from pyannote.audio import Pipeline
            import torch
            logger.info("Loading pyannote speaker-diarization-3.1 ...")
            t0 = time.time()
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.hf_token,
            )
            if torch.cuda.is_available():
                _diarization_pipeline.to(torch.device("cuda"))
                logger.info("Pyannote on GPU")
            else:
                logger.info("Pyannote on CPU (CUDA unavailable)")
            logger.info("Pyannote loaded in %.1fs", time.time() - t0)
        except Exception as e:
            logger.error("Pyannote load failed: %s", e)
            return None
    return _diarization_pipeline


def get_glossary_prompt() -> str:
    """Загружает словарь терминов для initial_prompt Whisper."""
    from app.config import settings
    try:
        if settings.glossary_path.exists():
            text = settings.glossary_path.read_text("utf-8").strip()
            if text:
                logger.info("Glossary loaded: %d terms", len(text.splitlines()))
                return text
    except Exception as e:
        logger.warning("Failed to load glossary: %s", e)
    return ""


# ======================================================================
# Data models
# ======================================================================

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""
    words: list = field(default_factory=list)
    avg_logprob: float = 0.0

    @property
    def confidence(self) -> float:
        """Уверенность 0-1 на основе avg_logprob. >0.7 хорошо, <0.4 сомнительно."""
        return max(0.0, min(1.0, math.exp(self.avg_logprob)))


@dataclass
class TranscriptionResult:
    segments: List['Segment'] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    processing_time: float = 0.0
    summary: str = ""
    topics: list = field(default_factory=list)


# ======================================================================
# Audio helpers
# ======================================================================

def prepare_audio(input_path: str, output_path: str) -> float:
    """Конвертирует любой аудио/видео в WAV 16kHz mono. Возвращает длительность."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
         str(output_path)],
        check=True, capture_output=True,
    )
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(output_path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def concat_audio_files(file_paths: list[str], output_path: str) -> float:
    """Объединяет несколько файлов в один WAV."""
    if len(file_paths) == 1:
        return prepare_audio(file_paths[0], output_path)

    list_file = Path(output_path).with_suffix(".list")
    with open(list_file, "w") as f:
        for p in file_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
         str(output_path)],
        check=True, capture_output=True,
    )
    list_file.unlink(missing_ok=True)

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(output_path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ======================================================================
# Transcription
# ======================================================================

def transcribe(
    audio_path: str,
    language: str = "",
    on_progress: Optional[Callable] = None,
) -> tuple[List[Segment], str]:
    """Транскрибация через faster-whisper. Возвращает (List[Segment], detected_language)."""
    model = get_whisper_model()
    logger.info("Transcribing: %s (language=%s)", audio_path, language or "auto")
    t0 = time.time()

    glossary = get_glossary_prompt()
    kwargs = {}
    if language and language != "auto":
        kwargs["language"] = language
    if glossary:
        kwargs["initial_prompt"] = glossary

    segments_iter, info = model.transcribe(str(audio_path), **kwargs)

    segments = []
    for seg in segments_iter:
        segments.append(Segment(
            start=round(seg.start, 3),
            end=round(seg.end, 3),
            text=seg.text.strip(),
            words=list(getattr(seg, "words", []) or []),
            avg_logprob=getattr(seg, "avg_logprob", 0.0),
        ))
        if on_progress and segments:
            on_progress(segments[-1].end)

    detected_lang = getattr(info, "language", language or "")
    logger.info(
        "Transcribed %d segments in %.0fs (language=%s, prob=%.2f)",
        len(segments), time.time() - t0, detected_lang,
        getattr(info, "language_probability", 0),
    )
    return segments, detected_lang


# ======================================================================
# Diarization
# ======================================================================

def diarize(
    audio_path: str,
    segments: List[Segment],
    num_speakers: int = 0,
) -> tuple[List[Segment], dict[str, str]]:
    """Назначает спикера каждому сегменту через pyannote. Возвращает (segments, name_map)."""
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        logger.warning("Pyannote unavailable, falling back to pause-based diarization")
        return _diarize_by_pauses(segments), {}

    logger.info("Running pyannote diarization...")
    t0 = time.time()

    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(audio_path), **kwargs)

    # Build speaker timeline
    speaker_timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_timeline.append((turn.start, turn.end, speaker))

    # Assign speaker to each segment based on overlap
    for seg in segments:
        seg_mid = (seg.start + seg.end) / 2
        best_speaker = "Unknown"
        for s_start, s_end, s_label in speaker_timeline:
            if s_start <= seg_mid <= s_end:
                best_speaker = s_label
                break
            # Fallback: closest overlap
            overlap_start = max(seg.start, s_start)
            overlap_end = min(seg.end, s_end)
            if overlap_end > overlap_start:
                best_speaker = s_label
        seg.speaker = best_speaker

    # Try to match with saved voice profiles
    name_map: dict[str, str] = {}
    try:
        from app.speakers import extract_speaker_embeddings_from_diarization, match_speakers, load_speaker_profiles
        profiles = load_speaker_profiles()
        if profiles:
            logger.info("Matching against %d voice profiles...", len(profiles))
            diar_embeddings = extract_speaker_embeddings_from_diarization(
                audio_path, diarization, segments
            )
            name_map = match_speakers(diar_embeddings)
            if name_map:
                logger.info("Matched speakers: %s", name_map)
    except Exception as e:
        logger.warning("Speaker matching failed: %s", e)

    unique_speakers = set(seg.speaker for seg in segments)
    logger.info("Diarization done in %.0fs — %d speakers", time.time() - t0, len(unique_speakers))
    return segments, name_map


def _diarize_by_pauses(segments: List[Segment]) -> List[Segment]:
    """Эвристическая диаризация по паузам (fallback)."""
    current_speaker = "Speaker_1"
    speaker_idx = 1
    for i in range(len(segments)):
        if i > 0 and (segments[i].start - segments[i - 1].end) > 2.0:
            speaker_idx += 1
            current_speaker = f"Speaker_{speaker_idx}"
        segments[i].speaker = current_speaker
    return segments


# ======================================================================
# Post-processing
# ======================================================================

def merge_segments(segments: List[Segment], max_gap: float = 0.5) -> List[Segment]:
    """Объединяет последовательные сегменты одного спикера."""
    if not segments:
        return segments
    merged = [Segment(
        start=segments[0].start, end=segments[0].end,
        text=segments[0].text, speaker=segments[0].speaker,
        avg_logprob=segments[0].avg_logprob,
    )]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker == prev.speaker and (seg.start - prev.end) <= max_gap:
            prev.end = seg.end
            prev.text = prev.text + " " + seg.text
            prev.avg_logprob = (prev.avg_logprob + seg.avg_logprob) / 2
        else:
            merged.append(Segment(
                start=seg.start, end=seg.end, text=seg.text,
                speaker=seg.speaker, avg_logprob=seg.avg_logprob,
            ))
    return merged


def rename_speakers(segments: List[Segment], name_map: dict[str, str]) -> List[Segment]:
    """Переименовывает SPEAKER_00 → Спикер 1 или по карте имён."""
    if not name_map:
        # Auto-generate: SPEAKER_00 → Спикер 1
        unique = {}
        for seg in segments:
            if seg.speaker not in unique:
                unique[seg.speaker] = f"Спикер {len(unique) + 1}"
        name_map = unique

    for seg in segments:
        seg.speaker = name_map.get(seg.speaker, seg.speaker)
    return segments


# ======================================================================
# Summary templates
# ======================================================================

SUMMARY_TEMPLATES = {
    "meeting": (
        "Создай структурированное резюме встречи:\n"
        "1. Краткое описание (2-3 предложения)\n"
        "2. Основные темы обсуждения\n"
        "3. Принятые решения\n"
        "4. Action items (кто, что, когда)\n"
        "5. Открытые вопросы"
    ),
    "standup": (
        "Это стендап-встреча. Для каждого участника выдели:\n"
        "1. Что сделал вчера\n"
        "2. Что планирует сегодня\n"
        "3. Блокеры и проблемы\n"
        "В конце — общий статус команды."
    ),
    "interview": (
        "Это интервью. Создай резюме:\n"
        "1. Участники и их роли\n"
        "2. Ключевые вопросы и ответы (кратко)\n"
        "3. Сильные стороны кандидата/собеседника\n"
        "4. Области для улучшения\n"
        "5. Общее впечатление"
    ),
    "lecture": (
        "Это лекция/презентация. Создай конспект:\n"
        "1. Тема и спикер\n"
        "2. Основные тезисы (по пунктам)\n"
        "3. Ключевые примеры и данные\n"
        "4. Выводы\n"
        "5. Вопросы из аудитории (если были)"
    ),
    "brainstorm": (
        "Это мозговой штурм. Выдели:\n"
        "1. Тема/проблема\n"
        "2. Все предложенные идеи (списком)\n"
        "3. Наиболее поддержанные идеи\n"
        "4. Критика и контраргументы\n"
        "5. Решение по итогу (если было)"
    ),
}


def get_summary_templates() -> dict[str, str]:
    """Загрузить шаблоны: встроенные + пользовательские."""
    from app.config import settings
    templates = dict(SUMMARY_TEMPLATES)
    try:
        if settings.summary_templates_path.exists():
            custom = json.loads(settings.summary_templates_path.read_text("utf-8"))
            templates.update(custom)
    except Exception:
        pass
    return templates


# ======================================================================
# LLM: summarize / translate
# ======================================================================

def summarize(text: str, template: str = "meeting") -> str:
    """Суммаризация через Ollama с выбором шаблона."""
    import httpx
    from app.config import settings

    templates = get_summary_templates()
    prompt = templates.get(template, templates.get("meeting", settings.system_prompt))

    # Truncate very long texts
    lines = text.split("\n")
    if len(lines) > 2000:
        text = "\n".join(lines[:2000]) + "\n\n[... текст обрезан ...]"

    full_prompt = f"Ты — полезный русскоязычный ассистент.\n{prompt}\n\nВот транскрипт:\n\n{text}"

    logger.info("Summarizing via Ollama (%s)...", settings.ollama_model)
    try:
        r = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={"model": settings.ollama_model, "prompt": full_prompt, "stream": False},
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        logger.error("Ollama summarization failed: %s", e)
        return f"[Суммаризация недоступна: {e}]"


LANG_NAMES = {
    "ru": "русский", "en": "English", "uk": "українська",
    "de": "Deutsch", "fr": "français", "es": "español",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "pt": "Portuguese", "it": "italiano",
    "pl": "polski", "tr": "Türkçe", "kk": "қазақша",
}


def translate_segments(segments: List[Segment], target_lang: str) -> List[Segment]:
    """Переводит текст сегментов через Ollama. Возвращает новый список сегментов."""
    import httpx
    from app.config import settings

    lang_name = LANG_NAMES.get(target_lang, target_lang)
    translated = []

    # Process in batches of 20 segments
    batch_size = 20
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {s.text}" for j, s in enumerate(batch))

        prompt = (
            f"Translate the following numbered lines to {lang_name}. "
            f"Return ONLY the numbered translations, one per line, preserving the numbers. "
            f"Do not add any commentary.\n\n{numbered}"
        )

        try:
            r = httpx.post(
                f"{settings.ollama_host}/api/generate",
                json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
                timeout=120.0,
            )
            r.raise_for_status()
            response = r.json().get("response", "").strip()

            lines = response.split("\n")
            for j, seg in enumerate(batch):
                text = seg.text
                for line in lines:
                    line = line.strip().rstrip(".")
                    if line and (line[0].isdigit()):
                        parts = line.split(".", 1)
                        if len(parts) == 2:
                            num_str = parts[0].strip()
                            if num_str.isdigit() and int(num_str) == j + 1:
                                text = parts[1].strip()
                                break

                translated.append(Segment(
                    start=seg.start, end=seg.end, text=text,
                    speaker=seg.speaker,
                ))
        except Exception as e:
            logger.error("Translation batch failed: %s", e)
            translated.extend(batch)

    return translated


# ======================================================================
# Topics detection
# ======================================================================

def detect_topics(segments: List[Segment]) -> list[dict]:
    """Определяет темы по паузам и длине блоков."""
    if not segments:
        return []

    topics = []
    current_block = []
    for i, seg in enumerate(segments):
        current_block.append(seg)
        is_break = False
        if i < len(segments) - 1:
            gap = segments[i + 1].start - seg.end
            if gap > 5.0:
                is_break = True
            elif len(current_block) > 15:
                is_break = True

        if is_break or i == len(segments) - 1:
            if len(current_block) >= 3:
                block_text = " ".join(s.text for s in current_block)
                words = block_text.split()
                title = " ".join(words[:8]).strip() + "..."
                topics.append({
                    "title": title,
                    "start": current_block[0].start,
                    "end": current_block[-1].end,
                    "start_idx": segments.index(current_block[0]),
                })
            current_block = []

    return topics


# ======================================================================
# Full pipeline
# ======================================================================

def process(
    audio_paths: list[str],
    language: str = "",
    diarize_flag: bool = True,
    summarize_flag: bool = False,
    summary_template: str = "meeting",
    num_speakers: int = 0,
    on_progress: Optional[Callable] = None,
    on_step: Optional[Callable] = None,
) -> TranscriptionResult:
    """
    Полный пайплайн: конкатенация → транскрибация → диаризация →
    объединение → темы → суммаризация.

    on_step(step, percent, message) — callback для прогресса по этапам.
    """
    _step = on_step or (lambda *a: None)
    t_start = time.time()

    # 1. Prepare audio
    _step("prepare", 0, "Подготовка аудио...")
    work_dir = Path(audio_paths[0]).parent
    combined_wav = str(work_dir / "_combined.wav")

    if len(audio_paths) > 1:
        duration = concat_audio_files(audio_paths, combined_wav)
    else:
        duration = prepare_audio(audio_paths[0], combined_wav)

    logger.info("Audio prepared: %.0fs (%.1f min)", duration, duration / 60)
    _step("prepare", 100, f"Аудио подготовлено ({duration/60:.1f} мин)")

    # 2. Transcribe
    _step("transcribe", 0, "Транскрибация (Whisper large-v3)...")

    def _transcribe_progress(t):
        if duration > 0:
            pct = int(min(100, t / duration * 100))
            _step("transcribe", pct, f"Транскрибация: {t:.0f}/{duration:.0f} мин")

    segments, detected_lang = transcribe(combined_wav, language, _transcribe_progress)
    logger.info("Транскрибация завершена: %d сегментов (язык: %s)", len(segments), detected_lang)

    # 3. Diarize
    name_map = {}
    if diarize_flag:
        _step("diarize", 0, "Диаризация (разделение по спикерам)...")
        segments, name_map = diarize(combined_wav, segments, num_speakers)
        unique = set(s.speaker for s in segments)
        _step("diarize", 100, f"Диаризация завершена: {len(unique)} спикеров")

    # 4. Post-process
    _step("postprocess", 0, "Объединение сегментов...")
    segments = merge_segments(segments)
    segments = rename_speakers(segments, name_map)
    topics = detect_topics(segments)
    _step("postprocess", 100, f"Постобработка: {len(segments)} блоков, {len(topics)} тем")

    # 5. Summarize
    summary = ""
    if summarize_flag:
        _step("summarize", 0, "Суммаризация (LLM)...")
        full_text = "\n".join(f"{s.speaker}: {s.text}" for s in segments)
        summary = summarize(full_text, summary_template)
        _step("summarize", 100, "Суммаризация завершена")

    # Cleanup temp combined wav
    try:
        if len(audio_paths) > 1:
            os.unlink(combined_wav)
    except OSError:
        pass

    speakers = sorted(set(s.speaker for s in segments))
    processing_time = round(time.time() - t_start, 1)

    _step("done", 100, "Готово")
    logger.info(
        "Pipeline done: %d segments, %d speakers, %d topics in %.0fs",
        len(segments), len(speakers), len(topics), processing_time,
    )

    return TranscriptionResult(
        segments=segments,
        speakers=speakers,
        language=detected_lang,
        duration=duration,
        processing_time=processing_time,
        summary=summary,
        topics=topics,
    )
