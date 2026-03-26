"""
Управление голосовыми профилями спикеров.
Извлечение эмбеддингов через pyannote, сопоставление по косинусному сходству.
"""

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_embedding_model = None


def _get_embedding_model():
    """Lazy-load pyannote embedding model."""
    global _embedding_model
    if _embedding_model is None:
        if not settings.hf_token:
            logger.warning("HF_TOKEN not set — speaker embedding unavailable")
            return None
        try:
            from pyannote.audio import Inference
            import torch
            _embedding_model = Inference(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=settings.hf_token,
            )
            if torch.cuda.is_available():
                _embedding_model.to(torch.device("cuda"))
            logger.info("Speaker embedding model loaded")
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            return None
    return _embedding_model


def _prepare_wav(input_path: str) -> Path:
    """Конвертирует аудио в WAV 16kHz mono для эмбеддинга."""
    out = Path(input_path).with_suffix(".emb.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
         str(out)],
        check=True, capture_output=True,
    )
    return out


def extract_embedding(audio_path: str) -> np.ndarray | None:
    """Извлекает голосовой эмбеддинг из аудиофайла."""
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        wav = _prepare_wav(audio_path)
        embedding = model(str(wav))
        wav.unlink(missing_ok=True)
        return embedding
    except Exception as e:
        logger.error("Embedding extraction failed: %s", e)
        return None


def save_voice_sample(speaker_name: str, audio_path: str) -> bool:
    """Сохраняет голосовой образец и эмбеддинг спикера."""
    embedding = extract_embedding(audio_path)
    if embedding is None:
        return False
    speaker_dir = settings.voice_samples_dir / speaker_name
    speaker_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(speaker_dir / "embedding.npy"), embedding)
    logger.info("Saved voice profile for '%s'", speaker_name)
    return True


def load_speaker_profiles() -> dict[str, np.ndarray]:
    """Загружает все сохранённые профили: {name: embedding}."""
    profiles = {}
    if not settings.voice_samples_dir.exists():
        return profiles
    for d in settings.voice_samples_dir.iterdir():
        if not d.is_dir():
            continue
        npy = d / "embedding.npy"
        if npy.exists():
            try:
                profiles[d.name] = np.load(str(npy))
            except Exception as e:
                logger.warning("Failed to load profile %s: %s", d.name, e)
    return profiles


def list_speakers() -> list[str]:
    """Список зарегистрированных спикеров."""
    if not settings.voice_samples_dir.exists():
        return []
    return sorted([
        d.name for d in settings.voice_samples_dir.iterdir()
        if d.is_dir() and (d / "embedding.npy").exists()
    ])


def delete_speaker(speaker_name: str) -> bool:
    """Удаляет профиль спикера."""
    speaker_dir = settings.voice_samples_dir / speaker_name
    if speaker_dir.exists():
        shutil.rmtree(speaker_dir)
        return True
    return False


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def match_speakers(
    diarization_embeddings: dict[str, np.ndarray],
    threshold: float = 0.65,
) -> dict[str, str]:
    """
    Сопоставляет эмбеддинги из диаризации с сохранёнными профилями.
    Возвращает {pyannote_label: real_name} для совпавших.
    """
    profiles = load_speaker_profiles()
    if not profiles or not diarization_embeddings:
        return {}

    matched = {}
    used_profiles = set()
    for label, emb in diarization_embeddings.items():
        best_name = None
        best_score = threshold
        for name, profile_emb in profiles.items():
            if name in used_profiles:
                continue
            score = _cosine_similarity(emb, profile_emb)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name:
            matched[label] = best_name
            used_profiles.add(best_name)
            logger.info("Matched %s → %s (score=%.3f)", label, best_name, best_score)

    return matched


def extract_speaker_embeddings_from_diarization(
    audio_path: str,
    diarization,
    segments,
) -> dict[str, np.ndarray]:
    """
    Извлекает эмбеддинги для каждого спикера из результата диаризации.
    Берёт 3 самых длинных сегмента каждого спикера.
    """
    model = _get_embedding_model()
    if model is None:
        return {}

    from pyannote.core import Segment as PyannoteSegment

    # Group segments by speaker
    speaker_turns: dict[str, list] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.setdefault(speaker, []).append((turn.start, turn.end))

    embeddings = {}
    for speaker, turns in speaker_turns.items():
        # Take 3 longest segments
        turns.sort(key=lambda x: x[1] - x[0], reverse=True)
        top_turns = turns[:3]

        try:
            speaker_embs = []
            for start, end in top_turns:
                excerpt = PyannoteSegment(start, end)
                emb = model.crop(str(audio_path), excerpt)
                speaker_embs.append(emb)

            if speaker_embs:
                embeddings[speaker] = np.mean(speaker_embs, axis=0)
        except Exception as e:
            logger.warning("Failed to extract embedding for %s: %s", speaker, e)

    return embeddings
