"""Конфигурация ASVO-Transcriber"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Auth
    hf_token: str = ""
    telegram_bot_token: str = ""

    # Ollama
    ollama_host: str = "http://host.docker.internal:11434"
    ollama_model: str = "gpt-oss:20b"
    system_prompt: str = (
        "Ты — полезный русскоязычный ассистент. Создай структурированное резюме встречи на русском языке:\n"
        "1. Краткое описание (2-3 предложения)\n"
        "2. Основные темы обсуждения\n"
        "3. Принятые решения\n"
        "4. Action items (кто, что, когда)\n"
        "5. Открытые вопросы"
    )

    # Whisper
    whisper_model: str = "large-v3"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_language: str = ""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Paths
    upload_dir: Path = Path("/data/uploads")
    output_dir: Path = Path("/data/outputs")
    db_path: Path = Path("/data/transcripts.db")
    voice_samples_dir: Path = Path("/data/voice_samples")
    glossary_path: Path = Path("/data/glossary.txt")
    summary_templates_path: Path = Path("/data/summary_templates.json")

    # S3 (optional)
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "transcriber"
    s3_region: str = "us-east-1"

    # Limits
    max_file_size_mb: int = 500
    max_audio_duration_min: int = 180

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.voice_samples_dir.mkdir(parents=True, exist_ok=True)
