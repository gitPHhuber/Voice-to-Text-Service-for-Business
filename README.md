# ASVO-Transcriber

Корпоративный сервис транскрибации, диаризации и анализа аудио/видео с веб-интерфейсом и Telegram-ботом.

## Возможности

**Транскрибация**
- faster-whisper (large-v3 / medium) на GPU — в 4x быстрее OpenAI Whisper
- 15+ языков с автоопределением
- Словарь терминов (glossary) для повышения точности

**Диаризация**
- pyannote.audio 3.1 — автоматическое разделение по спикерам
- Голосовые профили — сохранение образцов голоса, автоматическое узнавание спикеров (wespeaker embeddings + косинусное сходство)
- Fallback на эвристику по паузам при недоступности pyannote

**Анализ**
- Суммаризация через LLM (Ollama) с шаблонами: встреча, стендап, интервью, лекция, переговоры, мозговой штурм
- Перевод транскриптов на 15 языков через LLM
- Определение тем по паузам и длине блоков
- Полнотекстовый поиск по архиву (SQLite FTS5)

**Экспорт**
- DOCX — форматированный документ с таймкодами, спикерами, резюме, оглавлением тем
- TXT — текст с таймкодами
- SRT — субтитры
- JSON — полная структура с confidence score

**Веб-интерфейс**
- Drag & drop загрузка (мультифайл, до 500 МБ)
- Прогресс-бар по этапам обработки
- Просмотр транскрипта с аудиоплеером и подсветкой текущего сегмента
- Редактирование транскриптов в браузере
- История с полнотекстовым поиском
- Живая транскрибация с микрофона (WebSocket)

**Telegram-бот**
- Кнопочный интерфейс — inline-клавиатуры для всех действий
- Настройки: язык, диаризация, суммаризация, формат выхода
- Скачивание результатов: DOCX / TXT / SRT / JSON
- Перевод: EN / DE / FR одной кнопкой
- Поиск по архиву, история транскриптов
- Управление голосовыми профилями спикеров
- LLM-чат через Ollama
- Авторизация пользователей (whitelist), админ-панель
- SSE + polling для отслеживания статуса

## Архитектура

```
┌──────────┐     ┌──────────────────────────────────┐     ┌───────┐
│ Telegram │────▶│         Bot (python-telegram-bot) │────▶│       │
│  User    │◀────│         Inline UI, LLM chat       │◀────│       │
└──────────┘     └──────────────────────────────────┘     │       │
                                                          │Backend│
┌──────────┐     ┌──────────────────────────────────┐     │FastAPI│
│ Browser  │────▶│         Web UI (HTML/JS)          │────▶│+Celery│
│          │◀────│         Upload, Viewer, History   │◀────│       │
└──────────┘     └──────────────────────────────────┘     │       │
                                                          │  app/ │
                 ┌──────────────────────────────────┐     │engine │
                 │         app/ — Core Library       │     │  db   │
                 │  engine.py  — faster-whisper      │     │export │
                 │  db.py      — SQLite FTS5         │     │speak. │
                 │  export.py  — DOCX/TXT/SRT/JSON   │     │       │
                 │  speakers.py— voice profiles      │     └───┬───┘
                 │  storage.py — S3/local FS         │         │
                 │  tasks.py   — Celery tasks        │     ┌───▼───┐
                 └──────────────────────────────────┘     │ Redis │
                                                          └───────┘
```

**Сервисы (Docker Compose):**
- `backend` — FastAPI + Celery worker, GPU (NVIDIA CUDA)
- `bot` — Telegram-бот (легковесный, HTTP-клиент)
- `redis` — брокер задач и кеш

## Стек

| Компонент | Технология |
|-----------|-----------|
| Транскрибация | faster-whisper, CTranslate2 |
| Диаризация | pyannote.audio 3.1 |
| Спикеры | pyannote/wespeaker-voxceleb-resnet34-LM |
| LLM | Ollama (gpt-oss, llama и др.) |
| API | FastAPI, Uvicorn |
| Очередь | Celery + Redis |
| БД | SQLite с FTS5 |
| Экспорт | python-docx, pandoc |
| Бот | python-telegram-bot |
| GPU | NVIDIA CUDA 12.8, PyTorch |
| Хранилище | Локальная FS / S3 (MinIO) |

## Установка

```bash
git clone https://github.com/gitPHhuber/Voice-to-Text-Service-for-Business.git
cd Voice-to-Text-Service-for-Business
```

Создайте `.env`:

```env
HF_TOKEN=hf_ваш_токен
TELEGRAM_BOT_TOKEN=ваш_токен_бота
ADMIN_TELEGRAM_ID=ваш_telegram_id

OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=gpt-oss:20b

WHISPER_MODEL=medium
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto

REDIS_URL=redis://redis:6379/0
```

Запуск:

```bash
docker compose build
docker compose up -d
```

**Доступ:**
- Веб: `http://your-server:8001/`
- Бот: напишите `/start` боту в Telegram
- API: `http://your-server:8001/health`
- История: `http://your-server:8001/history`

## API

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| POST | `/transcribe` | Загрузить файл на транскрибацию |
| GET | `/status/{job_id}` | Статус задачи |
| GET | `/result/{task_id}?ext=docx` | Скачать результат |
| POST | `/cancel/{job_id}` | Отменить задачу |
| GET | `/events/{job_id}` | SSE-стрим статуса |
| GET | `/api/search?q=...` | Полнотекстовый поиск |
| GET | `/api/transcripts` | Список транскриптов |
| GET | `/api/transcript/{id}` | Полный транскрипт |
| PUT | `/api/transcript/{id}` | Редактирование |
| POST | `/api/speakers` | Добавить голосовой профиль |
| GET | `/api/speakers` | Список спикеров |
| DELETE | `/api/speakers/{name}` | Удалить спикера |
| POST | `/api/translate/{id}` | Перевод транскрипта |
| GET | `/api/glossary` | Словарь терминов |
| PUT | `/api/glossary` | Обновить словарь |
| GET | `/health` | Healthcheck |

## Модели Whisper

| Модель | VRAM | Скорость | Качество |
|--------|------|----------|----------|
| `small` | ~1 ГБ | Быстро | Базовое |
| `medium` | ~1.5 ГБ | Средне | Хорошее |
| `large-v3` | ~3 ГБ | Медленно | Лучшее |

## Требования

- Docker + Docker Compose
- NVIDIA GPU с CUDA (минимум 4 ГБ VRAM)
- HuggingFace токен (для pyannote)
- Ollama (опционально, для суммаризации/перевода)

## Структура проекта

```
app/                    # Core library
├── config.py           # Pydantic Settings
├── db.py               # SQLite FTS5
├── engine.py           # faster-whisper + pyannote
├── export.py           # DOCX, TXT, JSON, SRT
├── speakers.py         # Voice profiles
├── storage.py          # S3/local abstraction
├── tasks.py            # Celery tasks
└── worker.py           # Worker entry point
backend/
├── main.py             # FastAPI API + Web UI routes
├── templates/          # HTML (upload, viewer, live, history)
├── Dockerfile
└── requirements.txt
bot/
├── bot.py              # Telegram bot
├── Dockerfile
└── requirements.txt
docker-compose.yml
.env
```
