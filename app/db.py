"""
SQLite FTS5 — хранение и полнотекстовый поиск по транскриптам.
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)

DB_PATH = str(settings.db_path)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Создаёт таблицы если не существуют."""
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS transcripts (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                file_name TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT '',
                duration REAL NOT NULL DEFAULT 0,
                processing_time REAL NOT NULL DEFAULT 0,
                speakers TEXT NOT NULL DEFAULT '[]',
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES transcripts(task_id) ON DELETE CASCADE,
                speaker TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL DEFAULT '',
                start REAL NOT NULL DEFAULT 0,
                "end" REAL NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 1.0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                text, content=segments, content_rowid=id
            );

            CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
                INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
                INSERT INTO segments_fts(segments_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
                INSERT INTO segments_fts(segments_fts, rowid, text) VALUES('delete', old.id, old.text);
                INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
            END;
        """)

        # Migration: add confidence column if missing
        cols = [r[1] for r in conn.execute("PRAGMA table_info(segments)").fetchall()]
        if "confidence" not in cols:
            conn.execute("ALTER TABLE segments ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0")
            logger.info("Migrated: added 'confidence' column to segments")

    logger.info("Database initialized: %s", DB_PATH)


def save_transcript(task_id: str, result, title: str = "", file_name: str = ""):
    """Сохраняет транскрипт и сегменты в БД."""
    with _db() as conn:
        conn.execute("DELETE FROM segments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM transcripts WHERE task_id = ?", (task_id,))

        conn.execute(
            """
            INSERT INTO transcripts (task_id, title, file_name, language, duration,
                                     processing_time, speakers, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                title,
                file_name,
                result.language,
                result.duration,
                result.processing_time,
                json.dumps(result.speakers, ensure_ascii=False),
                result.summary,
                datetime.now().isoformat(),
            ),
        )

        conn.executemany(
            """
            INSERT INTO segments (task_id, speaker, text, start, "end", confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (task_id, s.speaker, s.text, s.start, s.end, getattr(s, "confidence", 1.0))
                for s in result.segments
            ],
        )
        logger.info("Saved transcript %s: %d segments", task_id, len(result.segments))


def search(query: str, limit: int = 20, offset: int = 0) -> list[dict]:
    """Полнотекстовый поиск по сегментам."""
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT s.task_id, s.speaker, s.text, s.start, s."end",
                   t.title, t.file_name, t.created_at,
                   snippet(segments_fts, 0, '<mark>', '</mark>', '...', 32) AS snippet
            FROM segments_fts
            JOIN segments s ON s.id = segments_fts.rowid
            JOIN transcripts t ON t.task_id = s.task_id
            WHERE segments_fts MATCH ?
            ORDER BY rank
            LIMIT ? OFFSET ?
            """,
            (query, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def list_transcripts(limit: int = 50, offset: int = 0) -> list[dict]:
    """Список всех транскриптов."""
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT task_id, title, file_name, language, duration,
                   processing_time, speakers, summary, created_at
            FROM transcripts
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["speakers"] = json.loads(d.get("speakers", "[]"))
            d["summary"] = d.get("summary", "")
            result.append(d)
        return result


def get_transcript(task_id: str) -> dict | None:
    """Получить полный транскрипт из БД."""
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM transcripts WHERE task_id = ?", (task_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["speakers"] = json.loads(d.get("speakers", "[]"))

        segs = conn.execute(
            """
            SELECT speaker, text, start, "end", confidence
            FROM segments WHERE task_id = ?
            ORDER BY start
            """,
            (task_id,),
        ).fetchall()
        d["segments"] = [dict(s) for s in segs]
        return d


def delete_transcript(task_id: str):
    """Удалить транскрипт из БД."""
    with _db() as conn:
        conn.execute("DELETE FROM segments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM transcripts WHERE task_id = ?", (task_id,))


def update_segments(task_id: str, segments: list[dict]):
    """Обновить сегменты транскрипта (после редактирования)."""
    with _db() as conn:
        conn.execute("DELETE FROM segments WHERE task_id = ?", (task_id,))
        conn.executemany(
            """
            INSERT INTO segments (task_id, speaker, text, start, "end")
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (task_id, s["speaker"], s["text"], s["start"], s.get("end", 0))
                for s in segments
            ],
        )
