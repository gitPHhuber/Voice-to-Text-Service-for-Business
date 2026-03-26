"""
Celery Worker — фоновая обработка задач транскрибации.
Запуск: celery -A app.tasks.app worker --loglevel=info
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("worker")


def run_worker():
    from app.tasks import app as celery_app
    from app.db import init_db

    init_db()
    logger.info("Worker started")
    celery_app.worker_main(["worker", "--loglevel=info", "-P", "solo"])


if __name__ == "__main__":
    run_worker()
