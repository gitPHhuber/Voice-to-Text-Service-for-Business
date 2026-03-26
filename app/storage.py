"""
Абстракция хранилища: локальная FS или S3/MinIO.
Если S3_ENDPOINT задан — используется S3, иначе — локальные файлы.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

_s3_client = None


def _is_s3_enabled() -> bool:
    return bool(settings.s3_endpoint and settings.s3_access_key)


def _get_s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        from botocore.config import Config
        _s3_client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=Config(signature_version="s3v4"),
        )
        try:
            _s3_client.head_bucket(Bucket=settings.s3_bucket)
        except Exception:
            try:
                _s3_client.create_bucket(Bucket=settings.s3_bucket)
                logger.info("Created S3 bucket: %s", settings.s3_bucket)
            except Exception as e:
                logger.warning("Could not create bucket: %s", e)
    return _s3_client


def upload_file(local_path: str, remote_key: str):
    """Загрузить файл в хранилище."""
    if not _is_s3_enabled():
        return
    try:
        _get_s3().upload_file(str(local_path), settings.s3_bucket, remote_key)
        logger.info("Uploaded to S3: %s", remote_key)
    except Exception as e:
        logger.error("S3 upload failed: %s", e)


def download_file(remote_key: str, local_path: str):
    """Скачать файл из хранилища."""
    if not _is_s3_enabled():
        if not Path(local_path).exists():
            return
        return
    try:
        _get_s3().download_file(settings.s3_bucket, remote_key, str(local_path))
    except Exception as e:
        logger.error("S3 download failed: %s", e)


def upload_results(task_id: str, output_dir: str):
    """Загрузить все результаты задачи в S3."""
    if not _is_s3_enabled():
        return
    out = Path(output_dir)
    for f in out.iterdir():
        if f.is_file() and f.name != "status.json":
            upload_file(str(f), f"outputs/{task_id}/{f.name}")


def upload_source(task_id: str, file_path: str):
    """Загрузить исходный файл в S3."""
    if not _is_s3_enabled():
        return
    name = Path(file_path).name
    upload_file(file_path, f"uploads/{task_id}/{name}")


def get_file_url(remote_key: str, expires: int = 3600) -> Optional[str]:
    """Получить presigned URL для скачивания."""
    if not _is_s3_enabled():
        return None
    try:
        return _get_s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": remote_key},
            ExpiresIn=expires,
        )
    except Exception as e:
        logger.error("S3 presigned URL failed: %s", e)
        return None


def list_files(prefix: str) -> list[str]:
    """Список файлов по префиксу."""
    if not _is_s3_enabled():
        return []
    try:
        resp = _get_s3().list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception as e:
        logger.error("S3 list failed: %s", e)
        return []
