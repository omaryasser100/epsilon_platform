"""MinIO / S3 object storage helpers.

Mirrors the reference EsplionRAG ingestion pipeline's storage module so
key layout and call signatures are consistent across the org:

    <bucket>/<prefix>/<channel_id>/raw/<doc_id>/<safe_name>.pdf
    <bucket>/<prefix>/<channel_id>/markdown/<doc_id>.md
    <bucket>/<prefix>/<channel_id>/metadata/<doc_id>.json

Bucket creation is lazy — `ensure_bucket()` is idempotent and is what
materialises the bucket on first ingest. Prefixes ("folders") don't
need to be created in S3 semantics; they appear when objects land.
"""
from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from minio import Minio
from minio.error import S3Error

from core.config import MinIOConfig


def _client(cfg: MinIOConfig) -> Minio:
    return Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )


def ensure_bucket(cfg: MinIOConfig) -> None:
    client = _client(cfg)
    if not client.bucket_exists(cfg.bucket):
        client.make_bucket(cfg.bucket)


def object_key(cfg: MinIOConfig, channel_id: str, *parts: str) -> str:
    base = "/".join(p for p in (cfg.prefix, channel_id, *parts) if p)
    return base.replace("\\", "/")


def put_bytes(
    cfg: MinIOConfig,
    object_name: str,
    data: bytes,
    content_type: str,
) -> None:
    client = _client(cfg)
    client.put_object(
        cfg.bucket,
        object_name,
        BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def put_json(cfg: MinIOConfig, object_name: str, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    put_bytes(cfg, object_name, raw, "application/json; charset=utf-8")


def safe_put(
    cfg: MinIOConfig,
    object_name: str,
    data: bytes,
    content_type: str,
) -> None:
    try:
        put_bytes(cfg, object_name, data, content_type)
    except S3Error as e:
        raise RuntimeError(f"MinIO upload failed for {object_name!r}: {e}") from e
