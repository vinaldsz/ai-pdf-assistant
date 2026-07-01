"""Cloudflare R2 storage — upload and download PDF bytes.

R2 is the source of truth for raw PDF bytes. Chunks and embeddings in Neon are
derived data; R2 lets us re-embed without asking users to resubmit files.

boto3 is used against R2's S3-compatible endpoint. The client is created lazily
and cached so we don't open connections at import time.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from io import BytesIO
from typing import Any

from app.obs.logging import get_logger
from app.settings import settings

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _client() -> Any:
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id.get_secret_value(),  # type: ignore[union-attr]
        aws_secret_access_key=settings.r2_secret_access_key.get_secret_value(),  # type: ignore[union-attr]
        region_name="auto",
    )


def _upload_sync(key: str, data: bytes) -> None:
    _client().upload_fileobj(BytesIO(data), settings.r2_bucket, key)
    log.info("r2.upload", key=key, size=len(data))


def _download_sync(key: str) -> bytes:
    buf = BytesIO()
    _client().download_fileobj(settings.r2_bucket, key, buf)
    return buf.getvalue()


async def upload(key: str, data: bytes) -> None:
    """Upload bytes to R2 under the given key. No-op if R2 is not configured."""
    if not settings.r2_enabled:
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _upload_sync, key, data)


async def download(key: str) -> bytes:
    """Download bytes from R2 by key."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _download_sync, key)
