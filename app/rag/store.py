"""pgvector read/write helpers using SQLAlchemy + asyncpg connection pool."""
from __future__ import annotations

from typing import Any

from app.rag.chunker import Chunk


async def get_document_by_sha256(sha256: str) -> dict[str, Any] | None:
    raise NotImplementedError("Day 3")


async def insert_document(
    *,
    sha256: str,
    source_url: str,
    title: str | None,
    pages: int | None,
    embedder_version: str,
    chunker_version: str,
) -> str:
    raise NotImplementedError("Day 3")


async def bulk_insert_chunks(
    *,
    doc_id: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    raise NotImplementedError("Day 3")
