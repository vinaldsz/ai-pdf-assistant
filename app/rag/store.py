"""pgvector read/write helpers backed by an asyncpg connection pool."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.rag.chunker import Chunk
from app.settings import settings

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                _pool = await _create_pool()
    return _pool


async def _create_pool() -> asyncpg.Pool:
    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    dsn, needs_ssl = _dsn()
    return await asyncpg.create_pool(
        dsn, init=_init, min_size=1, max_size=10, ssl=needs_ssl or None
    )


def _dsn() -> tuple[str, bool]:
    from urllib.parse import urlparse, urlunparse

    url = str(settings.database_url)
    url = url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )
    # Strip all query params — asyncpg rejects Neon's sslmode/channel_binding params.
    # SSL is passed explicitly via ssl= in create_pool().
    parsed = urlparse(url)
    needs_ssl = "sslmode=disable" not in (parsed.query or "") and parsed.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    )
    return urlunparse(parsed._replace(query="")), needs_ssl


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


async def get_document_by_sha256(sha256: str) -> dict[str, Any] | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, sha256, source_url, title, pages FROM documents WHERE sha256 = $1",
            sha256,
        )
        return dict(row) if row else None


async def get_chunks_by_doc(doc_id: str) -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, page, text FROM chunks WHERE doc_id = $1 ORDER BY id",
            uuid.UUID(doc_id),
        )
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


async def insert_document_with_chunks(
    *,
    sha256: str,
    source_url: str,
    title: str | None,
    pages: int | None,
    embedder_version: str,
    chunker_version: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> str:
    """Insert document + chunks in a single transaction — prevents orphaned document rows."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO documents
                    (sha256, source_url, title, pages, embedder_version, chunker_version)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                sha256,
                source_url,
                title,
                pages,
                embedder_version,
                chunker_version,
            )
            doc_id = str(row["id"])

            if chunks:
                await conn.executemany(
                    """
                    INSERT INTO chunks (doc_id, page, text, embedding)
                    VALUES ($1, $2, $3, $4)
                    """,
                    [
                        (
                            uuid.UUID(doc_id),
                            chunk.page,
                            chunk.text,
                            np.array(vector, dtype=np.float32),
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                )

            return doc_id


# ---------------------------------------------------------------------------
# Retrieval helpers (used by retriever.py)
# ---------------------------------------------------------------------------


async def dense_search(
    query_vector: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """Cosine similarity search via pgvector. Returns rows ordered by descending similarity."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, doc_id, page, text,
                   1 - (embedding <=> $1) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            np.array(query_vector, dtype=np.float32),
            limit,
        )
        return [dict(r) for r in rows]


async def sparse_search(query: str, limit: int) -> list[dict[str, Any]]:
    """Full-text keyword search via tsvector. Returns rows ordered by ts_rank_cd.

    Uses OR between query terms (via websearch_to_tsquery) so multi-term queries
    return chunks matching ANY term — ts_rank_cd naturally ranks chunks that match
    MORE terms higher. AND logic (plainto_tsquery) misses too many chunks in small
    corpora where related terms land in adjacent chunks rather than the same one.
    """
    # Join with "OR" so websearch_to_tsquery produces term1 | term2 | ...
    or_query = " OR ".join(query.split())
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, doc_id, page, text,
                   ts_rank_cd(tsv, websearch_to_tsquery('english', $1)) AS score
            FROM chunks
            WHERE tsv @@ websearch_to_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
            """,
            or_query,
            limit,
        )
        return [dict(r) for r in rows]
