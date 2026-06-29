"""Integration tests for store.py against a real pgvector database (testcontainers).

These tests spin up a real pgvector Postgres container, apply the schema directly
via asyncpg (bypassing alembic's env.py which reads from app.settings), and exercise
the store layer end-to-end: insert, dedup, retrieval, and transactional rollback.

Requires Docker to be running. Skip with: pytest -m "not integration"
"""
from __future__ import annotations

import asyncio

import asyncpg
import pytest
from pgvector.asyncpg import register_vector
from testcontainers.postgres import PostgresContainer

import app.rag.store as _store
from app.rag.chunker import Chunk

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Schema setup (mirrors migrations/versions/0001_baseline.py)
# ---------------------------------------------------------------------------


async def _apply_schema(conn: asyncpg.Connection) -> None:  # type: ignore[type-arg]
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            sha256          TEXT        NOT NULL,
            source_url      TEXT        NOT NULL,
            title           TEXT,
            pages           INTEGER,
            embedder_version TEXT       NOT NULL,
            chunker_version  TEXT       NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT documents_sha256_unique UNIQUE (sha256)
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id        UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id    UUID    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page      INTEGER NOT NULL,
            text      TEXT    NOT NULL,
            embedding vector(384),
            tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres() -> PostgresContainer:  # type: ignore[misc]
    """Start a pgvector Postgres container and apply the schema once per session."""
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        dsn = container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        async def _setup() -> None:
            conn = await asyncpg.connect(dsn)
            try:
                await _apply_schema(conn)
            finally:
                await conn.close()

        asyncio.run(_setup())
        yield container  # type: ignore[misc]


@pytest.fixture()
async def pool(postgres: PostgresContainer) -> asyncpg.Pool:  # type: ignore[type-arg]
    """Per-test asyncpg pool patched into store._pool. Cleans all rows after each test."""
    dsn = postgres.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

    async def _init(conn: asyncpg.Connection) -> None:  # type: ignore[type-arg]
        await register_vector(conn)

    _pool = await asyncpg.create_pool(dsn, init=_init, min_size=1, max_size=5)  # type: ignore[assignment]
    original = _store._pool
    _store._pool = _pool  # type: ignore[assignment]

    yield _pool  # type: ignore[misc]

    async with _pool.acquire() as conn:
        await conn.execute("DELETE FROM chunks")
        await conn.execute("DELETE FROM documents")

    await _pool.close()
    _store._pool = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks(n: int = 2) -> tuple[list[Chunk], list[list[float]]]:
    return (
        [Chunk(text=f"chunk text {i}", page=1, index=i) for i in range(n)],
        [[0.1] * 384 for _ in range(n)],
    )


async def _insert(sha256: str = "abc123", n: int = 2) -> str:
    chunks, vectors = _chunks(n)
    return await _store.insert_document_with_chunks(
        sha256=sha256,
        source_url="https://example.com/test.pdf",
        title="Test PDF",
        pages=5,
        embedder_version="bge-small",
        chunker_version="recursive-v1",
        chunks=chunks,
        vectors=vectors,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_insert_and_retrieve_chunks(pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
    """Documents and chunks round-trip correctly."""
    doc_id = await _insert(sha256="sha_roundtrip", n=3)

    assert doc_id  # non-empty UUID string

    stored = await _store.get_chunks_by_doc(doc_id)
    assert len(stored) == 3
    texts = {r["text"] for r in stored}
    assert "chunk text 0" in texts
    assert "chunk text 2" in texts


async def test_get_document_by_sha256_found(pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
    """get_document_by_sha256 returns the document when sha256 matches."""
    await _insert(sha256="sha_find_me")

    doc = await _store.get_document_by_sha256("sha_find_me")
    assert doc is not None
    assert doc["sha256"] == "sha_find_me"
    assert doc["source_url"] == "https://example.com/test.pdf"


async def test_get_document_by_sha256_missing(pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
    """get_document_by_sha256 returns None for an unknown sha256."""
    result = await _store.get_document_by_sha256("does_not_exist")
    assert result is None


async def test_sha256_dedup_raises_on_duplicate(pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
    """Inserting the same sha256 twice raises a unique constraint error."""
    await _insert(sha256="sha_dedup")

    with pytest.raises(Exception):  # asyncpg.UniqueViolationError
        await _insert(sha256="sha_dedup")

    # The first insert is still intact
    doc = await _store.get_document_by_sha256("sha_dedup")
    assert doc is not None


async def test_transaction_rollback_on_bad_vector(pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
    """If chunk insert fails, the document row is also rolled back (atomic transaction)."""
    chunks = [Chunk(text="test chunk", page=1, index=0)]
    bad_vectors = [[0.1] * 100]  # wrong dimension — vector(384) column rejects this

    with pytest.raises(Exception):
        await _store.insert_document_with_chunks(
            sha256="sha_rollback",
            source_url="https://example.com/rollback.pdf",
            title=None,
            pages=1,
            embedder_version="bge-small",
            chunker_version="recursive-v1",
            chunks=chunks,
            vectors=bad_vectors,
        )

    # The document must not have been persisted
    doc = await _store.get_document_by_sha256("sha_rollback")
    assert doc is None


async def test_dense_search_returns_results(pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
    """dense_search returns chunks ordered by cosine similarity."""
    await _insert(sha256="sha_dense", n=3)

    results = await _store.dense_search(query_vector=[0.1] * 384, limit=2)
    assert len(results) == 2
    assert all("score" in r for r in results)
    assert all("text" in r for r in results)


async def test_sparse_search_returns_results(pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
    """sparse_search returns chunks that match the query terms."""
    chunks = [Chunk(text="the transformer architecture uses attention", page=1, index=0)]
    vectors = [[0.1] * 384]
    await _store.insert_document_with_chunks(
        sha256="sha_sparse",
        source_url="https://example.com/sparse.pdf",
        title=None,
        pages=1,
        embedder_version="bge-small",
        chunker_version="recursive-v1",
        chunks=chunks,
        vectors=vectors,
    )

    results = await _store.sparse_search(query="transformer attention", limit=5)
    assert len(results) >= 1
    assert results[0]["text"] == "the transformer architecture uses attention"
