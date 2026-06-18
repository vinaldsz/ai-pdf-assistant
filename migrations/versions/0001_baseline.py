"""baseline schema: documents + chunks

Revision ID: 0001
Revises:
Create Date: 2026-06-16

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE documents (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            sha256      TEXT        NOT NULL,
            source_url  TEXT        NOT NULL,
            title       TEXT,
            pages       INTEGER,
            embedder_version TEXT   NOT NULL,
            chunker_version  TEXT   NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT documents_sha256_unique UNIQUE (sha256)
        )
    """)

    op.execute("""
        CREATE TABLE chunks (
            id        UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id    UUID    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page      INTEGER NOT NULL,
            text      TEXT    NOT NULL,
            embedding vector(384),
            tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        )
    """)

    # HNSW index with explicit params — pgvector defaults are conservative for 384-dim vectors
    op.execute("""
        CREATE INDEX chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    op.execute("CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
