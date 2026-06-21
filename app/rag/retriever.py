"""Hybrid retriever: dense cosine (pgvector) + sparse tsvector, fused with RRF."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.settings import settings

# embedder and store imported lazily inside retrieve() — keeps this module import-time cheap
# (no sentence-transformers or asyncpg loaded until retrieve() is actually called).

_RRF_K = 60  # standard RRF constant; higher = smoother rank blending


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    score: float  # cosine similarity from dense leg (used for below-threshold check)


async def retrieve(query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
    """
    Returns up to `top_k` chunks ranked by RRF fusion of dense + sparse search.
    Returns an empty list when the best dense score < MIN_SIMILARITY (triggers
    the "I don't know" path in the generator — no LLM call is made).
    """
    from app.rag import embedder, store  # noqa: PLC0415

    k = top_k if top_k is not None else settings.top_k

    # Embed query in a thread — encode_batch is CPU-bound and would block the event loop
    loop = asyncio.get_running_loop()
    query_vector: list[float] = (
        await loop.run_in_executor(None, embedder.encode_batch, [query])
    )[0]

    # Run dense and sparse searches concurrently
    dense_rows, sparse_rows = await asyncio.gather(
        store.dense_search(query_vector, k),
        store.sparse_search(query, k),
    )

    if not dense_rows:
        return []

    # Below-threshold short-circuit: skip LLM if retrieval confidence is too low
    best_score = float(dense_rows[0]["score"])
    if best_score < settings.min_similarity:
        return []

    dense = [_to_result(r) for r in dense_rows]
    sparse = [_to_result(r) for r in sparse_rows]

    return _rrf(dense, sparse)[:k]


def _to_result(row: dict) -> RetrievalResult:  # type: ignore[type-arg]
    return RetrievalResult(
        chunk_id=str(row["id"]),
        doc_id=str(row["doc_id"]),
        page=int(row["page"]),
        text=str(row["text"]),
        score=float(row["score"]),
    )


def _rrf(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion — combines two ranked lists without requiring score normalisation."""
    rrf_scores: dict[str, float] = {}
    by_id: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(dense):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_id[result.chunk_id] = result

    for rank, result in enumerate(sparse):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_id[result.chunk_id] = result

    return sorted(by_id.values(), key=lambda r: rrf_scores[r.chunk_id], reverse=True)
