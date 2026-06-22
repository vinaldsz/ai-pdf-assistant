"""Cross-encoder reranker using BAAI/bge-reranker-base.

Runs in a thread (CPU-bound). Same lazy-load pattern as embedder.py so that
importing this module never triggers a torch load.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from app.rag.retriever import RetrievalResult  # lightweight — no torch dep
from app.settings import settings

_warmed: bool = False  # set True after warmup(); checked by /ready


@lru_cache(maxsize=1)
def _model():  # type: ignore[return]
    from sentence_transformers import CrossEncoder  # lazy — avoids torch at import time
    return CrossEncoder(settings.reranker_model, device="cpu")


def _rerank_sync(query: str, chunks: list[RetrievalResult], k: int) -> list[RetrievalResult]:
    """Score every (query, chunk) pair and return the top-k by score.

    CrossEncoder scores are raw logits — not cosine similarities — so we only
    use them for ordering, not for the below-threshold check.
    """
    if not chunks:
        return []
    pairs = [(query, c.text) for c in chunks]
    scores = _model().predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:k]]


async def rerank(
    query: str,
    chunks: list[RetrievalResult],
    k: int | None = None,
) -> list[RetrievalResult]:
    """Async wrapper — offloads CPU-bound scoring to a thread pool executor."""
    top = k if k is not None else settings.rerank_k
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _rerank_sync, query, chunks, top)


async def warmup() -> None:
    """Load model at startup so the first real request doesn't block."""
    global _warmed
    dummy = [RetrievalResult(chunk_id="0", doc_id="0", page=1, text="warmup", score=1.0)]
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _rerank_sync, "warmup", dummy, 1)
    _warmed = True
