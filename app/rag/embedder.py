"""Lazy-loaded local embedder using BAAI/bge-small-en-v1.5 via sentence-transformers."""
from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.settings import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def encode_batch(texts: list[str]) -> list[list[float]]:
    """Encode texts to normalized vectors. CPU-bound — call via run_in_executor from async code."""
    if not texts:
        return []
    embeddings = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()  # type: ignore[return-value]


async def warmup() -> None:
    """Load and warm the model so the first real request doesn't stall the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, encode_batch, ["warmup"])
