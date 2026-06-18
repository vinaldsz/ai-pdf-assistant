"""Lazy-loaded local embedder using BAAI/bge-small-en-v1.5 via sentence-transformers."""
from __future__ import annotations


def encode_batch(texts: list[str]) -> list[list[float]]:
    raise NotImplementedError("Day 3")


async def warmup() -> None:
    raise NotImplementedError("Day 3")
