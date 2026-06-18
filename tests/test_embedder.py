"""Unit tests for app/rag/embedder.py.

Tests marked `slow` load the bge-small-en-v1.5 model (~90 MB first download).
Run fast suite only: uv run pytest -m 'not slow'
Run everything:     uv run pytest
"""
import math

import pytest


def test_encode_batch_empty_input_returns_empty():
    """No model load needed — the early-return guard runs before _model()."""
    from app.rag.embedder import encode_batch
    assert encode_batch([]) == []


@pytest.mark.slow
def test_encode_batch_returns_correct_number_of_vectors():
    from app.rag.embedder import encode_batch
    texts = ["Hello world", "Another sentence", "Third one"]
    result = encode_batch(texts)
    assert len(result) == 3


@pytest.mark.slow
def test_encode_batch_returns_384_dimensional_vectors():
    from app.rag.embedder import encode_batch
    result = encode_batch(["test sentence"])
    assert len(result[0]) == 384


@pytest.mark.slow
def test_encode_batch_vectors_are_unit_length():
    # normalize_embeddings=True means ‖v‖₂ ≈ 1.0 — critical for cosine similarity
    from app.rag.embedder import encode_batch
    result = encode_batch(["unit length check"])
    vec = result[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5


@pytest.mark.slow
def test_encode_batch_is_deterministic():
    from app.rag.embedder import encode_batch
    r1 = encode_batch(["consistent input"])
    r2 = encode_batch(["consistent input"])
    assert r1 == r2


@pytest.mark.slow
def test_encode_batch_different_texts_produce_different_vectors():
    from app.rag.embedder import encode_batch
    result = encode_batch(["cats and dogs", "quantum mechanics"])
    assert result[0] != result[1]
