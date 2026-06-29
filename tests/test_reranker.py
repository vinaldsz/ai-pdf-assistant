"""Tests for app/rag/reranker.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.reranker import _rerank_sync
from app.rag.retriever import RetrievalResult


def _chunk(chunk_id: str, text: str = "text", score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, doc_id="doc1", page=1, text=text, score=score)


def _mock_model(scores: list[float]) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = np.array(scores)
    return model


# ---------------------------------------------------------------------------
# Fast unit tests — model is mocked, no torch load
# ---------------------------------------------------------------------------


def test_rerank_empty_input_returns_empty():
    with patch("app.rag.reranker._model") as mock:
        result = _rerank_sync("query", [], k=5)
    mock.assert_not_called()
    assert result == []


def test_rerank_returns_exactly_k_results():
    chunks = [_chunk(str(i)) for i in range(5)]
    scores = [0.1, 0.9, 0.3, 0.8, 0.5]
    with patch("app.rag.reranker._model", return_value=_mock_model(scores)):
        result = _rerank_sync("query", chunks, k=3)
    assert len(result) == 3


def test_rerank_orders_by_score_descending():
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    scores = [0.2, 0.9, 0.5]  # b is highest, then c, then a
    with patch("app.rag.reranker._model", return_value=_mock_model(scores)):
        result = _rerank_sync("query", chunks, k=3)
    assert result[0].chunk_id == "b"
    assert result[1].chunk_id == "c"
    assert result[2].chunk_id == "a"


def test_rerank_k_larger_than_input_returns_all():
    chunks = [_chunk(str(i)) for i in range(3)]
    scores = [0.9, 0.5, 0.1]
    with patch("app.rag.reranker._model", return_value=_mock_model(scores)):
        result = _rerank_sync("query", chunks, k=10)
    assert len(result) == 3


def test_rerank_passes_query_chunk_pairs_to_model():
    chunks = [_chunk("a", text="alpha"), _chunk("b", text="beta")]
    with patch("app.rag.reranker._model", return_value=_mock_model([0.5, 0.9])) as mock_factory:
        _rerank_sync("my query", chunks, k=2)
    call_args = mock_factory.return_value.predict.call_args
    pairs = call_args[0][0]
    assert pairs == [("my query", "alpha"), ("my query", "beta")]


# ---------------------------------------------------------------------------
# Slow tests — loads the real ~570 MB bge-reranker-base model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reranker_ranks_more_relevant_chunk_higher():
    """A chunk that directly answers the query must rank above an off-topic chunk."""
    from app.rag.reranker import _rerank_sync as real_rerank

    relevant = _chunk("rel", text="The Transformer model uses multi-head self-attention.")
    irrelevant = _chunk("irr", text="The recipe calls for two cups of flour and one egg.")

    result = real_rerank(
        "How does the Transformer attention mechanism work?", [irrelevant, relevant], k=2
    )

    assert result[0].chunk_id == "rel", "More relevant chunk should rank first"


@pytest.mark.slow
async def test_reranker_async_wrapper_returns_top_k():
    from app.rag.reranker import rerank

    chunks = [_chunk(str(i), text=f"chunk number {i}") for i in range(8)]
    result = await rerank("test query", chunks, k=3)
    assert len(result) == 3
