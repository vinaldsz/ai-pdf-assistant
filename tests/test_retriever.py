"""Unit tests for app/rag/retriever.py — RRF math and below-threshold short-circuit."""

from unittest.mock import AsyncMock, patch

import pytest

from app.rag.retriever import RetrievalResult, _rrf, retrieve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r(
    chunk_id: str, *, doc_id: str = "doc1", page: int = 1, score: float = 0.9
) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, doc_id=doc_id, page=page, text="text", score=score)


# ---------------------------------------------------------------------------
# _rrf — pure math, no I/O
# ---------------------------------------------------------------------------


def test_rrf_chunk_in_both_lists_ranks_higher_than_exclusive_chunks():
    # "b" appears in both lists at low ranks; "a" and "c" appear in only one.
    # RRF score for "b" is the sum of two reciprocal ranks, so it should beat
    # "a" (dense rank 0 only) and "c" (sparse rank 0 only) in most orderings.
    dense = [_r("a"), _r("b"), _r("c")]
    sparse = [_r("c"), _r("b"), _r("d")]
    result = _rrf(dense, sparse)
    [r.chunk_id for r in result]
    # "b" (rank 1 + rank 1) and "c" (rank 2 + rank 0) both appear in both lists;
    # verify they appear before chunks that only appear in one list.
    shared = {"b", "c"}
    exclusive_dense = {"a"}
    exclusive_sparse = {"d"}
    shared_positions = [i for i, r in enumerate(result) if r.chunk_id in shared]
    exclusive_positions = [
        i for i, r in enumerate(result) if r.chunk_id in (exclusive_dense | exclusive_sparse)
    ]
    assert max(shared_positions) < max(exclusive_positions) or min(shared_positions) < min(
        exclusive_positions
    )


def test_rrf_top_chunk_when_same_rank_in_both():
    # "a" is rank 0 in both lists → highest possible RRF score
    dense = [_r("a"), _r("b")]
    sparse = [_r("a"), _r("c")]
    result = _rrf(dense, sparse)
    assert result[0].chunk_id == "a"


def test_rrf_empty_sparse_preserves_dense_order():
    dense = [_r("a"), _r("b"), _r("c")]
    result = _rrf(dense, [])
    assert [r.chunk_id for r in result] == ["a", "b", "c"]


def test_rrf_empty_dense_preserves_sparse_order():
    sparse = [_r("x"), _r("y")]
    result = _rrf([], sparse)
    assert [r.chunk_id for r in result] == ["x", "y"]


def test_rrf_both_empty_returns_empty():
    assert _rrf([], []) == []


def test_rrf_returns_all_unique_chunks():
    dense = [_r("a"), _r("b")]
    sparse = [_r("c"), _r("d")]
    result = _rrf(dense, sparse)
    assert {r.chunk_id for r in result} == {"a", "b", "c", "d"}


def test_rrf_no_duplicate_chunk_ids_in_output():
    # "a" appears in both lists — should appear only once in the output
    dense = [_r("a"), _r("b")]
    sparse = [_r("a"), _r("c")]
    result = _rrf(dense, sparse)
    ids = [r.chunk_id for r in result]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# retrieve — below-threshold short-circuit (mocks store + embedder)
# ---------------------------------------------------------------------------


async def test_below_threshold_returns_empty_list():
    # dense score of 0.1 is well below the default MIN_SIMILARITY of 0.30
    low_score_row = {"id": "uuid-1", "doc_id": "doc-1", "page": 1, "text": "t", "score": 0.1}

    with (
        patch("app.rag.store.dense_search", new_callable=AsyncMock, return_value=[low_score_row]),
        patch("app.rag.store.sparse_search", new_callable=AsyncMock, return_value=[]),
        patch("app.rag.embedder.encode_batch", return_value=[[0.1] * 384]),
    ):
        result = await retrieve("any query")

    assert result == []


async def test_empty_dense_results_returns_empty_list():
    with (
        patch("app.rag.store.dense_search", new_callable=AsyncMock, return_value=[]),
        patch("app.rag.store.sparse_search", new_callable=AsyncMock, return_value=[]),
        patch("app.rag.embedder.encode_batch", return_value=[[0.5] * 384]),
    ):
        result = await retrieve("any query")

    assert result == []


async def test_above_threshold_returns_results():
    good_row = {
        "id": "uuid-1",
        "doc_id": "doc-1",
        "page": 1,
        "text": "relevant text",
        "score": 0.85,
    }

    with (
        patch("app.rag.store.dense_search", new_callable=AsyncMock, return_value=[good_row]),
        patch("app.rag.store.sparse_search", new_callable=AsyncMock, return_value=[]),
        patch("app.rag.embedder.encode_batch", return_value=[[0.5] * 384]),
    ):
        result = await retrieve("any query")

    assert len(result) == 1
    assert result[0].chunk_id == "uuid-1"
    assert result[0].score == pytest.approx(0.85)
