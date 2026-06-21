"""Unit tests for app/rag/generator.py — prompt ceiling, pure logic, no Groq calls."""
from app.rag.generator import _MAX_CONTEXT_CHARS, _build_context


def _r(text: str, page: int = 1, score: float = 0.9):  # type: ignore[return]
    from app.rag.retriever import RetrievalResult
    return RetrievalResult(chunk_id="cid", doc_id="did", page=page, text=text, score=score)


def test_build_context_empty_returns_empty_string():
    assert _build_context([]) == ""


def test_build_context_includes_page_number():
    context = _build_context([_r("some text", page=7)])
    assert "page 7" in context


def test_build_context_includes_chunk_text():
    context = _build_context([_r("important content")])
    assert "important content" in context


def test_build_context_ceiling_enforced_on_single_large_chunk():
    # One chunk whose text alone exceeds the ceiling
    context = _build_context([_r("x" * (_MAX_CONTEXT_CHARS + 500))])
    assert len(context) <= _MAX_CONTEXT_CHARS


def test_build_context_ceiling_enforced_across_many_chunks():
    # 20 chunks × 500 chars = 10,000 chars total, well over the 6,000 ceiling
    chunks = [_r("a" * 500, page=i) for i in range(20)]
    context = _build_context(chunks)
    assert len(context) <= _MAX_CONTEXT_CHARS


def test_build_context_small_chunks_all_fit():
    # 3 short chunks should all appear in the output
    chunks = [_r(f"chunk {i} text", page=i) for i in range(3)]
    context = _build_context(chunks)
    for i in range(3):
        assert f"chunk {i} text" in context


def test_build_context_ceiling_independent_of_chunk_count():
    # Ceiling holds whether there are 5 chunks or 50
    for n in (5, 20, 50):
        chunks = [_r("b" * 300) for _ in range(n)]
        context = _build_context(chunks)
        assert len(context) <= _MAX_CONTEXT_CHARS, f"ceiling exceeded with {n} chunks"
