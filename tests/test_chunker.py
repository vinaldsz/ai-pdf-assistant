"""Unit tests for app/rag/chunker.py — pure logic, no DB or network."""
from app.rag.chunker import Chunk, chunk_text


def test_empty_string_returns_empty():
    assert chunk_text("", page=1) == []


def test_whitespace_only_returns_empty():
    assert chunk_text("   \n\n   ", page=1) == []


def test_text_shorter_than_chunk_size():
    result = chunk_text("Hello world", page=1, chunk_size=200, chunk_overlap=0)
    assert len(result) == 1
    assert result[0].text == "Hello world"
    assert result[0].page == 1
    assert result[0].index == 0


def test_exact_chunk_size_produces_one_chunk():
    # 800 chars with no separators — fits exactly in one 800-char window
    text = "a" * 800
    result = chunk_text(text, page=1, chunk_size=800, chunk_overlap=0)
    assert len(result) == 1
    assert len(result[0].text) == 800


def test_page_and_index_metadata_preserved():
    text = ("word " * 40).strip()  # ~200 chars
    chunks = chunk_text(text, page=5, chunk_size=50, chunk_overlap=0)
    assert len(chunks) >= 2
    assert all(c.page == 5 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_carry_over():
    # 10 x 8-char words = 89 chars; chunk_size=40 forces a split, overlap=10 carries tail
    text = " ".join(["abcdefgh"] * 10)
    chunks = chunk_text(text, page=1, chunk_size=40, chunk_overlap=10)
    assert len(chunks) >= 2
    tail = chunks[0].text[-10:]
    assert tail.strip() in chunks[1].text


def test_no_natural_separators_falls_back_to_hard_split():
    # A solid run of 'a' chars — no whitespace or punctuation to split on.
    # The empty-string separator hard-splits into 200-char pieces.
    text = "a" * 1000
    chunks = chunk_text(text, page=1, chunk_size=200, chunk_overlap=0)
    assert len(chunks) == 5
    assert all(len(c.text) == 200 for c in chunks)


def test_paragraph_boundaries_preferred_over_mid_sentence():
    # Splitter should break at \n\n before resorting to ". " or space
    first = "First paragraph with some content here."
    second = "Second paragraph with different content."
    text = f"{first}\n\n{second}"
    chunks = chunk_text(text, page=1, chunk_size=50, chunk_overlap=0)
    texts = [c.text for c in chunks]
    assert any("First paragraph" in t for t in texts)
    assert any("Second paragraph" in t for t in texts)


def test_chunk_text_returns_chunk_dataclass_instances():
    result = chunk_text("some text", page=2, chunk_size=200, chunk_overlap=0)
    assert all(isinstance(c, Chunk) for c in result)
