from __future__ import annotations

from dataclasses import dataclass

from app.settings import settings

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int  # ordinal position within the document


def chunk_text(
    text: str,
    page: int,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    pieces = _split(text.strip(), size, _SEPARATORS)
    windows = _merge(pieces, size, overlap)
    return [Chunk(text=w, page=page, index=i) for i, w in enumerate(windows) if w.strip()]


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively split text, trying each separator in order until pieces fit in `size`."""
    if len(text) <= size:
        return [text] if text.strip() else []

    sep, *rest = separators

    if not sep:
        # Last resort: hard character split
        return [text[i : i + size] for i in range(0, len(text), size)]

    pieces: list[str] = []
    for part in text.split(sep):
        part = part.strip()
        if not part:
            continue
        if len(part) <= size:
            pieces.append(part)
        else:
            pieces.extend(_split(part, size, rest or [""]))

    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Merge small pieces into windows of at most `size` chars, with `overlap` tail carry-over."""
    if not pieces:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for piece in pieces:
        plen = len(piece)
        sep_cost = 1 if buf else 0
        if buf and buf_len + sep_cost + plen > size:
            chunk = " ".join(buf)
            chunks.append(chunk)
            tail = chunk[-overlap:].strip() if overlap else ""
            buf = [tail, piece] if tail else [piece]
            buf_len = (len(tail) + 1 + plen) if tail else plen
        else:
            buf.append(piece)
            buf_len += sep_cost + plen

    if buf:
        chunks.append(" ".join(buf))

    return chunks
