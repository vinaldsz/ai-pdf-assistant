"""PDF ingestion pipeline: download → sha256 dedup → parse → chunk → embed → store."""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import socket
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network
from urllib.parse import urljoin, urlparse

import httpx
from pypdf import PdfReader

from app.rag.chunker import Chunk, chunk_text
from app.settings import settings

CHUNKER_VERSION = "recursive-v1"
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB — keeps peak RAM safe on 512 MB Fly VM
MAX_PDF_PAGES = 500  # ~30k chunks at this limit; beyond that risks OOM on 512 MB VM
_MAX_REDIRECTS = 5
_BLOCKED_NETWORKS = [
    ip_network("10.0.0.0/8"),  # RFC1918 private
    ip_network("172.16.0.0/12"),  # RFC1918 private
    ip_network("192.168.0.0/16"),  # RFC1918 private
    ip_network("169.254.0.0/16"),  # link-local / cloud metadata (AWS, GCP, Azure, Fly)
    ip_network("127.0.0.0/8"),  # loopback
    ip_network("0.0.0.0/8"),  # "this" network
    ip_network("::1/128"),  # IPv6 loopback
    ip_network("fc00::/7"),  # IPv6 unique-local (covers Fly.io fd00::/8)
]


@dataclass
class IngestResult:
    doc_id: str
    skipped: bool
    chunk_count: int


async def ingest_from_url(source_url: str) -> IngestResult:
    pdf_bytes = await _download(source_url)
    return await ingest_bytes(pdf_bytes, source_url=source_url)


async def ingest_bytes(pdf_bytes: bytes, *, source_url: str) -> IngestResult:
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    from app.rag import store  # implemented Day 3

    existing = await store.get_document_by_sha256(sha256)
    if existing is not None:
        return IngestResult(doc_id=str(existing["id"]), skipped=True, chunk_count=0)

    from app.ingest import r2

    await r2.upload(f"{sha256}.pdf", pdf_bytes)

    from app.rag import embedder  # implemented Day 3

    loop = asyncio.get_running_loop()
    # _parse_pdf is CPU-bound (pypdf can be slow on large/complex PDFs); run in a thread
    # so it cannot block the event loop and stall concurrent requests.
    try:
        pages = await asyncio.wait_for(
            loop.run_in_executor(None, _parse_pdf, pdf_bytes),
            timeout=60.0,
        )
    except TimeoutError:
        raise ValueError("PDF parsing timed out (60 s limit)") from None
    chunks = _chunk_pages(pages)
    texts = [c.text for c in chunks]
    vectors: list[list[float]] = await loop.run_in_executor(None, embedder.encode_batch, texts)

    doc_id = await store.insert_document_with_chunks(
        sha256=sha256,
        source_url=source_url,
        title=_extract_title(pdf_bytes),
        pages=len(pages),
        embedder_version=settings.embedding_model,
        chunker_version=CHUNKER_VERSION,
        chunks=chunks,
        vectors=vectors,
    )
    return IngestResult(doc_id=doc_id, skipped=False, chunk_count=len(chunks))


async def _validate_url(url: str) -> tuple[str, str]:
    """Validate url is safe to fetch. Returns (hostname, resolved_ip): the IP
    validated here is the one _download must connect to — resolving again at
    connect time would let DNS answer differently the second time (rebinding)."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only https:// URLs are accepted")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    ip = await _resolve_and_assert_host_is_public(hostname)
    return hostname, ip


async def _resolve_and_assert_host_is_public(hostname: str) -> str:
    # Bare IP submitted directly — check immediately without DNS lookup
    try:
        bare_addr = ip_address(hostname)
    except ValueError:
        bare_addr = None
    if bare_addr is not None:
        _assert_ip_is_public(bare_addr)
        return str(bare_addr)

    # Resolve all DNS records and check every returned address, so a hostname
    # can't sneak a blocked address in as one of several A/AAAA records.
    # run_in_executor offloads the blocking getaddrinfo call to a thread.
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.run_in_executor(None, socket.getaddrinfo, hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname {hostname!r}") from exc

    resolved = [ip_address(sockaddr[0]) for *_, sockaddr in infos]
    for addr in resolved:
        _assert_ip_is_public(addr)
    return str(resolved[0])


def _assert_ip_is_public(addr: IPv4Address | IPv6Address) -> None:
    # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:169.254.169.254) before
    # the blocklist check — otherwise they bypass all IPv4 network rules.
    if isinstance(addr, IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    for net in _BLOCKED_NETWORKS:
        if addr in net:
            raise ValueError(f"URL resolves to a blocked address ({addr})")


def _pin_to_ip(url: str, ip: str) -> str:
    """Rewrite url's host to the already-validated IP, preserving scheme/port/path.
    The caller must send the original hostname via the Host header and SNI —
    connecting to a bare IP skips DNS (and thus rebinding) entirely."""
    parsed = urlparse(url)
    host = f"[{ip}]" if ":" in ip else ip
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return parsed._replace(netloc=netloc).geturl()


async def _download(url: str) -> bytes:
    hostname, ip = await _validate_url(url)

    base_headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-PDF-Assistant/1.0)"}
    async with httpx.AsyncClient(follow_redirects=False, timeout=60.0) as client:
        current_url = url
        for hop in range(_MAX_REDIRECTS + 1):
            request_headers = {**base_headers, "Host": hostname}
            async with client.stream(
                "GET",
                _pin_to_ip(current_url, ip),
                headers=request_headers,
                extensions={"sni_hostname": hostname},
            ) as response:
                if response.is_redirect:
                    if hop == _MAX_REDIRECTS:
                        raise ValueError(f"Too many redirects (max {_MAX_REDIRECTS})")
                    location = response.headers.get("location", "")
                    if not location:
                        raise ValueError("Redirect response missing Location header")
                    # urljoin handles both absolute and relative Location values
                    next_url = urljoin(current_url, location)
                    hostname, ip = await _validate_url(next_url)
                    current_url = next_url
                    continue  # exits this async with, re-enters loop with validated URL

                response.raise_for_status()

                content_length = int(response.headers.get("content-length", 0))
                if content_length > MAX_PDF_BYTES:
                    raise ValueError(
                        f"PDF too large: {content_length / 1024 / 1024:.1f} MB "
                        f"(limit {MAX_PDF_BYTES // 1024 // 1024} MB)"
                    )

                # Stream body — abort mid-download if server lied about Content-Length
                buf: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes(chunk_size=65_536):
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise ValueError(
                            f"PDF exceeded {MAX_PDF_BYTES // 1024 // 1024} MB limit while streaming"
                        )
                    buf.append(chunk)

                return b"".join(buf)

    raise ValueError(f"Too many redirects (max {_MAX_REDIRECTS})")


def _parse_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    if not pdf_bytes[:4] == b"%PDF":
        raise ValueError(f"Response is not a PDF (starts with {pdf_bytes[:20]!r})")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"PDF could not be parsed: {exc}") from exc
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(
            f"PDF has {len(reader.pages)} pages (limit {MAX_PDF_PAGES}). "
            "Split the document and re-submit each part."
        )
    _REFS_PATTERN = re.compile(r"(?m)^(References|Bibliography|Works Cited)\s*$")
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        # Stop at the references/bibliography section — it pollutes retrieval
        # by matching citation text instead of methodology content.
        # Truncate the page at the heading if it appears mid-page.
        match = _REFS_PATTERN.search(text)
        if match:
            pre = text[: match.start()].strip()
            if pre:
                pages.append((i + 1, pre))
            break
        pages.append((i + 1, text))
    return pages


def _chunk_pages(pages: list[tuple[int, str]]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page_num, text in pages:
        all_chunks.extend(chunk_text(text, page=page_num))
    return [Chunk(text=c.text, page=c.page, index=i) for i, c in enumerate(all_chunks)]


def _extract_title(pdf_bytes: bytes) -> str | None:
    try:
        meta = PdfReader(io.BytesIO(pdf_bytes)).metadata
        return meta.title if meta and meta.title else None
    except Exception:
        return None
