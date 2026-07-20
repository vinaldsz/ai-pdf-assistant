"""Unit tests for app/ingest/pdf.py — URL validation and sha256 dedup.

These tests are network-free: bare IP tests bypass DNS. The dedup test mocks
the store so no DB connection is needed.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.ingest.pdf import _pin_to_ip, _validate_url, ingest_bytes

# ---------------------------------------------------------------------------
# _validate_url — scheme enforcement
# ---------------------------------------------------------------------------


async def test_rejects_http_scheme():
    with pytest.raises(ValueError, match="Only https://"):
        await _validate_url("http://example.com/file.pdf")


async def test_rejects_ftp_scheme():
    with pytest.raises(ValueError, match="Only https://"):
        await _validate_url("ftp://example.com/file.pdf")


async def test_rejects_no_hostname():
    with pytest.raises(ValueError):
        await _validate_url("https:///file.pdf")


# ---------------------------------------------------------------------------
# _validate_url — RFC1918 + special-purpose blocks (bare IPs, no DNS needed)
# ---------------------------------------------------------------------------


async def test_rejects_rfc1918_10_block():
    with pytest.raises(ValueError, match="blocked"):
        await _validate_url("https://10.0.0.1/file.pdf")


async def test_rejects_rfc1918_172_block():
    with pytest.raises(ValueError, match="blocked"):
        await _validate_url("https://172.16.0.1/file.pdf")


async def test_rejects_rfc1918_192_block():
    with pytest.raises(ValueError, match="blocked"):
        await _validate_url("https://192.168.1.1/file.pdf")


async def test_rejects_link_local_metadata_endpoint():
    # 169.254.169.254 is the AWS/GCP/Azure/Fly instance-metadata endpoint
    with pytest.raises(ValueError, match="blocked"):
        await _validate_url("https://169.254.169.254/latest/meta-data/")


async def test_rejects_loopback():
    with pytest.raises(ValueError, match="blocked"):
        await _validate_url("https://127.0.0.1/file.pdf")


async def test_rejects_ipv6_loopback():
    with pytest.raises(ValueError, match="blocked"):
        await _validate_url("https://[::1]/file.pdf")


async def test_accepts_public_ip():
    # 93.184.216.34 is example.com — clearly public, no DNS lookup needed
    await _validate_url("https://93.184.216.34/file.pdf")


# ---------------------------------------------------------------------------
# _validate_url — returns the resolved IP so _download can pin the connection
# to it, closing the DNS-rebinding gap between validation and connect.
# ---------------------------------------------------------------------------


async def test_validate_url_returns_hostname_and_ip_for_bare_ip():
    hostname, ip = await _validate_url("https://93.184.216.34/file.pdf")
    assert hostname == "93.184.216.34"
    assert ip == "93.184.216.34"


# ---------------------------------------------------------------------------
# _pin_to_ip — rewrites the URL host to a pre-validated IP without touching
# scheme, port, or path
# ---------------------------------------------------------------------------


def test_pin_to_ip_ipv4():
    assert _pin_to_ip("https://example.com/file.pdf", "93.184.216.34") == (
        "https://93.184.216.34/file.pdf"
    )


def test_pin_to_ip_preserves_explicit_port():
    assert _pin_to_ip("https://example.com:8443/file.pdf", "93.184.216.34") == (
        "https://93.184.216.34:8443/file.pdf"
    )


def test_pin_to_ip_ipv6_gets_bracketed():
    assert _pin_to_ip("https://example.com/file.pdf", "2606:2800:220:1:1:1:1:1") == (
        "https://[2606:2800:220:1:1:1:1:1]/file.pdf"
    )


# ---------------------------------------------------------------------------
# ingest_bytes — sha256 dedup
# ---------------------------------------------------------------------------


async def test_sha256_dedup_returns_skipped_on_second_call():
    fake_pdf = b"fake pdf bytes for sha256 dedup test"
    existing_doc = {"id": "123e4567-e89b-12d3-a456-426614174000"}

    with patch(
        "app.rag.store.get_document_by_sha256",
        new_callable=AsyncMock,
        return_value=existing_doc,
    ):
        result = await ingest_bytes(fake_pdf, source_url="https://example.com/doc.pdf")

    assert result.skipped is True
    assert result.chunk_count == 0
    assert result.doc_id == str(existing_doc["id"])


async def test_sha256_dedup_does_not_call_insert_when_duplicate():
    fake_pdf = b"another fake payload"
    existing_doc = {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}

    with (
        patch(
            "app.rag.store.get_document_by_sha256",
            new_callable=AsyncMock,
            return_value=existing_doc,
        ) as mock_get,
        patch(
            "app.rag.store.insert_document_with_chunks",
            new_callable=AsyncMock,
        ) as mock_insert,
    ):
        await ingest_bytes(fake_pdf, source_url="https://example.com/dup.pdf")

    mock_get.assert_called_once()
    mock_insert.assert_not_called()
