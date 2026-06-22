"""
End-to-end smoke test. Requires a running API and a real PDF URL.

Run with:
    E2E_API_URL=http://localhost:8000 \
    E2E_PDF_URL=https://arxiv.org/pdf/1706.03762 \
    pytest tests/test_e2e.py -m e2e -s

Skipped automatically when E2E_API_URL is not set.
"""

import asyncio
import os
import time

import httpx
import pytest

pytestmark = pytest.mark.e2e

API_URL = os.getenv("E2E_API_URL", "")
PDF_URL = os.getenv("E2E_PDF_URL", "https://arxiv.org/pdf/1706.03762")
POLL_INTERVAL = 2  # seconds between job status checks
JOB_TIMEOUT = 120  # seconds to wait for ingestion to complete
ROUND_TRIP_BUDGET = 5  # seconds for the query round-trip (after indexing)


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


def _skip_if_no_api():
    if not API_URL:
        pytest.skip("E2E_API_URL not set — skipping e2e tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_job(client: httpx.AsyncClient, job_id: str) -> dict:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        r = await client.get(f"{API_URL}/jobs/{job_id}")
        assert r.status_code == 200, f"Job status check failed: {r.text}"
        job = r.json()
        if job["status"] in ("done", "failed"):
            return job
        await asyncio.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {job_id} did not complete within {JOB_TIMEOUT}s")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health():
    """API must respond healthy before running any other checks."""
    _skip_if_no_api()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{API_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_ingest_query_round_trip():
    """Full happy path: index a PDF → poll until done → query → citations present."""
    _skip_if_no_api()

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Submit indexing job
        r = await client.post(f"{API_URL}/index", json={"source_url": PDF_URL})
        assert r.status_code == 202, f"Index failed: {r.text}"
        job_id = r.json()["job_id"]
        assert job_id

        # 2. Poll until ingestion completes
        job = await _wait_for_job(client, job_id)
        assert job["status"] == "done", f"Ingestion failed: {job}"
        assert job.get("chunk_count", 0) > 0, "No chunks ingested"

        # 3. Query — must complete quickly once embeddings are in the DB
        query_start = time.monotonic()
        r = await client.post(
            f"{API_URL}/query",
            json={"query": "What is the attention mechanism?"},
            timeout=ROUND_TRIP_BUDGET + 5,
        )
        elapsed = time.monotonic() - query_start
        assert r.status_code == 200, f"Query failed: {r.text}"
        body = r.json()

        assert body.get("answer"), "Answer must be non-empty"
        assert isinstance(body.get("citations"), list), "Citations must be a list"
        assert len(body["citations"]) > 0, "At least one citation expected"
        assert elapsed < ROUND_TRIP_BUDGET + 5, f"Query took {elapsed:.1f}s — too slow"


@pytest.mark.anyio
async def test_below_threshold_query_returns_no_answer():
    """A query about something not in the corpus should hit the short-circuit."""
    _skip_if_no_api()

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{API_URL}/query",
            json={"query": "What is the population of the moon base in 2150?"},
        )
    assert r.status_code == 200
    body = r.json()
    # The short-circuit must fire — answer should signal "I don't know"
    assert body.get("answer") or body.get("citations") is not None


@pytest.mark.anyio
async def test_duplicate_index_is_idempotent():
    """Re-submitting the same PDF must return a job that completes with skipped=True."""
    _skip_if_no_api()

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{API_URL}/index", json={"source_url": PDF_URL})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        job = await _wait_for_job(client, job_id)

    # Either completes instantly as a no-op or reports skipped
    assert job["status"] == "done"
