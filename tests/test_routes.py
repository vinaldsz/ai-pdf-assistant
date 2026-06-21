"""Route-level tests — no real DB, Groq, or model calls."""
import pytest
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    """Test client with embedder warmup mocked so no model loads during lifespan."""
    with patch("app.rag.embedder.warmup", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

async def test_health_returns_200(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_health_sets_request_id_header(client: AsyncClient):
    response = await client.get("/health")
    assert "X-Request-ID" in response.headers


# ---------------------------------------------------------------------------
# /ready
# ---------------------------------------------------------------------------

async def test_ready_returns_503_when_db_unreachable(client: AsyncClient):
    with patch("app.rag.store.get_pool", side_effect=Exception("connection refused")), \
         patch("app.rag.embedder._warmed", new=True):
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["db"].startswith("error:")


async def test_ready_returns_503_when_embedder_not_warm(client: AsyncClient):
    # _warmed=False means the embedder check fails regardless of DB state
    with patch("app.rag.embedder._warmed", new=False), \
         patch("app.rag.store.get_pool", side_effect=Exception("skip db")):
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["embedder"] == "not warm"


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

async def test_query_empty_retrieval_returns_no_answer(client: AsyncClient):
    with patch("app.rag.retriever.retrieve", new_callable=AsyncMock, return_value=[]):
        response = await client.post("/query", json={"query": "what is X?"})

    assert response.status_code == 200
    data = response.json()
    assert data["citations"] == []
    assert "don't" in data["answer"].lower()


async def test_query_empty_retrieval_does_not_call_groq(client: AsyncClient):
    with patch("app.rag.retriever.retrieve", new_callable=AsyncMock, return_value=[]), \
         patch("app.rag.generator.AsyncGroq") as mock_groq:
        await client.post("/query", json={"query": "anything"})

    mock_groq.assert_not_called()


async def test_query_missing_body_returns_422(client: AsyncClient):
    response = await client.post("/query", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /index + /jobs
# ---------------------------------------------------------------------------

async def test_index_enqueues_job_and_returns_202(client: AsyncClient):
    with patch("app.routes.index._run_ingestion", new_callable=AsyncMock):
        response = await client.post("/index", json={"url": "https://example.com/doc.pdf"})

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


async def test_get_job_returns_404_for_unknown_id(client: AsyncClient):
    response = await client.get("/jobs/nonexistent-id")
    assert response.status_code == 404


async def test_get_job_returns_status_after_enqueue(client: AsyncClient):
    with patch("app.routes.index._run_ingestion", new_callable=AsyncMock):
        index_resp = await client.post("/index", json={"url": "https://example.com/doc.pdf"})

    job_id = index_resp.json()["job_id"]
    job_resp = await client.get(f"/jobs/{job_id}")
    assert job_resp.status_code == 200
    assert job_resp.json()["job_id"] == job_id
