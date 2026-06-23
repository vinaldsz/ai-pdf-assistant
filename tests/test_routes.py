"""Route-level tests — no real DB, Groq, or model calls."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from app.main import app
from app.rag.retriever import RetrievalResult


def _make_chunk(chunk_id: str = "c1") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, doc_id="doc1", page=1, text="some text", score=0.8)


@pytest.fixture
async def client():
    """Test client with both model warmups mocked so no torch loads during lifespan."""
    with patch("app.rag.embedder.warmup", new_callable=AsyncMock), \
         patch("app.rag.reranker.warmup", new_callable=AsyncMock):
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
         patch("app.rag.embedder._warmed", new=True), \
         patch("app.rag.reranker._warmed", new=True):
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["db"].startswith("error:")


async def test_ready_returns_503_when_embedder_not_warm(client: AsyncClient):
    with patch("app.rag.embedder._warmed", new=False), \
         patch("app.rag.reranker._warmed", new=True), \
         patch("app.rag.store.get_pool", side_effect=Exception("skip db")):
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["embedder"] == "not warm"


async def test_ready_returns_503_when_reranker_not_warm(client: AsyncClient):
    with patch("app.rag.embedder._warmed", new=True), \
         patch("app.rag.reranker._warmed", new=False), \
         patch("app.rag.store.get_pool", side_effect=Exception("skip db")):
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["reranker"] == "not warm"


# ---------------------------------------------------------------------------
# /query (JSON)
# ---------------------------------------------------------------------------

async def test_query_empty_retrieval_returns_no_answer(client: AsyncClient):
    with patch("app.rag.retriever.retrieve", new_callable=AsyncMock, return_value=[]):
        response = await client.post("/query", json={"query": "what is X?"})

    assert response.status_code == 200
    data = response.json()
    assert data["citations"] == []
    assert "don't" in data["answer"].lower()


async def test_query_empty_retrieval_does_not_call_llm(client: AsyncClient):
    with patch("app.rag.retriever.retrieve", new_callable=AsyncMock, return_value=[]), \
         patch("app.rag.generator.AsyncOpenAI") as mock_openai:
        await client.post("/query", json={"query": "anything"})

    mock_openai.assert_not_called()


async def test_query_calls_reranker_before_generator(client: AsyncClient):
    chunk = _make_chunk()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "The answer."

    with patch("app.rag.retriever.retrieve", new_callable=AsyncMock, return_value=[chunk]), \
         patch("app.rag.reranker.rerank", new_callable=AsyncMock, return_value=[chunk]) as mock_rerank, \
         patch("app.rag.generator.generate", new_callable=AsyncMock) as mock_gen:
        from app.rag.generator import GeneratorResponse
        mock_gen.return_value = GeneratorResponse(
            answer="The answer.",
            citations=[{"doc_id": "doc1", "page": 1, "snippet": "some text", "score": 0.8}],
        )
        response = await client.post("/query", json={"query": "test?"})

    assert response.status_code == 200
    mock_rerank.assert_called_once()
    mock_gen.assert_called_once()


async def test_query_response_has_citations_fields(client: AsyncClient):
    chunk = _make_chunk()
    with patch("app.rag.retriever.retrieve", new_callable=AsyncMock, return_value=[chunk]), \
         patch("app.rag.reranker.rerank", new_callable=AsyncMock, return_value=[chunk]):
        from app.rag.generator import GeneratorResponse
        with patch("app.rag.generator.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GeneratorResponse(
                answer="Answer.",
                citations=[{"doc_id": "doc1", "page": 1, "snippet": "some text", "score": 0.8}],
            )
            response = await client.post("/query", json={"query": "test?"})

    citation = response.json()["citations"][0]
    assert {"doc_id", "page", "snippet", "score"} <= citation.keys()


async def test_query_missing_body_returns_422(client: AsyncClient):
    response = await client.post("/query", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /query/stream (SSE)
# ---------------------------------------------------------------------------

async def test_query_stream_empty_retrieval_returns_no_answer_event(client: AsyncClient):
    with patch("app.rag.retriever.retrieve", new_callable=AsyncMock, return_value=[]):
        response = await client.post("/query/stream", json={"query": "unknown?"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = _parse_sse(response.text)
    tokens = "".join(e["token"] for e in events if "token" in e)
    assert "don't" in tokens.lower()
    done_events = [e for e in events if e.get("done")]
    assert done_events, "Must emit a done event"
    assert done_events[0]["citations"] == []


async def test_query_stream_produces_tokens_and_done_event(client: AsyncClient):
    chunk = _make_chunk()

    async def _fake_stream(query, chunks):
        for word in ["Hello", " world"]:
            yield word

    with patch("app.rag.retriever.retrieve", new_callable=AsyncMock, return_value=[chunk]), \
         patch("app.rag.reranker.rerank", new_callable=AsyncMock, return_value=[chunk]), \
         patch("app.rag.generator.generate_stream", side_effect=_fake_stream):
        response = await client.post("/query/stream", json={"query": "test?"})

    events = _parse_sse(response.text)
    tokens = [e["token"] for e in events if "token" in e]
    assert tokens == ["Hello", " world"]
    done_events = [e for e in events if e.get("done")]
    assert done_events
    assert isinstance(done_events[0]["citations"], list)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(body: str) -> list[dict]:  # type: ignore[type-arg]
    """Parse SSE body into a list of JSON objects from data: lines."""
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events
