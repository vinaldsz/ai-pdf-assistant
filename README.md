# AI PDF Assistant

A production-grade RAG service: upload PDFs, ask questions, get grounded answers with page citations. Built entirely on free tiers — $0/month to run.

---

## Motivation

Most RAG demos are toy examples: one PDF, one embedding call, one LLM call. They work in a notebook but fall apart in practice — hallucinations on out-of-corpus questions, no deduplication, no observability, no way to know why an answer was wrong.

This project builds RAG the way it should be built:

- **Hybrid retrieval** — dense vector search alone misses exact terms (model names, paper IDs, codes). Combining it with sparse full-text search and fusing the results with Reciprocal Rank Fusion gives meaningfully better recall.
- **Local reranking** — a cross-encoder reranker runs in-process to re-score the top-20 candidates. No API cost, deterministic latency, and a significant quality jump over bi-encoder ranking alone.
- **Hallucination guard** — if retrieval confidence is too low, the LLM is never called. The system returns "I don't know" rather than inventing an answer.
- **Built to understand** — no RAG framework wrapping the internals. The full pipeline is ~150 lines of explicit, readable Python so every design choice is visible and changeable.

---

## Overview

The system has two paths:

**Ingestion** — a user submits a PDF URL. The API validates it (SSRF guard, size/page limits), downloads it, deduplicates by SHA-256, splits it into overlapping chunks, embeds each chunk locally, and stores everything in Postgres with a vector index. The API returns a job ID immediately; processing runs in the background.

**Query** — a user asks a question. The query is embedded locally, then retrieved via hybrid search (dense cosine similarity + sparse `tsvector`, fused with RRF). If the best score is below a threshold, the LLM is skipped entirely. Otherwise, a local cross-encoder reranks the top-20 candidates to top-5, which are assembled into a prompt and streamed through Groq's `llama-3.3-70b`. The response includes page-level citations so every claim is traceable back to the source document.

Every request is traced in Langfuse (retrieve → rerank → generate spans with latency and token counts) and logged as structured JSON.

---

## How it works

```
POST /index  →  download PDF  →  sha256 dedup  →  chunk  →  embed (local)  →  store in pgvector
POST /query  →  embed query   →  hybrid search (dense + sparse, fused RRF)
                              →  rerank top-20 → top-5 (local cross-encoder)
                              →  Groq llama-3.3-70b  →  streamed answer + citations
```

If retrieval score is below the similarity threshold, the LLM is never called — "I don't know" is returned directly. This prevents hallucinations on out-of-corpus questions and saves Groq quota.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Embedder | `BAAI/bge-small-en-v1.5` (local, 90 MB) |
| Reranker | `BAAI/bge-reranker-base` (local, 570 MB) |
| Vector DB | Neon Postgres + pgvector (HNSW cosine + GIN tsvector) |
| LLM | Groq `llama-3.3-70b-versatile` (streamed SSE) |
| PDF storage | Cloudflare R2 |
| UI | Streamlit |
| Observability | Langfuse Cloud (traces) + structlog (JSON logs) |

---

## Quick start (local)

**Prerequisites:** Docker, [uv](https://github.com/astral-sh/uv), a Groq API key.

```bash
# 1. Clone and install
git clone https://github.com/vinaldsz/ai-pdf-assistant
cd ai-pdf-assistant
uv sync

# 2. Start Postgres + pgvector
docker compose up -d

# 3. Create .env
cat > .env <<EOF
GROQ_API_KEY=gsk_...
DATABASE_URL=postgresql://ai:ai@localhost:5433/ai
# Optional — Langfuse tracing
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
EOF

# 4. Run migrations
uv run alembic upgrade head

# 5. Start the API
uv run uvicorn app.main:app --reload
```

The API is now at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

**Start the UI (optional, separate terminal):**

```bash
uv run streamlit run ui/streamlit_app.py
```

---

## Usage

**Index a PDF:**
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/pdf/1706.03762"}'
# → {"job_id": "...", "status": "queued"}
```

**Check job status:**
```bash
curl http://localhost:8000/jobs/<job_id>
# → {"status": "done", "chunk_count": 66}
```

**Ask a question (streaming):**
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the attention mechanism?"}'
# → data: {"token": "The"}
# → data: {"token": " attention"}
# → ...
# → data: {"citations": [...], "done": true}
```

**Ask a question (JSON):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the attention mechanism?"}'
# → {"answer": "...", "citations": [{"doc_id": "...", "page": 3, "snippet": "...", "score": 0.91}]}
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check (DB + models warm) |
| `POST` | `/index` | Submit a PDF URL for indexing |
| `GET` | `/jobs/{id}` | Poll ingestion job status |
| `POST` | `/query` | Ask a question, get JSON response |
| `POST` | `/query/stream` | Ask a question, stream SSE tokens |

---

## Development

```bash
# Lint + format
uv run ruff check app/ tests/
uv run ruff format app/ tests/

# Type check
uv run mypy app/

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_chunker.py

# Run a single test by name
uv run pytest tests/test_chunker.py::test_overlap_respected

# Skip slow (real-model) tests
uv run pytest -m "not slow"
```

---

## Configuration

All settings live in `app/settings.py` (Pydantic `BaseSettings`). Set via `.env` or environment variables.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | One of these two | — | Groq API key |
| `OPENROUTER_API_KEY` | One of these two | — | OpenRouter API key (higher limits; recommended for eval) |
| `DATABASE_URL` | Yes | — | Postgres connection URL |
| `LANGFUSE_PUBLIC_KEY` | No | — | Enables Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | — | Enables Langfuse tracing |
| `EMBEDDING_MODEL` | No | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `RERANKER_MODEL` | No | `BAAI/bge-reranker-base` | Cross-encoder model |
| `LLM_MODEL` | No | `llama-3.3-70b-versatile` | Model ID (Groq) or `meta-llama/llama-3.3-70b-instruct` (OpenRouter) |
| `MIN_SIMILARITY` | No | `0.30` | Below this score → "I don't know" |
| `TOP_K` | No | `20` | Candidates retrieved before reranking |
| `RERANK_K` | No | `5` | Chunks passed to the LLM |
| `CHUNK_SIZE` | No | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | No | `120` | Overlap between chunks |

---

## Free-tier budget

| Service | Limit | Used for |
|---|---|---|
| Groq | 30 RPM, ~1M tokens/day | LLM completions |
| Neon Postgres | 0.5 GB | Chunks + embeddings |
| Cloudflare R2 | 10 GB, $0 egress | Raw PDF storage |
| Fly.io | 512 MB RAM | API + local ML models |
| Hugging Face Spaces | Free CPU | Streamlit UI |
| Langfuse Cloud | 50k observations/mo | Request tracing |

**Target monthly cost: $0**

---

## Project structure

```
app/
  ingest/       # PDF download, SSRF guard, chunking, embedding, store
  obs/          # structlog JSON logging + Langfuse tracing
  rag/          # embedder, retriever (hybrid RRF), reranker, generator
  routes/       # FastAPI route handlers
  settings.py   # All config via Pydantic BaseSettings
  main.py       # App factory, middleware, lifespan
ui/
  streamlit_app.py   # Streamlit chat UI (pure HTTP, no app/ imports)
tests/               # pytest unit + integration tests
migrations/          # Alembic schema migrations
```
