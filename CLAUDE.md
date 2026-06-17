# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A production-grade RAG service: users upload PDFs → system indexes them into pgvector → users ask questions → service retrieves relevant chunks and returns grounded answers with citations. Built on free tiers only ($0/month target).

This is a **rebuild** of a legacy phidata-based demo (`legacy/`). The new stack owns the RAG pipeline directly (~150 lines) instead of using a framework, to avoid fighting API churn.

## Dev environment

Package manager is **uv**. The venv is at `.venv/` (not the legacy `agentic-ai/`).

```bash
# Activate
source .venv/bin/activate

# Add a dependency
uv add <package>

# Run anything in the venv without activating
uv run <command>
```

## Common commands

```bash
# Start local Postgres + pgvector
docker compose up -d

# Run database migrations
uv run alembic upgrade head

# Start the API (dev mode, hot reload)
uv run uvicorn app.main:app --reload

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

# Run eval harness
uv run python -m eval.run
```

## Architecture

### Request flow (query path)
`POST /query` → embed query (local bge-small) → hybrid retrieval from pgvector (dense cosine + tsvector sparse, fused via RRF) → if best score < `MIN_SIMILARITY`, return "I don't know" without calling the LLM → rerank top-20 → top-5 (local bge-reranker) → Groq llama-3.3-70b (streamed SSE) → response with citations.

### Request flow (ingestion path)
`POST /index` → store raw PDF in Cloudflare R2 → return job ID immediately → `BackgroundTask` pulls from R2 → sha256 dedup check → pypdf parse → chunk → batch embed → bulk insert into Neon Postgres.

### Key design decisions
- **No framework lock-in**: uses `groq` SDK and `sentence-transformers` directly. The RAG pipeline lives entirely in `app/rag/`.
- **Embedder + reranker run in-process** on the Fly.io VM (512 MB). Both models must fit in that budget — bge-small (~90 MB) + bge-reranker (~570 MB) is tight but viable.
- **Below-threshold short-circuit**: if retrieval score < `MIN_SIMILARITY`, the LLM is never called. This is load-bearing — it prevents hallucinations on out-of-corpus questions and saves Groq quota.
- **R2 is source of truth for PDFs**: Neon holds derived data (chunks + embeddings) only. Embeddings can be regenerated from R2 if the model changes.
- **Idempotent ingestion**: keyed on sha256 of raw PDF bytes. Re-submitting the same file is always a no-op.

### Configuration
All config lives in `app/settings.py` (Pydantic `BaseSettings`). Import via `from app.settings import settings`. Never read env vars directly elsewhere. Required vars (`GROQ_API_KEY`, `DATABASE_URL`) raise a `ValidationError` at import time if missing — this is intentional.

### Data model
Two tables in Neon Postgres:
- `documents(id, sha256 UNIQUE, source_url, title, pages, embedder_version, chunker_version, created_at)` — one row per PDF; `sha256` is the dedup key; version columns enable incremental re-embedding when the model changes.
- `chunks(id, doc_id FK, page, text, embedding vector(384), tsv tsvector)` — HNSW index (cosine) on `embedding`; GIN index on `tsv`.

### Observability
Every request gets a Langfuse trace (Cloud free tier). Spans: `retrieve`, `rerank`, `generate`. Token counts + latency captured per span. `structlog` JSON logger with request-ID propagated via context var. Request ID is set in middleware and flows through all spans.

## Project-specific slash commands

- `/security-check` — threat-model review across 8 attack surfaces specific to this RAG service (SSRF, secrets, injection, info leakage, deps, rate limiting, auth, PDF parsing)
- `/arch-review` — solution architect review comparing current implementation against `plan.md` and `ARCHITECTURE.md`

## What's in `legacy/`

The old phidata-based code. Do not import from it. It exists as reference only.

## Free-tier limits to keep in mind

- **Groq**: 30 RPM, ~1M tokens/day — eval runs + chat traffic share this quota
- **Neon**: 0.5 GB storage — monitor chunk table growth
- **Fly.io**: 512 MB RAM — both local ML models must fit alongside the FastAPI process
