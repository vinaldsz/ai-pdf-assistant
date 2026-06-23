# AI PDF Assistant — Production-Readiness Plan

A 2-week, free-tier plan to take this repo from "demo script" to "deployable RAG service." Tasks are ordered by completion sequence; each later task assumes earlier ones are done.

---

## Target Stack (all free tiers / open source)

| Layer         | Choice                                                  | Notes                                                          |
| ------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| API           | FastAPI + Pydantic                                      | Replaces `app_api.py` shim                                     |
| LLM           | Groq `llama-3.3-70b-versatile`                          | 30 RPM, ~1M tokens/day free                                    |
| Embeddings    | `BAAI/bge-small-en-v1.5` (local, sentence-transformers) | 384-dim, no API                                                |
| Reranker      | `BAAI/bge-reranker-base` (local cross-encoder)          | Free quality boost                                             |
| Vector DB     | Neon Postgres + pgvector                                | 0.5 GB free                                                    |
| PDF storage   | Cloudflare R2                                           | 10 GB free, $0 egress                                          |
| Queue         | FastAPI `BackgroundTasks` → Upstash Redis if needed     | Start simple                                                   |
| Observability | Langfuse **Cloud** (free tier) + `structlog`            | 50k observations/mo free; lighter on dev RAM than self-hosting |
| Eval          | Ragas + Groq-as-judge                                   | Free                                                           |
| Hosting (API) | Fly.io (shared-cpu-1x, 512 MB)                          | Free tier                                                      |
| Hosting (UI)  | Hugging Face Spaces                                     | Free CPU                                                       |
| CI / Registry | GitHub Actions + GHCR                                   | Free for public repos                                          |
| Errors        | Sentry free tier                                        | Optional                                                       |

**Architectural decision:** rip out `phi` / phidata. Replace with direct `groq` SDK + `sentence-transformers` + pgvector. ~150 lines of owned code beats fighting a moving framework API.

---

## Target Repo Layout

```
ai-pdf-assistant/
├─ app/
│  ├─ main.py              FastAPI app
│  ├─ settings.py          Pydantic BaseSettings
│  ├─ deps.py              DI: db pool, embedder, llm client
│  ├─ routes/{health,index,query}.py
│  ├─ rag/{chunker,embedder,store,retriever,reranker,generator}.py
│  ├─ ingest/{pdf,jobs}.py
│  └─ obs/logging.py
├─ migrations/             Alembic
├─ eval/{gold.jsonl,run.py}
├─ tests/
├─ ui/streamlit_app.py     pure HTTP client
├─ Dockerfile
├─ docker-compose.yml
├─ fly.toml
├─ pyproject.toml          uv-managed
└─ .github/workflows/ci.yml
```

---

# Week 1 — Foundations & Make It Deployable

## Day 1 — Settings, structure, kill the shim

- [x] Add `pyproject.toml` (uv-managed); pin deps with a lockfile
- [x] Create `app/` package with the layout above (empty modules first)
- [x] Write `app/settings.py` using Pydantic `BaseSettings` (`GROQ_API_KEY`, `DATABASE_URL`, `R2_*`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `RERANK_K`, `MIN_SIMILARITY`, `LOG_LEVEL`)
- [x] Fail fast at import if required vars are missing (clear error message)
- [ ] Delete the broken `os.environ["GROQ_API_KEY"] = os.getenv(...)` lines
- [ ] Delete `app_api.py` (regex-based recovery shim) — replaced in Days 4–5
- [ ] Remove duplicate `duckduckgo-search` in `requirements.txt`; pin all versions
- [x] Add `__pycache__/`, `.DS_Store`, `.env`, `agentic-ai/` to `.gitignore`
- [ ] **Tests:** `tests/test_settings.py` — `ValidationError` raised when `GROQ_API_KEY` or `DATABASE_URL` missing; all defaults match expected values; `chunk_overlap >= chunk_size` raises

## Day 2 — Documents table + idempotent ingestion

- [x] Add Alembic; create `migrations/` and baseline migration
- [x] Schema: `documents(id, sha256 UNIQUE, source_url, title, pages, created_at, embedder_version, chunker_version)`
- [x] Schema: `chunks(id, doc_id FK, page, text, embedding vector(384), tsv tsvector)`
- [x] Run `CREATE EXTENSION IF NOT EXISTS vector` as the first statement in the baseline migration (required before creating `vector` columns on Neon and fresh local DBs)
- [x] Indexes: HNSW on `chunks.embedding` (cosine) with `m=16, ef_construction=64`; GIN on `chunks.tsv` — use explicit params, pgvector defaults are conservative for 384-dim vectors
- [x] `app/ingest/pdf.py`: download → sha256 → skip if exists → pypdf parse → recursive chunker → batch embed → bulk insert
- [x] Recursive chunker in `app/rag/chunker.py` (configurable size/overlap)
- [x] **Tests:** `tests/test_chunker.py` — empty string, text shorter than `chunk_size`, exact boundary, overlap carry-over into next chunk, recursive split on long text with no natural separators; `tests/test_pdf_ingest.py` — `_validate_url` rejects `http://`, RFC1918 IPs, bare `169.254.169.254`; sha256 dedup returns `skipped=True` on second call

## Day 3 — Local embeddings + retriever

- [x] `app/rag/embedder.py`: lazy-loaded `BAAI/bge-small-en-v1.5` via `sentence-transformers`; batch encode; CPU-friendly
- [x] Pre-warm embedder at startup (call `encode(["warmup"])` during app lifespan) so the first real request doesn't block for 5–15s while the model loads from disk; `/ready` should only return 200 after warm-up completes
- [x] `app/rag/store.py`: pgvector read/write helpers using SQLAlchemy + asyncpg pool
- [x] `app/rag/retriever.py`: dense (cosine via pgvector) + sparse (`ts_rank_cd`) → reciprocal rank fusion
- [x] Below-threshold short-circuit: if best score < `MIN_SIMILARITY`, return "I don't know" path
- [x] **Tests:** `tests/test_retriever.py` — `_rrf` correct ordering when chunk appears in both lists, handles empty sparse list, handles empty dense list; below-threshold returns `[]`; `tests/test_embedder.py` — `encode_batch` returns correct length, vectors are unit-length (dot product with self ≈ 1.0)

## Day 4 — Generator + FastAPI routes

- [x] `app/rag/generator.py`: prompt builder w/ retrieved chunks + Groq call via `groq` SDK directly
- [x] Enforce a hard token ceiling in the prompt builder (regardless of `RERANK_K` config) so a misconfigured reranker or disabled rerank flag cannot silently overflow the context window
- [x] `tenacity` retries on 429/5xx; per-call timeout (20s)
- [x] `app/main.py`: FastAPI app, request ID middleware, exception handler that returns opaque error IDs (never tracebacks)
- [x] `POST /query` → `{answer, citations: [{doc_id, page, snippet, score}]}`
- [x] `POST /index` → enqueues with `BackgroundTasks`; returns job ID
- [x] `GET /jobs/{id}` → simple in-memory status dict (`queued | running | done | failed`); without this users have no visibility into silent ingestion failures
- [x] `GET /health` (process alive), `GET /ready` (DB + Groq 1-token check + embedder warm)
- [x] **Tests:** `tests/test_generator.py` — prompt builder enforces token ceiling regardless of chunk count; `tests/test_routes.py` — `/health` returns 200; `/ready` returns 503 when DB unreachable; `/query` with empty retrieval returns "I don't know" without calling Groq (mock Groq client asserts zero calls)

## Day 5 — Dockerize + local compose

- [x] Multi-stage `Dockerfile`: uv → wheels stage → slim runtime
- [x] `docker-compose.yml`: `pgvector/pgvector:pg16`, API (Langfuse is Cloud, not in compose); `model_cache` volume persists HF models across restarts
- [x] `entrypoint.sh`: runs `alembic upgrade head` then starts uvicorn; `.dockerignore` excludes dev artifacts
- [ ] One-command local dev: `docker compose up` brings the whole system up (verify manually)
- [x] **Tests:** `tests/test_e2e.py` — ingest a real PDF via `POST /index`, poll until `done`, query via `POST /query`, assert non-empty answer + citations; `/query` round-trip asserted; duplicate index idempotent; below-threshold query short-circuits. Skipped unless `E2E_API_URL` env var is set.
- [ ] **Dev-machine note (8 GB laptop):** for day-to-day work, prefer running Python natively via `uv run` and only Postgres in Docker — `docker compose` is the reproducible reference, not a daily-driver requirement.

## Day 6 — Reranker + citations + Streamlit client

- [x] `app/rag/reranker.py`: `BAAI/bge-reranker-base` cross-encoder; rerank top 20 → top 5; same lazy-load pattern as embedder; `_warmed` flag; warmup wired into lifespan
- [x] Wire reranker into `/query` pipeline: retrieve (top-20 RRF) → rerank → top-5 → Groq
- [x] Citations surfaced in `/query` response (`doc_id`, `page`, `snippet`, `score`)
- [x] `ui/streamlit_app.py` as pure HTTP client — sidebar PDF indexing with job polling, chat interface calling `/query/stream`
- [x] `POST /query/stream` SSE endpoint: yields `{"token": "..."}` events then `{"citations": [...], "done": true}`; `generator.generate_stream()` streams tokens from Groq
- [x] `/ready` updated to check `reranker._warmed` alongside embedder
- [x] **Tests:** `tests/test_reranker.py` — 5 fast unit tests (mocked model): empty→empty, returns top-k, orders by score descending, k > len returns all, correct query/chunk pairs passed to model; 2 slow tests (real model); `tests/test_routes.py` — updated fixture mocks both warmups, adds reranker warm check test, query pipeline calls reranker before generator, SSE stream test

## Day 7 — Structured logging + Langfuse

- [x] `structlog` JSON logger; request ID propagated via context var
- [x] Sign up for **Langfuse Cloud** (free tier, 50k observations/mo); create project, copy public + secret keys into `.env`
- [x] Langfuse SDK wired: one trace per request; spans for `retrieve`, `rerank`, `generate`
- [x] Token counts + latency captured per span
- [x] Self-hosting Langfuse is deferred — revisit only if Cloud free tier becomes insufficient or data-residency requires it
- [x] **Tests:** `tests/test_logging.py` — request ID present in all log lines for a single request; Langfuse SDK is a no-op (no exception) when `LANGFUSE_PUBLIC_KEY` is not set

---

# Week 2 — Quality, Deploy, Polish

## Day 8 — Eval harness

- [ ] Write `eval/gold.jsonl` — 30 (question, expected_doc, expected_answer) pairs from the Thai Recipes PDF (or your chosen corpus)
- [ ] `eval/run.py`: Ragas with `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- [ ] Groq as the judge model (uses free quota)
- [ ] Commit baseline numbers in `eval/baseline.json`
- [ ] **Performance baseline:** `eval/run.py` also measures and commits per-phase latency (embed, retrieve, rerank, generate) and end-to-end p50/p95 across the 30 gold questions; store in `eval/perf_baseline.json` — CI warns if any phase regresses > 20% vs baseline
- [ ] Document how to re-run: `python -m eval.run`

## Day 9 — CI pipeline

- [ ] Integration tests using `testcontainers` (real pgvector in CI) — covers `store.py` insert + query round-trip, sha256 dedup, transactional rollback on chunk failure
- [ ] `.github/workflows/ci.yml`: `ruff` lint, `mypy`/`pyright`, full `pytest` suite (unit + integration), `pip-audit`, `bandit`
- [ ] Eval job runs on PR (warn-only initially; gate later) — note: eval uses Groq as both generator and Ragas judge; a 30-question set can consume 50–100 Groq requests per run, so consider running on a schedule rather than every PR push to avoid exhausting the 1M token/day free quota
- [ ] Build + push image to GHCR on push to `main`

## Day 10 — Cloudflare R2 for PDFs

- [ ] Sign up Cloudflare R2 (no credit card needed for 10 GB tier)
- [ ] Create bucket; mint scoped API token
- [ ] `boto3` against R2's S3-compatible endpoint
- [ ] `POST /index` accepts URL **or** multipart upload → store original PDF in R2 → ingestion job pulls from R2
- [ ] R2 keys/secrets via Pydantic settings

## Day 11 — Neon Postgres

- [ ] Create Neon free project; enable `pgvector` extension
- [ ] Run Alembic migrations against Neon
- [ ] Switch local `DATABASE_URL` to Neon for a smoke test
- [ ] Verify HNSW index built; sanity-check query latency
- [ ] Document the connection-pooling caveat (Neon sleeps; use pooled URL)

## Day 12 — Deploy API to Fly.io + UI to HF Spaces

- [ ] `fly launch` → `shared-cpu-1x@512MB`
- [ ] Set secrets: `GROQ_API_KEY`, `DATABASE_URL` (Neon pooled), `R2_*`, `LANGFUSE_*`
- [ ] Confirm `/health` + `/ready` green on Fly
- [ ] Deploy Streamlit UI to Hugging Face Spaces pointing at the Fly URL
- [ ] End-to-end smoke test against production URLs

## Day 13 — Hardening

**Rate limiting & abuse**

- [ ] `slowapi` rate limiting: 10 req/min/IP on `/query`, 5 req/min/IP on `/index` — do this before sharing any public URL; one script can exhaust the Groq 30 RPM free-tier quota in seconds
- [ ] Cap concurrent background ingestion jobs with `asyncio.Semaphore(3)` in `_run_ingestion` — prevents OOM on the 512 MB Fly VM from simultaneous download+embed+write pipelines

**Input validation**

- [ ] `QueryRequest.query`: add `Field(..., min_length=1, max_length=2000)` and a `field_validator` that strips control characters (`ch < " "` except `\t\n`) — closes unbounded CPU spike on embedder and token-burn on Groq
- [ ] PDF page-count guard in `_parse_pdf`: reject if `len(reader.pages) > 500` before extracting text — a valid 50 MB PDF can produce ~30k chunks otherwise

**Error & info leakage**

- [ ] `app/routes/health.py:27`: replace `f"error: {exc}"` with `"error: unreachable"` — raw asyncpg exceptions can contain DSN passwords in the response body
- [ ] `app/routes/index.py:66`: catch `ValueError` separately (surface message) vs `Exception` (return generic `"ingestion failed"`) — prevents infra details (hostnames, ports) leaking via `GET /jobs/{id}`
- [ ] Disable OpenAPI docs in prod: `FastAPI(docs_url=None, redoc_url=None)` when `settings.environment == "prod"`

**SSRF hardening**

- [ ] Block IPv4-mapped IPv6 addresses (`::ffff:169.254.169.254` etc.) by unwrapping `addr.ipv4_mapped` before blocklist check in `_assert_ip_is_public`
- [ ] DNS rebinding mitigation: resolve hostname once, assert IP, pass resolved IP directly to httpx with `Host` header — eliminates the second DNS lookup between validation and connect (low practical risk for v0.1 but required before prod)
- [x] SSRF guard already in place: `https`-only, RFC1918 + loopback + `169.254.0.0/16` blocklist, redirect re-validation, 50 MB streaming cap

**PDF parsing safety**

- [ ] Wrap `_parse_pdf` in `run_in_executor` + `asyncio.wait_for(..., timeout=60.0)` — currently runs synchronously in a BackgroundTask on the event loop; a decompression-bomb PDF can stall all other requests
- [ ] Catch `pypdf.errors.PdfReadError` in `_parse_pdf` and re-raise as `ValueError("PDF could not be read")` — prevents raw pypdf error messages reaching the job store

**Existing items**

- [ ] Confirm no tracebacks leak in responses (only opaque error IDs)
- [x] PDF size guard in `app/ingest/pdf.py`: streaming download aborts if body exceeds 50 MB
- [ ] Sentry SDK wired (free tier, 5k events/mo)
- [ ] Pin and audit dependencies (`pip-audit`); confirm Docker/CI uses `uv sync --frozen`

## Day 14 — Docs + Runbook

- [ ] README rewrite: architecture diagram (Mermaid), env-var table, quickstart (`docker compose up`), deploy guide
- [ ] `RUNBOOK.md` covering: Groq rate-limited, Neon sleeping/cold start, Langfuse down, vector DB full, indexing job stuck
- [ ] `ARCHITECTURE.md` short doc explaining the RAG pipeline + retrieval strategy
- [ ] Cut a `v0.1.0` git tag

---

# Deferred (do later when justified)

These are _real_ production needs but premature for week 1–2:

- [ ] Auth (OIDC / JWT) — only matters at >1 user
- [ ] Multi-tenancy / per-tenant collections — needs auth first
- [ ] Celery / RQ / Arq — only when `BackgroundTasks` proves insufficient
- [ ] Hybrid retrieval can launch dense-only; add `tsvector` when retrieval misses on proper nouns
- [ ] Semantic answer cache (GPTCache / Redis) — cost optimization when traffic justifies
- [ ] PII scanning (Presidio) — when user-uploaded PDFs become a concern
- [ ] Output moderation (Llama Guard) — when surface area justifies
- [ ] Re-embedding migration tooling — when embedding model changes
- [ ] Kubernetes — Fly.io scales surprisingly far before this is needed

---

# Free-Tier Limits to Watch

| Service       | Limit                             | Risk                                             |
| ------------- | --------------------------------- | ------------------------------------------------ |
| Groq          | 30 RPM, ~1M tokens/day            | Eval runs + chat traffic combined                |
| Neon          | 0.5 GB storage, 190 compute-hr/mo | Cold starts, vector growth                       |
| Fly.io        | 3× shared-cpu-1x VMs              | Memory ceiling with sentence-transformers loaded |
| R2            | 10 GB storage, 1M Class-A ops/mo  | Plenty for PDFs                                  |
| HF Spaces     | Sleeps after 48h idle             | First request slow                               |
| Upstash Redis | 10k commands/day                  | Only if queue introduced                         |

---

# Definition of Done (end of Week 2)

- [ ] Public Fly.io URL serves `/health`, `/ready`, `/query`, `/index`
- [ ] Public HF Spaces UI lets a user upload a PDF and chat with it
- [ ] Langfuse trace exists for every request, with retrieval + LLM spans
- [ ] CI green on `main`; image published to GHCR
- [ ] Eval baseline committed; CI eval job runs (warn-only)
- [ ] README + runbook adequate for someone else to deploy from scratch
- [ ] Zero secrets in git; all config via env vars
- [ ] No tracebacks returned to clients
- [ ] Monthly cost: **$0**
