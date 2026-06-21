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

- [ ] Multi-stage `Dockerfile`: uv → wheels stage → slim runtime
- [ ] `docker-compose.yml`: `pgvector/pgvector:pg16`, API, UI (Langfuse is Cloud, not in compose)
- [ ] One-command local dev: `docker compose up` brings the whole system up
- [ ] Verify ingestion + query end-to-end against compose stack
- [ ] **Tests:** end-to-end smoke against compose stack — ingest a real PDF via `POST /index`, poll until `done`, query it via `POST /query`, assert answer is non-empty and citations reference the correct doc; assert `/query` round-trip < 5s (catches in-process model latency regressions)
- [ ] **Dev-machine note (8 GB laptop):** for day-to-day work, prefer running Python natively via `uv run` and only Postgres in Docker — `docker compose` is the reproducible reference, not a daily-driver requirement. Install [OrbStack](https://orbstack.dev/) instead of Docker Desktop on macOS to cut idle VM RAM ~10×.

## Day 6 — Reranker + citations + Streamlit client

- [ ] `app/rag/reranker.py`: `BAAI/bge-reranker-base` cross-encoder; rerank top 20 → top 5
- [ ] Wire reranker into the retrieval pipeline behind a config flag
- [ ] Citations surfaced in `/query` response and rendered in UI
- [ ] Rewrite `ui/streamlit_app.py` as pure HTTP client (no Python imports from `app/`)
- [ ] Server-Sent Events streaming on `/query`; UI renders tokens as they arrive
- [ ] **Tests:** `tests/test_reranker.py` — reranker returns exactly `RERANK_K` results, ordering differs from input order on a known pair; `/query` response includes `citations` with `doc_id`, `page`, `snippet`, `score` fields; SSE stream produces at least one token before closing

## Day 7 — Structured logging + Langfuse

- [ ] `structlog` JSON logger; request ID propagated via context var
- [ ] Sign up for **Langfuse Cloud** (free tier, 50k observations/mo); create project, copy public + secret keys into `.env`
- [ ] Langfuse SDK wired: one trace per request; spans for `retrieve`, `rerank`, `generate`
- [ ] Token counts + latency captured per span
- [ ] Self-hosting Langfuse is deferred — revisit only if Cloud free tier becomes insufficient or data-residency requires it
- [ ] **Tests:** `tests/test_logging.py` — request ID present in all log lines for a single request; Langfuse SDK is a no-op (no exception) when `LANGFUSE_PUBLIC_KEY` is not set

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

- [ ] `slowapi` rate limiting (10 req/min/IP on `/query`)
- [ ] Confirm no tracebacks leak in responses (only opaque error IDs)
- [ ] SSRF guard on `/index` URL ingestion: `https` only, block RFC1918 / 169.254.169.254 / link-local
- [ ] Input validation: max prompt length, strip control chars
- [x] PDF size guard in `app/ingest/pdf.py`: HEAD request checks `Content-Length` before download; second check on actual body size in case server omits the header (limit: 50 MB)
- [ ] Sentry SDK wired (free tier, 5k events/mo)
- [ ] Pin and audit dependencies (`pip-audit`)

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
