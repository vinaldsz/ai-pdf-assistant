# AI PDF Assistant — Production-Readiness Plan

A 2-week, free-tier plan to take this repo from "demo script" to "deployable RAG service." Tasks are ordered by completion sequence; each later task assumes earlier ones are done.

---

## Target Stack (all free tiers / open source)

| Layer | Choice | Notes |
|---|---|---|
| API | FastAPI + Pydantic | Replaces `app_api.py` shim |
| LLM | Groq `llama-3.3-70b-versatile` | 30 RPM, ~1M tokens/day free |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, sentence-transformers) | 384-dim, no API |
| Reranker | `BAAI/bge-reranker-base` (local cross-encoder) | Free quality boost |
| Vector DB | Neon Postgres + pgvector | 0.5 GB free |
| PDF storage | Cloudflare R2 | 10 GB free, $0 egress |
| Queue | FastAPI `BackgroundTasks` → Upstash Redis if needed | Start simple |
| Observability | Langfuse **Cloud** (free tier) + `structlog` | 50k observations/mo free; lighter on dev RAM than self-hosting |
| Eval | Ragas + Groq-as-judge | Free |
| Hosting (API) | Fly.io (shared-cpu-1x, 512 MB) | Free tier |
| Hosting (UI) | Hugging Face Spaces | Free CPU |
| CI / Registry | GitHub Actions + GHCR | Free for public repos |
| Errors | Sentry free tier | Optional |

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
- [ ] Add `pyproject.toml` (uv-managed); pin deps with a lockfile
- [ ] Create `app/` package with the layout above (empty modules first)
- [ ] Write `app/settings.py` using Pydantic `BaseSettings` (`GROQ_API_KEY`, `DATABASE_URL`, `R2_*`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `RERANK_K`, `MIN_SIMILARITY`, `LOG_LEVEL`)
- [ ] Fail fast at import if required vars are missing (clear error message)
- [ ] Delete the broken `os.environ["GROQ_API_KEY"] = os.getenv(...)` lines
- [ ] Delete `app_api.py` (regex-based recovery shim) — replaced in Days 4–5
- [ ] Remove duplicate `duckduckgo-search` in `requirements.txt`; pin all versions
- [ ] Add `__pycache__/`, `.DS_Store`, `.env`, `agentic-ai/` to `.gitignore`

## Day 2 — Documents table + idempotent ingestion
- [ ] Add Alembic; create `migrations/` and baseline migration
- [ ] Schema: `documents(id, sha256 UNIQUE, source_url, title, pages, created_at, embedder_version, chunker_version)`
- [ ] Schema: `chunks(id, doc_id FK, page, text, embedding vector(384), tsv tsvector)`
- [ ] Indexes: HNSW on `chunks.embedding` (cosine); GIN on `chunks.tsv`
- [ ] `app/ingest/pdf.py`: download → sha256 → skip if exists → pypdf parse → recursive chunker → batch embed → bulk insert
- [ ] Recursive chunker in `app/rag/chunker.py` (configurable size/overlap)

## Day 3 — Local embeddings + retriever
- [ ] `app/rag/embedder.py`: lazy-loaded `BAAI/bge-small-en-v1.5` via `sentence-transformers`; batch encode; CPU-friendly
- [ ] `app/rag/store.py`: pgvector read/write helpers using SQLAlchemy + asyncpg pool
- [ ] `app/rag/retriever.py`: dense (cosine via pgvector) + sparse (`ts_rank_cd`) → reciprocal rank fusion
- [ ] Below-threshold short-circuit: if best score < `MIN_SIMILARITY`, return "I don't know" path

## Day 4 — Generator + FastAPI routes
- [ ] `app/rag/generator.py`: prompt builder w/ retrieved chunks + Groq call via `groq` SDK directly
- [ ] `tenacity` retries on 429/5xx; per-call timeout (20s)
- [ ] `app/main.py`: FastAPI app, request ID middleware, exception handler that returns opaque error IDs (never tracebacks)
- [ ] `POST /query` → `{answer, citations: [{doc_id, page, snippet, score}]}`
- [ ] `POST /index` → enqueues with `BackgroundTasks`; returns job ID
- [ ] `GET /health` (process alive), `GET /ready` (DB + Groq 1-token check)

## Day 5 — Dockerize + local compose
- [ ] Multi-stage `Dockerfile`: uv → wheels stage → slim runtime
- [ ] `docker-compose.yml`: `pgvector/pgvector:pg16`, API, UI (Langfuse is Cloud, not in compose)
- [ ] One-command local dev: `docker compose up` brings the whole system up
- [ ] Verify ingestion + query end-to-end against compose stack
- [ ] **Dev-machine note (8 GB laptop):** for day-to-day work, prefer running Python natively via `uv run` and only Postgres in Docker — `docker compose` is the reproducible reference, not a daily-driver requirement. Install [OrbStack](https://orbstack.dev/) instead of Docker Desktop on macOS to cut idle VM RAM ~10×.

## Day 6 — Reranker + citations + Streamlit client
- [ ] `app/rag/reranker.py`: `BAAI/bge-reranker-base` cross-encoder; rerank top 20 → top 5
- [ ] Wire reranker into the retrieval pipeline behind a config flag
- [ ] Citations surfaced in `/query` response and rendered in UI
- [ ] Rewrite `ui/streamlit_app.py` as pure HTTP client (no Python imports from `app/`)
- [ ] Server-Sent Events streaming on `/query`; UI renders tokens as they arrive

## Day 7 — Structured logging + Langfuse
- [ ] `structlog` JSON logger; request ID propagated via context var
- [ ] Sign up for **Langfuse Cloud** (free tier, 50k observations/mo); create project, copy public + secret keys into `.env`
- [ ] Langfuse SDK wired: one trace per request; spans for `retrieve`, `rerank`, `generate`
- [ ] Token counts + latency captured per span
- [ ] Self-hosting Langfuse is deferred — revisit only if Cloud free tier becomes insufficient or data-residency requires it

---

# Week 2 — Quality, Deploy, Polish

## Day 8 — Eval harness
- [ ] Write `eval/gold.jsonl` — 30 (question, expected_doc, expected_answer) pairs from the Thai Recipes PDF (or your chosen corpus)
- [ ] `eval/run.py`: Ragas with `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- [ ] Groq as the judge model (uses free quota)
- [ ] Commit baseline numbers in `eval/baseline.json`
- [ ] Document how to re-run: `python -m eval.run`

## Day 9 — Tests + CI
- [ ] `pytest` unit tests: chunker boundaries, retriever fusion math, prompt builder
- [ ] Integration tests using `testcontainers` (real pgvector in CI)
- [ ] `.github/workflows/ci.yml`: `ruff` lint, `mypy`/`pyright`, `pytest`, `pip-audit`, `bandit`
- [ ] Eval job runs on PR (warn-only initially; gate later)
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
- [ ] Sentry SDK wired (free tier, 5k events/mo)
- [ ] Pin and audit dependencies (`pip-audit`)

## Day 14 — Docs + Runbook
- [ ] README rewrite: architecture diagram (Mermaid), env-var table, quickstart (`docker compose up`), deploy guide
- [ ] `RUNBOOK.md` covering: Groq rate-limited, Neon sleeping/cold start, Langfuse down, vector DB full, indexing job stuck
- [ ] `ARCHITECTURE.md` short doc explaining the RAG pipeline + retrieval strategy
- [ ] Cut a `v0.1.0` git tag

---

# Deferred (do later when justified)

These are *real* production needs but premature for week 1–2:

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

| Service | Limit | Risk |
|---|---|---|
| Groq | 30 RPM, ~1M tokens/day | Eval runs + chat traffic combined |
| Neon | 0.5 GB storage, 190 compute-hr/mo | Cold starts, vector growth |
| Fly.io | 3× shared-cpu-1x VMs | Memory ceiling with sentence-transformers loaded |
| R2 | 10 GB storage, 1M Class-A ops/mo | Plenty for PDFs |
| HF Spaces | Sleeps after 48h idle | First request slow |
| Upstash Redis | 10k commands/day | Only if queue introduced |

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
