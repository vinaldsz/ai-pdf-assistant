# Runbook — AI PDF Assistant

Operational reference for running this service in production. Each section covers one failure mode or maintenance task: what happens, how to verify it, and what to do.

---

## 1. Groq rate limited (429)

**Symptoms:**
- `/query` returns HTTP 502 or a JSON error body with `"rate_limited"` or Groq's `429` status
- Langfuse traces show `generate` span failing
- Logs show: `groq.RateLimitError` with `x-ratelimit-remaining-requests: 0`

**Check:**
```bash
# Inspect recent logs from the HF Space
# (HF Spaces streams logs via the Spaces UI — Settings > Logs)
# Or hit the API directly to confirm the error class:
curl -s https://vinaldsz-ai-pdf-assistant.hf.space/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' | jq .
```

**Groq console:** https://console.groq.com/usage — shows tokens/day and RPM remaining.

**Mitigation:**
- Wait. Groq's RPM window resets every 60 seconds; the daily token budget resets at UTC midnight.
- If an eval run is in progress, stop it (`Ctrl-C`) — it shares the same key and quota.
- If traffic is sustained, reduce `TOP_K` and `RERANK_K` in `app/settings.py` to send fewer tokens per request.
- Do not rotate the key unless it's compromised — a new key has the same quota.

---

## 2. Neon sleeping / cold start latency

**What happens:** Neon's free tier suspends the compute endpoint after 5 minutes of inactivity. The first query after suspension takes 2-4 seconds for Neon to wake before the connection succeeds. Combined with HF Space cold start (see section 3), total first-request latency can reach 15-20 seconds.

**This is expected behavior, not a bug.** The API's `/ready` endpoint will return `503` until the DB connection succeeds.

**Warm up Neon manually:**
```bash
curl -s https://vinaldsz-ai-pdf-assistant.hf.space/ready
# Repeat until {"status": "ok"} — usually 1-2 attempts
```

**Check Neon status:** https://neon.tech/docs/introduction/auto-suspend — the Neon console shows compute state (Active / Idle / Suspended).

**Check storage usage:**
```sql
-- Connect via psql or Neon console SQL editor
SELECT pg_size_pretty(pg_database_size(current_database()));
-- Alert if approaching 450 MB (limit is 512 MB)

-- Break down by table
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

---

## 3. HF Space sleeping

**What happens:** HuggingFace Spaces (free tier) pauses the container after 48 hours of no incoming HTTP traffic. The next request wakes it, but model loading adds ~10-15 seconds to the cold start.

**Check Space status:**
```bash
# HF Spaces API — returns "running", "paused", "building", etc.
curl -s "https://huggingface.co/api/spaces/vinaldsz/ai-pdf-assistant" \
  -H "Authorization: Bearer $HF_TOKEN" | jq .runtime.stage
```

**Wake the Space:**
```bash
# Any HTTP request wakes it; /health is cheapest
curl -s https://vinaldsz-ai-pdf-assistant.hf.space/health
# First response may time out (30s+) — retry once
curl -s https://vinaldsz-ai-pdf-assistant.hf.space/ready
```

**Prevent sleeping (during demos):** Send a lightweight ping every 30 minutes via a cron job or UptimeRobot on the free tier. Note that persistent wakefulness is not guaranteed on free HF Spaces.

**Force restart (if Space is stuck):**
```bash
# Restart via HF Spaces API
curl -X POST "https://huggingface.co/api/spaces/vinaldsz/ai-pdf-assistant/restart" \
  -H "Authorization: Bearer $HF_TOKEN"
```

---

## 4. Langfuse down

**Impact: none on the API.** The Langfuse client is initialized with `flush_at=1` and all tracing calls are wrapped so that any exception is caught and logged at `WARNING` level. If Langfuse is unreachable, the API continues to serve requests normally — traces are simply not recorded.

**Verify it's degraded gracefully:**
```bash
# API should still return 200
curl -s https://vinaldsz-ai-pdf-assistant.hf.space/health | jq .

# Logs will show warnings like:
# {"level": "warning", "event": "langfuse_flush_failed", "error": "..."}
```

**When Langfuse recovers:** Traces resume automatically on the next request. In-flight traces that failed to flush are lost — there is no local buffer or retry queue.

---

## 5. Indexing job stuck

**Symptoms:** `GET /jobs/{job_id}` stays at `{"status": "processing"}` for more than 5 minutes.

**Check job status:**
```bash
curl -s https://vinaldsz-ai-pdf-assistant.hf.space/jobs/<job_id> | jq .
```

**What to look for in logs (HF Spaces > Settings > Logs):**
- `"event": "ingest_started"` — job picked up by BackgroundTask
- `"event": "r2_fetch_failed"` — R2 credentials wrong or object missing
- `"event": "pdf_parse_error"` — pypdf couldn't parse the file (corrupted, encrypted, image-only PDF)
- `"event": "embed_batch_failed"` — embedder OOM or model not loaded
- `"event": "db_insert_failed"` — Neon connection dropped mid-insert

**Recovery:**
- If the job is stuck in `processing` with no recent log activity, the BackgroundTask likely died silently. The job will not auto-retry — resubmit the same URL. Ingestion is idempotent, so if the PDF was already partially written to Neon, the sha256 dedup check will short-circuit and return `skipped`.
- If the PDF URL is the problem, verify it's publicly accessible and under the size/page limits enforced by the SSRF guard in `app/ingest/`.

---

## 6. Vector DB storage full

**Limit:** Neon free tier is 0.5 GB total database size.

**Check current usage:**
```sql
-- In Neon console SQL editor or via psql
SELECT pg_size_pretty(pg_database_size(current_database())) AS total;

SELECT
  relname AS table,
  pg_size_pretty(pg_total_relation_size(relid)) AS size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

**Find large documents to prune:**
```sql
SELECT d.id, d.title, d.source_url, d.pages, COUNT(c.id) AS chunk_count
FROM documents d
JOIN chunks c ON c.doc_id = d.id
GROUP BY d.id
ORDER BY chunk_count DESC
LIMIT 10;
```

**Delete a document and its chunks:**
```sql
-- Chunks have ON DELETE CASCADE from the FK, so deleting the document is enough
DELETE FROM documents WHERE id = '<doc_uuid>';
```

**After bulk deletes, reclaim space:**
```sql
VACUUM ANALYZE chunks;
VACUUM ANALYZE documents;
```

---

## 7. Re-embed after model change

When `EMBEDDING_MODEL` is changed, existing chunk embeddings are stale. The `embedder_version` column on `documents` identifies which rows need updating.

**Step 1 — identify stale documents:**
```sql
SELECT id, title, embedder_version FROM documents
WHERE embedder_version != 'BAAI/bge-small-en-v1.5';
-- Or list all distinct versions present:
SELECT DISTINCT embedder_version, COUNT(*) FROM documents GROUP BY 1;
```

**Step 2 — delete stale chunks (keep documents row for dedup):**
```sql
DELETE FROM chunks
WHERE doc_id IN (
  SELECT id FROM documents WHERE embedder_version = '<old_version>'
);
```

**Step 3 — re-ingest from R2:**

R2 is the source of truth. For each stale document, retrieve its `source_url` (which is the R2 object key or original URL) and resubmit to `/index`:

```bash
# For each stale source_url:
curl -X POST https://vinaldsz-ai-pdf-assistant.hf.space/index \
  -H "Content-Type: application/json" \
  -d '{"url": "<source_url>"}'
```

Because the `sha256` already exists in `documents`, the dedup check will fire — **first update the `embedder_version` on those rows to force a re-index**, or temporarily drop and recreate the documents rows.

The cleaner approach is a one-off script that reads from R2, re-chunks, re-embeds with the new model, and bulk-inserts new chunk rows while updating `embedder_version`. This is tracked in `plan.md` as deferred tooling.

---

## 8. Deploy a new version

**API Space (`vinaldsz/ai-pdf-assistant`):**
```bash
# One-time setup: add the HF remote
git remote add hf https://huggingface.co/spaces/vinaldsz/ai-pdf-assistant

# Deploy — push triggers a Docker build on HF infrastructure
git push hf main

# Watch the build log in HF Spaces UI (Settings > Build logs)
# Build + model download (first deploy only): ~5-10 minutes
# Subsequent deploys (models cached in image): ~2-3 minutes
```

**UI Space (`vinaldsz/ai-pdf-assistant-ui`, separate Streamlit space):**
```bash
# The UI is a standalone Streamlit app — copy to a temp deploy dir
mkdir -p /tmp/hf-ui-deploy
cp -r ui/ /tmp/hf-ui-deploy/
cp requirements-ui.txt /tmp/hf-ui-deploy/requirements.txt   # if separate

cd /tmp/hf-ui-deploy
git init
git remote add hf https://huggingface.co/spaces/vinaldsz/ai-pdf-assistant-ui
git add .
git commit -m "deploy $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push hf main --force
```

**Alembic migrations:** `entrypoint.sh` runs `alembic upgrade head` before uvicorn starts on every container boot. New migrations deploy automatically with the image — no manual step needed.

---

## 9. Run the eval harness

The eval harness uses Ragas to score retrieval and answer quality. It hits the live API and uses OpenRouter as the judge LLM, so both keys must be set.

```bash
# From the repo root, with .env populated
source .venv/bin/activate

GROQ_API_KEY=gsk_... \
OPENROUTER_API_KEY=sk-or-... \
DATABASE_URL=postgresql://... \
uv run python -m eval.run
```

Expected output: a table of Ragas metrics (faithfulness, answer relevancy, context recall) printed to stdout. Results are also written to `eval/results/` with a timestamp.

**Note on quota:** Each eval question makes one `/query` call (Groq) and one judge call (OpenRouter). A 20-question eval uses ~20 Groq requests against the 30 RPM limit. Run evals outside of peak usage or add a `--delay` flag if rate limiting occurs.

---

## 10. Secrets rotation

Secrets are stored as HuggingFace Spaces environment variables (encrypted at rest). Rotation does not require a code change or redeploy — the container reads them from the environment on each boot.

**Rotate a secret via the HF Spaces API:**
```bash
HF_TOKEN="hf_..."          # your HF write token
SPACE_ID="vinaldsz/ai-pdf-assistant"
SECRET_NAME="GROQ_API_KEY"
NEW_VALUE="gsk_newvalue..."

curl -X PUT \
  "https://huggingface.co/api/spaces/${SPACE_ID}/secrets" \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"key\": \"${SECRET_NAME}\", \"value\": \"${NEW_VALUE}\"}"
```

**After rotation:** Restart the Space so the container picks up the new value:
```bash
curl -X POST \
  "https://huggingface.co/api/spaces/${SPACE_ID}/restart" \
  -H "Authorization: Bearer ${HF_TOKEN}"
```

**Secrets that may need rotation:**

| Secret | When to rotate |
|---|---|
| `GROQ_API_KEY` | Key compromised or leaked in logs |
| `DATABASE_URL` | Neon password rotated (Neon console > Project > Connection details) |
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | R2 key compromised (Cloudflare dashboard > R2 > API tokens) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse project reset (Langfuse dashboard > Settings > API keys) |
| `HF_TOKEN` | HF token compromised — rotate in HF profile settings, then update any CI secrets |
