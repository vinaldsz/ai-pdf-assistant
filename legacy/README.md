# Legacy

Pre-rewrite code, kept for reference. Not imported by the new `app/` package.

| File | What it was | Why it's here |
|---|---|---|
| `pdf_assistant.py` | Typer CLI built on `phi` Agent + `PgAgentStorage` + `PgVector2` + Gemini embedder. Single-PDF demo. | Replaced by `app/main.py` (FastAPI), `app/rag/*` (direct pgvector + sentence-transformers + Groq SDK), `app/ingest/*`. |
| `app_api.py` | Thin "API" layer that tried to call the `phi` Agent via `hasattr` probing and recovered from failed tool-calls by regex-parsing error strings. | Brittle shim around a moving framework. Replaced by typed FastAPI routes (Day 4). |
| `streamlit_app.py` | Streamlit UI that imported `app_api.index_url` / `query_text` directly. Coupled to `app_api.py`'s shape. | Replaced by `ui/streamlit_app.py` — a pure HTTP client against the FastAPI service (Day 6). |
| `playground.py` | Phi `playground` web-UI launcher — registered the PDF agent + a finance agent and served the phi playground. | Not part of the new architecture; we use Streamlit + FastAPI instead. |
| `finance_agent.py` | Standalone phi agent with yfinance + DuckDuckGo tools. Unrelated to PDF QA. | Out of scope for this project. |

History is preserved via `git mv`, so:

```bash
git log --follow legacy/pdf_assistant.py
```

shows the full evolution.

## Want to run the old code?

It still works in isolation (assuming a local Postgres + `.env`). From the repo root:

```bash
uv run python legacy/pdf_assistant.py
```

Note: the new `app/` package does **not** depend on anything in `legacy/`. If you delete this directory entirely, the new service is unaffected.
