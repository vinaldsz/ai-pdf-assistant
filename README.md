# pdf_assistant (project)

## Purpose & motivation

The pdf_assistant project is designed as a compact, easy-to-reproduce example of a retrieval-augmented assistant that can understand and answer questions based on PDF documents. It serves as a developer-friendly demo and a starting point for building production-grade tools that:

- Index content from PDFs (either via URLs or uploaded files) into a vector database for semantic search.
- Combine information retrieval from a knowledge base with a powerful language model (RAG) to generate grounded, context-aware answers.
- Showcase how to connect embedders, a pgvector-enabled Postgres database, and an agent-based assistant (phi) with a simple interface for exploration.

Common use cases:

- Prototyping document-based Q&A systems for product manuals, research papers, or legal and policy documents.
- Building an internal knowledge assistant that can respond to business or team-specific queries from PDFs.
- Quickly experimenting with different embedders, models, and vector database settings (like dimensions and indexing strategies).

## Quick overview

`pdf_assistant` indexes PDF documents (URLs or local files) into a vector store and exposes an assistant that can answer questions using retrieval-augmented generation (RAG). The code uses the `phi` library, a pgvector-backed PostgreSQL vector database, and an LLM provider (configured via environment variables).

Contents referenced here:

- `pdf_assistant.py` — CLI entrypoint that constructs the knowledge base, DB storage and runs the Agent.
- `streamlit_app.py` — small Streamlit UI to index PDFs and ask queries.
- `app_api.py` — programmatic wrapper used by the Streamlit UI.

## Prerequisites

- Python 3.10+ (use the system python or install with Homebrew / pyenv)
- Docker (optional but recommended for local Postgres + pgvector)
- git

## Create and activate a virtual environment (recommended)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you already have an existing venv in the repo (for example `agentic-ai/`), activate that instead:

```bash
source agentic-ai/bin/activate
```

## Environment variables

Copy `.env.example` (if present) to `.env` and fill in real secrets. The project looks for the following variables (examples):

- `GROQ_API_KEY` — Groq API key (if using Groq model provider)
- `GOOGLE_API_KEY` — Google API key (if using Google embedders)
- `PHI_API_KEY` — (if required by phi tooling)

Notes:

- Do NOT commit `.env` to version control. Add `.env` to `.gitignore`.

## Run local Postgres + pgvector (recommended)

Start the Docker container used in development (the repo uses image `phidata/pgvector`):

```bash
docker run -d \
	-e POSTGRES_DB=ai \
	-e POSTGRES_USER=ai \
	-e POSTGRES_PASSWORD=ai \
	-e PGDATA=/var/lib/postgresql/data/pgdata \
	-v pgvolume:/var/lib/postgresql/data \
	-p 5532:5432 \
	--name pgvector \
	phidata/pgvector:16
```

Replace port, credentials and image as needed. The default `db_url` in `pdf_assistant.py` is:

```
postgresql+psycopg://ai:ai@localhost:5532/ai
```

If you prefer docker-compose, add a simple `docker-compose.yml` with the above image and environment.

## Run the CLI assistant

Indexing is done automatically when `pdf_assistant.py` constructs the knowledge base. To run the CLI assistant:

```bash
# from project root, with venv active and docker DB running
python pdf_assistant.py
```

This will:

- construct a `PDFUrlKnowledgeBase` using the URL(s) configured in `pdf_assistant.py`
- create or upsert vectors into the `PgVector2` collection (table)
- start an interactive assistant CLI

If you add more URLs to the knowledge base or want to re-index, edit the `urls` list in `pdf_assistant.py` or use the Streamlit UI (below) to index additional PDFs.

## Streamlit demo UI

The repository includes a small Streamlit app entrypoint `streamlit_app.py` that uses `app_api.py` to index a URL and run single-turn queries.

Run the Streamlit UI:

```bash
# with venv activated
streamlit run streamlit_app.py
```

Open http://localhost:8501

Notes:

- Indexing is synchronous in the simple demo (it may block the UI for large PDFs). For production, run indexing in a background job.
- The Streamlit UI displays only the extracted assistant text (the code includes extraction logic to avoid showing internal object reprs).

## Troubleshooting

- Error: expected 1536 dimensions, not 384

  - Cause: your vector DB column was created for 1536-dimensional vectors while the embedder produced 384-dim vectors (or vice-versa). Fix by either:
    1. Changing the embedder/model to one that returns the DB's expected dimension, or
    2. Recreate/alter the DB vector column to match the embedder's dimension and re-run indexing.

- Error: provider/tool call failed (e.g. groq BadRequestError / tool_use_failed)

  - The model attempted to call a tool (for example `search_knowledge_base`) and the tool invocation failed. Check:
    - The tool is registered under the expected name.
    - The tool accepts the JSON arguments the model sent (e.g. `{"query": "..."}`).
    - The DB and vector collection exist and are accessible.

- OpenAI / provider API key issues
  - Ensure your API key is set in `.env` and exported into the environment or mapped at runtime. See the code near the top of `pdf_assistant.py` for mapping behavior.

## Development tips

- Use the Streamlit UI for quick manual indexing and queries during development.
- If you change embedding model or vector dimension, reindex all documents — dimensions must match across the DB.

## Next steps (optional)

- Add asynchronous indexing in `streamlit_app.py` (background worker) to avoid blocking the UI.
- Add upload support to the Streamlit app to index local PDF files.
- Pin dependency versions in `requirements.txt` (optional: add a `requirements-dev.txt` for linters/tests).

---
