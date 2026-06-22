# Stage 1 — install dependencies into an isolated virtualenv
# Uses uv for fast, reproducible installs from the lockfile.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Cache layer: only re-runs when pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2 — slim runtime image (no uv, no build tools)
FROM python:3.12-slim

WORKDIR /app

# Pull the venv from the builder — keeps the runtime image lean.
COPY --from=builder /app/.venv /app/.venv

# Application source and Alembic migrations
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Store HF model cache inside the container workdir so it can be volume-mounted.
    HF_HOME="/app/.cache/huggingface"

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
