# Stage 1 — install Python dependencies
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Cache layer: only re-runs when pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Replace CUDA-enabled torch with CPU-only wheel — cuts image by ~2 GB and
# runtime RAM by ~250 MB. Must run after uv sync so it overwrites the lock-resolved wheel.
# Version pinned to match uv.lock's torch (bump both together). The PyTorch CPU
# index only mirrors setuptools up to 78.1.0, but torch>=2.13.0 requires >=83.0.0 —
# --extra-index-url + unsafe-best-match lets that one resolve from PyPI instead
# while torch itself still resolves to the +cpu build (pinned explicitly so
# there's no ambiguity with the CUDA build PyPI hosts under the same version).
RUN uv pip install torch==2.13.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --reinstall --quiet

# Stage 2 — download HuggingFace models into a cache layer.
# Baking models into the image means cold-start is seconds, not minutes.
# bge-small-en-v1.5 ~90 MB + bge-reranker-base ~570 MB = ~660 MB total.
FROM python:3.12-slim AS model-downloader

COPY --from=builder /app/.venv /app/.venv

ARG HF_TOKEN=""
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface" \
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); \
CrossEncoder('BAAI/bge-reranker-base', device='cpu'); \
print('Models downloaded.')"

# Stage 3 — slim runtime image
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=model-downloader /app/.cache /app/.cache

COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/app/.cache/huggingface"

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
