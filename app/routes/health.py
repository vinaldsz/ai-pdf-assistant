"""GET /health and GET /ready endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.rag import embedder, reranker, store

router = APIRouter()


@router.get("/health")
async def health() -> dict:  # type: ignore[type-arg]
    return {"status": "ok"}


@router.get("/ready")
async def ready(response: Response) -> dict:  # type: ignore[type-arg]
    checks: dict[str, str] = {}

    # DB connectivity check
    try:
        pool = await store.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["db"] = "ok"
    except Exception:
        checks["db"] = "error: unreachable"

    # Model warm checks — both must be loaded before serving queries
    checks["embedder"] = "ok" if embedder._warmed else "not warm"
    checks["reranker"] = "ok" if reranker._warmed else "not warm"

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        response.status_code = 503
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
