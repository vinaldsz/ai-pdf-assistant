"""FastAPI application factory — middleware, lifespan, and router wiring."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.obs.logging import configure_logging, get_logger, request_id_var
from app.rag import embedder, reranker
from app.routes import health, index, query

log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(log_level="INFO")
    log.info("startup: warming up models")
    await embedder.warmup()
    await reranker.warmup()
    log.info("startup: complete")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="AI PDF Assistant", lifespan=_lifespan)

    @app.middleware("http")
    async def _attach_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # Never expose tracebacks — return an opaque error ID the user can quote
        error_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "error_id": error_id},
        )

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(index.router)

    return app


app = create_app()
