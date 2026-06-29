"""structlog JSON logger with request-ID context propagation.

Usage:
    from app.obs.logging import get_logger, request_id_var

    log = get_logger(__name__)
    log.info("retrieved chunks", count=5, query="what is attention?")
"""
from __future__ import annotations

import contextvars
import logging
import sys
from typing import Any

import structlog

# Context variable — set once per request in middleware, read by any log call
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def _add_request_id(
    logger: object, method: str, event_dict: dict  # type: ignore[type-arg]
) -> dict:  # type: ignore[type-arg]
    event_dict["request_id"] = request_id_var.get()
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Call once at app startup to configure structlog + stdlib logging."""
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_request_id,
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
