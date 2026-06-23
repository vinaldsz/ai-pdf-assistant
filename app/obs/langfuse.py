"""Langfuse observability — optional. All functions are no-ops when keys are absent.

One trace per /query request. Spans for retrieve, rerank, and generate.
Import only `get_client` and `trace_context` — never import langfuse directly
from pipeline modules so the dep stays optional.
"""
from __future__ import annotations

import contextvars
import time
from typing import Any

from app.settings import settings

# Holds the active Langfuse trace for the current request (None when Langfuse is off)
_trace_var: contextvars.ContextVar[Any] = contextvars.ContextVar("langfuse_trace", default=None)


def _is_enabled() -> bool:
    return bool(
        settings.langfuse_public_key and settings.langfuse_secret_key
    )


def _client():  # type: ignore[return]
    """Lazy singleton — only instantiated when keys are present."""
    if not _is_enabled():
        return None
    from langfuse import Langfuse  # noqa: PLC0415
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),  # type: ignore[union-attr]
        secret_key=settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
        host=settings.langfuse_host,
    )


def start_trace(name: str, input: dict) -> None:  # type: ignore[type-arg]
    """Start a new Langfuse trace for this request. Stores it in the context var."""
    client = _client()
    if client is None:
        return
    trace = client.trace(name=name, input=input)
    _trace_var.set(trace)


def end_trace(output: dict) -> None:  # type: ignore[type-arg]
    """Update the trace output and flush."""
    trace = _trace_var.get()
    if trace is None:
        return
    trace.update(output=output)
    _client().flush()  # type: ignore[union-attr]


class span:
    """Context manager that wraps a pipeline step in a Langfuse span.

    Usage:
        with span("retrieve", input={"query": q}) as s:
            results = await retrieve(q)
            s.set_output({"count": len(results)})
    """

    def __init__(self, name: str, input: dict) -> None:  # type: ignore[type-arg]
        self.name = name
        self.input = input
        self._span: Any = None
        self._start = time.perf_counter()

    def __enter__(self) -> span:
        trace = _trace_var.get()
        if trace is not None:
            self._span = trace.span(name=self.name, input=self.input)
        return self

    def set_output(self, output: dict) -> None:  # type: ignore[type-arg]
        if self._span is not None:
            elapsed_ms = int((time.perf_counter() - self._start) * 1000)
            self._span.end(output={**output, "latency_ms": elapsed_ms})

    def __exit__(self, *_: object) -> None:
        if self._span is not None and not hasattr(self._span, "_ended"):
            self._span.end()
