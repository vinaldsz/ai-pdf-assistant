"""Langfuse v4 observability wrapper — no-op when keys absent.

Langfuse v4 uses OpenTelemetry under the hood. The v2 low-level API
(client.trace() / trace.span()) is gone; everything goes through
start_as_current_observation() context managers and update_current_span().

One trace per /query request. Spans for retrieve, rerank, generate.
Import only start_trace / end_trace / span from pipeline modules.
"""
from __future__ import annotations

import contextvars
import time
from typing import Any

from app.settings import settings

# Holds the active trace context manager so end_trace() can close it.
_trace_cm_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "langfuse_trace_cm", default=None
)

_client_instance: Any = None


def _is_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _client() -> Any:
    """Lazy singleton — created once and reused so flush() works on the right instance."""
    global _client_instance
    if not _is_enabled():
        return None
    if _client_instance is None:
        from langfuse import Langfuse
        _client_instance = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),  # type: ignore[union-attr]
            secret_key=settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
            host=settings.langfuse_host,
        )
    return _client_instance


def start_trace(name: str, input: dict) -> None:  # type: ignore[type-arg]
    """Open a root trace for this request. Stores the context manager in a contextvar."""
    client = _client()
    if client is None:
        return
    cm = client.start_as_current_observation(name=name, input=input)
    cm.__enter__()
    _trace_cm_var.set(cm)


def end_trace(output: dict) -> None:  # type: ignore[type-arg]
    """Close the root trace and flush pending events to Langfuse Cloud."""
    cm = _trace_cm_var.get()
    if cm is None:
        return
    client = _client()
    if client is not None:
        try:
            client.update_current_span(output=output)
        except Exception:
            pass
        cm.__exit__(None, None, None)
        client.flush()


class span:
    """Sync context manager that wraps a pipeline step in a Langfuse child span.

    Usage (works inside async functions — sync with blocks can contain awaits):

        with span("retrieve", input={"query": q}) as s:
            results = await retrieve(q)
            s.set_output({"count": len(results)})
    """

    def __init__(self, name: str, input: dict) -> None:  # type: ignore[type-arg]
        self.name = name
        self.input = input
        self._cm: Any = None
        self._ended = False
        self._start = time.perf_counter()

    def __enter__(self) -> span:
        client = _client()
        # Only create a child span when there is an active root trace
        if client is not None and _trace_cm_var.get() is not None:
            self._cm = client.start_as_current_observation(
                name=self.name, input=self.input
            )
            self._cm.__enter__()
        return self

    def set_output(self, output: dict) -> None:  # type: ignore[type-arg]
        """Record output and latency, then close the span."""
        if self._ended:
            return
        client = _client()
        if client is not None and self._cm is not None:
            elapsed_ms = int((time.perf_counter() - self._start) * 1000)
            try:
                client.update_current_span(output={**output, "latency_ms": elapsed_ms})
            except Exception:
                pass
        self._ended = True

    def __exit__(self, *_: object) -> None:
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
