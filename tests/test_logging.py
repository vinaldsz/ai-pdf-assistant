"""Tests for structured logging and Langfuse no-op behaviour."""

from __future__ import annotations

import json
import logging


def test_configure_logging_outputs_json(capsys):
    from app.obs.logging import configure_logging

    configure_logging(log_level="DEBUG")
    logging.getLogger("test_json_output").info("hello world")

    captured = capsys.readouterr()
    # At least one line should be valid JSON containing our message
    json_lines = [ln for ln in captured.out.splitlines() if ln.strip().startswith("{")]
    assert json_lines, "No JSON output found"
    payload = json.loads(json_lines[-1])
    assert "event" in payload or "message" in payload


def test_request_id_propagation(capsys):
    from app.obs.logging import configure_logging, get_logger, request_id_var

    configure_logging()
    token = request_id_var.set("test-request-123")
    try:
        log = get_logger("test_request_id")
        log.info("propagation check")
        captured = capsys.readouterr()
        json_lines = [ln for ln in captured.out.splitlines() if ln.strip().startswith("{")]
        assert json_lines
        payload = json.loads(json_lines[-1])
        assert payload.get("request_id") == "test-request-123"
    finally:
        request_id_var.reset(token)


def test_request_id_defaults_to_dash(capsys):
    from app.obs.logging import configure_logging, get_logger, request_id_var

    configure_logging()
    # Ensure no request ID is set in this context
    request_id_var.set("-")
    log = get_logger("test_default_id")
    log.info("default id check")
    captured = capsys.readouterr()
    json_lines = [ln for ln in captured.out.splitlines() if ln.strip().startswith("{")]
    assert json_lines
    payload = json.loads(json_lines[-1])
    assert payload.get("request_id") == "-"


def test_langfuse_noop_when_keys_absent(monkeypatch):
    """When LANGFUSE_PUBLIC_KEY is not set, all functions must be silent no-ops."""
    from app.obs import langfuse as lf

    # Patch _is_enabled so it reports False regardless of env
    monkeypatch.setattr(lf, "_is_enabled", lambda: False)

    # None of these should raise
    lf.start_trace("test", input={"q": "hello"})
    lf.end_trace(output={"answer": "bye"})

    with lf.span("retrieve", input={"q": "hello"}) as s:
        s.set_output({"count": 3})


def test_langfuse_span_no_side_effects_without_trace(monkeypatch):
    """span() is safe to call even when no trace is active in the context var."""
    from unittest.mock import MagicMock

    from app.obs import langfuse as lf

    mock_client = MagicMock()
    # Patch _client directly so no real Langfuse connection is attempted
    monkeypatch.setattr(lf, "_client", lambda: mock_client)

    # _trace_cm_var defaults to None — span.__enter__ must not create a child observation
    with lf.span("generate", input={"x": 1}) as s:
        s.set_output({"y": 2})

    mock_client.start_as_current_observation.assert_not_called()
