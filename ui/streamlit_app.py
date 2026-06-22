"""Streamlit UI — pure HTTP client. No imports from app/.

Run with:
    streamlit run ui/streamlit_app.py

Set API_URL env var to point at a running instance (default: localhost:8000).
"""
from __future__ import annotations

import json
import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="AI PDF Assistant", page_icon="📄")
st.title("📄 AI PDF Assistant")

# ---------------------------------------------------------------------------
# Sidebar — ingest
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Index a PDF")
    pdf_url = st.text_input("PDF URL (https only)", placeholder="https://arxiv.org/pdf/1706.03762")

    if st.button("Index", disabled=not pdf_url):
        with st.spinner("Submitting…"):
            try:
                r = httpx.post(f"{API_URL}/index", json={"url": pdf_url}, timeout=10)
                r.raise_for_status()
                job = r.json()
                st.session_state["job_id"] = job["job_id"]
                st.success(f"Job queued: `{job['job_id']}`")
            except httpx.HTTPStatusError as e:
                st.error(f"Error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(str(e))

    # Poll job status if one is in progress
    job_id = st.session_state.get("job_id")
    if job_id:
        with st.spinner("Checking job…"):
            try:
                r = httpx.get(f"{API_URL}/jobs/{job_id}", timeout=5)
                job = r.json()
                status = job["status"]
                if status == "done":
                    st.success(f"Done — {job.get('chunk_count', '?')} chunks indexed")
                    del st.session_state["job_id"]
                elif status == "failed":
                    st.error(f"Failed: {job.get('error', 'unknown error')}")
                    del st.session_state["job_id"]
                else:
                    st.info(f"Status: {status} — refresh to update")
            except Exception as e:
                st.warning(str(e))

    st.divider()
    st.caption(f"API: `{API_URL}`")

# ---------------------------------------------------------------------------
# Main — chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            _render_citations(msg["citations"])  # type: ignore[name-defined]

if query := st.chat_input("Ask a question about your PDFs…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        citations: list[dict] = []  # type: ignore[type-arg]
        full_text = ""

        try:
            with httpx.Client(timeout=60) as client:
                with client.stream(
                    "POST",
                    f"{API_URL}/query/stream",
                    json={"query": query},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        if "token" in event:
                            full_text += event["token"]
                            placeholder.markdown(full_text + "▌")
                        if event.get("done"):
                            citations = event.get("citations", [])

            placeholder.markdown(full_text)

        except httpx.HTTPStatusError as e:
            full_text = f"Error {e.response.status_code}: {e.response.text}"
            placeholder.error(full_text)
        except Exception as e:
            full_text = str(e)
            placeholder.error(full_text)

        if citations:
            st.divider()
            st.caption("**Sources**")
            for i, c in enumerate(citations, 1):
                st.caption(
                    f"[{i}] page {c['page']} · score {c['score']:.3f} · doc `{c['doc_id'][:8]}…`"
                    f"\n> {c['snippet'][:120]}…"
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "citations": citations}
        )


def _render_citations(citations: list[dict]) -> None:  # type: ignore[type-arg]
    if not citations:
        return
    st.divider()
    st.caption("**Sources**")
    for i, c in enumerate(citations, 1):
        st.caption(
            f"[{i}] page {c['page']} · score {c['score']:.3f} · doc `{c['doc_id'][:8]}…`"
            f"\n> {c['snippet'][:120]}…"
        )
