"""Streamlit UI — pure HTTP client. No imports from app/.

Run with:
    streamlit run ui/streamlit_app.py

Set API_URL env var to point at a running instance (default: localhost:8000).
"""
from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="PDF Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { min-width: 320px; max-width: 320px; }
        .block-container { padding-top: 2rem; }
        .source-card {
            background: #f8f9fa;
            border-left: 3px solid #dee2e6;
            padding: 0.6rem 0.9rem;
            margin: 0.4rem 0;
            border-radius: 0 4px 4px 0;
            font-size: 0.82rem;
            color: #495057;
        }
        .source-meta { font-weight: 600; margin-bottom: 0.2rem; color: #212529; }
        h1 { font-size: 1.5rem !important; font-weight: 700 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — document ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### PDF Assistant")
    st.caption("Upload a document, then ask questions about its content.")
    st.divider()

    st.markdown("**Add a document**")
    pdf_url = st.text_input(
        "Document URL",
        placeholder="https://arxiv.org/pdf/1706.03762",
        label_visibility="collapsed",
    )

    if st.button("Index document", disabled=not pdf_url, use_container_width=True):
        with st.spinner("Submitting..."):
            try:
                r = httpx.post(f"{API_URL}/index", json={"url": pdf_url}, timeout=10)
                r.raise_for_status()
                job = r.json()
                st.session_state["job_id"] = job["job_id"]
                st.success("Document queued for indexing.")
            except httpx.HTTPStatusError as e:
                st.error(f"Error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(str(e))

    job_id = st.session_state.get("job_id")
    if job_id:
        with st.spinner("Indexing in progress..."):
            try:
                r = httpx.get(f"{API_URL}/jobs/{job_id}", timeout=5)
                job = r.json()
                status = job["status"]
                if status == "done":
                    st.success(f"Indexed — {job.get('chunk_count', '?')} chunks ready.")
                    del st.session_state["job_id"]
                elif status == "failed":
                    st.error(f"Failed: {job.get('error', 'unknown error')}")
                    del st.session_state["job_id"]
                else:
                    st.info("Still processing — refresh to check status.")
            except Exception as e:
                st.warning(str(e))

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"API endpoint: `{API_URL}`")

# ---------------------------------------------------------------------------
# Main — chat
# ---------------------------------------------------------------------------
st.markdown("## PDF Assistant")
st.caption("Ask questions about any document you have indexed.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _preview(text: str) -> str:
    sentences = text.replace("\n", " ").split(". ")
    preview = ". ".join(sentences[:2])
    return preview + ("." if not preview.endswith(".") else "")


def _render_citations(citations: list[dict]) -> None:  # type: ignore[type-arg]
    if not citations:
        return
    st.markdown("**Sources**")
    for i, c in enumerate(citations, 1):
        score_pct = min(100, int(c["score"] * 100))
        snippet = c["snippet"]
        preview = _preview(snippet)
        with st.expander(f"Source {i}  ·  Page {c['page']}  ·  Relevance {score_pct}%"):
            st.markdown(f"_{preview}_")
            if len(snippet) > len(preview) + 5:
                st.caption(snippet[len(preview):].strip())


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            _render_citations(msg["citations"])

if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        citations: list[dict] = []  # type: ignore[type-arg]
        full_text = ""

        try:
            with httpx.Client(timeout=120) as client:
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
                            placeholder.markdown(full_text + " |")
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
            _render_citations(citations)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "citations": citations}
        )
