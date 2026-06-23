"""POST /query — hybrid RAG pipeline: retrieve → rerank → generate → citations.
POST /query/stream — same pipeline but streams tokens via Server-Sent Events.
"""
from __future__ import annotations

import json
import unicodedata

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIStatusError
from pydantic import BaseModel, Field, field_validator

from app.obs import langfuse as lf
from app.obs.logging import get_logger
from app.rag import generator, reranker, retriever

router = APIRouter()
log = get_logger(__name__)

_NO_ANSWER = (
    "I don't have enough information in the indexed documents to answer that question."
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def _strip_control_chars(cls, v: str) -> str:
        # Remove C0/C1 control characters (null bytes, escape sequences, etc.)
        # but keep normal whitespace (\n, \t, space).
        return "".join(
            ch for ch in v if unicodedata.category(ch) != "Cc" or ch in "\n\t"
        )


class Citation(BaseModel):
    doc_id: str
    page: int
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest) -> QueryResponse:
    log.info("query", query=body.query[:120])
    lf.start_trace("query", input={"query": body.query})

    with lf.span("retrieve", input={"query": body.query}) as s:
        chunks = await retriever.retrieve(body.query)
        s.set_output({"count": len(chunks)})

    if not chunks:
        log.info("query.no_results")
        lf.end_trace(output={"answer": "no_results"})
        return QueryResponse(answer=_NO_ANSWER, citations=[])

    with lf.span("rerank", input={"candidates": len(chunks)}) as s:
        reranked = await reranker.rerank(body.query, chunks)
        s.set_output({"count": len(reranked)})

    try:
        with lf.span("generate", input={"chunks": len(reranked)}) as s:
            result = await generator.generate(body.query, reranked)
            s.set_output({"answer_len": len(result.answer)})
    except APIStatusError as exc:
        log.warning("generate.upstream_error", status_code=exc.status_code)
        lf.end_trace(output={"error": exc.status_code})
        status = 429 if exc.status_code == 429 else 503
        raise HTTPException(status_code=status, detail="LLM service temporarily unavailable") from exc

    log.info("query.done", citations=len(result.citations))
    lf.end_trace(output={"answer": result.answer[:200]})
    return QueryResponse(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
    )


@router.post("/query/stream")
async def query_stream_endpoint(body: QueryRequest) -> StreamingResponse:
    """SSE endpoint — yields `data: {"token": "..."}` events, then a final
    `data: {"citations": [...], "done": true}` event."""
    log.info("query.stream", query=body.query[:120])
    lf.start_trace("query.stream", input={"query": body.query})

    with lf.span("retrieve", input={"query": body.query}) as s:
        chunks = await retriever.retrieve(body.query)
        s.set_output({"count": len(chunks)})

    if not chunks:
        log.info("query.stream.no_results")
        lf.end_trace(output={"answer": "no_results"})

        async def _no_answer_sse():
            yield f"data: {json.dumps({'token': _NO_ANSWER})}\n\n"
            yield f"data: {json.dumps({'citations': [], 'done': True})}\n\n"

        return StreamingResponse(_no_answer_sse(), media_type="text/event-stream")

    with lf.span("rerank", input={"candidates": len(chunks)}) as s:
        reranked = await reranker.rerank(body.query, chunks)
        s.set_output({"count": len(reranked)})

    citations = [
        {
            "doc_id": c.doc_id,
            "page": c.page,
            "snippet": c.text[:400],
            "score": round(c.score, 4),
        }
        for c in reranked
    ]

    async def _stream_sse():
        token_count = 0
        try:
            with lf.span("generate", input={"chunks": len(reranked)}) as s:
                async for token in generator.generate_stream(body.query, reranked):
                    token_count += 1
                    yield f"data: {json.dumps({'token': token})}\n\n"
                s.set_output({"tokens": token_count})
        except APIStatusError as exc:
            log.warning("generate.upstream_error", status_code=exc.status_code)
            lf.end_trace(output={"error": exc.status_code})
            yield f"data: {json.dumps({'error': 'LLM service temporarily unavailable', 'done': True})}\n\n"
            return
        lf.end_trace(output={"tokens": token_count, "citations": len(citations)})
        log.info("query.stream.done", tokens=token_count, citations=len(citations))
        yield f"data: {json.dumps({'citations': citations, 'done': True})}\n\n"

    return StreamingResponse(_stream_sse(), media_type="text/event-stream")
