"""POST /query — hybrid RAG pipeline: retrieve → rerank → generate → citations.
POST /query/stream — same pipeline but streams tokens via Server-Sent Events.
"""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag import generator, reranker, retriever

router = APIRouter()

_NO_ANSWER = (
    "I don't have enough information in the indexed documents to answer that question."
)


class QueryRequest(BaseModel):
    query: str


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
    chunks = await retriever.retrieve(body.query)

    if not chunks:
        return QueryResponse(answer=_NO_ANSWER, citations=[])

    reranked = await reranker.rerank(body.query, chunks)
    result = await generator.generate(body.query, reranked)
    return QueryResponse(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
    )


@router.post("/query/stream")
async def query_stream_endpoint(body: QueryRequest) -> StreamingResponse:
    """SSE endpoint — yields `data: {"token": "..."}` events, then a final
    `data: {"citations": [...], "done": true}` event."""
    chunks = await retriever.retrieve(body.query)

    if not chunks:
        async def _no_answer_sse():
            yield f"data: {json.dumps({'token': _NO_ANSWER})}\n\n"
            yield f"data: {json.dumps({'citations': [], 'done': True})}\n\n"

        return StreamingResponse(_no_answer_sse(), media_type="text/event-stream")

    reranked = await reranker.rerank(body.query, chunks)
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
        async for token in generator.generate_stream(body.query, reranked):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'citations': citations, 'done': True})}\n\n"

    return StreamingResponse(_stream_sse(), media_type="text/event-stream")
