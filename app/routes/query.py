"""POST /query — hybrid RAG pipeline: retrieve → generate → citations."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.rag import retriever
from app.rag.generator import generate

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

    result = await generate(body.query, chunks)
    return QueryResponse(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
    )
