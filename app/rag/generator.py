"""Prompt builder and Groq LLM completion with tenacity retries."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import APIStatusError, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.obs.logging import get_logger
from app.settings import settings

if TYPE_CHECKING:
    from app.rag.retriever import RetrievalResult

log = get_logger(__name__)

# Hard ceiling on context fed to the LLM — prevents context overflow regardless of
# RERANK_K config or a disabled reranker letting too many chunks through.
_MAX_CONTEXT_CHARS = 6_000  # ~1,500 tokens for English text

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based solely on the provided "
    "document excerpts. If the excerpts do not contain enough information, say you don't "
    "know — do not invent facts. Cite the page numbers you used."
)


@dataclass
class GeneratorResponse:
    answer: str
    citations: list[dict]  # type: ignore[type-arg]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504)


def _build_messages(query: str, context: str) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Document excerpts:\n\n{context}\n\nQuestion: {query}"},
    ]


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def generate(query: str, chunks: list[RetrievalResult]) -> GeneratorResponse:
    """Build prompt from retrieved chunks and call Groq. Retries on 429/5xx."""
    context = _build_context(chunks)
    citations = [
        {
            "doc_id": c.doc_id,
            "page": c.page,
            "snippet": c.text[:200],
            "score": round(c.score, 4),
        }
        for c in chunks
    ]

    client = AsyncOpenAI(
        api_key=settings.groq_api_key.get_secret_value(), base_url="https://api.groq.com/openai/v1"
    )
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=_build_messages(query, context),
        temperature=0.1,
        max_tokens=512,
        timeout=20.0,
    )

    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    log.info(
        "generate.done",
        answer_len=len(answer),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return GeneratorResponse(
        answer=answer,
        citations=citations,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _create_stream(
    client: AsyncOpenAI, messages: list[ChatCompletionMessageParam]
) -> AsyncStream[ChatCompletionChunk]:
    """Retryable stream creation — retry here, not inside the async generator."""
    return await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
        timeout=20.0,
        stream=True,
    )


async def generate_stream(query: str, chunks: list[RetrievalResult]) -> AsyncGenerator[str, None]:
    """Stream token strings from Groq. Yields raw token strings one at a time."""
    context = _build_context(chunks)
    client = AsyncOpenAI(
        api_key=settings.groq_api_key.get_secret_value(), base_url="https://api.groq.com/openai/v1"
    )
    stream = await _create_stream(client, _build_messages(query, context))
    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


def _build_context(chunks: list[RetrievalResult]) -> str:
    """Concatenate chunk texts, strictly respecting _MAX_CONTEXT_CHARS."""
    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks):
        entry = f"[{i + 1}] (page {chunk.page})\n{chunk.text}"
        sep = "\n\n" if parts else ""
        cost = len(sep) + len(entry)
        if total + cost > _MAX_CONTEXT_CHARS:
            remaining = _MAX_CONTEXT_CHARS - total - len(sep)
            if remaining > 100:
                parts.append(entry[:remaining])
            break
        parts.append(entry)
        total += cost
    return "\n\n".join(parts)
