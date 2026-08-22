"""Retrieval and context construction.

The pipeline:

    query -> scope filter -> embed -> vector search (top_k)
          -> rerank (rerank_top_k) -> token-budgeted context -> citations

Scoping is the part that makes this usable rather than a demo. Retrieving from
every document ever uploaded for every question produces confident nonsense, so
a query is scoped, in order of precedence:

1. explicit ``document_ids`` (the "ask this document" flow),
2. otherwise the current channel's documents,
3. otherwise everything.

Citations are built from the same chunks that entered the context, and the
numbered ``[1]``, ``[2]`` markers in the context text line up with the citation
list indices, so a model that cites ``[2]`` is pointing at a real source the UI
can render.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.core.models import Citation, RetrievedChunk
from app.core.tokens import estimate_tokens, truncate_to_tokens
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.reranker import Reranker, get_reranker
from app.rag.store import Filter, VectorStore, get_vector_store

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    #: Ready to paste into a prompt. Empty when nothing was retrieved.
    context: str = ""
    context_tokens: int = 0

    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    #: Number of candidates before reranking, for observability.
    candidates: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def metrics(self) -> dict[str, float | int]:
        return {
            "candidates": self.candidates,
            "returned": len(self.chunks),
            "context_tokens": self.context_tokens,
            "embed_ms": round(self.embed_ms, 1),
            "search_ms": round(self.search_ms, 1),
            "rerank_ms": round(self.rerank_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


def build_scope_filter(
    *, document_ids: list[str] | None = None, channel_id: str | None = None
) -> Filter | None:
    """Translate a scope into a store filter. See module docstring for precedence."""
    if document_ids:
        if len(document_ids) == 1:
            return {"document_id": document_ids[0]}
        return {"document_id": {"$in": list(document_ids)}}
    if channel_id:
        return {"channel_id": channel_id}
    return None


class Retriever:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._embeddings = embedding_provider
        self._store = vector_store
        self._reranker = reranker

    # Resolved lazily so tests can swap the globals after construction.
    @property
    def embeddings(self) -> EmbeddingProvider:
        return self._embeddings or get_embedding_provider()

    @property
    def store(self) -> VectorStore:
        return self._store or get_vector_store()

    @property
    def reranker(self) -> Reranker:
        return self._reranker or get_reranker()

    async def retrieve(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        channel_id: str | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        token_budget: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        result = RetrievalResult(query=query)

        query = (query or "").strip()
        if not query:
            return result

        top_k = top_k or settings.rag_top_k
        rerank_top_k = rerank_top_k or settings.rag_rerank_top_k
        token_budget = token_budget or settings.context_rag_token_budget
        min_score = min_score if min_score is not None else settings.rag_min_score

        where = build_scope_filter(document_ids=document_ids, channel_id=channel_id)

        # --- embed ---------------------------------------------------------
        embed_started = time.perf_counter()
        try:
            query_vector = await self.embeddings.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"failed to embed query: {exc}") from exc
        result.embed_ms = (time.perf_counter() - embed_started) * 1000

        # --- search --------------------------------------------------------
        search_started = time.perf_counter()
        try:
            candidates = await self.store.search(query_vector, top_k=top_k, where=where)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"vector search failed: {exc}") from exc
        result.search_ms = (time.perf_counter() - search_started) * 1000
        result.candidates = len(candidates)

        if not candidates and where is not None and channel_id and not document_ids:
            # Documents uploaded outside any channel are still worth searching
            # before telling the user nothing was found.
            try:
                candidates = await self.store.search(query_vector, top_k=top_k, where=None)
                result.candidates = len(candidates)
            except Exception as exc:  # noqa: BLE001
                raise RetrievalError(f"vector search failed: {exc}") from exc

        if not candidates:
            result.total_ms = (time.perf_counter() - started) * 1000
            return result

        # --- rerank --------------------------------------------------------
        rerank_started = time.perf_counter()
        try:
            ranked = await self.reranker.rerank(query, candidates, top_k=rerank_top_k)
        except Exception as exc:  # noqa: BLE001 - degrade rather than fail the query
            logger.error("reranking failed, using vector order: %s", exc)
            ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:rerank_top_k]
        result.rerank_ms = (time.perf_counter() - rerank_started) * 1000

        if min_score > 0:
            ranked = [c for c in ranked if c.final_score >= min_score]

        # --- build context -------------------------------------------------
        result.chunks = ranked
        result.context, result.context_tokens = build_context(ranked, token_budget)
        result.citations = [chunk.to_citation() for chunk in ranked]
        result.total_ms = (time.perf_counter() - started) * 1000

        logger.debug(
            "retrieved %d/%d chunks in %.0fms",
            len(ranked),
            result.candidates,
            result.total_ms,
            extra={"query": query[:80], "channel_id": channel_id},
        )
        return result


def format_source_label(chunk: RetrievedChunk) -> str:
    if chunk.page is not None:
        return f"{chunk.filename}, page {chunk.page}"
    return chunk.filename


def build_context(chunks: list[RetrievedChunk], token_budget: int) -> tuple[str, int]:
    """Render chunks into a numbered, token-bounded context block.

    Chunks are emitted best-first, and the last one to partially fit is
    truncated rather than dropped - a partial best-match passage is more useful
    than none. Anything that does not fit at all is left out silently; the
    citation list still reflects only what was included.
    """
    if not chunks or token_budget <= 0:
        return "", 0

    header = (
        "Excerpts from the user's documents. Cite them by their bracketed "
        "number, for example [1]. If the excerpts do not answer the question, "
        "say so rather than guessing.\n"
    )
    parts: list[str] = [header]
    used = estimate_tokens(header)
    included = 0

    for index, chunk in enumerate(chunks, start=1):
        label = f"\n[{index}] {format_source_label(chunk)}\n"
        label_tokens = estimate_tokens(label)
        remaining = token_budget - used - label_tokens

        if remaining <= 32:
            break

        text = chunk.text.strip()
        text_tokens = estimate_tokens(text)
        if text_tokens > remaining:
            text = truncate_to_tokens(text, remaining)
            text_tokens = estimate_tokens(text)

        parts.append(label)
        parts.append(text)
        used += label_tokens + text_tokens
        included += 1

    if included == 0:
        return "", 0

    # Trim the citation list to what actually made it into the context so the
    # numbers the model sees match the sources the user is shown.
    del chunks[included:]

    return "".join(parts).strip(), used


default_retriever = Retriever()
