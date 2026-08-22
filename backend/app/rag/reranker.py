"""Reranking.

Vector search is recall-oriented: it is good at pulling twenty plausible
passages out of thousands, and mediocre at ordering those twenty. Reranking
fixes the ordering, which is what actually reaches the model's context window.

The default ``lexical`` reranker is BM25 over the retrieved candidates, fused
with the vector score. It costs microseconds, needs no model, and reliably
rescues the case vector search handles worst: an exact term the user typed
verbatim ("Section 4.2", a product code) that the embedding blurred away.

``cross_encoder`` is stronger and slower, and is the right choice on the GPU
machine.
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from collections import Counter

from app.core.config import settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.core.models import RetrievedChunk
from app.rag.embeddings import tokenize

logger = get_logger(__name__)


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        """Reorder ``chunks`` by relevance to ``query`` and truncate to ``top_k``."""


class NoopReranker(Reranker):
    """Keeps vector order. Useful as a benchmark baseline."""

    name = "none"

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ordered[:top_k]


class LexicalReranker(Reranker):
    """BM25 over the candidate set, fused with the vector score.

    BM25 statistics are computed over the retrieved candidates rather than the
    whole corpus. That is a deliberate approximation: it is cheap, needs no
    global index, and within a candidate set of ~20 it discriminates well.
    """

    name = "lexical"

    #: Term-frequency saturation. 1.2 is the standard default.
    K1 = 1.2
    #: Length normalization strength.
    B = 0.75
    #: Weight on the lexical score when fusing with the vector score. Vector
    #: similarity still leads; BM25 breaks ties and rescues exact matches.
    LEXICAL_WEIGHT = 0.4

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = await asyncio.to_thread(self._bm25_scores, query, chunks)

        highest = max(scores) if scores else 0.0
        for chunk, lexical in zip(chunks, scores):
            normalized = (lexical / highest) if highest > 0 else 0.0
            chunk.rerank_score = (
                (1.0 - self.LEXICAL_WEIGHT) * chunk.score
                + self.LEXICAL_WEIGHT * normalized
            )

        chunks.sort(key=lambda c: c.final_score, reverse=True)
        return chunks[:top_k]

    def _bm25_scores(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * len(chunks)

        documents = [tokenize(c.text) for c in chunks]
        lengths = [len(d) for d in documents]
        average_length = (sum(lengths) / len(lengths)) if lengths else 0.0
        if average_length == 0.0:
            return [0.0] * len(chunks)

        total_documents = len(documents)
        document_frequency = Counter()
        term_counts = []
        for document in documents:
            counts = Counter(document)
            term_counts.append(counts)
            for term in counts:
                document_frequency[term] += 1

        scores: list[float] = []
        for index, counts in enumerate(term_counts):
            score = 0.0
            length_ratio = lengths[index] / average_length
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                # Robertson/Sparck-Jones IDF with the +0.5 smoothing that keeps
                # it non-negative for terms present in every candidate.
                df = document_frequency[term]
                idf = math.log(1.0 + (total_documents - df + 0.5) / (df + 0.5))
                numerator = frequency * (self.K1 + 1.0)
                denominator = frequency + self.K1 * (
                    1.0 - self.B + self.B * length_ratio
                )
                score += idf * (numerator / denominator)
            scores.append(score)
        return scores


class CrossEncoderReranker(Reranker):
    """A cross-encoder scoring each (query, passage) pair jointly."""

    name = "cross_encoder"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = None
        self._lock = asyncio.Lock()
        self._fallback = LexicalReranker()

    async def _ensure_model(self):
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model

            def _load():
                from sentence_transformers import CrossEncoder

                return CrossEncoder(self.model_name)

            logger.info("loading reranker model %s", self.model_name)
            self._model = await asyncio.to_thread(_load)
        return self._model

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            model = await self._ensure_model()
        except ImportError:
            logger.warning(
                "RERANKER=cross_encoder but sentence-transformers is not installed; "
                "falling back to lexical reranking"
            )
            return await self._fallback.rerank(query, chunks, top_k=top_k)

        pairs = [(query, chunk.text) for chunk in chunks]

        def _score():
            return model.predict(pairs, show_progress_bar=False)

        try:
            raw = await asyncio.to_thread(_score)
        except Exception as exc:  # noqa: BLE001 - never fail a query on reranking
            logger.error("cross-encoder scoring failed, using lexical: %s", exc)
            return await self._fallback.rerank(query, chunks, top_k=top_k)

        # Cross-encoder logits are unbounded; squash to 0..1 so the score means
        # the same thing as it does for the other rerankers.
        for chunk, value in zip(chunks, raw):
            chunk.rerank_score = 1.0 / (1.0 + math.exp(-float(value)))

        chunks.sort(key=lambda c: c.final_score, reverse=True)
        return chunks[:top_k]


def create_reranker(name: str | None = None) -> Reranker:
    backend = (name or settings.reranker).lower()
    if backend == "none":
        return NoopReranker()
    if backend == "lexical":
        return LexicalReranker()
    if backend == "cross_encoder":
        return CrossEncoderReranker()
    raise ConfigurationError(
        f"unknown RERANKER '{backend}'",
        details={"supported": ["none", "lexical", "cross_encoder"]},
    )


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = create_reranker()
        logger.info("reranker ready", extra={"reranker": _reranker.name})
    return _reranker


def set_reranker(reranker: Reranker | None) -> None:
    global _reranker
    _reranker = reranker
