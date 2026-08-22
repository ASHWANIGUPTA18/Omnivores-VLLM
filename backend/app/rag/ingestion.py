"""The document ingestion pipeline.

    VALIDATE -> STORE -> PARSE -> CHUNK -> EMBED -> INDEX -> READY

Each stage advances the document's ``status``/``stage``/``progress`` and calls
the progress hook, which the document service turns into WebSocket events. That
is what lets the UI show a real progress bar instead of a spinner that lies.

Failures are terminal for the document and recorded on it: a document that
failed at PARSE stays visible in the UI with the reason attached, rather than
disappearing or sitting at 40% forever.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from app.core.config import settings
from app.core.exceptions import DocumentParseError, EmbeddingError, VectorStoreError
from app.core.logging import get_logger
from app.core.models import Document, DocumentChunk, DocumentStatus, utcnow
from app.rag.chunker import Chunker
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.parser import ParserRegistry, parser_registry
from app.rag.store import VectorStore, get_vector_store

logger = get_logger(__name__)

ProgressHook = Callable[[Document], Awaitable[None]]

#: Stage -> progress fraction reported when that stage *completes*.
_STAGE_PROGRESS = {
    "stored": 0.10,
    "parsed": 0.35,
    "chunked": 0.50,
    "embedded": 0.85,
    "indexed": 1.00,
}


@dataclass
class IngestionResult:
    document: Document
    chunks: list[DocumentChunk]
    parse_ms: float = 0.0
    chunk_ms: float = 0.0
    embed_ms: float = 0.0
    index_ms: float = 0.0
    total_ms: float = 0.0

    def metrics(self) -> dict[str, float | int]:
        return {
            "chunks": len(self.chunks),
            "parse_ms": round(self.parse_ms, 1),
            "chunk_ms": round(self.chunk_ms, 1),
            "embed_ms": round(self.embed_ms, 1),
            "index_ms": round(self.index_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


class IngestionPipeline:
    def __init__(
        self,
        *,
        parsers: ParserRegistry | None = None,
        chunker: Chunker | None = None,
        embeddings: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.parsers = parsers or parser_registry
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store

    @property
    def chunker(self) -> Chunker:
        # Built per access so a settings change between runs is picked up.
        return self._chunker or Chunker()

    @property
    def embeddings(self) -> EmbeddingProvider:
        return self._embeddings or get_embedding_provider()

    @property
    def store(self) -> VectorStore:
        return self._store or get_vector_store()

    async def ingest(
        self,
        document: Document,
        path: Path,
        *,
        on_progress: ProgressHook | None = None,
    ) -> IngestionResult:
        started = time.perf_counter()
        result = IngestionResult(document=document, chunks=[])

        async def advance(stage: str, status: DocumentStatus | None = None) -> None:
            document.stage = stage
            document.progress = _STAGE_PROGRESS.get(stage, document.progress)
            if status is not None:
                document.status = status
            document.updated_at = utcnow()
            if on_progress:
                await on_progress(document)

        try:
            await advance("stored", DocumentStatus.PROCESSING)

            # --- parse -----------------------------------------------------
            parse_started = time.perf_counter()
            parsed = await asyncio.to_thread(self.parsers.parse, path)
            result.parse_ms = (time.perf_counter() - parse_started) * 1000

            if parsed.is_empty:
                raise DocumentParseError(
                    f"{document.filename} contains no extractable text",
                    user_message="No readable text was found in that file.",
                )

            document.page_count = parsed.page_count
            document.character_count = parsed.character_count
            await advance("parsed")

            # --- chunk -----------------------------------------------------
            chunk_started = time.perf_counter()
            chunks = await asyncio.to_thread(
                self.chunker.chunk_document,
                parsed,
                document_id=document.id,
                filename=document.filename,
                channel_id=document.channel_id,
            )
            result.chunk_ms = (time.perf_counter() - chunk_started) * 1000

            if not chunks:
                raise DocumentParseError(
                    f"{document.filename} produced no chunks",
                    user_message="That document could not be split into indexable text.",
                )

            result.chunks = chunks
            document.chunk_count = len(chunks)
            await advance("chunked")

            # --- embed -----------------------------------------------------
            await advance("embedding", DocumentStatus.INDEXING)
            embed_started = time.perf_counter()
            try:
                vectors = await self._embed_in_batches(
                    [c.text for c in chunks], document, on_progress
                )
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"embedding failed: {exc}") from exc
            result.embed_ms = (time.perf_counter() - embed_started) * 1000
            await advance("embedded")

            # --- index -----------------------------------------------------
            index_started = time.perf_counter()
            try:
                await self.store.add(chunks, vectors)
            except VectorStoreError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"indexing failed: {exc}") from exc
            result.index_ms = (time.perf_counter() - index_started) * 1000

            document.indexed_at = utcnow()
            document.error = None
            await advance("indexed", DocumentStatus.READY)

            result.total_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "indexed %s", document.filename, extra={"document_id": document.id, **result.metrics()}
            )
            return result

        except Exception as exc:
            document.status = DocumentStatus.FAILED
            document.stage = "failed"
            document.error = getattr(exc, "user_message", None) or str(exc)
            document.updated_at = utcnow()
            logger.error(
                "ingestion failed for %s: %s",
                document.filename,
                exc,
                extra={"document_id": document.id},
            )
            if on_progress:
                await on_progress(document)
            raise

    async def _embed_in_batches(
        self,
        texts: list[str],
        document: Document,
        on_progress: ProgressHook | None,
    ) -> list[list[float]]:
        """Embed in batches so a large document reports incremental progress.

        Progress is interpolated between the 'chunked' and 'embedded'
        checkpoints, which is the slow part of ingesting a real PDF.
        """
        batch_size = max(1, settings.embedding_batch_size)
        vectors: list[list[float]] = []
        floor = _STAGE_PROGRESS["chunked"]
        ceiling = _STAGE_PROGRESS["embedded"]

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self.embeddings.embed_documents(batch))

            if on_progress:
                fraction = min(1.0, (start + len(batch)) / max(1, len(texts)))
                document.progress = floor + (ceiling - floor) * fraction
                document.stage = "embedding"
                document.updated_at = utcnow()
                await on_progress(document)

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedding provider returned {len(vectors)} vectors for {len(texts)} chunks"
            )
        return vectors


default_pipeline = IngestionPipeline()
