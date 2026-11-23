"""Vector stores.

Two backends behind one interface:

``memory``
    Pure Python, persisted to a JSON file, no extra dependencies. This is the
    default because it makes the whole RAG path runnable anywhere, including
    CI. It brute-force scans, which is exactly right up to tens of thousands of
    chunks and wrong beyond that.

``chroma``
    A persistent Chroma collection with an HNSW index, for the GPU machine and
    larger corpora.

The filter language is deliberately small - equality and ``$in`` over metadata
keys - because that is all document scoping needs, and it is the intersection
of what both backends support natively.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import ConfigurationError, VectorStoreError
from app.core.logging import get_logger
from app.core.models import DocumentChunk, RetrievedChunk

logger = get_logger(__name__)

Vector = Sequence[float]

#: Metadata filter, e.g. {"channel_id": "general"} or
#: {"document_id": {"$in": ["doc_a", "doc_b"]}}
Filter = dict[str, Any]


def matches_filter(metadata: dict[str, Any], where: Filter | None) -> bool:
    """Evaluate a filter against one record's metadata."""
    if not where:
        return True
    for key, condition in where.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition:
                if value not in condition["$in"]:
                    return False
            if "$ne" in condition and value == condition["$ne"]:
                return False
            if "$eq" in condition and value != condition["$eq"]:
                return False
        elif value != condition:
            return False
    return True


@dataclass
class StoredChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    vector: list[float]

    def to_retrieved(self, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=self.id,
            document_id=str(self.metadata.get("document_id", "")),
            filename=str(self.metadata.get("filename", "")),
            text=self.text,
            chunk_index=int(self.metadata.get("chunk_index", 0)),
            page=self.metadata.get("page"),
            channel_id=self.metadata.get("channel_id"),
            score=score,
        )


class VectorStore(ABC):
    name: str = "base"

    @abstractmethod
    async def add(self, chunks: list[DocumentChunk], vectors: list[Vector]) -> None:
        """Index chunks. Re-adding an existing id replaces it."""

    @abstractmethod
    async def search(
        self, query_vector: Vector, *, top_k: int, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks, best first, scored 0..1."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        """Remove every chunk of a document. Returns how many were removed."""

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def document_ids(self) -> set[str]:
        ...

    async def health(self) -> tuple[str, str | None]:
        try:
            total = await self.count()
            return "healthy", f"{total} chunks indexed"
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return "unavailable", f"{type(exc).__name__}: {exc}"

    async def close(self) -> None:
        return None


class MemoryVectorStore(VectorStore):
    """Brute-force store with JSON persistence.

    Uses numpy for the scan when it is importable and falls back to pure Python
    otherwise, so the store works on any interpreter that can run FastAPI -
    including a Python version too new to have numpy wheels yet.
    """

    name = "memory"

    def __init__(self, path: Path | None = None, *, persist: bool = True) -> None:
        self.path = Path(path or settings.vector_store_path) / "chunks.json"
        self.persist_enabled = persist
        self._records: dict[str, StoredChunk] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._matrix = None  # numpy array cache, invalidated on write
        self._matrix_ids: list[str] = []
        self._np = self._try_import_numpy()
        if self.persist_enabled:
            self._load()

    @staticmethod
    def _try_import_numpy():
        try:
            import numpy

            return numpy
        except ImportError:
            logger.info("numpy unavailable - memory vector store using pure Python scan")
            return None

    # --- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("vector store at %s is unreadable: %s", self.path, exc)
            # Refuse to silently start empty: a corrupt index that looks like an
            # empty one produces confidently wrong "no results" answers.
            raise VectorStoreError(
                f"vector store file {self.path} is corrupt: {exc}",
                user_message=(
                    "The document index could not be loaded. Delete "
                    f"{self.path} to rebuild it from scratch."
                ),
            ) from exc

        for record in payload.get("chunks", []):
            self._records[record["id"]] = StoredChunk(
                id=record["id"],
                text=record["text"],
                metadata=record.get("metadata", {}),
                vector=record["vector"],
            )
        logger.info("loaded %d chunks from %s", len(self._records), self.path)

    def _save(self) -> None:
        if not self.persist_enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunks": [
                {
                    "id": r.id,
                    "text": r.text,
                    "metadata": r.metadata,
                    "vector": r.vector,
                }
                for r in self._records.values()
            ],
        }
        # Write-then-rename so an interrupted save cannot truncate the index.
        temp_path = self.path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload), encoding="utf-8")
        temp_path.replace(self.path)
        self._dirty = False

    async def flush(self) -> None:
        if self._dirty:
            await asyncio.to_thread(self._save_locked)

    def _save_locked(self) -> None:
        with self._lock:
            self._save()

    # --- writes ------------------------------------------------------------

    async def add(self, chunks: list[DocumentChunk], vectors: list[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors"
            )
        if not chunks:
            return

        with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._records[chunk.id] = StoredChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata(),
                    vector=[float(v) for v in vector],
                )
            self._matrix = None
            self._dirty = True
        await asyncio.to_thread(self._save_locked)

    async def delete_document(self, document_id: str) -> int:
        with self._lock:
            doomed = [
                cid
                for cid, record in self._records.items()
                if record.metadata.get("document_id") == document_id
            ]
            for cid in doomed:
                del self._records[cid]
            if doomed:
                self._matrix = None
                self._dirty = True
        if doomed:
            await asyncio.to_thread(self._save_locked)
        return len(doomed)

    # --- reads -------------------------------------------------------------

    async def search(
        self, query_vector: Vector, *, top_k: int, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        with self._lock:
            candidates = [
                record
                for record in self._records.values()
                if matches_filter(record.metadata, where)
            ]
        if not candidates:
            return []

        scored = await asyncio.to_thread(
            self._score, candidates, [float(v) for v in query_vector]
        )
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [record.to_retrieved(score) for record, score in scored[:top_k]]

    def _score(
        self, candidates: list[StoredChunk], query: list[float]
    ) -> list[tuple[StoredChunk, float]]:
        expected = len(query)
        usable = [c for c in candidates if len(c.vector) == expected]
        if len(usable) != len(candidates):
            # Happens when EMBEDDING_PROVIDER changed without reindexing.
            logger.warning(
                "skipped %d chunks with mismatched embedding dimensions - "
                "re-index after changing EMBEDDING_PROVIDER",
                len(candidates) - len(usable),
            )
        if not usable:
            return []

        if self._np is not None:
            np = self._np
            matrix = np.asarray([c.vector for c in usable], dtype=np.float32)
            q = np.asarray(query, dtype=np.float32)
            matrix_norms = np.linalg.norm(matrix, axis=1)
            q_norm = float(np.linalg.norm(q))
            if q_norm == 0.0:
                return [(c, 0.0) for c in usable]
            denominator = np.where(matrix_norms == 0.0, 1.0, matrix_norms) * q_norm
            similarities = (matrix @ q) / denominator
            return [
                (usable[i], self._to_unit_scale(float(similarities[i])))
                for i in range(len(usable))
            ]

        q_norm = math.sqrt(sum(v * v for v in query))
        results: list[tuple[StoredChunk, float]] = []
        for record in usable:
            norm = math.sqrt(sum(v * v for v in record.vector))
            if norm == 0.0 or q_norm == 0.0:
                results.append((record, 0.0))
                continue
            dot = sum(a * b for a, b in zip(record.vector, query, strict=True))
            results.append((record, self._to_unit_scale(dot / (norm * q_norm))))
        return results

    @staticmethod
    def _to_unit_scale(cosine: float) -> float:
        """Map cosine [-1, 1] onto [0, 1].

        Hashed embeddings genuinely produce negative similarities, and a
        negative "score" in the UI is meaningless, so the whole range is
        rescaled rather than clamped.
        """
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    async def count(self) -> int:
        with self._lock:
            return len(self._records)

    async def document_ids(self) -> set[str]:
        with self._lock:
            return {
                str(r.metadata.get("document_id"))
                for r in self._records.values()
                if r.metadata.get("document_id")
            }

    async def close(self) -> None:
        await self.flush()


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection.

    We pass vectors in explicitly and never let Chroma embed for us, so the
    ``EmbeddingProvider`` abstraction stays authoritative and swapping stores
    does not silently swap embedding models.
    """

    name = "chroma"

    def __init__(
        self, path: Path | None = None, collection_name: str | None = None
    ) -> None:
        self.path = Path(path or settings.vector_store_path) / "chroma"
        self.collection_name = collection_name or settings.chroma_collection
        self._collection = None
        self._lock = asyncio.Lock()

    async def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        async with self._lock:
            if self._collection is not None:
                return self._collection

            def _connect():
                try:
                    import chromadb
                    from chromadb.config import Settings as ChromaSettings
                except ImportError as exc:
                    raise ConfigurationError(
                        "VECTOR_STORE=chroma requires chromadb. Install it with "
                        "`pip install -r requirements-chroma.txt`, or set "
                        "VECTOR_STORE=memory.",
                        user_message="The Chroma vector store is not installed.",
                    ) from exc

                self.path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(self.path),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                return client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

            self._collection = await asyncio.to_thread(_connect)
            logger.info(
                "chroma collection ready",
                extra={"collection": self.collection_name, "path": str(self.path)},
            )
        return self._collection

    async def add(self, chunks: list[DocumentChunk], vectors: list[Vector]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise VectorStoreError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        collection = await self._ensure_collection()

        def _upsert():
            collection.upsert(
                ids=[c.id for c in chunks],
                embeddings=[[float(v) for v in vec] for vec in vectors],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata() for c in chunks],
            )

        try:
            await asyncio.to_thread(_upsert)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"chroma upsert failed: {exc}") from exc

    @staticmethod
    def _to_chroma_where(where: Filter | None) -> dict[str, Any] | None:
        """Translate our filter language into Chroma's ``where`` syntax."""
        if not where:
            return None
        clauses: list[dict[str, Any]] = []
        for key, condition in where.items():
            if isinstance(condition, dict):
                clauses.append({key: condition})
            else:
                clauses.append({key: {"$eq": condition}})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    async def search(
        self, query_vector: Vector, *, top_k: int, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        collection = await self._ensure_collection()
        chroma_where = self._to_chroma_where(where)

        def _query():
            return collection.query(
                query_embeddings=[[float(v) for v in query_vector]],
                n_results=top_k,
                where=chroma_where,
                include=["documents", "metadatas", "distances"],
            )

        try:
            response = await asyncio.to_thread(_query)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"chroma query failed: {exc}") from exc

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            # Chroma's cosine space returns distance = 1 - cosine, so it lands
            # in [0, 2]. Halving maps it onto the same 0..1 scale the memory
            # store reports, keeping thresholds portable between backends.
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance / 2.0))
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    document_id=str(metadata.get("document_id", "")),
                    filename=str(metadata.get("filename", "")),
                    text=documents[index] if index < len(documents) else "",
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    page=metadata.get("page"),
                    channel_id=metadata.get("channel_id"),
                    score=score,
                )
            )
        return results

    async def delete_document(self, document_id: str) -> int:
        collection = await self._ensure_collection()

        def _delete() -> int:
            existing = collection.get(where={"document_id": document_id}, include=[])
            ids = existing.get("ids") or []
            if ids:
                collection.delete(ids=ids)
            return len(ids)

        try:
            return await asyncio.to_thread(_delete)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"chroma delete failed: {exc}") from exc

    async def count(self) -> int:
        collection = await self._ensure_collection()
        return int(await asyncio.to_thread(collection.count))

    async def document_ids(self) -> set[str]:
        collection = await self._ensure_collection()

        def _scan() -> set[str]:
            response = collection.get(include=["metadatas"])
            return {
                str(m.get("document_id"))
                for m in (response.get("metadatas") or [])
                if m and m.get("document_id")
            }

        return await asyncio.to_thread(_scan)


def create_vector_store(name: str | None = None) -> VectorStore:
    backend = (name or settings.vector_store).lower()
    if backend == "memory":
        return MemoryVectorStore()
    if backend == "chroma":
        return ChromaVectorStore()
    raise ConfigurationError(
        f"unknown VECTOR_STORE '{backend}'", details={"supported": ["memory", "chroma"]}
    )


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = create_vector_store()
        logger.info("vector store ready", extra={"store": _store.name})
    return _store


def set_vector_store(store: VectorStore | None) -> None:
    global _store
    _store = store


async def close_vector_store() -> None:
    global _store
    if _store is not None:
        await _store.close()
        _store = None
