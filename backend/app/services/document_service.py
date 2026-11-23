"""Document lifecycle.

Owns the path from an uploaded byte stream to a searchable index, and the
metadata the UI renders along the way:

    UPLOADING -> PROCESSING -> INDEXING -> READY
                     \\-> FAILED

Validation happens before anything touches disk, and it validates three things
independently: the declared size, the actual streamed size, and the extension.
Trusting a client-declared ``Content-Length`` alone is how you end up with a
50 GB file on a "50 MB limit" server.

Ingestion runs as a background task so the HTTP upload returns immediately and
progress arrives over the WebSocket.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    DocumentNotFound,
    FileTooLarge,
    UnsupportedFileType,
    ValidationError,
)
from app.core.logging import get_logger
from app.core.models import Document, DocumentStatus, utcnow
from app.rag.ingestion import IngestionPipeline, default_pipeline
from app.rag.store import VectorStore, get_vector_store

logger = get_logger(__name__)

EventHook = Callable[[dict[str, Any]], Awaitable[None]]

#: Anything outside this is stripped from a stored filename.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]")


def sanitize_filename(filename: str) -> str:
    """Reduce an uploaded name to something safe to display and store.

    Path traversal is defeated by taking the basename of both separators (a
    Windows client can send backslashes to a Linux server, where ``os.path``
    would not treat them as separators) and by rejecting the result if it is
    empty or still relative.
    """
    name = unicodedata.normalize("NFKC", filename or "").strip()
    name = name.replace("\\", "/").split("/")[-1]
    name = _SAFE_FILENAME_RE.sub("_", name)
    name = name.lstrip(".") or "upload"
    if len(name) > 200:
        stem, _, suffix = name.rpartition(".")
        name = (stem[:180] + "." + suffix) if suffix else name[:200]
    return name


class DocumentService:
    def __init__(
        self,
        *,
        pipeline: IngestionPipeline | None = None,
        store: VectorStore | None = None,
        upload_dir: Path | None = None,
    ) -> None:
        self._pipeline = pipeline or default_pipeline
        self._store = store
        self.upload_dir = Path(upload_dir or settings.upload_dir)
        self._documents: dict[str, Document] = {}
        self._tasks: set[asyncio.Task] = set()
        #: In-flight ingestions by document id, so delete can cancel one.
        self._ingestions: dict[str, asyncio.Task] = {}
        self._on_event: EventHook | None = None

    @property
    def store(self) -> VectorStore:
        return self._store or get_vector_store()

    def set_event_hook(self, hook: EventHook | None) -> None:
        """Wire document events to the broadcast hub."""
        self._on_event = hook

    async def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event:
            try:
                await self._on_event(event)
            except Exception as exc:  # noqa: BLE001 - never fail ingestion on a socket
                logger.debug("document event hook failed: %s", exc)

    # --- validation --------------------------------------------------------

    def validate_upload(self, filename: str, declared_size: int | None) -> tuple[str, str]:
        """Check name, extension and declared size. Returns (safe_name, extension)."""
        safe_name = sanitize_filename(filename)
        extension = Path(safe_name).suffix.lower()

        if not extension:
            raise UnsupportedFileType(
                f"'{safe_name}' has no file extension",
                user_message="The file needs an extension so we know how to read it.",
            )
        if extension not in settings.allowed_extension_set:
            raise UnsupportedFileType(
                f"extension '{extension}' is not allowed",
                user_message=(
                    f"'{extension}' files are not supported. Allowed: "
                    + ", ".join(sorted(settings.allowed_extension_set))
                ),
            )
        if declared_size is not None and declared_size > settings.max_upload_size_bytes:
            raise FileTooLarge(
                f"{declared_size} bytes exceeds the limit",
                user_message=(
                    f"That file is larger than the {settings.max_upload_size_mb} MB limit."
                ),
            )
        return safe_name, extension

    # --- upload ------------------------------------------------------------

    async def upload(
        self,
        *,
        filename: str,
        stream: AsyncIterator[bytes],
        content_type: str = "application/octet-stream",
        channel_id: str | None = None,
        declared_size: int | None = None,
        uploaded_by: str = "user",
        process: bool = True,
    ) -> Document:
        """Store an uploaded file and start indexing it."""
        safe_name, extension = self.validate_upload(filename, declared_size)

        document = Document(
            filename=safe_name,
            extension=extension,
            content_type=content_type,
            channel_id=channel_id,
            uploaded_by=uploaded_by,
            status=DocumentStatus.UPLOADING,
            stage="uploading",
        )

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # Store under the document id so two uploads of "report.pdf" cannot
        # collide, and so a crafted filename can never escape the directory.
        target = self.upload_dir / f"{document.id}{extension}"
        document.storage_path = str(target)

        written = 0
        try:
            with target.open("wb") as handle:
                async for piece in stream:
                    written += len(piece)
                    if written > settings.max_upload_size_bytes:
                        raise FileTooLarge(
                            f"upload exceeded {settings.max_upload_size_mb} MB mid-stream",
                            user_message=(
                                f"That file is larger than the "
                                f"{settings.max_upload_size_mb} MB limit."
                            ),
                        )
                    handle.write(piece)
        except Exception:
            target.unlink(missing_ok=True)
            raise

        if written == 0:
            target.unlink(missing_ok=True)
            raise ValidationError(
                "uploaded file is empty", user_message="That file is empty."
            )

        document.size_bytes = written
        document.updated_at = utcnow()
        self._documents[document.id] = document

        from app.core import events as ev

        await self._emit(ev.document_uploaded(document))
        logger.info(
            "stored upload %s (%d bytes)",
            document.filename,
            written,
            extra={"document_id": document.id},
        )

        if process:
            self._spawn_ingestion(document, target)
        return document

    def _spawn_ingestion(self, document: Document, path: Path) -> None:
        task = asyncio.create_task(
            self._ingest(document, path), name=f"ingest-{document.id}"
        )
        self._tasks.add(task)
        self._ingestions[document.id] = task
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(
            lambda finished, doc_id=document.id: self._ingestions.pop(doc_id, None)
            if self._ingestions.get(doc_id) is finished
            else None
        )

    async def _cancel_ingestion(self, document_id: str) -> None:
        """Stop an in-flight ingestion and wait for it to actually stop.

        Deleting a document while its ingestion is still running leaves chunks
        the task writes *after* the store was cleared - searchable rows with no
        document to attribute them to.
        """
        task = self._ingestions.pop(document_id, None)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    async def _ingest(self, document: Document, path: Path) -> None:
        from app.core import events as ev

        async def on_progress(doc: Document) -> None:
            if doc.status == DocumentStatus.FAILED:
                await self._emit(ev.document_failed(doc))
            elif doc.status == DocumentStatus.READY:
                await self._emit(ev.document_ready(doc))
            else:
                await self._emit(ev.document_processing(doc))

        try:
            await self._pipeline.ingest(document, path, on_progress=on_progress)
        except Exception:
            # The pipeline has already set FAILED and emitted; nothing to add
            # beyond not letting the task die noisily.
            logger.debug("ingestion task ended in failure for %s", document.id)

    async def process_now(self, document_id: str) -> Document:
        """Ingest synchronously. Used by tests and by the reindex endpoint."""
        document = self.get(document_id)
        if not document.storage_path:
            raise DocumentNotFound(f"no stored file for {document_id}")
        await self._ingest(document, Path(document.storage_path))
        return document

    async def wait_for_processing(self, timeout: float = 60.0) -> None:
        """Await every in-flight ingestion. Used by tests and by shutdown."""
        if not self._tasks:
            return
        await asyncio.wait(set(self._tasks), timeout=timeout)

    # --- queries -----------------------------------------------------------

    def get(self, document_id: str) -> Document:
        try:
            return self._documents[document_id]
        except KeyError:
            raise DocumentNotFound(f"no document '{document_id}'") from None

    def list(
        self, *, channel_id: str | None = None, include_deleted: bool = False
    ) -> list[Document]:
        documents = [
            d
            for d in self._documents.values()
            if include_deleted or d.status != DocumentStatus.DELETED
        ]
        if channel_id is not None:
            documents = [d for d in documents if d.channel_id == channel_id]
        return sorted(documents, key=lambda d: d.created_at, reverse=True)

    def ready_documents(self, channel_id: str | None = None) -> list[Document]:
        return [d for d in self.list(channel_id=channel_id) if d.status == DocumentStatus.READY]

    # --- deletion ----------------------------------------------------------

    async def delete(self, document_id: str) -> None:
        """Remove a document, its chunks, and its file."""
        document = self.get(document_id)

        # Must precede the store wipe, or the task races us and re-adds chunks.
        await self._cancel_ingestion(document_id)

        removed = await self.store.delete_document(document_id)
        logger.info(
            "deleted %d chunks for %s",
            removed,
            document.filename,
            extra={"document_id": document_id},
        )

        if document.storage_path:
            Path(document.storage_path).unlink(missing_ok=True)

        document.status = DocumentStatus.DELETED
        document.updated_at = utcnow()
        self._documents.pop(document_id, None)

        from app.core import events as ev

        await self._emit(ev.document_deleted(document_id, document.channel_id))

    async def reindex(self, document_id: str) -> Document:
        """Drop and rebuild a document's chunks.

        Needed after changing ``EMBEDDING_PROVIDER`` or chunking settings, since
        vectors from different models are not comparable.
        """
        document = self.get(document_id)
        if not document.storage_path or not Path(document.storage_path).exists():
            raise DocumentNotFound(
                f"the stored file for {document.filename} is missing",
                user_message="The original file is no longer on disk; re-upload it.",
            )

        await self.store.delete_document(document_id)
        document.status = DocumentStatus.PROCESSING
        document.progress = 0.0
        document.stage = "reindexing"
        document.chunk_count = 0
        document.error = None
        document.updated_at = utcnow()

        self._spawn_ingestion(document, Path(document.storage_path))
        return document

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for document in self._documents.values():
            by_status[document.status.value] = by_status.get(document.status.value, 0) + 1
        return {
            "documents": len(self._documents),
            "by_status": by_status,
            "chunks": sum(d.chunk_count for d in self._documents.values()),
            "in_flight": len(self._tasks),
        }


_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    global _service
    if _service is None:
        _service = DocumentService()
    return _service


def set_document_service(service: DocumentService | None) -> None:
    global _service
    _service = service
