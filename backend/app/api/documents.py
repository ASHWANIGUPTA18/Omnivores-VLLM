"""Document upload and management.

Uploads stream to disk in chunks rather than being read into memory. A 50 MB
limit read with ``await file.read()`` is a 50 MB allocation per concurrent
upload; streaming keeps it constant and lets the size check abort early.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.api.deps import require_api_key
from app.core.config import settings
from app.core.logging import get_logger
from app.services.document_service import get_document_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"], dependencies=[Depends(require_api_key)])

#: Read size for streaming an upload to disk.
_CHUNK_BYTES = 1024 * 1024


async def _stream(upload: UploadFile) -> AsyncIterator[bytes]:
    while True:
        piece = await upload.read(_CHUNK_BYTES)
        if not piece:
            break
        yield piece


@router.post("", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    channel_id: str | None = Form(default=None),
) -> dict[str, Any]:
    """Accept a file and start indexing it.

    Returns 202: the document is stored but not yet searchable. Progress
    arrives as ``document.*`` events on the WebSocket, and the document's
    ``status`` field reaches ``ready`` when it can be queried.
    """
    service = get_document_service()

    declared_size = None
    if file.size is not None:
        declared_size = int(file.size)

    document = await service.upload(
        filename=file.filename or "upload",
        stream=_stream(file),
        content_type=file.content_type or "application/octet-stream",
        channel_id=channel_id,
        declared_size=declared_size,
    )
    return {"document": document.public(), "status": "accepted"}


@router.get("")
async def list_documents(
    channel_id: str | None = Query(default=None),
) -> dict[str, Any]:
    service = get_document_service()
    documents = service.list(channel_id=channel_id)
    return {
        "documents": [d.public() for d in documents],
        "stats": service.stats(),
    }


@router.get("/{document_id}")
async def get_document(document_id: str) -> dict[str, Any]:
    return {"document": get_document_service().get(document_id).public()}


@router.delete("/{document_id}", status_code=200)
async def delete_document(document_id: str) -> dict[str, Any]:
    service = get_document_service()
    document = service.get(document_id)
    await service.delete(document_id)
    return {"deleted": document_id, "filename": document.filename}


@router.post("/{document_id}/reindex", status_code=202)
async def reindex_document(document_id: str) -> dict[str, Any]:
    """Rebuild a document's chunks.

    Needed after changing ``EMBEDDING_PROVIDER``, ``CHUNK_SIZE_TOKENS`` or
    ``CHUNK_OVERLAP_TOKENS`` - vectors produced by different models or
    settings are not comparable, and mixing them degrades every search.
    """
    document = await get_document_service().reindex(document_id)
    return {"document": document.public(), "status": "reindexing"}


@router.get("/limits/info")
async def upload_limits() -> dict[str, Any]:
    return {
        "max_upload_size_mb": settings.max_upload_size_mb,
        "allowed_extensions": sorted(settings.allowed_extension_set),
    }
