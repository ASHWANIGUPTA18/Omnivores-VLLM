"""The document pipeline end to end: upload -> parse -> chunk -> embed ->
index -> query -> cite -> delete.

This is the flow the product is judged on ("upload a PDF, ask a question about
it"), so it is tested as one continuous path rather than as isolated units.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    DocumentNotFound,
    FileTooLarge,
    UnsupportedFileType,
    ValidationError,
)
from app.core.models import DocumentStatus
from app.rag.ingestion import IngestionPipeline
from app.rag.retriever import Retriever
from app.services.document_service import DocumentService, sanitize_filename


async def stream_bytes(data: bytes, piece: int = 512):
    for start in range(0, len(data), piece):
        yield data[start : start + piece]


@pytest.fixture
def service(tmp_path, embeddings, vector_store) -> DocumentService:
    return DocumentService(
        pipeline=IngestionPipeline(embeddings=embeddings, store=vector_store),
        store=vector_store,
        upload_dir=tmp_path / "uploads",
    )


@pytest.fixture
def retriever_for(embeddings, vector_store) -> Retriever:
    return Retriever(embedding_provider=embeddings, vector_store=vector_store)


# --- filename safety -------------------------------------------------------


def test_path_traversal_is_stripped():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"


def test_leading_dots_are_stripped():
    assert sanitize_filename(".env") == "env"


def test_dangerous_characters_are_replaced():
    assert "/" not in sanitize_filename("a/b;c.pdf")
    assert ";" not in sanitize_filename("a/b;c.pdf")


def test_empty_filename_gets_a_fallback():
    assert sanitize_filename("") == "upload"


# --- validation ------------------------------------------------------------


async def test_unsupported_extension_is_rejected(service):
    with pytest.raises(UnsupportedFileType):
        await service.upload(filename="virus.exe", stream=stream_bytes(b"x"))


async def test_missing_extension_is_rejected(service):
    with pytest.raises(UnsupportedFileType):
        await service.upload(filename="README", stream=stream_bytes(b"x"))


async def test_declared_oversize_is_rejected(service):
    from app.core.config import settings

    with pytest.raises(FileTooLarge):
        await service.upload(
            filename="huge.txt",
            stream=stream_bytes(b"x"),
            declared_size=settings.max_upload_size_bytes + 1,
        )


async def test_oversize_stream_is_stopped_mid_upload(service, monkeypatch):
    """A client can lie about Content-Length; the actual bytes are what count."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_upload_size_mb", 0.001)
    with pytest.raises(FileTooLarge):
        await service.upload(
            filename="big.txt", stream=stream_bytes(b"x" * 50_000), declared_size=None
        )


async def test_empty_file_is_rejected(service):
    with pytest.raises(ValidationError):
        await service.upload(filename="empty.txt", stream=stream_bytes(b""))


async def test_a_rejected_upload_leaves_no_file_behind(service, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_upload_size_mb", 0.001)
    with pytest.raises(FileTooLarge):
        await service.upload(filename="big.txt", stream=stream_bytes(b"x" * 50_000))

    assert list(service.upload_dir.glob("*")) == []


# --- the full path ---------------------------------------------------------


async def test_upload_indexes_and_becomes_queryable(service, retriever_for, policy_text):
    document = await service.upload(
        filename="employee_handbook.md",
        stream=stream_bytes(policy_text.encode("utf-8")),
        channel_id="hr",
    )
    await service.wait_for_processing()

    assert document.status == DocumentStatus.READY
    assert document.chunk_count > 0
    assert document.progress == 1.0
    assert document.indexed_at is not None

    result = await retriever_for.retrieve("how many days of annual leave", channel_id="hr")
    assert not result.is_empty
    assert "20 days" in result.context


async def test_answers_carry_citations_with_a_page(service, retriever_for, policy_text):
    await service.upload(
        filename="employee_handbook.md",
        stream=stream_bytes(policy_text.encode("utf-8")),
        channel_id="hr",
    )
    await service.wait_for_processing()

    result = await retriever_for.retrieve("notice period for managers", channel_id="hr")
    citation = result.citations[0]

    assert citation.filename == "employee_handbook.md"
    assert citation.page is not None
    assert citation.snippet
    assert "employee_handbook.md" in citation.label


async def test_retrieval_finds_the_right_section(service, retriever_for, policy_text):
    await service.upload(
        filename="handbook.md",
        stream=stream_bytes(policy_text.encode("utf-8")),
        channel_id="hr",
    )
    await service.wait_for_processing()

    result = await retriever_for.retrieve("remote work approval", channel_id="hr")
    assert "remote" in result.context.lower()


async def test_progress_events_are_emitted_in_order(service, policy_text):
    seen: list[tuple[str, float]] = []

    async def hook(event):
        document = event["data"].get("document")
        if document:
            seen.append((document["status"], document["progress"]))

    service.set_event_hook(hook)
    await service.upload(
        filename="handbook.md", stream=stream_bytes(policy_text.encode("utf-8"))
    )
    await service.wait_for_processing()

    statuses = [s for s, _ in seen]
    assert statuses[0] == "uploading"
    assert statuses[-1] == "ready"
    progresses = [p for _, p in seen]
    assert progresses == sorted(progresses), "progress must never go backwards"


async def test_a_corrupt_document_fails_visibly(service):
    """A failed document stays in the list with a reason attached, rather than
    vanishing or sitting at 40% forever."""
    document = await service.upload(
        filename="broken.pdf", stream=stream_bytes(b"this is not a real PDF file")
    )
    await service.wait_for_processing()

    assert document.status == DocumentStatus.FAILED
    assert document.error
    assert document in service.list()


async def test_documents_are_scoped_by_channel(service, retriever_for):
    await service.upload(
        filename="hr_doc.md",
        stream=stream_bytes(b"# HR\n\nThe leave allowance is 20 days."),
        channel_id="hr",
    )
    await service.upload(
        filename="eng_doc.md",
        stream=stream_bytes(b"# Engineering\n\nThe leave allowance is 25 days."),
        channel_id="engineering",
    )
    await service.wait_for_processing()

    result = await retriever_for.retrieve("leave allowance", channel_id="hr")
    assert {c.filename for c in result.citations} == {"hr_doc.md"}


async def test_explicit_document_scope_overrides_the_channel(service, retriever_for):
    first = await service.upload(
        filename="a.md", stream=stream_bytes(b"# A\n\nProject alpha uses Postgres."), channel_id="general"
    )
    await service.upload(
        filename="b.md", stream=stream_bytes(b"# B\n\nProject beta uses MySQL."), channel_id="general"
    )
    await service.wait_for_processing()

    result = await retriever_for.retrieve(
        "which database", channel_id="general", document_ids=[first.id]
    )
    assert {c.document_id for c in result.citations} == {first.id}


async def test_deleting_a_document_removes_it_from_search(service, retriever_for, policy_text):
    document = await service.upload(
        filename="handbook.md",
        stream=stream_bytes(policy_text.encode("utf-8")),
        channel_id="hr",
    )
    await service.wait_for_processing()
    assert not (await retriever_for.retrieve("annual leave", channel_id="hr")).is_empty

    await service.delete(document.id)

    assert (await retriever_for.retrieve("annual leave", channel_id="hr")).is_empty
    with pytest.raises(DocumentNotFound):
        service.get(document.id)


async def test_deleting_removes_the_stored_file(service, policy_text):
    from pathlib import Path

    document = await service.upload(
        filename="handbook.md", stream=stream_bytes(policy_text.encode("utf-8"))
    )
    await service.wait_for_processing()
    path = Path(document.storage_path)
    assert path.exists()

    await service.delete(document.id)
    assert not path.exists()


async def test_reindexing_rebuilds_the_chunks(service, vector_store, policy_text):
    document = await service.upload(
        filename="handbook.md", stream=stream_bytes(policy_text.encode("utf-8"))
    )
    await service.wait_for_processing()
    original_count = await vector_store.count()

    await service.reindex(document.id)
    await service.wait_for_processing()

    assert await vector_store.count() == original_count
    assert document.status == DocumentStatus.READY


async def test_two_uploads_of_the_same_name_do_not_collide(service):
    first = await service.upload(filename="report.md", stream=stream_bytes(b"# One\n\nalpha"))
    second = await service.upload(filename="report.md", stream=stream_bytes(b"# Two\n\nbeta"))
    await service.wait_for_processing()

    assert first.id != second.id
    assert first.storage_path != second.storage_path
    assert len(service.list()) == 2


async def test_listing_filters_by_channel(service):
    await service.upload(filename="a.md", stream=stream_bytes(b"# A\n\nx"), channel_id="hr")
    await service.upload(filename="b.md", stream=stream_bytes(b"# B\n\ny"), channel_id="general")
    await service.wait_for_processing()

    assert [d.filename for d in service.list(channel_id="hr")] == ["a.md"]


async def test_stats_reflect_indexed_documents(service, policy_text):
    await service.upload(
        filename="handbook.md", stream=stream_bytes(policy_text.encode("utf-8"))
    )
    await service.wait_for_processing()

    stats = service.stats()
    assert stats["documents"] == 1
    assert stats["by_status"]["ready"] == 1
    assert stats["chunks"] > 0


async def test_an_empty_query_retrieves_nothing(retriever_for):
    assert (await retriever_for.retrieve("")).is_empty


async def test_retrieval_reports_timings(service, retriever_for, policy_text):
    await service.upload(
        filename="handbook.md",
        stream=stream_bytes(policy_text.encode("utf-8")),
        channel_id="hr",
    )
    await service.wait_for_processing()

    result = await retriever_for.retrieve("annual leave", channel_id="hr")
    metrics = result.metrics()
    assert metrics["candidates"] > 0
    assert metrics["total_ms"] >= 0
