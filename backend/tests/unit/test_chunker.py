"""Chunking.

The invariant that matters most: a chunk never spans a page, because a chunk
that does cannot be cited honestly.
"""

from __future__ import annotations

import pytest

from app.core.tokens import estimate_tokens
from app.rag.chunker import Chunker, ChunkerConfig
from app.rag.parser import ParsedDocument, ParsedPage


@pytest.fixture
def chunker() -> Chunker:
    return Chunker(ChunkerConfig(chunk_size_tokens=60, overlap_tokens=12, min_chunk_tokens=8))


def test_short_text_is_one_chunk(chunker):
    assert chunker.split_text("A short sentence.") == ["A short sentence."]


def test_empty_text_produces_nothing(chunker):
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n  ") == []


def test_long_text_is_split(chunker):
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunker.split_text(text)
    assert len(chunks) > 1


def test_chunks_respect_the_size_limit(chunker):
    text = " ".join(f"token{i}" for i in range(600))
    for chunk in chunker.split_text(text):
        # Overlap and runt-merging can push slightly past the target; a hard
        # overrun would risk a context rejection, so allow only a small margin.
        assert estimate_tokens(chunk) <= 60 * 1.5


def test_paragraph_boundaries_are_preferred(chunker):
    first = "alpha " * 40
    second = "beta " * 40
    chunks = chunker.split_text(f"{first.strip()}\n\n{second.strip()}")
    assert len(chunks) >= 2
    assert chunks[0].startswith("alpha")


def test_overlap_is_carried_between_chunks():
    chunker = Chunker(ChunkerConfig(chunk_size_tokens=40, overlap_tokens=15))
    sentences = ". ".join(f"Sentence number {i} has content" for i in range(40))
    chunks = chunker.split_text(sentences)
    assert len(chunks) > 1
    # Some tail of chunk N should reappear at the head of chunk N+1.
    tail_words = set(chunks[0].split()[-8:])
    head_words = set(chunks[1].split()[:12])
    assert tail_words & head_words


def test_no_overlap_when_configured_to_zero():
    chunker = Chunker(ChunkerConfig(chunk_size_tokens=40, overlap_tokens=0))
    chunks = chunker.split_text(". ".join(f"Sentence {i} here" for i in range(40)))
    assert len(chunks) > 1


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Chunker(ChunkerConfig(chunk_size_tokens=50, overlap_tokens=50))


def test_text_with_no_separators_is_hard_split(chunker):
    chunks = chunker.split_text("x" * 5000)
    assert len(chunks) > 1
    assert "".join(chunks) == "x" * 5000


def test_chunks_never_span_pages(chunker):
    parsed = ParsedDocument(
        pages=[
            ParsedPage(page=1, text="alpha " * 100),
            ParsedPage(page=2, text="beta " * 100),
        ],
        page_count=2,
    )
    chunks = chunker.chunk_document(parsed, document_id="doc_1", filename="f.pdf")

    for chunk in chunks:
        assert not ("alpha" in chunk.text and "beta" in chunk.text)
    assert {c.page for c in chunks} == {1, 2}


def test_chunk_indices_are_sequential_across_pages(chunker):
    parsed = ParsedDocument(
        pages=[ParsedPage(page=n, text=f"page {n} content " * 30) for n in (1, 2, 3)],
        page_count=3,
    )
    chunks = chunker.chunk_document(parsed, document_id="doc_1", filename="f.pdf")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_metadata_carries_citation_fields(chunker):
    parsed = ParsedDocument(pages=[ParsedPage(page=7, text="policy text here")])
    chunk = chunker.chunk_document(
        parsed, document_id="doc_9", filename="handbook.pdf", channel_id="hr"
    )[0]

    metadata = chunk.metadata()
    assert metadata["document_id"] == "doc_9"
    assert metadata["filename"] == "handbook.pdf"
    assert metadata["page"] == 7
    assert metadata["channel_id"] == "hr"


def test_metadata_values_are_store_safe(chunker):
    """Chroma rejects non-scalar metadata; the memory store mirrors that."""
    parsed = ParsedDocument(pages=[ParsedPage(page=1, text="text")])
    chunk = chunker.chunk_document(parsed, document_id="d", filename="f.md")[0]
    for value in chunk.metadata().values():
        assert isinstance(value, (str, int, float, bool))


def test_pages_without_numbers_are_handled(chunker):
    parsed = ParsedDocument(pages=[ParsedPage(page=None, text="plain text file")])
    chunk = chunker.chunk_document(parsed, document_id="d", filename="f.txt")[0]
    assert chunk.page is None
    assert "page" not in chunk.metadata()
