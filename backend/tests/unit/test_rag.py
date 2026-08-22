"""Embeddings, the vector store, reranking and context construction.

The hashed embedding provider is a real lexical model, so these tests assert on
actual retrieval quality rather than just plumbing.
"""

from __future__ import annotations

import pytest

from app.core.models import DocumentChunk, RetrievedChunk
from app.core.tokens import estimate_tokens
from app.rag.embeddings import HashingEmbeddingProvider, cosine_similarity, tokenize
from app.rag.parser import MarkdownParser, TextParser, normalize_whitespace
from app.rag.reranker import LexicalReranker, NoopReranker
from app.rag.retriever import build_context, build_scope_filter
from app.rag.store import MemoryVectorStore, matches_filter


# --- embeddings ------------------------------------------------------------


async def test_embeddings_are_deterministic(embeddings):
    first = await embeddings.embed_query("annual leave policy")
    second = await embeddings.embed_query("annual leave policy")
    assert first == second


async def test_embeddings_have_the_configured_dimensions(embeddings):
    assert len(await embeddings.embed_query("text")) == embeddings.dimensions


async def test_embeddings_are_normalized(embeddings):
    vector = await embeddings.embed_query("normalization check")
    magnitude = sum(v * v for v in vector) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-6)


async def test_similar_text_scores_higher_than_unrelated_text(embeddings):
    query = await embeddings.embed_query("annual leave entitlement days")
    related = (await embeddings.embed_documents(
        ["Employees receive 20 days of paid annual leave entitlement."]
    ))[0]
    unrelated = (await embeddings.embed_documents(
        ["The build pipeline compiles the Rust binary and uploads artifacts."]
    ))[0]

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


async def test_empty_text_gives_a_zero_vector(embeddings):
    assert set(await embeddings.embed_query("")) == {0.0}


def test_tokenizer_drops_stopwords_and_punctuation():
    assert tokenize("The policy is, in fact, about LEAVE!") == ["policy", "fact", "leave"]


async def test_embedding_health(embeddings):
    status, _ = await embeddings.health()
    assert status == "healthy"


# --- vector store ----------------------------------------------------------


def chunk(text: str, *, doc: str = "doc_1", index: int = 0, page: int | None = 1,
          channel: str | None = "hr") -> DocumentChunk:
    return DocumentChunk(
        document_id=doc,
        filename=f"{doc}.md",
        text=text,
        chunk_index=index,
        page=page,
        channel_id=channel,
        token_estimate=estimate_tokens(text),
    )


async def test_store_round_trip(vector_store, embeddings):
    chunks = [chunk("paid annual leave is 20 days", index=0)]
    vectors = await embeddings.embed_documents([c.text for c in chunks])
    await vector_store.add(chunks, vectors)

    assert await vector_store.count() == 1
    results = await vector_store.search(
        await embeddings.embed_query("annual leave"), top_k=5
    )
    assert results[0].document_id == "doc_1"
    assert results[0].page == 1


async def test_store_scores_are_within_unit_range(vector_store, embeddings):
    chunks = [chunk("alpha beta gamma"), chunk("delta epsilon", index=1)]
    await vector_store.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))

    for result in await vector_store.search(await embeddings.embed_query("alpha"), top_k=5):
        assert 0.0 <= result.score <= 1.0


async def test_store_orders_by_relevance(vector_store, embeddings):
    chunks = [
        chunk("the notice period for managers is 60 days", index=0),
        chunk("the office kitchen is restocked on Mondays", index=1),
    ]
    await vector_store.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))

    results = await vector_store.search(
        await embeddings.embed_query("notice period managers"), top_k=2
    )
    assert "notice period" in results[0].text


async def test_store_filters_by_document(vector_store, embeddings):
    chunks = [chunk("shared topic text", doc="doc_a"), chunk("shared topic text", doc="doc_b")]
    await vector_store.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))

    results = await vector_store.search(
        await embeddings.embed_query("shared topic"),
        top_k=10,
        where={"document_id": "doc_a"},
    )
    assert {r.document_id for r in results} == {"doc_a"}


async def test_store_filters_by_document_list(vector_store, embeddings):
    chunks = [chunk("text", doc=f"doc_{n}") for n in "abc"]
    await vector_store.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))

    results = await vector_store.search(
        await embeddings.embed_query("text"),
        top_k=10,
        where={"document_id": {"$in": ["doc_a", "doc_c"]}},
    )
    assert {r.document_id for r in results} == {"doc_a", "doc_c"}


async def test_deleting_a_document_removes_its_chunks(vector_store, embeddings):
    chunks = [chunk("keep this", doc="doc_a"), chunk("remove this", doc="doc_b")]
    await vector_store.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))

    removed = await vector_store.delete_document("doc_b")
    assert removed == 1
    assert await vector_store.count() == 1
    assert await vector_store.document_ids() == {"doc_a"}


async def test_readding_a_chunk_id_replaces_it(vector_store, embeddings):
    original = chunk("first version")
    await vector_store.add([original], await embeddings.embed_documents([original.text]))

    original.text = "second version"
    await vector_store.add([original], await embeddings.embed_documents([original.text]))

    assert await vector_store.count() == 1


async def test_mismatched_vector_count_is_rejected(vector_store):
    from app.core.exceptions import VectorStoreError

    with pytest.raises(VectorStoreError):
        await vector_store.add([chunk("a"), chunk("b", index=1)], [[0.1] * 128])


async def test_search_on_an_empty_store_returns_nothing(vector_store, embeddings):
    assert await vector_store.search(await embeddings.embed_query("anything"), top_k=5) == []


async def test_store_persists_across_instances(tmp_path, embeddings):
    path = tmp_path / "persist"
    first = MemoryVectorStore(path=path)
    chunks = [chunk("persisted content")]
    await first.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))

    second = MemoryVectorStore(path=path)
    assert await second.count() == 1


def test_filter_matching():
    metadata = {"document_id": "d1", "channel_id": "hr", "page": 3}
    assert matches_filter(metadata, None)
    assert matches_filter(metadata, {"channel_id": "hr"})
    assert not matches_filter(metadata, {"channel_id": "general"})
    assert matches_filter(metadata, {"document_id": {"$in": ["d1", "d2"]}})
    assert not matches_filter(metadata, {"document_id": {"$in": ["d9"]}})


def test_scope_filter_precedence():
    """Explicit documents win over the channel - "ask this document" must not
    silently widen to the whole channel."""
    assert build_scope_filter(document_ids=["d1"], channel_id="hr") == {"document_id": "d1"}
    assert build_scope_filter(document_ids=["d1", "d2"]) == {
        "document_id": {"$in": ["d1", "d2"]}
    }
    assert build_scope_filter(channel_id="hr") == {"channel_id": "hr"}
    assert build_scope_filter() is None


# --- reranking -------------------------------------------------------------


def retrieved(text: str, score: float, *, chunk_id: str = "c") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d",
        filename="f.md",
        text=text,
        chunk_index=0,
        page=1,
        score=score,
    )


async def test_lexical_reranker_promotes_an_exact_term_match():
    """The case vector search handles worst: a verbatim term the embedding
    blurred away."""
    chunks = [
        retrieved("general discussion of company benefits and perks", 0.61, chunk_id="a"),
        retrieved("the notice period is 60 days for managers", 0.60, chunk_id="b"),
    ]
    ranked = await LexicalReranker().rerank("notice period", chunks, top_k=2)
    assert ranked[0].chunk_id == "b"


async def test_reranker_truncates_to_top_k():
    chunks = [retrieved(f"text {i}", 0.5, chunk_id=str(i)) for i in range(10)]
    assert len(await LexicalReranker().rerank("text", chunks, top_k=3)) == 3


async def test_reranker_sets_a_bounded_score():
    chunks = [retrieved("annual leave policy", 0.8)]
    ranked = await LexicalReranker().rerank("leave", chunks, top_k=1)
    assert ranked[0].rerank_score is not None
    assert 0.0 <= ranked[0].final_score <= 1.0


async def test_noop_reranker_keeps_vector_order():
    chunks = [retrieved("a", 0.2, chunk_id="a"), retrieved("b", 0.9, chunk_id="b")]
    ranked = await NoopReranker().rerank("anything", chunks, top_k=2)
    assert [c.chunk_id for c in ranked] == ["b", "a"]


async def test_reranking_an_empty_list_is_safe():
    assert await LexicalReranker().rerank("q", [], top_k=5) == []


# --- context construction --------------------------------------------------


def test_context_numbers_sources_for_citation():
    chunks = [
        retrieved("Employees get 20 days of leave.", 0.9, chunk_id="a"),
        retrieved("Notice period is 30 days.", 0.8, chunk_id="b"),
    ]
    context, tokens = build_context(chunks, token_budget=2000)

    assert "[1]" in context and "[2]" in context
    assert "f.md, page 1" in context
    assert tokens > 0


def test_context_respects_the_token_budget():
    chunks = [retrieved("word " * 500, 0.9, chunk_id=str(i)) for i in range(5)]
    _, tokens = build_context(chunks, token_budget=200)
    assert tokens <= 200


def test_context_trims_the_citation_list_to_what_it_included():
    """Citation numbers must line up with the markers the model actually saw."""
    chunks = [retrieved("word " * 400, 0.9, chunk_id=str(i)) for i in range(5)]
    context, _ = build_context(chunks, token_budget=250)

    assert len(chunks) < 5
    assert f"[{len(chunks)}]" in context
    assert f"[{len(chunks) + 1}]" not in context


def test_context_is_empty_when_there_is_nothing_to_include():
    assert build_context([], token_budget=1000) == ("", 0)
    assert build_context([retrieved("text", 0.5)], token_budget=0) == ("", 0)


def test_citations_carry_page_and_snippet():
    citation = retrieved("A long passage about the leave policy.", 0.77).to_citation()
    assert citation.page == 1
    assert citation.label == "f.md - page 1"
    assert "leave policy" in citation.snippet


# --- parsers ---------------------------------------------------------------


def test_whitespace_normalization_rejoins_hyphenated_line_breaks():
    assert normalize_whitespace("depart-\nment") == "department"


def test_whitespace_normalization_keeps_paragraph_breaks():
    assert normalize_whitespace("one\n\n\n\ntwo") == "one\n\ntwo"


def test_text_parser_reads_a_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("plain content", encoding="utf-8")
    parsed = TextParser().parse(path)
    assert parsed.text == "plain content"
    assert parsed.pages[0].page is None


def test_markdown_parser_splits_on_headings(policy_file):
    parsed = MarkdownParser().parse(policy_file)
    assert parsed.page_count == 3
    assert "Leave Policy" in parsed.pages[0].text
    assert "Notice Period" in parsed.pages[1].text


def test_markdown_without_headings_is_one_section(tmp_path):
    path = tmp_path / "flat.md"
    path.write_text("just a paragraph of text", encoding="utf-8")
    parsed = MarkdownParser().parse(path)
    assert len(parsed.pages) == 1


def test_parser_registry_rejects_unsupported_types(tmp_path):
    from app.core.exceptions import UnsupportedFileType
    from app.rag.parser import parser_registry

    with pytest.raises(UnsupportedFileType):
        parser_registry.for_extension(".xlsx")
