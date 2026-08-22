#!/usr/bin/env python3
"""Benchmark the retrieval pipeline: ingestion throughput and query latency.

Runs entirely offline against synthetic documents, so it works without a GPU
and is the right tool for answering "did that chunking change make retrieval
slower?".

    python benchmarks/benchmark_rag.py --documents 20 --queries 50

    # compare embedding providers
    EMBEDDING_PROVIDER=mock  python benchmarks/benchmark_rag.py --output results/rag-mock.json
    EMBEDDING_PROVIDER=local python benchmarks/benchmark_rag.py --output results/rag-local.json

Reported separately because they have very different cost profiles:

* **Ingestion** is a one-off per document but dominated by embedding, which is
  where a local model on GPU beats a hashed provider on quality and loses on
  speed.
* **Query latency** happens on every message and is the one that shows up as
  perceived slowness. Its components (embed / search / rerank) are broken out
  so you can see which one grew.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Result,
    Sample,
    base_parser,
    describe_environment,
    print_environment,
    print_result,
    save_results,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.rag.chunker import Chunker  # noqa: E402
from app.rag.embeddings import create_embedding_provider  # noqa: E402
from app.rag.parser import ParsedDocument, ParsedPage  # noqa: E402
from app.rag.reranker import create_reranker  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.rag.store import MemoryVectorStore  # noqa: E402

TOPICS = [
    ("leave policy", "annual leave entitlement carry over accrual sabbatical"),
    ("notice period", "resignation termination notice garden leave handover"),
    ("remote work", "hybrid working from home office attendance approval"),
    ("expenses", "reimbursement travel per diem receipts approval limits"),
    ("performance", "review cycle objectives ratings promotion calibration"),
    ("security", "access control passwords incident reporting device encryption"),
    ("onboarding", "first week buddy equipment accounts training induction"),
    ("benefits", "insurance pension wellness allowance dental vision"),
]

QUERIES = [
    "how many days of annual leave am I entitled to",
    "what is the notice period for a manager",
    "can I work from home three days a week",
    "how do I claim travel expenses",
    "when does the performance review happen",
    "what do I do if I lose my laptop",
    "what happens in my first week",
    "is dental cover included",
]


def synthetic_document(index: int, sections: int) -> ParsedDocument:
    """Build a document with realistic structure and repeated vocabulary."""
    pages = []
    for section in range(sections):
        topic, vocabulary = TOPICS[(index + section) % len(TOPICS)]
        words = vocabulary.split()
        body = " ".join(
            f"The {topic} states that {words[n % len(words)]} applies under "
            f"condition {n} in document {index}."
            for n in range(60)
        )
        pages.append(
            ParsedPage(page=section + 1, text=f"# {topic.title()}\n\n{body}")
        )
    return ParsedDocument(pages=pages, page_count=sections)


async def benchmark_ingestion(
    embeddings, store, chunker, documents: int, sections: int
) -> tuple[Result, int]:
    samples: list[Sample] = []
    total_chunks = 0
    started = time.perf_counter()

    for index in range(documents):
        document_started = time.perf_counter()
        parsed = synthetic_document(index, sections)

        chunks = chunker.chunk_document(
            parsed,
            document_id=f"doc_{index}",
            filename=f"document_{index}.md",
            channel_id="general",
        )
        vectors = await embeddings.embed_documents([c.text for c in chunks])
        await store.add(chunks, vectors)

        total_chunks += len(chunks)
        samples.append(
            Sample(
                latency_s=time.perf_counter() - document_started,
                output_tokens=sum(c.token_estimate for c in chunks),
            )
        )

    return (
        Result(
            name="ingestion (per document)",
            samples=samples,
            wall_time_s=time.perf_counter() - started,
        ),
        total_chunks,
    )


async def benchmark_queries(retriever: Retriever, count: int) -> tuple[Result, dict]:
    samples: list[Sample] = []
    stages = {"embed_ms": 0.0, "search_ms": 0.0, "rerank_ms": 0.0}
    started = time.perf_counter()

    for n in range(count):
        query = QUERIES[n % len(QUERIES)]
        query_started = time.perf_counter()
        result = await retriever.retrieve(query, channel_id="general")
        samples.append(
            Sample(
                latency_s=time.perf_counter() - query_started,
                output_tokens=result.context_tokens,
                ok=not result.is_empty,
                error=None if not result.is_empty else "no results",
            )
        )
        stages["embed_ms"] += result.embed_ms
        stages["search_ms"] += result.search_ms
        stages["rerank_ms"] += result.rerank_ms

    for key in stages:
        stages[key] = round(stages[key] / max(1, count), 2)

    return (
        Result(name="retrieval (per query)", samples=samples,
               wall_time_s=time.perf_counter() - started),
        stages,
    )


async def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--documents", type=int, default=20, help="synthetic documents to index")
    parser.add_argument("--sections", type=int, default=5, help="sections per document")
    parser.add_argument("--queries", type=int, default=50, help="queries to run")
    args = parser.parse_args()

    print_environment()

    embeddings = create_embedding_provider()
    store = MemoryVectorStore(persist=False)
    reranker = create_reranker()
    chunker = Chunker()
    retriever = Retriever(
        embedding_provider=embeddings, vector_store=store, reranker=reranker
    )

    print(f"\n  indexing {args.documents} documents x {args.sections} sections...")
    ingestion, total_chunks = await benchmark_ingestion(
        embeddings, store, chunker, args.documents, args.sections
    )
    print_result(ingestion)
    print(f"    {total_chunks} chunks indexed, {await store.count()} in the store")

    if args.warmup:
        for _ in range(args.warmup):
            await retriever.retrieve(QUERIES[0], channel_id="general")

    print(f"\n  running {args.queries} queries...")
    queries, stages = await benchmark_queries(retriever, args.queries)
    print_result(queries)
    print(
        f"    stage breakdown  embed {stages['embed_ms']}ms  "
        f"search {stages['search_ms']}ms  rerank {stages['rerank_ms']}ms"
    )

    await embeddings.close()

    if args.output:
        save_results(
            [ingestion, queries],
            args.output,
            extra={
                "environment": describe_environment(),
                "corpus": {
                    "documents": args.documents,
                    "sections": args.sections,
                    "chunks": total_chunks,
                },
                "stage_breakdown_ms": stages,
            },
        )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
