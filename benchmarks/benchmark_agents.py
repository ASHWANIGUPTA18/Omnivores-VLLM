#!/usr/bin/env python3
"""Benchmark whole agent turns: mention in, answer out.

Where ``benchmark_inference.py`` measures the model, this measures the
product. It runs four scenarios that stress different parts of the stack:

    plain      one model call, no tools     - the floor
    rag        retrieval + one model call   - adds embedding and search
    tool       one tool round               - adds a second model call
    multi-tool two tool rounds              - adds a third

The interesting number is not any single scenario but the *gap* between them.
If `tool` costs far more than two `plain` runs, the overhead is in the
application - context rebuilding, tool dispatch - not the model.

    python benchmarks/benchmark_agents.py --requests 10
    LLM_PROVIDER=vllm python benchmarks/benchmark_agents.py --concurrency 1 4
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

from app.agents.registry import AgentRegistry  # noqa: E402
from app.agents.runtime import AgentRuntime  # noqa: E402
from app.core.models import Author, AuthorType, Message  # noqa: E402
from app.inference.client import create_inference_client  # noqa: E402
from app.rag.chunker import Chunker  # noqa: E402
from app.rag.embeddings import create_embedding_provider  # noqa: E402
from app.rag.parser import ParsedDocument, ParsedPage  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.rag.store import MemoryVectorStore  # noqa: E402

USER = Author(id="bench", type=AuthorType.USER, display_name="Benchmark")

CORPUS = """\
# Leave Policy

Full-time employees receive 20 days of paid annual leave per calendar year.
Leave must be requested at least two weeks in advance through the HR portal.
Up to five unused days may be carried into the following year.

# Notice Period

The standard notice period is 30 days below manager level and 60 days for
managers and above. Notice must be given in writing.

# Remote Work

Employees may work remotely up to three days per week with written manager
approval. Fully remote arrangements require director sign-off and a review
every six months.
"""

SCENARIOS = {
    "plain": {
        "agent": "researcher",
        "tools": [],
        "prompt": "Summarise what makes a multi-agent chat system hard to build.",
    },
    "rag": {
        "agent": "hr",
        "tools": [],
        "prompt": "How many days of annual leave do full-time employees receive?",
    },
    "tool": {
        "agent": "researcher",
        "tools": ["search_knowledge_base"],
        "prompt": "What does the policy document say about the notice period?",
    },
    "multi_tool": {
        "agent": "researcher",
        "tools": ["search_knowledge_base", "web_search"],
        "prompt": (
            "Search the policy document about remote work, then search the web "
            "for current best practice, and compare them."
        ),
    },
}


async def build_corpus(embeddings, store) -> None:
    parsed = ParsedDocument(
        pages=[ParsedPage(page=1, text=CORPUS)], page_count=1
    )
    chunks = Chunker().chunk_document(
        parsed,
        document_id="doc_benchmark",
        filename="employee_handbook.md",
        channel_id="general",
    )
    await store.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))


async def one_turn(runtime, definition, prompt: str) -> Sample:
    message = Message(channel_id="general", author=USER, content=prompt)
    started = time.perf_counter()

    result = await runtime.run(
        definition=definition,
        channel_id="general",
        message_id=f"bench_{int(started * 1e6)}",
        user_message=message,
        history=[],
    )

    latency = time.perf_counter() - started
    if not result.ok:
        return Sample(
            latency_s=latency,
            ok=False,
            error=result.error.code if result.error else "unknown",
        )

    metrics = result.metrics
    return Sample(
        latency_s=latency,
        ttft_s=metrics.first_ttft if metrics else None,
        input_tokens=metrics.input_tokens if metrics else 0,
        output_tokens=metrics.output_tokens if metrics else 0,
    )


async def run_scenario(runtime, definition, prompt, requests, concurrency) -> Result:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded() -> Sample:
        async with semaphore:
            return await one_turn(runtime, definition, prompt)

    started = time.perf_counter()
    samples = await asyncio.gather(*(guarded() for _ in range(requests)))
    return Result(
        name="",
        samples=list(samples),
        wall_time_s=time.perf_counter() - started,
        concurrency=concurrency,
    )


async def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        default="all",
        help="which scenario to run",
    )
    args = parser.parse_args()

    print_environment()

    client = create_inference_client()
    health = await client.health()
    print(f"\n    inference status     {health.status}  {health.detail or ''}")
    if health.status == "unavailable":
        print("\n    Cannot benchmark: the inference backend is unreachable.\n")
        await client.close()
        return 1

    embeddings = create_embedding_provider()
    store = MemoryVectorStore(persist=False)
    await build_corpus(embeddings, store)

    retriever = Retriever(embedding_provider=embeddings, vector_store=store)
    runtime = AgentRuntime(inference_client=client, retriever=retriever)
    registry = AgentRegistry()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results: list[Result] = []

    try:
        for name in names:
            scenario = SCENARIOS[name]
            base = registry.get(scenario["agent"])
            # Copy the definition so the tool set can be varied per scenario
            # without mutating the registry.
            from dataclasses import replace

            definition = replace(base, tools=list(scenario["tools"]))

            if args.warmup:
                for _ in range(args.warmup):
                    await one_turn(runtime, definition, scenario["prompt"])

            for concurrency in args.concurrency:
                print(
                    f"\n  running: scenario={name} concurrency={concurrency} "
                    f"requests={args.requests}"
                )
                result = await run_scenario(
                    runtime, definition, scenario["prompt"], args.requests, concurrency
                )
                result.name = f"{name} (@{scenario['agent']})"
                results.append(result)
                print_result(result)
    finally:
        await client.close()
        await embeddings.close()

    print("\n" + "=" * 62)
    print("  Scenario comparison")
    print("=" * 62)
    print(f"    {'scenario':<28} {'p50 lat':>9} {'p95 lat':>9} {'out tok':>9}")
    for result in results:
        summary = result.summary()
        if not summary.get("latency_s"):
            print(f"    {summary['name']:<28} {'FAILED':>9}")
            continue
        print(
            f"    {summary['name']:<28} {summary['latency_s']['p50']:>8}s "
            f"{summary['latency_s']['p95']:>8}s "
            f"{summary['tokens']['output_total']:>9}"
        )

    if args.output:
        save_results(results, args.output, extra={"environment": describe_environment()})

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
