#!/usr/bin/env python3
"""Benchmark raw inference: time to first token, decode speed, concurrency.

Talks to the configured ``InferenceClient`` directly, bypassing agents, tools
and RAG, so the numbers describe the model server and nothing else.

    # against vLLM
    LLM_PROVIDER=vllm python benchmarks/benchmark_inference.py \\
        --concurrency 1 4 8 16 --requests 20

    # compare two models: run once per model, save, then diff the JSON
    python benchmarks/benchmark_inference.py --output results/qwen3-8b.json

What to read:

* **TTFT** is dominated by prefill, so it grows with prompt length. It is what
  makes the UI feel responsive.
* **Decode speed** (tok/s per request) is what makes a long answer bearable.
  It falls as concurrency rises.
* **Aggregate output tok/s** is what continuous batching improves. It should
  rise with concurrency well past the point where per-request speed drops -
  that gap is the whole argument for vLLM over one-request-at-a-time serving.
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

from app.core.tokens import estimate_tokens  # noqa: E402
from app.inference.base import ChatMessage  # noqa: E402
from app.inference.client import create_inference_client  # noqa: E402

PROMPTS = {
    "short": "In two sentences, what is continuous batching?",
    "medium": (
        "Explain how a retrieval-augmented generation pipeline works, covering "
        "chunking, embedding, vector search, reranking and citation. Keep it "
        "under 300 words."
    ),
    "long": (
        "Write a technical design note for a multi-agent chat system where "
        "agents are first-class participants addressed by @mention. Cover the "
        "agent loop, tool calling, context budgeting, streaming, and how "
        "document retrieval is scoped per channel. Include the failure modes "
        "you would guard against."
    ),
}


async def one_request(client, prompt: str, max_tokens: int) -> Sample:
    messages = [
        ChatMessage.system("You are a concise technical assistant."),
        ChatMessage.user(prompt),
    ]
    started = time.perf_counter()
    ttft: float | None = None
    text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0

    try:
        async for chunk in client.stream(messages, max_tokens=max_tokens):
            if chunk.content:
                if ttft is None:
                    ttft = time.perf_counter() - started
                text_parts.append(chunk.content)
            if chunk.usage:
                input_tokens = chunk.usage.input_tokens
                output_tokens = chunk.usage.output_tokens
    except Exception as exc:  # noqa: BLE001 - a failed request is a data point
        return Sample(
            latency_s=time.perf_counter() - started,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    latency = time.perf_counter() - started
    text = "".join(text_parts)

    # Fall back to estimates when the server does not report usage, so a
    # provider without usage reporting still yields comparable numbers.
    if not output_tokens:
        output_tokens = estimate_tokens(text)
    if not input_tokens:
        input_tokens = await client.count_tokens(messages)

    return Sample(
        latency_s=latency,
        ttft_s=ttft,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def run_level(client, prompt: str, requests: int, concurrency: int, max_tokens: int) -> Result:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded() -> Sample:
        async with semaphore:
            return await one_request(client, prompt, max_tokens)

    started = time.perf_counter()
    samples = await asyncio.gather(*(guarded() for _ in range(requests)))
    wall = time.perf_counter() - started

    return Result(name="", samples=list(samples), wall_time_s=wall, concurrency=concurrency)


async def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--prompt",
        choices=[*PROMPTS, "all"],
        default="medium",
        help="prompt size to use",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    print_environment()

    client = create_inference_client()

    health = await client.health()
    print(f"\n    inference status     {health.status}  {health.detail or ''}")
    if health.status == "unavailable":
        print("\n    Cannot benchmark: the inference backend is unreachable.\n")
        await client.close()
        return 1

    prompt_names = list(PROMPTS) if args.prompt == "all" else [args.prompt]
    results: list[Result] = []

    try:
        if args.warmup:
            print(f"\n  warming up ({args.warmup} untimed requests)...")
            await asyncio.gather(
                *(
                    one_request(client, PROMPTS[prompt_names[0]], 64)
                    for _ in range(args.warmup)
                )
            )

        for prompt_name in prompt_names:
            prompt = PROMPTS[prompt_name]
            for concurrency in args.concurrency:
                print(
                    f"\n  running: prompt={prompt_name} concurrency={concurrency} "
                    f"requests={args.requests}"
                )
                result = await run_level(
                    client, prompt, args.requests, concurrency, args.max_tokens
                )
                result.name = f"{prompt_name} prompt"
                results.append(result)
                print_result(result)
    finally:
        await client.close()

    if len(results) > 1:
        print("\n" + "=" * 62)
        print("  Scaling")
        print("=" * 62)
        print(f"    {'level':<24} {'req/s':>8} {'p95 lat':>10} {'agg tok/s':>11}")
        for result in results:
            summary = result.summary()
            if not summary.get("latency_s"):
                continue
            label = f"{summary['name']} c={summary['concurrency']}"
            print(
                f"    {label:<24} {summary['requests_per_second']:>8} "
                f"{summary['latency_s']['p95']:>9}s "
                f"{summary['tokens']['output_per_second']:>11}"
            )

    if args.output:
        save_results(results, args.output, extra={"environment": describe_environment()})

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
