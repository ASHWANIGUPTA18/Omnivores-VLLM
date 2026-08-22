"""Shared benchmark machinery.

Two things here that make the numbers trustworthy:

**Percentiles, not just averages.** An average latency hides the tail, and the
tail is what users experience as "it's slow sometimes".

**A warmup phase.** The first request against a freshly started vLLM server
pays for CUDA graph capture and a cold KV cache. Including it makes every
comparison meaningless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Make `app` importable when running from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


@dataclass
class Sample:
    latency_s: float
    ttft_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True
    error: str | None = None

    @property
    def tokens_per_second(self) -> float:
        generation = self.latency_s - (self.ttft_s or 0.0)
        if generation <= 0 or not self.output_tokens:
            return 0.0
        return self.output_tokens / generation


@dataclass
class Result:
    name: str
    samples: list[Sample] = field(default_factory=list)
    wall_time_s: float = 0.0
    concurrency: int = 1

    @property
    def ok_samples(self) -> list[Sample]:
        return [s for s in self.samples if s.ok]

    @property
    def error_count(self) -> int:
        return len(self.samples) - len(self.ok_samples)

    def summary(self) -> dict:
        good = self.ok_samples
        if not good:
            return {
                "name": self.name,
                "concurrency": self.concurrency,
                "requests": len(self.samples),
                "errors": self.error_count,
                "error_sample": next((s.error for s in self.samples if s.error), None),
            }

        latencies = [s.latency_s for s in good]
        ttfts = [s.ttft_s for s in good if s.ttft_s is not None]
        rates = [s.tokens_per_second for s in good if s.tokens_per_second]

        return {
            "name": self.name,
            "concurrency": self.concurrency,
            "requests": len(self.samples),
            "errors": self.error_count,
            "wall_time_s": round(self.wall_time_s, 2),
            "requests_per_second": round(len(good) / self.wall_time_s, 2)
            if self.wall_time_s
            else 0.0,
            "latency_s": {
                "mean": round(statistics.mean(latencies), 3),
                "p50": round(percentile(latencies, 50), 3),
                "p95": round(percentile(latencies, 95), 3),
                "max": round(max(latencies), 3),
            },
            "ttft_ms": {
                "mean": round(statistics.mean(ttfts) * 1000, 1),
                "p50": round(percentile(ttfts, 50) * 1000, 1),
                "p95": round(percentile(ttfts, 95) * 1000, 1),
            }
            if ttfts
            else None,
            "tokens_per_second": {
                "mean": round(statistics.mean(rates), 1),
                "p50": round(percentile(rates, 50), 1),
            }
            if rates
            else None,
            "tokens": {
                "input_total": sum(s.input_tokens for s in good),
                "output_total": sum(s.output_tokens for s in good),
                "output_per_second": round(
                    sum(s.output_tokens for s in good) / self.wall_time_s, 1
                )
                if self.wall_time_s
                else 0.0,
            },
        }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def print_result(result: Result) -> None:
    summary = result.summary()
    print(f"\n  {summary['name']}  (concurrency {summary['concurrency']})")
    print("  " + "-" * 58)

    if not summary.get("latency_s"):
        print(f"    all {summary['requests']} request(s) failed")
        if summary.get("error_sample"):
            print(f"    {summary['error_sample']}")
        return

    latency = summary["latency_s"]
    print(f"    requests        {summary['requests']}  ({summary['errors']} errors)")
    print(f"    wall time       {summary['wall_time_s']}s")
    print(f"    throughput      {summary['requests_per_second']} req/s")
    print(
        f"    latency         mean {latency['mean']}s   "
        f"p50 {latency['p50']}s   p95 {latency['p95']}s   max {latency['max']}s"
    )
    if summary.get("ttft_ms"):
        ttft = summary["ttft_ms"]
        print(
            f"    time to first   mean {ttft['mean']}ms   "
            f"p50 {ttft['p50']}ms   p95 {ttft['p95']}ms"
        )
    if summary.get("tokens_per_second"):
        rate = summary["tokens_per_second"]
        print(f"    decode speed    mean {rate['mean']} tok/s   p50 {rate['p50']} tok/s")
    tokens = summary["tokens"]
    print(
        f"    tokens          in {tokens['input_total']}  out {tokens['output_total']}  "
        f"({tokens['output_per_second']} out-tok/s aggregate)"
    )


def save_results(results: list[Result], path: Path, extra: dict | None = None) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extra or {}),
        "results": [r.summary() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  results written to {path}")


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--requests", type=int, default=10, help="requests per concurrency level"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1],
        help="concurrency levels to test, e.g. --concurrency 1 4 8 16",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="untimed requests first; the first call to a cold server is not "
        "representative",
    )
    parser.add_argument("--output", type=Path, help="write JSON results here")
    return parser


def describe_environment() -> dict:
    from app.core.config import settings

    return {
        "provider": settings.llm_provider,
        "model": settings.resolved_model_name,
        "embedding_provider": settings.embedding_provider,
        "vector_store": settings.vector_store,
        "reranker": settings.reranker,
        "max_output_tokens": settings.max_output_tokens,
        "temperature": settings.temperature,
    }


def print_environment() -> None:
    environment = describe_environment()
    print("\n" + "=" * 62)
    print("  Omnivores benchmark")
    print("=" * 62)
    for key, value in environment.items():
        print(f"    {key:<20} {value}")
    if environment["provider"] == "mock":
        print(
            "\n    NOTE: LLM_PROVIDER=mock. These numbers measure the "
            "application\n          overhead only, not a real model. Set "
            "LLM_PROVIDER=vllm to\n          benchmark inference."
        )


def asdict_sample(sample: Sample) -> dict:
    return asdict(sample)
