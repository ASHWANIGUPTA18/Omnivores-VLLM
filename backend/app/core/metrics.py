"""Inference and agent metrics.

Captured from the first request rather than bolted on later, because the point
of moving to vLLM is throughput, and you cannot tune what you do not measure.
Every agent run produces an ``AgentMetrics`` record; every model call produces
an ``InferenceMetrics`` record nested inside it.

The registry keeps a bounded in-memory window so ``GET /health/metrics`` can
report live percentiles without a metrics backend.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceMetrics:
    """One model call."""

    model: str = ""
    provider: str = ""
    agent_id: str | None = None
    streaming: bool = False

    start_time: float = field(default_factory=time.perf_counter)
    #: Time to first token, seconds. None for non-streaming calls.
    time_to_first_token: float | None = None
    total_time: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    #: True when token counts came from the server rather than our estimator.
    tokens_reported: bool = False

    finish_reason: str | None = None
    error: str | None = None

    def mark_first_token(self) -> None:
        if self.time_to_first_token is None:
            self.time_to_first_token = time.perf_counter() - self.start_time

    def finish(self) -> InferenceMetrics:
        self.total_time = time.perf_counter() - self.start_time
        return self

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second of generation time.

        Measured after the first token so that queueing and prefill do not
        depress the number - this is a decode-speed metric.
        """
        if not self.output_tokens:
            return 0.0
        generation_time = self.total_time - (self.time_to_first_token or 0.0)
        if generation_time <= 0:
            return 0.0
        return self.output_tokens / generation_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "agent_id": self.agent_id,
            "streaming": self.streaming,
            "ttft_ms": round(self.time_to_first_token * 1000, 1)
            if self.time_to_first_token is not None
            else None,
            "total_ms": round(self.total_time * 1000, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_reported": self.tokens_reported,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


@dataclass
class ToolMetrics:
    name: str
    duration_ms: float = 0.0
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 1),
            "ok": self.ok,
            "error": self.error,
        }


@dataclass
class AgentMetrics:
    """One end-to-end agent run, from mention to final message."""

    agent_id: str
    channel_id: str
    message_id: str

    start_time: float = field(default_factory=time.perf_counter)
    total_time: float = 0.0

    rounds: int = 0
    inference_calls: list[InferenceMetrics] = field(default_factory=list)
    tool_calls: list[ToolMetrics] = field(default_factory=list)

    retrieved_chunks: int = 0
    retrieval_ms: float = 0.0
    context_tokens: int = 0

    error: str | None = None

    def finish(self) -> AgentMetrics:
        self.total_time = time.perf_counter() - self.start_time
        return self

    @property
    def input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.inference_calls)

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.inference_calls)

    @property
    def first_ttft(self) -> float | None:
        for m in self.inference_calls:
            if m.time_to_first_token is not None:
                return m.time_to_first_token
        return None

    def to_dict(self) -> dict[str, Any]:
        ttft = self.first_ttft
        return {
            "agent_id": self.agent_id,
            "channel_id": self.channel_id,
            "message_id": self.message_id,
            "total_ms": round(self.total_time * 1000, 1),
            "rounds": self.rounds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
            "tokens_per_second": round(
                max((m.tokens_per_second for m in self.inference_calls), default=0.0), 1
            ),
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "retrieved_chunks": self.retrieved_chunks,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "context_tokens": self.context_tokens,
            "error": self.error,
        }

    def summary_line(self) -> str:
        ttft = self.first_ttft
        parts = [
            f"agent={self.agent_id}",
            f"rounds={self.rounds}",
            f"in={self.input_tokens}",
            f"out={self.output_tokens}",
        ]
        if ttft is not None:
            parts.append(f"ttft={ttft * 1000:.0f}ms")
        tps = max((m.tokens_per_second for m in self.inference_calls), default=0.0)
        if tps:
            parts.append(f"{tps:.1f}tok/s")
        if self.tool_calls:
            parts.append("tools=" + ",".join(t.name for t in self.tool_calls))
        parts.append(f"total={self.total_time:.2f}s")
        return " ".join(parts)


class MetricsRegistry:
    """Bounded in-memory window of recent runs.

    Not a replacement for a real metrics backend - it exists so that the health
    endpoint and the benchmark scripts have something to read without one.
    """

    def __init__(self, window: int = 200) -> None:
        self._lock = threading.Lock()
        self._agent_runs: deque[dict[str, Any]] = deque(maxlen=window)
        self._inference_calls: deque[dict[str, Any]] = deque(maxlen=window * 2)
        self._counters: dict[str, int] = {}

    def record_agent_run(self, metrics: AgentMetrics) -> None:
        with self._lock:
            self._agent_runs.append(metrics.to_dict())
            self._counters["agent_runs"] = self._counters.get("agent_runs", 0) + 1
            if metrics.error:
                self._counters["agent_errors"] = self._counters.get("agent_errors", 0) + 1

    def record_inference(self, metrics: InferenceMetrics) -> None:
        with self._lock:
            self._inference_calls.append(metrics.to_dict())
            self._counters["inference_calls"] = (
                self._counters.get("inference_calls", 0) + 1
            )
            if metrics.error:
                self._counters["inference_errors"] = (
                    self._counters.get("inference_errors", 0) + 1
                )

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            runs = list(self._agent_runs)
            calls = list(self._inference_calls)
            counters = dict(self._counters)

        ttfts = [c["ttft_ms"] for c in calls if c.get("ttft_ms") is not None]
        totals = [c["total_ms"] for c in calls if c.get("total_ms") is not None]
        tps = [c["tokens_per_second"] for c in calls if c.get("tokens_per_second")]

        return {
            "counters": counters,
            "window": {"agent_runs": len(runs), "inference_calls": len(calls)},
            "inference": {
                "ttft_ms_p50": round(self._percentile(ttfts, 50), 1),
                "ttft_ms_p95": round(self._percentile(ttfts, 95), 1),
                "latency_ms_p50": round(self._percentile(totals, 50), 1),
                "latency_ms_p95": round(self._percentile(totals, 95), 1),
                "tokens_per_second_avg": round(sum(tps) / len(tps), 1) if tps else 0.0,
                "output_tokens_total": sum(c.get("output_tokens", 0) for c in calls),
                "input_tokens_total": sum(c.get("input_tokens", 0) for c in calls),
            },
            "recent_agent_runs": runs[-10:],
        }

    def reset(self) -> None:
        with self._lock:
            self._agent_runs.clear()
            self._inference_calls.clear()
            self._counters.clear()


metrics_registry = MetricsRegistry()
