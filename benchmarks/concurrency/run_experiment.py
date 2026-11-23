#!/usr/bin/env python3
"""Sequential vs parallel multi-agent dispatch - experiment runner.

Executes the condition matrix against the live backend and writes one
immutable record per run. Nothing is aggregated here: analysis reads the raw
JSONL afterwards, so a summary can never be mistaken for the measurement.

    python run_experiment.py --setup          # ingest corpus, install agents
    python run_experiment.py --dry-run        # verify plumbing, no real timing
    python run_experiment.py --sanity         # the 20-run sanity experiment

Pairing: sequential and parallel runs are matched on
``(agent_count, prompt_id, repetition)``. Both arms of a pair see the same
prompt, the same agents and the same pinned history, which is what makes a
paired statistical test valid rather than merely convenient.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets pyyaml")
try:
    import yaml
except ImportError:
    sys.exit("pip install websockets pyyaml")

from instrument import GpuSampler, capture_environment, wait_until_drained

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
AGENTS_D = HERE / "agents.d"
RESULTS = HERE / "results"


# --- plan -------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    agent_count: int
    dispatch_mode: str
    users: int
    prompt_id: str
    repetition: int

    @property
    def pair_key(self) -> str:
        """Identifies the matched sequential/parallel observation."""
        return f"a{self.agent_count}:u{self.users}:{self.prompt_id}:r{self.repetition}"

    @property
    def condition_id(self) -> str:
        return f"a{self.agent_count}:u{self.users}:{self.dispatch_mode}"


def build_matrix(spec: dict[str, Any], prompts: list[dict[str, Any]]) -> list[Condition]:
    """Full crossing of the spec. Prompts cycle so a pair always shares one."""
    prompt_ids = [p["id"] for p in prompts]
    out: list[Condition] = []
    for agent_count in spec["agent_count"]:
        for users in spec["users"]:
            for repetition in range(spec["repetitions"]):
                # Deterministic: repetition r always uses the same prompt, so
                # the sequential and parallel members of a pair match.
                prompt_id = prompt_ids[repetition % len(prompt_ids)]
                for mode in spec["dispatch_mode"]:
                    out.append(
                        Condition(agent_count, mode, users, prompt_id, repetition)
                    )
    return out


# --- one measured run -------------------------------------------------------


@dataclass
class AgentTiming:
    agent_id: str
    started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None
    chars: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    content: str = ""
    failed: str | None = None


class RunObserver:
    """Collects per-agent timing from the event stream of one request."""

    def __init__(self) -> None:
        self.agents: dict[str, AgentTiming] = {}
        self.msg_to_agent: dict[str, str] = {}
        self.first_token_at: float | None = None
        self.error: str | None = None

    def _agent(self, agent_id: str) -> AgentTiming:
        return self.agents.setdefault(agent_id, AgentTiming(agent_id))

    def handle(self, frame: dict[str, Any]) -> bool:
        """Returns True once every started agent has finished."""
        kind = frame.get("type")
        data = frame.get("data") or {}
        now = time.perf_counter()
        agent_id = data.get("agent_id")

        if kind == "agent.started" and agent_id:
            timing = self._agent(agent_id)
            timing.started_at = timing.started_at or now
            if data.get("message_id"):
                self.msg_to_agent[data["message_id"]] = agent_id

        elif kind == "message.created":
            message = data.get("message") or {}
            author = message.get("author") or {}
            if author.get("type") == "agent" and message.get("id"):
                self.msg_to_agent[message["id"]] = author.get("id", "")

        elif kind == "message.delta":
            owner = self.msg_to_agent.get(data.get("message_id", ""))
            piece = data.get("content") or ""
            if owner and piece:
                timing = self._agent(owner)
                if timing.first_token_at is None:
                    timing.first_token_at = now
                if self.first_token_at is None:
                    self.first_token_at = now
                timing.chars += len(piece)

        elif kind == "agent.completed" and agent_id:
            timing = self._agent(agent_id)
            timing.completed_at = now
            timing.metrics = data.get("metrics") or {}
            timing.citations = data.get("citations") or []

        elif kind == "agent.failed" and agent_id:
            timing = self._agent(agent_id)
            timing.completed_at = now
            timing.failed = f"{data.get('code')}: {data.get('error')}"

        elif kind == "message.completed":
            message = data.get("message") or {}
            owner = self.msg_to_agent.get(message.get("id", ""))
            if owner:
                self._agent(owner).content = message.get("content", "")

        elif kind == "error":
            self.error = json.dumps(data)[:400]

        return bool(self.agents) and all(
            a.completed_at is not None for a in self.agents.values()
        )


async def execute_run(
    *,
    ws_url: str,
    channel: str,
    agents: list[str],
    prompt: str,
    timeout_s: float,
) -> tuple[RunObserver, float, float]:
    """Send one request, stream it to completion. Returns (observer, t0, t1)."""
    observer = RunObserver()
    async with websockets.connect(ws_url, max_size=None) as socket:
        await socket.send(json.dumps(
            {"type": "client.subscribe", "data": {"channel_id": channel}}
        ))
        mention = " ".join(f"@{a}" for a in agents)
        started = time.perf_counter()
        await socket.send(json.dumps({
            "type": "client.send_message",
            "data": {"channel_id": channel, "content": f"{mention} {prompt}"},
        }))
        while True:
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                observer.error = f"timeout after {timeout_s}s"
                return observer, started, time.perf_counter()
            frame = json.loads(raw)
            data = frame.get("data") or {}
            message = data.get("message") or {}
            if (data.get("channel_id") or message.get("channel_id")) != channel:
                continue
            if observer.handle(frame):
                return observer, started, time.perf_counter()


# --- orchestration ----------------------------------------------------------


def http_json(url: str, *, method: str = "GET", timeout: float = 10.0) -> Any:
    request = urllib.request.Request(url, method=method)  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode())


def set_dispatch_mode(backend: str, mode: str) -> str:
    query = urllib.parse.urlencode({"mode": mode})
    payload = http_json(f"{backend}/config/dispatch-mode?{query}", method="POST")
    return payload["dispatch_mode"]


def sha256_dir(path: Path, pattern: str = "*") -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.glob(pattern))
        if p.is_file()
    }


def preflight(backend: str, vllm: str, *, strict: bool) -> dict[str, Any]:
    """Controls that must hold before any timed run."""
    checks: dict[str, Any] = {}

    try:
        health = http_json(f"{backend}/health")
        checks["backend_healthy"] = health.get("status") == "healthy"
        checks["backend_provider"] = health.get("provider")
        checks["backend_model"] = health.get("model")
    except Exception as exc:
        checks["backend_healthy"] = False
        checks["backend_error"] = str(exc)

    # The experiment is defined on vLLM. Ollama answering would silently
    # change the system under test.
    checks["provider_is_vllm"] = checks.get("backend_provider") == "vllm"

    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        checks["gpu_processes"] = [ln.strip() for ln in out.splitlines() if ln.strip()]
        checks["ollama_on_gpu"] = any("ollama" in ln.lower() for ln in out.splitlines())
    except Exception as exc:
        checks["gpu_processes"] = [f"unavailable: {exc}"]
        checks["ollama_on_gpu"] = None

    checks["drained"] = wait_until_drained(vllm, timeout_s=30)

    problems = []
    if not checks.get("backend_healthy"):
        problems.append("backend is not healthy")
    if not checks.get("provider_is_vllm"):
        problems.append(f"provider is {checks.get('backend_provider')!r}, expected 'vllm'")
    if checks.get("ollama_on_gpu"):
        problems.append("Ollama is holding GPU memory")
    if not checks.get("drained"):
        problems.append("vLLM did not drain to zero")
    checks["problems"] = problems

    if problems and strict:
        raise SystemExit("preflight failed:\n  - " + "\n  - ".join(problems))
    return checks


def run_experiment(args: argparse.Namespace) -> int:
    config = yaml.safe_load((HERE / "benchmark.yaml").read_text(encoding="utf-8"))
    prompts = {p["id"]: p for p in config["prompts"]}
    execution = config["execution"]

    spec = dict(config["sanity"]) if args.sanity else dict(config["conditions"])
    if not args.sanity:
        spec["repetitions"] = spec.pop("repetitions_full")
        spec.pop("repetitions_sanity", None)

    matrix = build_matrix(spec, config["prompts"])
    rng = random.Random(execution["seed"])
    if execution.get("randomise_order", True):
        rng.shuffle(matrix)

    prefix = "DRYRUN" if args.dry_run else "EXP"
    experiment_id = f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    out_dir = RESULTS / experiment_id
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    (out_dir / "responses").mkdir(parents=True)

    print(f"\n  experiment : {experiment_id}")
    print(f"  mode       : {'DRY RUN - no real measurements' if args.dry_run else 'LIVE'}")
    print(f"  runs       : {len(matrix)}")
    print(f"  output     : {out_dir}\n")

    checks = preflight(args.backend, args.vllm, strict=not args.dry_run)
    if checks["problems"]:
        print("  preflight problems (tolerated in dry run):")
        for problem in checks["problems"]:
            print(f"    ! {problem}")

    environment = capture_environment(args.vllm, args.backend)
    environment.update({
        "experiment_id": experiment_id,
        "dry_run": args.dry_run,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "corpus_sha256": sha256_dir(CORPUS, "*.md"),
        "agents_sha256": sha256_dir(AGENTS_D, "*.yaml"),
        "benchmark_config_sha256": hashlib.sha256(
            (HERE / "benchmark.yaml").read_bytes()).hexdigest(),
        "preflight": checks,
        "spec": spec,
        "seed": execution["seed"],
    })
    (out_dir / "env.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    raw_path = out_dir / "raw.jsonl"
    agent_pool = config["experiment"]["agents"]
    channel = config["experiment"]["channel"]
    ws_url = args.backend.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    completed = failed = 0
    for order_index, condition in enumerate(matrix):
        prompt = prompts[condition.prompt_id]
        agents = agent_pool[: condition.agent_count]
        label = (f"[{order_index + 1:>3}/{len(matrix)}] {condition.condition_id:<18}"
                 f" {condition.prompt_id} rep{condition.repetition}")

        record: dict[str, Any] = {
            "run_id": f"{experiment_id}-{order_index:04d}",
            "experiment_id": experiment_id,
            "dry_run": args.dry_run,
            "order_index": order_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair_key": condition.pair_key,
            "condition_id": condition.condition_id,
            "dispatch_mode": condition.dispatch_mode,
            "agent_count": condition.agent_count,
            "user_count": condition.users,
            "repetition": condition.repetition,
            "prompt_id": condition.prompt_id,
            "workload_type": prompt["type"],
            "agents": agents,
            "corpus_id": environment["benchmark_config_sha256"][:12],
            "git_commit": environment.get("git_commit"),
            "model_name": environment.get("model", {}).get("id"),
            "vllm_version": environment.get("vllm_version"),
            "gpu": environment.get("gpu", {}),
        }

        if args.dry_run:
            # Deliberately no timing fields: a dry-run record must never be
            # mistakable for a measurement.
            record["dry_run_note"] = "plumbing check only - no request was sent"
            record["request_success"] = None
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"  {label}  (dry)")
            completed += 1
            continue

        try:
            if execution.get("drain_before_run", True):
                record["drained_before"] = wait_until_drained(
                    args.vllm, timeout_s=execution.get("drain_timeout_s", 60))
            record["dispatch_mode_applied"] = set_dispatch_mode(
                args.backend, condition.dispatch_mode)

            with GpuSampler(args.vllm) as sampler:
                observer, t0, t1 = asyncio.run(execute_run(
                    ws_url=ws_url, channel=channel, agents=agents,
                    prompt=prompt["text"],
                    timeout_s=execution.get("per_run_timeout_s", 300),
                ))
            record["gpu_during_run"] = sampler.summary()

            timings = observer.agents
            record["end_to_end_latency_ms"] = round((t1 - t0) * 1000, 2)
            record["ttft_ms"] = (
                round((observer.first_token_at - t0) * 1000, 2)
                if observer.first_token_at else None
            )
            record["agents_completed"] = sum(
                1 for a in timings.values() if a.failed is None)
            record["agents_failed"] = sum(
                1 for a in timings.values() if a.failed is not None)
            record["per_agent"] = [
                {
                    "agent_id": a.agent_id,
                    "latency_ms": round((a.completed_at - a.started_at) * 1000, 2)
                    if a.started_at and a.completed_at else None,
                    "ttft_ms": round((a.first_token_at - t0) * 1000, 2)
                    if a.first_token_at else None,
                    "chars": a.chars,
                    "input_tokens": a.metrics.get("input_tokens"),
                    "output_tokens": a.metrics.get("output_tokens"),
                    "rounds": a.metrics.get("rounds"),
                    "retrieved_chunks": a.metrics.get("retrieved_chunks"),
                    "retrieval_ms": a.metrics.get("retrieval_ms"),
                    "citation_count": len(a.citations),
                    "failed": a.failed,
                }
                for a in timings.values()
            ]
            record["input_tokens"] = sum(
                (a.metrics.get("input_tokens") or 0) for a in timings.values())
            record["output_tokens"] = sum(
                (a.metrics.get("output_tokens") or 0) for a in timings.values())
            elapsed = max(t1 - t0, 1e-9)
            record["output_tokens_per_second"] = round(
                record["output_tokens"] / elapsed, 2)
            record["request_success"] = (
                observer.error is None and record["agents_failed"] == 0
                and record["agents_completed"] == condition.agent_count
            )
            record["error"] = observer.error

            # Responses live beside the measurements but are scored later, so
            # no judge call can ever land inside a timed window.
            for agent in timings.values():
                target = out_dir / "responses" / f"{record['run_id']}.{agent.agent_id}.json"
                target.write_text(json.dumps({
                    "run_id": record["run_id"],
                    "pair_key": condition.pair_key,
                    "dispatch_mode": condition.dispatch_mode,
                    "agent_id": agent.agent_id,
                    "prompt_id": condition.prompt_id,
                    "prompt": prompt["text"],
                    "required_facts": prompt.get("required_facts", []),
                    "expected_sources": prompt.get("expected_sources", []),
                    "content": agent.content,
                    "citations": agent.citations,
                }, indent=2), encoding="utf-8")

            completed += 1
            print(f"  {label}  {record['end_to_end_latency_ms']:>9.0f} ms"
                  f"  ttft {record['ttft_ms'] or 0:>7.0f} ms"
                  f"  {'ok' if record['request_success'] else 'FAILED'}")

        except Exception as exc:  # one bad run must not end the experiment
            record["request_success"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
            print(f"  {label}  EXCEPTION {type(exc).__name__}: {exc}")

        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    if args.dry_run:
        (out_dir / "DRY_RUN_NOT_REAL_DATA").write_text(
            "This directory is the output of --dry-run.\n"
            "No request was sent and no timing was measured.\n"
            "These records must never be analysed as results.\n", encoding="utf-8")

    print(f"\n  wrote {completed} record(s), {failed} failure(s)")
    print(f"  raw  : {raw_path}")
    print(f"  env  : {out_dir / 'env.json'}\n")
    return 0


# --- setup ------------------------------------------------------------------


def setup(args: argparse.Namespace) -> int:
    """Install benchmark agents and ingest the frozen corpus. Run once."""
    backend_agents = HERE.parents[1] / "backend" / "agents.d"
    backend_agents.mkdir(parents=True, exist_ok=True)
    for src in sorted(AGENTS_D.glob("*.yaml")):
        (backend_agents / src.name).write_bytes(src.read_bytes())
        print(f"  installed agent  {src.name}")
    print("\n  RESTART THE BACKEND so the agents are registered, then:")
    print("    python run_experiment.py --setup --ingest\n")

    if not args.ingest:
        return 0

    import mimetypes
    manifest = []
    for doc in sorted(CORPUS.glob("*.md")):
        body = doc.read_bytes()
        boundary = uuid.uuid4().hex
        parts = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{doc.name}"\r\n'
            f"Content-Type: {mimetypes.guess_type(doc.name)[0] or 'text/markdown'}\r\n\r\n"
        ).encode() + body + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="channel_id"\r\n\r\ngeneral\r\n'
            f"--{boundary}--\r\n"
        ).encode()
        request = urllib.request.Request(  # noqa: S310
            f"{args.backend}/api/documents", data=parts, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            payload = json.loads(response.read().decode())
        document = payload["document"]
        manifest.append({
            "filename": doc.name,
            "document_id": document["id"],
            "sha256": hashlib.sha256(body).hexdigest(),
        })
        print(f"  ingested {doc.name} -> {document['id']}")

    (HERE / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  manifest: {HERE / 'corpus_manifest.json'}")
    print("  wait for every document to reach status 'ready' before measuring\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--sanity", action="store_true",
                        help="the 4-condition, 5-repetition sanity experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="verify plumbing without sending any request")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--ingest", action="store_true",
                        help="with --setup: upload the frozen corpus")
    args = parser.parse_args()

    if args.setup:
        return setup(args)
    if not (args.sanity or args.dry_run):
        parser.error("choose --sanity, --dry-run or --setup")
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
