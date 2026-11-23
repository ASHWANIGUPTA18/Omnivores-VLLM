#!/usr/bin/env python3
"""Prove that two simultaneous questions are answered in parallel.

Opens two independent WebSocket connections, fires a different question down
each at the same instant, and reports when the first token arrived on each and
how much the two answer streams overlapped in time.

Serial execution would show the second answer starting only after the first
finished, and an overlap near zero.

IMPORTANT: the two runs use DIFFERENT CHANNELS on purpose. The backend's
MessageService.broadcast() fans every event to every subscriber - the frontend
filters by channel - so a socket also receives the other run's token deltas.
Every frame here is therefore filtered by channel_id, or the measurement would
be nonsense.

    pip install websockets
    python scripts/concurrency-test.py
    python scripts/concurrency-test.py --url ws://localhost:8000/ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")


def channel_of(kind: str, data: dict) -> str | None:
    """Pull channel_id out of a server frame, whatever its shape."""
    # message.created / message.completed nest the whole Message object.
    if isinstance(data.get("message"), dict):
        return data["message"].get("channel_id")
    return data.get("channel_id")


class Run:
    def __init__(self, label: str, channel: str, question: str) -> None:
        self.label = label
        self.channel = channel
        self.question = question
        self.sent_at = 0.0
        self.first_token_at = 0.0
        self.done_at = 0.0
        self.chars = 0
        self.tool_calls = 0
        self.error: str | None = None

    @property
    def ttft(self) -> float:
        return self.first_token_at - self.sent_at if self.first_token_at else 0.0

    @property
    def total(self) -> float:
        return self.done_at - self.sent_at if self.done_at else 0.0


async def one(url: str, agent: str, run: Run, gate: asyncio.Event) -> None:
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(
            {"type": "client.subscribe", "data": {"channel_id": run.channel}}
        ))

        # Both sockets are connected and subscribed before either sends, so the
        # two questions really do start together.
        await gate.wait()

        run.sent_at = time.perf_counter()
        await ws.send(json.dumps({
            "type": "client.send_message",
            "data": {"channel_id": run.channel, "content": f"@{agent} {run.question}"},
        }))

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=180)
            except asyncio.TimeoutError:
                run.error = "timed out after 180s"
                run.done_at = time.perf_counter()
                return

            frame = json.loads(raw)
            kind = frame.get("type")
            data = frame.get("data") or {}

            # Drop everything belonging to the other run.
            if channel_of(kind, data) != run.channel:
                continue

            if kind == "message.delta":
                piece = data.get("content") or ""
                if piece:
                    if not run.first_token_at:
                        run.first_token_at = time.perf_counter()
                    run.chars += len(piece)

            elif kind == "agent.tool_started":
                run.tool_calls += 1

            elif kind in ("agent.completed", "message.completed"):
                run.done_at = time.perf_counter()
                return

            elif kind in ("agent.failed", "message.failed", "error"):
                run.error = json.dumps(data)[:300]
                run.done_at = time.perf_counter()
                return


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--agent", default="researcher",
                    help="must be a member of BOTH channels below")
    ap.add_argument("--channel-a", default="research")
    ap.add_argument("--channel-b", default="engineering")
    ap.add_argument("--token", default="", help="API_KEY, if the backend has one")
    args = ap.parse_args()

    if args.channel_a == args.channel_b:
        sys.exit("channel-a and channel-b must differ: the backend broadcasts "
                 "every event to every socket, so same-channel runs cannot be "
                 "told apart.")

    url = f"{args.url}?token={args.token}" if args.token else args.url
    runs = [
        Run("A", args.channel_a,
            "In exactly three sentences, explain what a vector database is."),
        Run("B", args.channel_b,
            "In exactly three sentences, explain what continuous batching is."),
    ]
    gate = asyncio.Event()

    tasks = [asyncio.create_task(one(url, args.agent, r, gate)) for r in runs]
    await asyncio.sleep(1.0)  # let both sockets connect and subscribe
    gate.set()
    await asyncio.gather(*tasks)

    print()
    print(f"{'run':<5}{'channel':<13}{'ttft':>9}{'total':>10}{'chars':>8}{'tools':>7}")
    print("-" * 52)
    for r in runs:
        if r.error:
            print(f"{r.label:<5}{r.channel:<13}  FAILED: {r.error}")
        else:
            print(f"{r.label:<5}{r.channel:<13}{r.ttft:>8.2f}s{r.total:>9.2f}s"
                  f"{r.chars:>8}{r.tool_calls:>7}")

    if any(r.error for r in runs):
        return 1

    a, b = runs
    overlap = min(a.done_at, b.done_at) - max(a.sent_at, b.sent_at)
    longest = max(a.total, b.total)
    combined = a.total + b.total
    wall = max(a.done_at, b.done_at) - min(a.sent_at, b.sent_at)
    ratio = overlap / longest if longest else 0.0

    print()
    print(f"  overlap           {overlap:.2f}s of a {longest:.2f}s longest run  ({ratio:.0%})")
    print(f"  wall clock        {wall:.2f}s")
    print(f"  serial would be   {combined:.2f}s")
    print()

    if ratio > 0.5:
        print("  PARALLEL - both answers streamed at the same time.")
        return 0
    print("  SERIALIZED - the second answer waited for the first. Investigate.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
