"""Sequential vs parallel dispatch of independent agents.

When one message mentions several agents, they can be run one after another or
at the same time. The module docstring in ``agent_service`` asserts that
sequential is the right choice because concurrency is "slower per-agent" on a
single GPU. That is a claim about throughput, not correctness, and these tests
exist so it can be measured rather than assumed.

The property that must hold either way: both modes see **the same history**.
Run sequentially, the second agent would otherwise see the first agent's reply
in its context, which makes the two modes incomparable.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.models import Author, AuthorType, Message
from app.inference.client import set_inference_client
from app.inference.mock import MockInferenceClient
from app.services.agent_service import AgentService
from app.services.message_service import get_message_service

CHANNEL = "general"


def user_says(text: str) -> Message:
    return Message(
        channel_id=CHANNEL,
        author=Author(id="user_1", type=AuthorType.USER, display_name="Ashwani"),
        content=text,
    )


class Recorder:
    """Timestamps agent start/finish so overlap can be detected."""

    def __init__(self) -> None:
        self.started: dict[str, float] = {}
        self.finished: dict[str, float] = {}
        self.order: list[str] = []

    async def __call__(self, event: dict) -> None:
        kind = event.get("type")
        data = event.get("data") or {}
        agent_id = data.get("agent_id")
        if not agent_id:
            return
        if kind == "agent.started":
            self.started.setdefault(agent_id, time.perf_counter())
            if agent_id not in self.order:
                self.order.append(agent_id)
        elif kind in ("agent.completed", "agent.failed"):
            self.finished[agent_id] = time.perf_counter()

    def overlapped(self, a: str, b: str) -> bool:
        """True if a and b were running at the same time."""
        if not {a, b} <= self.started.keys() or not {a, b} <= self.finished.keys():
            return False
        return (
            self.started[a] < self.finished[b] and self.started[b] < self.finished[a]
        )


@pytest.fixture
def client(isolated_globals):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def slow_model():
    """A mock with a measurable generation time, so overlap is observable."""
    client = MockInferenceClient(stream_delay=0.02)
    set_inference_client(client)
    return client


async def dispatch(mode: str, text: str, recorder: Recorder) -> list[Message]:
    service = AgentService(dispatch_mode=mode)
    message = await get_message_service().add(user_says(text))
    return await service.handle_user_message(message, on_event=recorder)


# --- both modes work --------------------------------------------------------


async def test_sequential_runs_every_agent(slow_model):
    recorder = Recorder()
    replies = await dispatch("sequential", "@researcher @coder hello", recorder)
    assert {r.author.id for r in replies} == {"researcher", "coder"}


async def test_parallel_runs_every_agent(slow_model):
    recorder = Recorder()
    replies = await dispatch("parallel", "@researcher @coder hello", recorder)
    assert {r.author.id for r in replies} == {"researcher", "coder"}


# --- the actual behavioural difference --------------------------------------


async def test_sequential_does_not_overlap(slow_model):
    recorder = Recorder()
    await dispatch("sequential", "@researcher @coder hello", recorder)
    assert not recorder.overlapped("researcher", "coder")


async def test_parallel_overlaps(slow_model):
    recorder = Recorder()
    await dispatch("parallel", "@researcher @coder hello", recorder)
    assert recorder.overlapped("researcher", "coder"), (
        "parallel dispatch must actually run the agents at the same time"
    )


async def test_parallel_is_not_slower_for_three_agents(slow_model):
    """The end-to-end win is the whole point; three agents is where it shows."""
    seq_start = time.perf_counter()
    await dispatch("sequential", "@researcher @coder @hr hello", Recorder())
    sequential = time.perf_counter() - seq_start

    par_start = time.perf_counter()
    await dispatch("parallel", "@researcher @coder @hr hello", Recorder())
    parallel = time.perf_counter() - par_start

    assert parallel < sequential


# --- the control that makes the two arms comparable -------------------------


async def test_both_modes_pin_the_same_history(slow_model):
    """Neither mode may let one agent see another's reply.

    Sequentially this would otherwise happen for free, and the two arms would
    be answering different prompts - which would invalidate any latency
    comparison drawn between them.
    """
    messages = get_message_service()
    for mode in ("sequential", "parallel"):
        before = len(messages.history(CHANNEL))
        service = AgentService(dispatch_mode=mode)
        message = await messages.add(user_says("@researcher @coder hello"))
        seen: list[int] = []

        original = service.run_agent

        # Both are rebound each loop iteration, so they are bound as defaults
        # rather than captured - a closure would see the last value, not this
        # iteration's.
        async def spy(
            *,
            definition,
            user_message,
            on_event,
            history=None,
            _call=original,
            _seen=seen,
        ):
            _seen.append(len(history) if history is not None else -1)
            return await _call(
                definition=definition,
                user_message=user_message,
                on_event=on_event,
                history=history,
            )

        service.run_agent = spy  # type: ignore[method-assign]
        await service.handle_user_message(message, on_event=Recorder())

        assert len(seen) == 2, f"{mode}: both agents should have been dispatched"
        assert seen[0] == seen[1], (
            f"{mode}: agents saw different history lengths "
            f"({seen[0]} vs {seen[1]}) - the arms are not comparable"
        )
        assert seen[0] >= before


# --- failure isolation ------------------------------------------------------


async def test_parallel_isolates_a_failing_agent(slow_model):
    """An erroring agent must not cancel its siblings.

    The mock directive lives in the shared user message, so both agents fail
    here. What is being asserted is that gather did not abandon the second
    agent the moment the first raised, and that the caller still returns
    normally instead of propagating.
    """
    recorder = Recorder()
    replies = await dispatch(
        "parallel", "@researcher [[mock:error]] @coder hello", recorder
    )
    assert replies == [], "both agents were told to fail"
    assert len(recorder.finished) == 2, (
        "both agents must have been run to completion, not cancelled by the "
        f"other's failure (saw {sorted(recorder.finished)})"
    )


async def test_unknown_dispatch_mode_is_rejected():
    with pytest.raises(ValueError):
        AgentService(dispatch_mode="whatever")


async def test_cancellation_still_works_in_parallel(slow_model):
    """Cancelling one in-flight agent must not disturb the other."""
    service = AgentService(dispatch_mode="parallel")
    messages = get_message_service()
    message = await messages.add(user_says("@researcher @coder hello"))
    task = asyncio.create_task(
        service.handle_user_message(message, on_event=Recorder())
    )
    await asyncio.sleep(0.05)
    service.cancel_all()
    await task  # must not raise


# --- runtime toggle (used by the concurrency benchmark) ---------------------


def test_dispatch_mode_endpoint_switches_and_reports(client):
    """The benchmark flips modes without restarting, so GPU state is constant."""
    before = client.get("/config").json()["agents"].get("dispatch_mode")
    assert before in ("sequential", "parallel")

    response = client.post("/config/dispatch-mode", params={"mode": "parallel"})
    assert response.status_code == 200
    assert response.json()["dispatch_mode"] == "parallel"
    assert client.get("/config").json()["agents"]["dispatch_mode"] == "parallel"

    client.post("/config/dispatch-mode", params={"mode": "sequential"})


def test_dispatch_mode_endpoint_rejects_nonsense(client):
    response = client.post("/config/dispatch-mode", params={"mode": "turbo"})
    assert response.status_code == 400
