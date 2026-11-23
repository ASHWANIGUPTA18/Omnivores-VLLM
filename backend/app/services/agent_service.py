"""Routing a user message to agents and streaming the reply.

Responsibilities:

* parse @mentions and decide who answers,
* enforce channel membership (an agent not in a channel does not answer there),
* create the placeholder message the agent streams into,
* forward the runtime's events and text deltas to the broadcast hub,
* keep the stored message and the streamed message in agreement.

Multiple mentioned agents run **sequentially** by default. Two agents streaming
into the same channel at once is harder to read, and the assumption has been
that on a single-GPU vLLM deployment it is also slower per-agent than doing
them in turn.

That assumption is testable, so ``AGENT_DISPATCH=parallel`` selects a
concurrent path instead. Both modes pin the channel history to a single
snapshot taken before any agent starts: run sequentially the second agent would
otherwise see the first one's reply, and the two modes would be answering
different prompts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from app.agents.mentions import parse_mentions
from app.agents.registry import AgentDefinition, AgentRegistry, get_agent_registry
from app.agents.runtime import AgentRuntime, default_runtime
from app.core import events as ev
from app.core.config import settings
from app.core.exceptions import OmnivoresError
from app.core.logging import get_logger, set_request_id
from app.core.models import Author, AuthorType, Message, MessageStatus
from app.services.message_service import MessageService, get_message_service

logger = get_logger(__name__)

EventHook = Callable[[dict[str, Any]], Awaitable[None]]

#: Deltas are coalesced above this many characters to keep the socket from
#: carrying one frame per word. The first few flush immediately so the user
#: sees output the moment generation starts.
_COALESCE_THRESHOLD = 16
_IMMEDIATE_FLUSHES = 3


class _DeltaBuffer:
    """Coalesces text increments into fewer WebSocket frames."""

    def __init__(self, flush: Callable[[str], Awaitable[None]]) -> None:
        self._flush = flush
        self._pending: list[str] = []
        self._size = 0
        self._flushes = 0

    async def add(self, text: str) -> None:
        self._pending.append(text)
        self._size += len(text)
        if self._flushes < _IMMEDIATE_FLUSHES or self._size >= _COALESCE_THRESHOLD:
            await self.flush()

    async def flush(self) -> None:
        if not self._pending:
            return
        payload = "".join(self._pending)
        self._pending.clear()
        self._size = 0
        self._flushes += 1
        await self._flush(payload)


class AgentService:
    def __init__(
        self,
        *,
        runtime: AgentRuntime | None = None,
        agent_registry: AgentRegistry | None = None,
        message_service: MessageService | None = None,
        dispatch_mode: str | None = None,
    ) -> None:
        self._runtime = runtime or default_runtime
        self._agents = agent_registry
        self._messages = message_service
        mode = dispatch_mode or settings.agent_dispatch
        if mode not in ("sequential", "parallel"):
            raise ValueError(
                f"dispatch_mode must be 'sequential' or 'parallel', got {mode!r}"
            )
        self.dispatch_mode = mode
        #: message_id -> running task, so a client can cancel a generation.
        self._running: dict[str, asyncio.Task] = {}

    @property
    def agents(self) -> AgentRegistry:
        return self._agents or get_agent_registry()

    @property
    def messages(self) -> MessageService:
        return self._messages or get_message_service()

    # --- routing -----------------------------------------------------------

    def resolve_targets(self, message: Message) -> tuple[list[AgentDefinition], list[str]]:
        """Decide which agents answer. Returns (agents, unknown_mentions)."""
        parse = parse_mentions(message.content, self.agents.ids)
        if not parse.agents:
            return [], parse.unknown

        available = set(
            self.messages.agents_for_channel(message.channel_id, self.agents.ids)
        )
        targets: list[AgentDefinition] = []
        for agent_id in parse.agents:
            if agent_id not in available:
                logger.info(
                    "ignoring @%s - not a member of #%s", agent_id, message.channel_id
                )
                continue
            targets.append(self.agents.get(agent_id))
        return targets, parse.unknown

    # --- dispatch ----------------------------------------------------------

    async def handle_user_message(
        self, message: Message, *, on_event: EventHook
    ) -> list[Message]:
        """Run every agent the message addresses. Returns their replies."""
        targets, unknown = self.resolve_targets(message)

        if unknown:
            names = ", ".join("@" + n for n in unknown)
            available = ", ".join("@" + a for a in self.agents.ids)
            note = await self.messages.add_system_message(
                message.channel_id,
                f"No agent named {names}. Available in this channel: {available}.",
            )
            await on_event(ev.message_created(note))

        if not targets:
            return []

        # One snapshot, shared by every agent in this turn. Taken here rather
        # than inside run_agent so that sequential and parallel dispatch give
        # the agents exactly the same context - without this, the second
        # sequential agent would see the first one's reply and the two modes
        # could not be compared.
        history = self.messages.history(message.channel_id)

        if self.dispatch_mode == "parallel":
            return await self._dispatch_parallel(targets, message, history, on_event)
        return await self._dispatch_sequential(targets, message, history, on_event)

    async def _dispatch_sequential(
        self,
        targets: list[AgentDefinition],
        message: Message,
        history: list[Message],
        on_event: EventHook,
    ) -> list[Message]:
        replies: list[Message] = []
        for definition in targets:
            reply = await self.run_agent(
                definition=definition,
                user_message=message,
                on_event=on_event,
                history=history,
            )
            if reply is not None:
                replies.append(reply)
        return replies

    async def _dispatch_parallel(
        self,
        targets: list[AgentDefinition],
        message: Message,
        history: list[Message],
        on_event: EventHook,
    ) -> list[Message]:
        """Run every agent at once, isolating failures.

        ``return_exceptions`` keeps one agent's crash from cancelling its
        siblings, matching the sequential path where each agent fails on its
        own. Replies are returned in mention order, not completion order, so
        the caller sees a stable result either way.
        """
        outcomes = await asyncio.gather(
            *(
                self.run_agent(
                    definition=definition,
                    user_message=message,
                    on_event=on_event,
                    history=history,
                )
                for definition in targets
            ),
            return_exceptions=True,
        )

        replies: list[Message] = []
        for definition, outcome in zip(targets, outcomes, strict=True):
            if isinstance(outcome, asyncio.CancelledError):
                continue
            if isinstance(outcome, BaseException):
                logger.error(
                    "agent %s failed during parallel dispatch: %s",
                    definition.id,
                    outcome,
                )
                continue
            if outcome is not None:
                replies.append(outcome)
        return replies

    async def run_agent(
        self,
        *,
        definition: AgentDefinition,
        user_message: Message,
        on_event: EventHook,
        history: list[Message] | None = None,
    ) -> Message | None:
        """Run one agent and stream its reply into the channel.

        ``history`` is the pinned snapshot from ``handle_user_message``. It is
        optional so that direct callers keep working, in which case the current
        history is read instead.
        """
        channel_id = user_message.channel_id
        author = Author(
            id=definition.id,
            type=AuthorType.AGENT,
            display_name=definition.display_name,
            avatar=definition.avatar,
        )

        reply = await self.messages.create_agent_message(
            channel_id=channel_id, author=author, reply_to=user_message.id
        )
        set_request_id(reply.id)

        await on_event(ev.typing_started(channel_id, definition.id, definition.display_name))
        # The client needs the placeholder before deltas can be applied to it.
        await on_event(ev.message_created(reply))

        async def flush_delta(text: str) -> None:
            await self.messages.append_delta(reply.id, text)
            await on_event(ev.message_delta(reply.id, channel_id, text))

        buffer = _DeltaBuffer(flush_delta)

        if history is None:
            history = self.messages.history(channel_id)

        task = asyncio.create_task(
            self._runtime.run(
                definition=definition,
                channel_id=channel_id,
                message_id=reply.id,
                user_message=user_message,
                history=history,
                document_ids=user_message.document_ids,
                on_event=on_event,
                on_delta=buffer.add,
            ),
            name=f"agent-{definition.id}-{reply.id}",
        )
        self._running[reply.id] = task

        try:
            result = await task
        except asyncio.CancelledError:
            await buffer.flush()
            await self.messages.fail_message(
                reply.id, code="cancelled", error="Generation was cancelled."
            )
            await on_event(
                ev.message_failed(
                    reply.id, channel_id, "cancelled", "Generation was cancelled."
                )
            )
            await on_event(ev.typing_stopped(channel_id, definition.id))
            return None
        finally:
            self._running.pop(reply.id, None)

        await buffer.flush()
        await on_event(ev.typing_stopped(channel_id, definition.id))

        if not result.ok:
            error: OmnivoresError = result.error  # type: ignore[assignment]
            await self.messages.fail_message(
                reply.id, code=error.code, error=error.user_message
            )
            # Preserve any partial text: half an answer plus a clear error beats
            # a message that silently vanishes.
            stored = self.messages.get_message(reply.id)
            if stored is not None and stored.content:
                stored.status = MessageStatus.FAILED
            await on_event(
                ev.agent_failed(
                    definition.id,
                    channel_id,
                    reply.id,
                    error.code,
                    error.user_message,
                    retryable=error.retryable,
                )
            )
            await on_event(
                ev.message_failed(
                    reply.id,
                    channel_id,
                    error.code,
                    error.user_message,
                    retryable=error.retryable,
                )
            )
            return None

        # The runtime's accumulated content is authoritative: streamed deltas
        # can be missing the final flush of a round, and this keeps the stored
        # message exactly equal to what the model produced.
        completed = await self.messages.complete_message(
            reply.id,
            content=result.content,
            citations=result.citations,
            tool_invocations=result.tool_invocations,
            metrics=result.metrics.to_dict() if result.metrics else None,
        )
        if completed is not None:
            await on_event(ev.message_completed(completed))
        return completed

    # --- cancellation ------------------------------------------------------

    def cancel(self, message_id: str) -> bool:
        task = self._running.get(message_id)
        if task is None or task.done():
            return False
        task.cancel()
        logger.info("cancelled generation for %s", message_id)
        return True

    def cancel_all(self) -> int:
        count = 0
        for task in list(self._running.values()):
            if not task.done():
                task.cancel()
                count += 1
        return count

    @property
    def running_count(self) -> int:
        return len(self._running)


_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService()
    return _service


def set_agent_service(service: AgentService | None) -> None:
    global _service
    _service = service
