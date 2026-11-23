"""Channels, messages, and the broadcast hub.

Storage is in-memory and bounded. That is a deliberate first-milestone choice,
not an oversight: Postgres, Redis and a job queue are the right answer once
persistence or multi-process concurrency is actually needed, and adding them
before then buys operational weight and no capability.

The seam is here, though. Everything goes through ``MessageService``, so
swapping the two dicts below for a repository is a change to this file only.
What would need to follow: history pagination, and moving the subscriber hub
onto a pub/sub channel so multiple workers see each other's events.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from app.core.config import settings
from app.core.exceptions import ChannelNotFound, ValidationError
from app.core.logging import get_logger
from app.core.models import Author, AuthorType, Channel, Message, MessageStatus

logger = get_logger(__name__)

Subscriber = Callable[[dict[str, Any]], Awaitable[None]]

#: Messages retained per channel. Older ones are evicted.
MAX_MESSAGES_PER_CHANNEL = 500

DEFAULT_CHANNELS = [
    Channel(
        id="general",
        name="general",
        description="Everything else. All agents are here.",
    ),
    Channel(
        id="research",
        name="research",
        description="Document and web research.",
        agent_ids=["researcher"],
    ),
    Channel(
        id="engineering",
        name="engineering",
        description="Code, debugging and execution.",
        agent_ids=["coder", "researcher"],
    ),
    Channel(
        id="hr",
        name="hr",
        description="Policy questions answered from uploaded HR documents.",
        agent_ids=["hr"],
    ),
]


class MessageService:
    def __init__(self, channels: Iterable[Channel] | None = None) -> None:
        self._channels: dict[str, Channel] = {
            c.id: c for c in (channels if channels is not None else DEFAULT_CHANNELS)
        }
        self._messages: dict[str, deque[Message]] = defaultdict(
            lambda: deque(maxlen=MAX_MESSAGES_PER_CHANNEL)
        )
        self._by_id: dict[str, Message] = {}
        self._subscribers: dict[str, Subscriber] = {}
        self._lock = asyncio.Lock()

    # --- channels ----------------------------------------------------------

    @property
    def channels(self) -> list[Channel]:
        return list(self._channels.values())

    def channel_payload(self) -> list[dict[str, Any]]:
        return [c.model_dump(mode="json") for c in self.channels]

    def get_channel(self, channel_id: str) -> Channel:
        try:
            return self._channels[channel_id]
        except KeyError:
            raise ChannelNotFound(
                f"no channel '{channel_id}'",
                details={"available": sorted(self._channels)},
            ) from None

    def has_channel(self, channel_id: str) -> bool:
        return channel_id in self._channels

    def agents_for_channel(self, channel_id: str, all_agent_ids: list[str]) -> list[str]:
        """Agents available in a channel. Empty ``agent_ids`` means all of them."""
        channel = self.get_channel(channel_id)
        if not channel.agent_ids:
            return list(all_agent_ids)
        return [a for a in channel.agent_ids if a in all_agent_ids]

    # --- messages ----------------------------------------------------------

    def history(self, channel_id: str, limit: int | None = None) -> list[Message]:
        self.get_channel(channel_id)
        messages = list(self._messages[channel_id])
        if limit is not None:
            return messages[-limit:]
        return messages

    def get_message(self, message_id: str) -> Message | None:
        return self._by_id.get(message_id)

    async def add(self, message: Message) -> Message:
        self.get_channel(message.channel_id)
        async with self._lock:
            bucket = self._messages[message.channel_id]
            if len(bucket) == bucket.maxlen:
                evicted = bucket[0]
                self._by_id.pop(evicted.id, None)
            bucket.append(message)
            self._by_id[message.id] = message
        return message

    async def create_user_message(
        self,
        *,
        channel_id: str,
        content: str,
        author: Author | None = None,
        document_ids: list[str] | None = None,
        mentions: list[str] | None = None,
    ) -> Message:
        content = (content or "").strip()
        if not content:
            raise ValidationError("message content is empty")
        if len(content) > settings.max_message_length:
            raise ValidationError(
                f"message exceeds {settings.max_message_length} characters",
                user_message=(
                    f"Messages are limited to {settings.max_message_length} characters."
                ),
            )

        message = Message(
            channel_id=channel_id,
            author=author
            or Author(id="user", type=AuthorType.USER, display_name="You"),
            content=content,
            mentions=mentions or [],
            document_ids=document_ids or [],
            status=MessageStatus.COMPLETED,
        )
        return await self.add(message)

    async def create_agent_message(
        self, *, channel_id: str, author: Author, reply_to: str | None = None
    ) -> Message:
        """Create the empty message an agent will stream into."""
        message = Message(
            channel_id=channel_id,
            author=author,
            content="",
            status=MessageStatus.STREAMING,
            reply_to=reply_to,
        )
        return await self.add(message)

    async def append_delta(self, message_id: str, delta: str) -> None:
        message = self._by_id.get(message_id)
        if message is None:
            logger.warning("delta for unknown message %s", message_id)
            return
        message.content += delta

    async def complete_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        citations: list | None = None,
        tool_invocations: list | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Message | None:
        message = self._by_id.get(message_id)
        if message is None:
            return None
        if content is not None:
            message.content = content
        if citations is not None:
            message.citations = citations
        if tool_invocations is not None:
            message.tool_invocations = tool_invocations
        if metrics is not None:
            message.metrics = metrics
        message.status = MessageStatus.COMPLETED
        return message

    async def fail_message(
        self, message_id: str, *, code: str, error: str
    ) -> Message | None:
        message = self._by_id.get(message_id)
        if message is None:
            return None
        message.status = MessageStatus.FAILED
        message.error = {"code": code, "message": error}
        if not message.content:
            message.content = ""
        return message

    async def add_system_message(self, channel_id: str, content: str) -> Message:
        return await self.add(
            Message(
                channel_id=channel_id,
                author=Author.system(),
                content=content,
                status=MessageStatus.COMPLETED,
            )
        )

    # --- broadcast ---------------------------------------------------------

    def subscribe(self, connection_id: str, callback: Subscriber) -> None:
        self._subscribers[connection_id] = callback

    def unsubscribe(self, connection_id: str) -> None:
        self._subscribers.pop(connection_id, None)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Fan an event out to every connection.

        Concurrent, and one dead socket never blocks or breaks the others - a
        slow client must not stall an agent that is mid-stream.
        """
        if not self._subscribers:
            return
        subscribers = list(self._subscribers.items())
        results = await asyncio.gather(
            *(callback(event) for _, callback in subscribers),
            return_exceptions=True,
        )
        for (connection_id, _), outcome in zip(subscribers, results, strict=True):
            if isinstance(outcome, Exception):
                logger.debug("dropping subscriber %s: %s", connection_id, outcome)
                self.unsubscribe(connection_id)

    def stats(self) -> dict[str, Any]:
        return {
            "channels": len(self._channels),
            "messages": sum(len(bucket) for bucket in self._messages.values()),
            "connections": len(self._subscribers),
        }


_service: MessageService | None = None


def get_message_service() -> MessageService:
    global _service
    if _service is None:
        _service = MessageService()
    return _service


def set_message_service(service: MessageService | None) -> None:
    global _service
    _service = service
