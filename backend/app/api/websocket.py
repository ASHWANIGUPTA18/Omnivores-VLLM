"""The WebSocket endpoint.

One socket per browser tab carries everything live: message creation, token
deltas, tool status, document progress and typing indicators.

Two things this gets right that a naive implementation does not:

**Dispatch does not block the receive loop.** Agent runs are started as tasks,
so the socket keeps reading while a model streams. That is what makes
``client.cancel`` work at all - a loop that awaited the agent inline could not
receive the cancel message that would stop it.

**A failure closes as little as possible.** A bad frame, a rejected message or
a failed agent produces an ``error`` event and the socket stays open. The
connection is only torn down when the transport itself is gone.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.agents.registry import get_agent_registry
from app.api.deps import authorize_websocket, message_rate_limiter
from app.core import events as ev
from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    OmnivoresError,
    RateLimitExceeded,
    ValidationError,
)
from app.core.logging import get_logger, set_request_id
from app.core.models import Author, AuthorType, new_id
from app.services.agent_service import get_agent_service
from app.services.message_service import get_message_service

logger = get_logger(__name__)

router = APIRouter()

#: Close codes.
_CLOSE_POLICY_VIOLATION = 1008
_CLOSE_TRY_AGAIN_LATER = 1013


class Connection:
    """One client socket."""

    def __init__(self, websocket: WebSocket, connection_id: str) -> None:
        self.websocket = websocket
        self.id = connection_id
        self.author = Author(
            id=f"user_{connection_id[:8]}", type=AuthorType.USER, display_name="You"
        )
        #: Agent runs started by this connection, cancelled if it goes away.
        self.tasks: set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()

    async def send(self, event: dict[str, Any]) -> None:
        """Send one event.

        Serialized behind a lock because deltas and document progress can be
        produced concurrently, and interleaved writes corrupt a WebSocket frame.
        """
        if self.websocket.client_state != WebSocketState.CONNECTED:
            raise RuntimeError("socket is closed")
        async with self._send_lock:
            await self.websocket.send_json(event)

    def track(self, task: asyncio.Task) -> None:
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def cancel_all(self) -> None:
        for task in list(self.tasks):
            if not task.done():
                task.cancel()


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, Connection] = {}

    @property
    def count(self) -> int:
        return len(self._connections)

    def add(self, connection: Connection) -> None:
        self._connections[connection.id] = connection

    def remove(self, connection_id: str) -> None:
        connection = self._connections.pop(connection_id, None)
        if connection is not None:
            connection.cancel_all()

    def at_capacity(self) -> bool:
        return self.count >= settings.max_websocket_connections


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, token: str | None = Query(default=None)
) -> None:
    try:
        authorize_websocket(token)
    except AuthenticationError:
        # Reject before the handshake completes so no frames are ever accepted.
        await websocket.close(code=_CLOSE_POLICY_VIOLATION, reason="unauthorized")
        return

    if manager.at_capacity():
        await websocket.close(code=_CLOSE_TRY_AGAIN_LATER, reason="server at capacity")
        logger.warning(
            "refused connection: at capacity (%d)", settings.max_websocket_connections
        )
        return

    await websocket.accept()

    connection = Connection(websocket, new_id("conn"))
    manager.add(connection)

    messages = get_message_service()
    agents = get_agent_service()
    registry = get_agent_registry()

    messages.subscribe(connection.id, connection.send)
    logger.info("client connected", extra={"connection_id": connection.id})

    try:
        await connection.send(
            ev.connection_ready(
                connection.id, messages.channel_payload(), registry.public()
            )
        )

        while True:
            raw = await websocket.receive_text()
            await _handle_frame(raw, connection, messages, agents)

    except WebSocketDisconnect:
        logger.info("client disconnected", extra={"connection_id": connection.id})
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - log and close cleanly
        logger.exception("websocket loop failed", extra={"connection_id": connection.id})
    finally:
        messages.unsubscribe(connection.id)
        manager.remove(connection.id)
        message_rate_limiter.forget(connection.id)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass


async def _handle_frame(raw: str, connection, messages, agents) -> None:
    """Parse and route one inbound frame. Never raises."""
    try:
        frame = json.loads(raw)
    except json.JSONDecodeError:
        await _safe_send(
            connection, ev.error_event("invalid_json", "Frame was not valid JSON.")
        )
        return

    if not isinstance(frame, dict):
        await _safe_send(
            connection, ev.error_event("invalid_frame", "Frame must be an object.")
        )
        return

    event_type = frame.get("type")
    data = frame.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    if event_type not in ev.CLIENT_EVENTS:
        await _safe_send(
            connection,
            ev.error_event(
                "unknown_event",
                f"Unsupported event type '{event_type}'.",
                details={"supported": sorted(ev.CLIENT_EVENTS)},
            ),
        )
        return

    try:
        if event_type == ev.EVT_CLIENT_PING:
            await connection.send(ev.pong(data.get("ts")))

        elif event_type == ev.EVT_CLIENT_SUBSCRIBE:
            await _handle_subscribe(data, connection, messages)

        elif event_type == ev.EVT_CLIENT_TYPING:
            await _handle_typing(data, connection, messages)

        elif event_type == ev.EVT_CLIENT_CANCEL:
            await _handle_cancel(data, connection, agents)

        elif event_type == ev.EVT_CLIENT_SEND_MESSAGE:
            await _handle_send_message(data, connection, messages, agents)

    except OmnivoresError as exc:
        await _safe_send(
            connection,
            ev.error_event(
                exc.code,
                exc.user_message,
                retryable=exc.retryable,
                channel_id=data.get("channel_id"),
                details=exc.details,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the socket
        logger.exception("error handling %s", event_type)
        await _safe_send(
            connection,
            ev.error_event(
                "internal_error",
                "The server hit an unexpected error handling that message.",
                details={"type": type(exc).__name__},
            ),
        )


async def _handle_subscribe(data, connection, messages) -> None:
    channel_id = data.get("channel_id")
    if not channel_id:
        raise ValidationError("channel_id is required")
    messages.get_channel(channel_id)
    limit = int(data.get("limit") or 100)
    history = messages.history(channel_id, limit=max(1, min(limit, 500)))
    await connection.send(ev.channel_history(channel_id, history))


async def _handle_typing(data, connection, messages) -> None:
    channel_id = data.get("channel_id")
    if not channel_id or not messages.has_channel(channel_id):
        return
    if data.get("active", True):
        event = ev.typing_started(
            channel_id, connection.author.id, connection.author.display_name
        )
    else:
        event = ev.typing_stopped(channel_id, connection.author.id)
    await messages.broadcast(event)


async def _handle_cancel(data, connection, agents) -> None:
    message_id = data.get("message_id")
    if not message_id:
        raise ValidationError("message_id is required")
    cancelled = agents.cancel(message_id)
    if not cancelled:
        await connection.send(
            ev.error_event(
                "not_cancellable",
                "That response has already finished.",
                message_id=message_id,
            )
        )


async def _handle_send_message(data, connection, messages, agents) -> None:
    channel_id = data.get("channel_id")
    content = data.get("content") or ""
    document_ids = data.get("document_ids") or []

    if not channel_id:
        raise ValidationError("channel_id is required")
    if not isinstance(document_ids, list):
        document_ids = []

    messages.get_channel(channel_id)

    try:
        message_rate_limiter.check(connection.id)
    except RateLimitExceeded as exc:
        await connection.send(
            ev.error_event(
                exc.code,
                exc.user_message,
                retryable=True,
                channel_id=channel_id,
                details=exc.details,
            )
        )
        return

    message = await messages.create_user_message(
        channel_id=channel_id,
        content=content,
        author=connection.author,
        document_ids=[str(d) for d in document_ids][:20],
    )
    set_request_id(message.id)

    # Echo to everyone, including the sender: the client renders from the
    # server's copy so ordering and ids are never ambiguous.
    await messages.broadcast(ev.message_created(message))

    async def broadcast(event: dict[str, Any]) -> None:
        await messages.broadcast(event)

    # Run agents off the receive loop so cancel and typing still work.
    task = asyncio.create_task(
        _dispatch(agents, message, broadcast), name=f"dispatch-{message.id}"
    )
    connection.track(task)


async def _dispatch(agents, message, broadcast) -> None:
    try:
        await agents.handle_user_message(message, on_event=broadcast)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - the service already reports agent errors
        logger.exception("dispatch failed", extra={"message_id": message.id})
        await broadcast(
            ev.error_event(
                "dispatch_failed",
                "Failed to run the agent for that message.",
                channel_id=message.channel_id,
                message_id=message.id,
            )
        )


async def _safe_send(connection, event: dict[str, Any]) -> None:
    try:
        await connection.send(event)
    except (RuntimeError, WebSocketDisconnect):
        pass
