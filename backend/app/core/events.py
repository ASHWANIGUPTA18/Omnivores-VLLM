"""The WebSocket event protocol.

Every byte that crosses the socket is one of these envelopes. Building them
here rather than scattering ``json.dumps({"type": ...})`` through the codebase
is what lets the frontend be a straightforward reducer over a closed set of
event types.

Envelope shape::

    {"type": "message.delta", "ts": "2026-...", "data": {...}}

Client to server events use the same envelope with the ``EVT_CLIENT_*`` types.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from app.core.models import Citation, Document, Message, ToolInvocation

# --- Server -> client ------------------------------------------------------

EVT_CONNECTION_READY = "connection.ready"
EVT_ERROR = "error"
EVT_PONG = "pong"

EVT_MESSAGE_CREATED = "message.created"
EVT_MESSAGE_DELTA = "message.delta"
EVT_MESSAGE_COMPLETED = "message.completed"
EVT_MESSAGE_FAILED = "message.failed"

EVT_AGENT_STARTED = "agent.started"
EVT_AGENT_THINKING = "agent.thinking"
EVT_AGENT_TOOL_STARTED = "agent.tool_started"
EVT_AGENT_TOOL_COMPLETED = "agent.tool_completed"
EVT_AGENT_COMPLETED = "agent.completed"
EVT_AGENT_FAILED = "agent.failed"

EVT_DOCUMENT_UPLOADED = "document.uploaded"
EVT_DOCUMENT_PROCESSING = "document.processing"
EVT_DOCUMENT_READY = "document.ready"
EVT_DOCUMENT_FAILED = "document.failed"
EVT_DOCUMENT_DELETED = "document.deleted"

EVT_TYPING_STARTED = "typing.started"
EVT_TYPING_STOPPED = "typing.stopped"

EVT_CHANNEL_HISTORY = "channel.history"

# --- Client -> server ------------------------------------------------------

EVT_CLIENT_SEND_MESSAGE = "client.send_message"
EVT_CLIENT_SUBSCRIBE = "client.subscribe"
EVT_CLIENT_TYPING = "client.typing"
EVT_CLIENT_PING = "client.ping"
EVT_CLIENT_CANCEL = "client.cancel"

CLIENT_EVENTS = frozenset(
    {
        EVT_CLIENT_SEND_MESSAGE,
        EVT_CLIENT_SUBSCRIBE,
        EVT_CLIENT_TYPING,
        EVT_CLIENT_PING,
        EVT_CLIENT_CANCEL,
    }
)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(type_: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a raw envelope. Prefer the named helpers below."""
    return {"type": type_, "ts": _ts(), "data": data or {}}


# --- Connection ------------------------------------------------------------


def connection_ready(
    connection_id: str, channels: list[dict[str, Any]], agents: list[dict[str, Any]]
) -> dict[str, Any]:
    return event(
        EVT_CONNECTION_READY,
        {"connection_id": connection_id, "channels": channels, "agents": agents},
    )


def error_event(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    channel_id: str | None = None,
    message_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return event(
        EVT_ERROR,
        {
            "code": code,
            "message": message,
            "retryable": retryable,
            "channel_id": channel_id,
            "message_id": message_id,
            "details": details or {},
        },
    )


def pong(client_ts: Any = None) -> dict[str, Any]:
    return event(EVT_PONG, {"client_ts": client_ts})


# --- Messages --------------------------------------------------------------


def message_created(message: Message) -> dict[str, Any]:
    return event(EVT_MESSAGE_CREATED, {"message": message.model_dump(mode="json")})


def message_delta(message_id: str, channel_id: str, content: str) -> dict[str, Any]:
    return event(
        EVT_MESSAGE_DELTA,
        {"message_id": message_id, "channel_id": channel_id, "content": content},
    )


def message_completed(message: Message) -> dict[str, Any]:
    return event(EVT_MESSAGE_COMPLETED, {"message": message.model_dump(mode="json")})


def message_failed(
    message_id: str, channel_id: str, code: str, error: str, *, retryable: bool = False
) -> dict[str, Any]:
    return event(
        EVT_MESSAGE_FAILED,
        {
            "message_id": message_id,
            "channel_id": channel_id,
            "code": code,
            "error": error,
            "retryable": retryable,
        },
    )


# --- Agent lifecycle -------------------------------------------------------


def agent_started(agent_id: str, channel_id: str, message_id: str) -> dict[str, Any]:
    return event(
        EVT_AGENT_STARTED,
        {"agent_id": agent_id, "channel_id": channel_id, "message_id": message_id},
    )


def agent_thinking(
    agent_id: str, channel_id: str, message_id: str, round_: int
) -> dict[str, Any]:
    return event(
        EVT_AGENT_THINKING,
        {
            "agent_id": agent_id,
            "channel_id": channel_id,
            "message_id": message_id,
            "round": round_,
        },
    )


def agent_tool_started(
    agent_id: str, channel_id: str, message_id: str, invocation: ToolInvocation
) -> dict[str, Any]:
    return event(
        EVT_AGENT_TOOL_STARTED,
        {
            "agent_id": agent_id,
            "channel_id": channel_id,
            "message_id": message_id,
            "invocation": invocation.model_dump(mode="json"),
        },
    )


def agent_tool_completed(
    agent_id: str, channel_id: str, message_id: str, invocation: ToolInvocation
) -> dict[str, Any]:
    return event(
        EVT_AGENT_TOOL_COMPLETED,
        {
            "agent_id": agent_id,
            "channel_id": channel_id,
            "message_id": message_id,
            "invocation": invocation.model_dump(mode="json"),
        },
    )


def agent_completed(
    agent_id: str,
    channel_id: str,
    message_id: str,
    citations: Iterable[Citation],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return event(
        EVT_AGENT_COMPLETED,
        {
            "agent_id": agent_id,
            "channel_id": channel_id,
            "message_id": message_id,
            "citations": [c.model_dump(mode="json") for c in citations],
            "metrics": metrics,
        },
    )


def agent_failed(
    agent_id: str,
    channel_id: str,
    message_id: str,
    code: str,
    error: str,
    *,
    retryable: bool = False,
) -> dict[str, Any]:
    return event(
        EVT_AGENT_FAILED,
        {
            "agent_id": agent_id,
            "channel_id": channel_id,
            "message_id": message_id,
            "code": code,
            "error": error,
            "retryable": retryable,
        },
    )


# --- Documents -------------------------------------------------------------


def document_event(type_: str, document: Document) -> dict[str, Any]:
    return event(type_, {"document": document.public()})


def document_uploaded(document: Document) -> dict[str, Any]:
    return document_event(EVT_DOCUMENT_UPLOADED, document)


def document_processing(document: Document) -> dict[str, Any]:
    return document_event(EVT_DOCUMENT_PROCESSING, document)


def document_ready(document: Document) -> dict[str, Any]:
    return document_event(EVT_DOCUMENT_READY, document)


def document_failed(document: Document) -> dict[str, Any]:
    return document_event(EVT_DOCUMENT_FAILED, document)


def document_deleted(document_id: str, channel_id: str | None) -> dict[str, Any]:
    return event(
        EVT_DOCUMENT_DELETED, {"document_id": document_id, "channel_id": channel_id}
    )


# --- Typing / history ------------------------------------------------------


def typing_started(channel_id: str, author_id: str, display_name: str) -> dict[str, Any]:
    return event(
        EVT_TYPING_STARTED,
        {
            "channel_id": channel_id,
            "author_id": author_id,
            "display_name": display_name,
        },
    )


def typing_stopped(channel_id: str, author_id: str) -> dict[str, Any]:
    return event(
        EVT_TYPING_STOPPED, {"channel_id": channel_id, "author_id": author_id}
    )


def channel_history(channel_id: str, messages: list[Message]) -> dict[str, Any]:
    return event(
        EVT_CHANNEL_HISTORY,
        {
            "channel_id": channel_id,
            "messages": [m.model_dump(mode="json") for m in messages],
        },
    )
